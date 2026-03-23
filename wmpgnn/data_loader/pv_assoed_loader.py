from tqdm import tqdm

import yaml
import glob

from multiprocessing.pool import ThreadPool
from functools import partial
import threading

import torch
import pytorch_lightning as L
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from wmpgnn.data_loader.weights_calculator import transform_pos_weight
from wmpgnn.data_loader.helper import *
from wmpgnn.analysis.load_module import load_module


class DFEIPVAssoModule(L.LightningModule):
    # Forward pass of the DFEI model, debatch the batched data to obtain per graph scores and masks
    def __init__(self, model, configs):
        super().__init__()
        self.name = "pv_asso_module"
        self.custom_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.custom_device)
        self.configs = configs
        self.node_thrs = configs["inference"]["node_prune_thr"]
        self.edge_thrs = configs["inference"]["edge_prune_thr"]
        self.use_pid = configs["DFEI"]["use_pid"]
        self.model.eval()
        self.lock = threading.Lock()

    @torch.no_grad()
    def forward(self, batch):
        with self.lock:  # trad looking
            if self.use_pid == "realistic":
                batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].real_pid], dim=1)
            elif self.use_pid == "true":
                batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)
            # Obtain the predicted quantity from the DFEI model
            outputs = self.model(batch.to(self.custom_device))

            lca_score = outputs[('tracks', 'to', 'tracks')].edges
            tracks_pred_y = self.model._blocks[-1].node_weights["tracks"].squeeze()
            tr_tr_pred_y = self.model._blocks[-1].edge_weights[('tracks', 'to', 'tracks')].squeeze()
            tr_pv_pred_y = self.model._blocks[-1].edge_weights[('tracks', 'to', 'pvs')]

            track_batch = batch["tracks"].batch
            pv_batch = batch['pvs'].batch
            tr_tr_edge_idx = batch[('tracks', 'tracks')].edge_index
            tr_pv_edge_idx = batch[('tracks', 'pvs')].edge_index

            n_graphs = track_batch.max().item() + 1
            graph_ids = torch.arange(n_graphs, device=self.custom_device)
            track_masks = track_batch.unsqueeze(1) == graph_ids.unsqueeze(0)  # Shape: [n_tracks, n_graphs]
            precomputed_slices = []
            for i in range(n_graphs):
                track_mask = track_masks[:, i]

                tr_tr_mask = track_masks[tr_tr_edge_idx[0], i] & track_masks[tr_tr_edge_idx[1], i]
                pv_mask = pv_batch == i
                tr_pv_mask = track_masks[tr_pv_edge_idx[0], i] & pv_mask[tr_pv_edge_idx[1]]

                # saving everything transferred to cpu for remaining parallel processing
                graph_tr_tr_pred_y = tr_tr_pred_y[tr_tr_mask]
                tr_tr_edge_selbool = graph_tr_tr_pred_y > self.edge_thrs
                precomputed_slices.append({
                    'tracks_pred_y': tracks_pred_y[track_mask].cpu(),
                    'lca_score': lca_score[tr_tr_mask].cpu(),
                    'tr_tr_pred_y': graph_tr_tr_pred_y.cpu(),
                    # "tr_tr_edge_selbool": tr_tr_edge_selbool.cpu(),
                    'pv_desc': tr_pv_pred_y[tr_pv_mask].cpu(),
                })
            torch.cuda.empty_cache()
            return precomputed_slices


class TruePVAssoModule(L.LightningModule):
    def __init__(self, configs):
        self.name = "ip/true"
        self.configs = configs
        self.custom_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, batch):
        batch.to(self.custom_device)

        if self.configs["settings"]["pv_model"] == "true":
            tr_pv_pred_y = batch[("tracks", "pvs")].y.squeeze()
        elif self.configs["settings"]["pv_model"] == "ip":
            tr_pv_pred_y = batch[("tracks", "pvs")].edges.squeeze()
        else:
            raise NotImplementedError
        tr_tr_pred_y = batch[("tracks", "tracks")].y != 0

        track_batch = batch["tracks"].batch
        pv_batch = batch['pvs'].batch
        tr_tr_edge_idx = batch[('tracks', 'tracks')].edge_index
        tr_pv_edge_idx = batch[('tracks', 'pvs')].edge_index

        n_graphs = track_batch.max().item() + 1
        graph_ids = torch.arange(n_graphs, device=self.custom_device)
        track_masks = track_batch.unsqueeze(1) == graph_ids.unsqueeze(0)  # Shape: [n_tracks, n_graphs]

        precomputed_slices = []
        for i in range(track_batch.max().item() + 1):
            pv_mask = pv_batch == i

            tr_pv_mask = track_masks[tr_pv_edge_idx[0], i] & pv_mask[tr_pv_edge_idx[1]]
            tr_tr_mask = track_masks[tr_tr_edge_idx[0], i] & track_masks[tr_tr_edge_idx[1], i]

            # saving everything transferred to cpu for remaining parallel processing
            precomputed_slices.append({
                "tr_tr_edge_selbool": tr_tr_pred_y[tr_tr_mask].cpu(),
                'pv_desc': tr_pv_pred_y[tr_pv_mask].cpu(),
            })
        batch.to("cpu")
        torch.cuda.empty_cache()
        return precomputed_slices


def obtain_pv_model(configs):
    # this needs to be adapted
    pv_model = configs["settings"]["pv_model"]
    if  pv_model == "true" or pv_model == "ip":
        print(f"Using {pv_model} information for association")
        pv_model = TruePVAssoModule(configs)
    elif isinstance(pv_model, int):
        print("Using DFEI model version:", configs["settings"]["pv_model"], "for PV association")
        log_dir = configs["log_dir"]
        hparams_file = f"{log_dir}/DFEI/version_{configs['settings']['pv_model']}/hparams.yaml"
        with open(hparams_file, "r") as file:
            hparams = yaml.safe_load(file)
        hparams['DFEI']['cpt'] = configs["settings"]["pv_model"]
        pos_weights = transform_pos_weight(None, None, mode="eval")
        module = load_module(hparams, pos_weights)
        pv_model = DFEIPVAssoModule(module.model, hparams)
    else:
        raise ValueError("Invalid config")
    return pv_model


def get_trn_val_loaders(configs):
    data_dir = configs["settings"]["data_dir"]
    # default 8 -> 2 parallel loading data and forward pass, 4 each during association
    ncpus =  int(configs["settings"]["ncpu"] / 4)
    nfiles = get_nfiles(configs["settings"])

    # Getting the PV model
    pv_model = obtain_pv_model(configs)

    nevts = {"training": {}, "validation": {}}

    print("Training:")
    load_train_dataset = partial(load_dataset, configs=configs, mode="train_weights", pv_asso_model=pv_model)
    trn_dataset = []
    weights = {}
    for sample, files in nfiles.items():
        nevts["training"][sample] = 0
        trn_paths = sorted(glob.glob(f'{data_dir}/{sample}/trn_data_*'))[:files]
        with ThreadPool(processes=ncpus) as pool:
            results = list(tqdm(pool.imap(load_train_dataset, trn_paths), total=len(trn_paths),
                                desc=f"Loading {sample} training dataset"))
        for r in results:
            trn_dataset.extend(r[0])
            for key, value in r[1].items():
                if key not in weights:
                    weights[key] = value
                else:
                    weights[key] += value
            nevts["training"][sample] += len(r[0])

    print("Validation:")
    load_val_dataset = partial(load_dataset, configs=configs, mode="val", pv_asso_model=pv_model)
    val_dataset = []
    for sample, files in nfiles.items():
        nevts["validation"][sample] = 0
        val_paths = sorted(glob.glob(f'{data_dir}/{sample}/val_data_*'))[:files]
        with ThreadPool(processes=ncpus) as pool:
            results = list(tqdm(pool.imap(load_val_dataset, val_paths), total=len(val_paths),
                                desc=f"Loading {sample} validation dataset"))
        for r in results:
            val_dataset.extend(r)
            nevts["validation"][sample] += len(r)
    print(f"Train dataset       : {len(trn_dataset)}")
    print(f"Validation dataset  : {len(val_dataset)}")
    print("=" * 30)
    # Creating the dataloaders
    batch_size = configs["settings"]["batch_size"]
    trn_loader = DataLoader(trn_dataset, batch_size=batch_size, num_workers=ncpus * 2, drop_last=True, shuffle=True)

    # Shuffle the initial dataset of validation as it is currently sorted by the samples
    generator = torch.Generator()
    shuffled_indices = torch.randperm(len(val_dataset), generator=generator).tolist()
    val_dataset_shuffled = Subset(val_dataset, shuffled_indices)
    val_loader = DataLoader(val_dataset_shuffled, batch_size=batch_size, num_workers=ncpus * 2, drop_last=True)

    return trn_loader, val_loader, weights, nevts


def get_tst_loader(configs, model="DFEI"):
    data_dir = configs["settings"]["data_dir"]
    # default 8 -> 2 parallel loading data and forward pass, 4 each during association
    ncpus = int(configs["settings"]["ncpu"] / 4)
    nfiles = get_nfiles(configs["evaluate"])

    # Getting the PV model
    pv_model = obtain_pv_model(configs)

    nevts = {"testing": {}}

    print("Testing:")
    load_tst_dataset = partial(load_dataset, configs=configs, mode="val", pv_asso_model=pv_model)
    tst_dataset = []
    for sample, files in nfiles.items():
        nevts["testing"][sample] = 0
        tst_paths = sorted(glob.glob(f'{data_dir}/{sample}/tst_data_*'))[:files]
        with ThreadPool(processes=ncpus) as pool:
            results = list(tqdm(pool.imap(load_tst_dataset, tst_paths), total=len(tst_paths),
                                desc=f"Loading {sample} testing dataset"))
        for r in results:
            tst_dataset.extend(r)
            nevts["testing"][sample] += len(r)

    # Shuffle the initial dataset of validation as it is currently sorted by the samples
    generator = torch.Generator()
    shuffled_indices = torch.randperm(len(tst_dataset), generator=generator).tolist()
    tst_dataset_shuffled = Subset(tst_dataset, shuffled_indices)
    tst_loader = DataLoader(tst_dataset_shuffled, batch_size=512, num_workers=2)
    return tst_loader, nevts
