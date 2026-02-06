from tqdm import tqdm

import yaml
import time
import copy
import glob

from multiprocessing.pool import ThreadPool
from functools import partial

import torch
import pytorch_lightning as L
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from wmpgnn.data_loader.weights_calculator import transform_pos_weight
from wmpgnn.data_loader.helper import *
from wmpgnn.lightning_module.exec_lightning import load_module


def check_data(evt):
    for key in evt["tracks"].keys():
        if evt["tracks"][key].shape[0] == 0:
            return False
    for key in evt["pvs"].keys():
        if evt["pvs"][key].shape[0] == 0:
            return False
    for key in evt[("tracks", "to", "tracks")].keys():
        if evt[("tracks", "to", "tracks")][key].shape[0] == 0:
            return False
    for key in evt[("tracks", "to", "pvs")].keys():
        if evt[("tracks", "to", "pvs")][key].shape[0] == 0:
            return False
    return True


class pv_asso_module(L.LightningModule):
    # Model split an event to a single pp collision based on DFEI decision
    def __init__(self, model, configs):
        super().__init__()
        self.model = model
        self.configs = configs
        self.node_thrs = configs["DFEI_pv_asso"]["settings"]["node_prune_thr"]
        self.use_pid = configs["DFEI_pv_asso"]["use_pid"]

    @torch.no_grad()
    def forward(self, batch):
        self.model.eval()
        pv_asso_data = []
        if self.use_pid:
            if self.use_pid == "realistic":
                batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].real_pid], dim=1)
            else:
                batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)

        original_data = copy.deepcopy(batch)

        # Obtain the predicted quantity from the DFEI model
        outputs = self.model(batch)
        original_data[('tracks', 'to', 'tracks')].lca = outputs[('tracks', 'to', 'tracks')].edges
        original_data["tracks"].pred_y = self.model._blocks[-1].node_weights["tracks"].squeeze()
        original_data[('tracks', 'to', 'tracks')].pred_y = self.model._blocks[-1].edge_weights[
            ('tracks', 'to', 'tracks')].squeeze()

        # here we need to get the threshold
        ntracks = torch.unique(outputs[("tracks", "to", "pvs")]["edge_index"][0]).shape[0]
        npvs = torch.unique(outputs[("tracks", "to", "pvs")]["edge_index"][1]).shape[0]
        pred_pv = torch.argmax(self.model._blocks[-1].edge_weights[('tracks', 'to', 'pvs')].view(ntracks, npvs), dim=1)
        node_selbool = self.model._blocks[-1].node_weights["tracks"].squeeze() >= self.node_thrs
        pv_oi = torch.unique(pred_pv[node_selbool])  # identify to which pv a sig node has an edge
        for pv in pv_oi:
            pv_oi_data = copy.deepcopy(original_data)
            nodes_asso_pv_selbool = pred_pv == pv
            pv_selbool = torch.zeros(pv_oi_data["pvs"].x.shape[0], dtype=torch.bool, device=self.device)
            pv_selbool[pv] = True

            # removes all the nodes associated to a different pv
            true_node_pruning(nodes_asso_pv_selbool, pv_oi_data,
                              "tracks", [('tracks', 'to', 'tracks'), ('tracks', 'to', 'pvs')])
            # lastly we need to remove the other pvs within the event and their edges to the tracks
            true_node_pruning(pv_selbool, pv_oi_data, "pvs", [('tracks', 'to', 'pvs')])
            del pv_oi_data["last_chunk"]
            if check_data(pv_oi_data):  # some data saftey checks
                pv_asso_data.append(pv_oi_data.to('cpu'))
        return pv_asso_data


class true_pv_asso(L.LightningModule):
    def __init__(self, configs):
        self.configs = configs
        # here do selection if either ip or true

    def forward(self, batch):
        pv_asso_data = []

        ntracks = torch.unique(batch[("tracks", "to", "pvs")]["edge_index"][0]).shape[0]
        npvs = torch.unique(batch[("tracks", "to", "pvs")]["edge_index"][1]).shape[0]
        if self.configs["settings"]["pv_model"] == "true":
            pred_pv = torch.argmax(batch[("tracks", "to", "pvs")].y.view(ntracks, npvs), dim=1)
        elif self.configs["settings"]["pv_model"] == "ip":
            pred_pv = torch.argmin(batch[("tracks", "to", "pvs")].edges.view(ntracks, npvs), dim=1)
        else:
            raise NotImplementedError

        node_selbool = batch["tracks"].ft != 1
        pv_oi = torch.unique(pred_pv[node_selbool])  # identify to which pv a sig node has an edge
        for pv in pv_oi:
            pv_oi_data = copy.deepcopy(batch)
            nodes_asso_pv_selbool = pred_pv == pv
            pv_selbool = torch.zeros(pv_oi_data["pvs"].x.shape[0], dtype=torch.bool)
            pv_selbool[pv] = True

            # removes all the nodes associated to a different pv
            true_node_pruning(nodes_asso_pv_selbool, pv_oi_data,
                              "tracks", [('tracks', 'to', 'tracks'), ('tracks', 'to', 'pvs')])
            # lastly we need to remove the other pvs within the event and their edges to the tracks
            true_node_pruning(pv_selbool, pv_oi_data, "pvs", [('tracks', 'to', 'pvs')])
            del pv_oi_data["last_chunk"]
            if check_data(pv_oi_data):  # some data safety checks
                pv_asso_data.append(pv_oi_data.to('cpu'))
        return pv_asso_data


def obtain_pv_model(configs):
    if configs["settings"]["pv_model"] == "true":
        print("Using truth information for association")
        pv_model = true_pv_asso(configs)
    elif configs["settings"]["pv_model"] == "ip":
        print("Using ip for association")
        pv_model = true_pv_asso(configs)
    elif isinstance(configs["settings"]["pv_model"], int):
        print("Using DFEI model version:", configs["settings"]["pv_model"])
        model = "DFEI_pv_asso"
        if 'pythia' in configs['settings']['data_dir']:
            log_dir = 'pythia_logs'
        elif 'LHCb' in configs['settings']['data_dir']:
            log_dir = 'LHCb_logs'
        else:
            raise ValueError("Invalid config")
        hparams_file = f"{log_dir}/DFEI/version_{configs['settings']['pv_model']}/hparams.yaml"
        with open(hparams_file, "r") as file:
            hparams = yaml.safe_load(file)
        configs[model] = hparams["DFEI"]
        configs[model]["cpt"] = configs["settings"]["pv_model"]

        pos_weights = transform_pos_weight(None, None, mode="eval")
        module = load_module(configs, pos_weights, model=model, is_train=False)
        pv_model = pv_asso_module(module.model, configs)
    else:
        raise ValueError("Invalid config")
    return pv_model


def get_trn_val_loaders(configs):
    samples = configs["settings"]["sample"]
    nfiles = {}
    for sample, nfile in zip(samples, configs["settings"]["nfiles"]):
        nfiles[sample] = nfile

    # Getting the PV model
    pv_model = obtain_pv_model(configs)

    nevts = {"training": {}, "validation": {}}
    print("Start reading in the data")
    data_dir = configs["settings"]["data_dir"]
    ncpus = configs["settings"]["ncpu"]

    start = time.time()
    print("Training:")
    load_train_dataset = partial(load_dataset, configs=configs, mode="train_weights", pv_asso_model=pv_model)
    trn_dataset = []
    weights = {}
    for sample in samples:
        nevts["training"][sample] = 0
        trn_paths = sorted(glob.glob(f'{data_dir}/{sample}/trn_data_*'))[:nfiles[sample]]
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
    for sample in samples:
        nevts["validation"][sample] = 0
        val_paths = sorted(glob.glob(f'{data_dir}/{sample}/val_data_*'))[:nfiles[sample]]
        with ThreadPool(processes=ncpus) as pool:
            results = list(tqdm(pool.imap(load_val_dataset, val_paths), total=len(val_paths),
                                desc=f"Loading {sample} validation dataset"))
        for r in results:
            val_dataset.extend(r)
            nevts["validation"][sample] += len(r)
    end = time.time()
    print(f"data read in, time needed {(end - start):.2f}")
    print(f"Train dataset       : {len(trn_dataset)}")
    print(f"Validation dataset  : {len(val_dataset)}")
    print("=" * 30)

    batch_size = configs["settings"]["batch_size"]
    trn_loader = DataLoader(trn_dataset, batch_size=batch_size,
                            num_workers=ncpus * 2, drop_last=True, shuffle=True)

    # Shuffle the initial dataset of validation as it is currently sorted by the samples
    generator = torch.Generator()
    shuffled_indices = torch.randperm(len(val_dataset), generator=generator).tolist()
    val_dataset_shuffled = Subset(val_dataset, shuffled_indices)
    val_loader = DataLoader(val_dataset_shuffled, batch_size=batch_size,
                            num_workers=ncpus * 2, drop_last=True)

    return trn_loader, val_loader, weights, nevts


def get_tst_loader(configs, model="DFEI"):
    sample = configs["evaluate"]["sample"]
    nfiles = configs["evaluate"]["nfiles"]
    configs = configs[model]
    nevts = {"testing": {sample: 0}}

    # Getting the PV model
    pv_model = obtain_pv_model(configs)

    print("Testing:")
    load_tst_dataset = partial(load_dataset, configs=configs, mode="val", pv_asso_model=pv_model)
    tst_dataset = []
    tst_paths = sorted(glob.glob(f'{configs["settings"]["data_dir"]}/{sample}/tst_data_*'))[:nfiles]
    with ThreadPool(processes=configs["settings"]["ncpu"]) as pool:
        results = list(
            tqdm(pool.imap(load_tst_dataset, tst_paths), total=len(tst_paths),
                 desc=f"Loading {sample} test dataset"))
    for r in results:
        tst_dataset.extend(r)
        nevts["testing"][sample] += len(r)

    # Shuffle the initial dataset of validation as it is currently sorted by the samples
    generator = torch.Generator()
    shuffled_indices = torch.randperm(len(tst_dataset), generator=generator).tolist()
    tst_dataset_shuffled = Subset(tst_dataset, shuffled_indices)
    tst_loader = DataLoader(tst_dataset_shuffled, batch_size=1, num_workers=configs["settings"]["ncpu"] * 2,
                            drop_last=True)
    return tst_loader, nevts
