import yaml

import torch
import pytorch_lightning as L

import threading

from wmpgnn.data_loader.weights_calculator import transform_pos_weight
from wmpgnn.lightning_module.load_module import load_module


def obtain_pv_model(configs):
    pv_model = configs["settings"]["pv_model"]
    if pv_model == "true" or pv_model == "ip":
        print(f"Using {pv_model} information for association")
        pv_model = TruePVAssoModule(configs)
    elif isinstance(pv_model, int):
        print("Using DFEI model version:", configs["settings"]["pv_model"], "for PV association")
        hparams_file = f"{configs['log_dir']}/DFEI/version_{configs['settings']['pv_model']}/hparams.yaml"
        with open(hparams_file, "r") as file:
            hparams = yaml.safe_load(file)
        hparams['DFEI']['cpt'] = configs["settings"]["pv_model"]
        pos_weights = transform_pos_weight(None, None, mode="eval")
        module, ckpt = load_module(hparams, pos_weights)
        if configs["settings"]["pv_model_name"] != "None":
            checkpoint = torch.load(configs["settings"]["pv_model_name"])
        else:
            checkpoint = torch.load(ckpt)
        module.load_state_dict(checkpoint["state_dict"])
        pv_model = DFEIPVAssoModule(module.model, hparams)
    else:
        raise ValueError("Invalid config")
    return pv_model


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
        with self.lock:  # thread looking
            if self.use_pid:
                batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)
            # Obtain the predicted quantity from the DFEI model
            outputs = self.model(batch.to(self.custom_device))

            lca_score = outputs[('tracks', 'to', 'tracks')].edges
            tracks_pred_y = self.model._blocks[-1].node_weights["tracks"].squeeze()
            tr_tr_pred_y = self.model._blocks[-1].edge_weights[('tracks', 'tracks')].squeeze()
            tr_pv_pred_y = self.model._blocks[-1].edge_weights[('tracks', 'pvs')]

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
                #tr_tr_edge_selbool = graph_tr_tr_pred_y > self.edge_thrs
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
