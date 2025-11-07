import pytorch_lightning as L

from collections import defaultdict
import copy

import torch.nn as nn

from wmpgnn.lightning_module.lightning_helper import *


class IFTLightningModule(L.LightningModule):
    # here we need to add the initial dfei model to pass thorugh and then to ift
    def __init__(self, model, dfei_model, optimizer_class, optimizer_params, configs, pos_weights, is_train=True):
        super().__init__()
        self.is_train = is_train
        if is_train:
            self.version = None
            self.save_hyperparameters({
                **configs,
                "pos_weights": make_loggable(pos_weights)
            })
        else:
            self.version = configs["IFT"]["cpt"].split("_")[0]
        self.signal = configs["evaluate"]["sample"]

        self.configs = configs["IFT"]["inference"]
        self.model = model
        self.dfei_model = dfei_model
        for param in self.dfei_model.parameters():
            param.requires_grad = False
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params

        if self.configs["frag"]:
            self.frag_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights["frag"])
        if self.configs["FT"]:
            self.ft_criterion = nn.CrossEntropyLoss(weight=pos_weights["FT"])

        self.trn_log, self.val_log = init_logs(configs)
        self.tst_log = init_logs(configs, mode="test")
        self.sig_df, self.evt_df = None, None

        # Pruning threshold for reco
        self.edge_prune = configs["IFT"]["settings"]["edge_prune_thr"]
        self.node_prune = configs["IFT"]["settings"]["node_prune_thr"]

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        return self.optimizer_class(self.model.parameters(), **self.optimizer_params)

    def shared_step(self, batch, batch_idx, log, mode="train"):
        optimizers = self.optimizers()
        loss = init_loss(self.device)


        data = copy.deepcopy(batch)
        outputs = self.dfei_model(batch)

        import pdb;
        pdb.set_trace()

        # data = data + outputs

        outputs_ft = self.model(data)

        selbool = y_ft != 1
        ift_loss = self.ft_criterion(outputs_ft["tracks"].x[selbool], y_ft[selbool])
        loss["ft_nodes"] += ift_loss
        combined_loss += ift_loss
        if mode == "test":
            ft_des = torch.softmax(outputs_ft["tracks"].x, dim=1)

        if mode == "test":
            frag_selbool = outputs_ft["tracks"].frag != -1
            frag_in_evt = outputs_ft["tracks"].frag[frag_selbool]
            frag_pid = outputs_ft["part_ids"][frag_selbool]

            if self.configs["node_prune"] or self.configs["edge_prune"]:
                node_selbool = block.node_weights["tracks"].squeeze() > self.node_prune
                edge_mask = true_node_pruning(node_selbool, outputs_ft, "tracks", [('tracks', 'to', 'tracks')])
                ft_des = ft_des[node_selbool]
                edge_selbool = block.edge_weights[('tracks', 'to', 'tracks')].squeeze()[edge_mask] > self.edge_prune
                edge_pruning(edge_selbool, outputs_ft, ('tracks', 'to', 'tracks'))
                outputs_ft[("tracks", "to", "tracks")].lca = outputs_ft[("tracks", "to", "tracks")].lca[edge_mask][
                    edge_selbool]
            outputs_ft["frag_y"] = frag_in_evt
            outputs_ft["frag_pid"] = frag_pid
            self.sig_df, self.evt_df = reco_event(outputs_ft, batch_idx, self.configs, self.signal, self.sig_df,
                                                  self.evt_df, ft_des)
        return ift_loss


    def on_test_start(self) -> None:
        self.sig_df, self.evt_df = init_test_df()

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, self.trn_log, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, self.val_log, mode="val")
        return loss

    def test_step(self, batch, batch_idx):
        _ = self.shared_step(batch, batch_idx, self.tst_log, mode="test")
        return {}

    def on_train_epoch_end(self):
        avg_losses = epoch_end_loggable(self.trn_log)
        for key, val in avg_losses.items():
            self.log(f"train_{key}", val, prog_bar=(key == "combined_loss"), on_epoch=True, on_step=False)
        self.trn_log = defaultdict(list)

    def on_validation_epoch_end(self):
        avg_losses = epoch_end_loggable(self.val_log)
        for key, val in avg_losses.items():
            self.log(f"val_{key}", val, prog_bar=(key == "combined_loss"), on_epoch=True, on_step=False)
        self.val_log = defaultdict(list)