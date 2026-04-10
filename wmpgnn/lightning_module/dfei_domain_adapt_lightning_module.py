import pytorch_lightning as L

from collections import defaultdict

import torch.nn as nn
import numpy as np

from wmpgnn.lightning_module.lightning_helper import *
from wmpgnn.reconstruction.reconstruction import EventReconstruction
from wmpgnn.performance.plotter import *
from wmpgnn.performance.reco_accuracy import acc_four_class, obtain_reco_accuracy, acc_pv_asso
from wmpgnn.performance.plot_results import plot_sig_pv_missasso, plot_sig_b_system_pv_missasso


class DFEIADLightningModule(L.LightningModule):
    def __init__(self, model, optimizer_class, optimizer_params, configs, pos_weights):
        super().__init__()
        if "model" in configs["settings"]:
            self.version = configs["settings"]["model"]
        else:
            self.version = None
            self.save_hyperparameters({
                **configs,
                "pos_weights": make_loggable(pos_weights)
            })

        self.signal = "_".join(configs["evaluate"]["sample"])
        if configs["evaluate"]["over_write"] != "":
            self.signal += "__" + configs["evaluate"]["over_write"]

        self.configs = configs["inference"]
        self.model = model
        self.use_pid = configs["DFEI"]["use_pid"]  # str holding what to do with pid information for DFEI

        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params

        # Loss functions + associated inference class for plotting
        if self.configs["LCA"]:
            self.lca_criterion = nn.CrossEntropyLoss(weight=pos_weights["LCA"])
        if self.configs["node_prune"]:
            self.node_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights["nodes"])
        if self.configs["edge_prune"]:
            self.edge_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights["edges"])
        if self.configs["pv_asso"]:
            self.pv_asso_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights["pv_asso"])
        self.da_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights["domain_adapt"])

        self.trn_log, self.val_log = init_logs(configs)
        self.tst_log = init_logs(configs, mode="test")
        # init event reconstruction class
        self.evt_reco = EventReconstruction(configs)

        # Pruning threshold for reco
        self.edge_prune = configs["inference"]["edge_prune_thr"]
        self.node_prune = configs["inference"]["node_prune_thr"]

        self.log_dir = configs["log_dir"]

    def forward(self, batch):
        if self.use_pid == "realistic":  # only for pythia
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].real_pid], dim=1)
        elif self.use_pid == "true":  # mc response for lhcb or onehot for pythia
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)
        return self.model(batch)

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_params)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,  # Reduce LR after 5 epochs of no improvement
            min_lr=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_combined_loss",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
        }

    def shared_step(self, batch, batch_idx, log, mode="train"):
        optimizers = self.optimizers()
        loss = init_loss(self.device)

        # modify batch to include pid information depending on use_pid or not
        if self.use_pid == "realistic":
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].real_pid], dim=1)
        elif self.use_pid == "true":
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)
        if mode == "test" and self.configs["pv_asso"]:
            minip = batch[("tracks", "to", "pvs")].edges.flatten()

        outputs = self.model(batch)

        # Domain adaptation
        loss["domain_adapt"] = self.da_criterion(self.model._domain_adapt.da_score.squeeze(), batch["da_label"])

        # Remaining DFEI
        if self.configs["LCA"]:
            y_LCA = outputs[('tracks', 'to', 'tracks')].y.to(torch.int64)
            outputs[('tracks', 'to', 'tracks')].lca = outputs[('tracks', 'to', 'tracks')].edges
            loss["LCA"] = self.lca_criterion(outputs[('tracks', 'to', 'tracks')].lca, y_LCA)
            log["LCA_loss"].append(loss["LCA"].item())
            acc_LCA = acc_four_class(outputs[('tracks', 'to', 'tracks')].lca, y_LCA)
            for key, values in acc_LCA.items():
                log[key].append(values)
        if self.configs["node_prune"]:
            y_nodes = (outputs["tracks"].ft != 1).to(torch.float32).unsqueeze(-1)
        if self.configs["edge_prune"]:
            y_edges = outputs[('tracks', 'to', 'tracks')].y > 0
            y_edges = y_edges.to(torch.float32).unsqueeze(-1)
        if self.configs["pv_asso"]:
            y_pv_asso = outputs[("tracks", "to", "pvs")].y.to(torch.float32)
            pv_filter = outputs[('tracks', 'pvs')].filter == 1

        for i, block in enumerate(self.model._blocks):
            if self.configs["node_prune"]:
                loss["t_nodes"] += self.node_criterion(block.node_logits['tracks'], y_nodes)
                if mode == "test" and self.configs["plt_nodes"]:
                    get_block_score(log, block.node_weights['tracks'].squeeze(), y_nodes, i, var="nodes")

            if self.configs["edge_prune"]:
                loss["tt_edges"] += self.edge_criterion(block.edge_logits[('tracks', 'to', 'tracks')], y_edges)
                if mode == "test" and self.configs["plt_edges"]:
                    get_block_score(log, block.edge_weights[('tracks', 'to', 'tracks')].squeeze(), y_edges, i,
                                    var="edges")
            if self.configs["pv_asso"]:
                loss["pv_asso"] += self.pv_asso_criterion(block.edge_logits[("tracks", "to", "pvs")][pv_filter],
                                                          y_pv_asso[pv_filter])
                if mode == "test" and self.configs["plt_pvs"]:
                    get_block_score(log, block.edge_weights[("tracks", "to", "pvs")].squeeze(), y_pv_asso, i,
                                    var="pv_asso")

        combined_loss = loss["LCA"] + loss["t_nodes"] + 33 * loss["tt_edges"] + loss["pv_asso"] + loss["domain_adapt"]

        # Apply reco
        if mode == "test":
            # Attaching pruning information to graph
            if self.configs["node_prune"]:
                outputs["node_weights"] = block.node_weights["tracks"].squeeze()
            if self.configs["edge_prune"]:
                outputs["edge_weights"] = block.edge_weights[('tracks', 'to', 'tracks')].squeeze()
            # Getting the PV decisions
            if self.configs["pv_asso"]:
                pv_asso_des = {"pred": block.edge_weights[('tracks', 'to', 'pvs')].squeeze(), "minIP": minip,
                               "true": y_pv_asso.squeeze(), "pv_filter": pv_filter}
            else:
                pv_asso_des = None
            self.evt_reco.reconstruct_heavyhadrons(outputs, pv_des=pv_asso_des)

        """Logging"""
        log = loss_logging(log, loss, self.configs, mode="DFEI")

        log["combined_loss"].append(combined_loss.item())
        return combined_loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, self.trn_log, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, self.val_log, mode="val")
        return loss

    def test_step(self, batch, batch_idx):
        _ = self.shared_step(batch, batch_idx, self.tst_log, mode="test")
        return {}

    def on_train_epoch_start(self):
        progress = self.current_epoch / max(self.trainer.max_epochs, 1)
        alpha = 2 / (1 + np.exp(-10 * progress)) - 1
        self.model._domain_adapt.grl.alpha = alpha
        self.log("grl_alpha", alpha, on_epoch=True)

    def on_train_epoch_end(self):
        avg_losses = epoch_end_loggable(self.trn_log)
        for key, val in avg_losses.items():
            self.log(f"train_{key}", val, prog_bar=(key == "combined_loss"), on_epoch=True, on_step=False)
        self.trn_log = defaultdict(list)

        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=False, on_epoch=True, on_step=False)

    def on_validation_epoch_end(self):
        avg_losses = epoch_end_loggable(self.val_log)
        for key, val in avg_losses.items():
            self.log(f"val_{key}", val, prog_bar=(key == "combined_loss"), on_epoch=True, on_step=False)
        self.val_log = defaultdict(list)

    def on_test_epoch_end(self):
        if self.version is None:
            self.version = self.logger.version
        # grab from the class and save to disk
        sig_df, evt_df = self.evt_reco.collect_results()
        sig_df.to_csv(f'{self.log_dir}/DFEI/version_{self.version}/signal_reco_df_{self.signal}.csv', index=False)
        evt_df.to_csv(f'{self.log_dir}/DFEI/version_{self.version}/event_reco_df_{self.signal}.csv', index=False)
        if self.configs["LCA"]:
            obtain_reco_accuracy(sig_df, self.version, self.signal, self.log_dir, model="DFEI")
        if self.configs["plt_nodes"]:
            for i in range(len(self.model._blocks)):
                plot_weights(self.tst_log[f"sig_nodes_score_{i}"], self.tst_log[f"bkg_nodes_score_{i}"],
                             [f"NN_nodes_{i}_decision", "sig", "bkg"], self.version,
                             model="DFEI", channel=self.signal, log_dir=self.log_dir)
                plot_roc_curve(self.tst_log[f"sig_nodes_score_{i}"], self.tst_log[f"bkg_nodes_score_{i}"],
                               [f"NN_nodes_{i}_roc", "sig", "bkg"], self.version,
                               model="DFEI", channel=self.signal, log_dir=self.log_dir)
        if self.configs["plt_edges"]:
            for i in range(len(self.model._blocks)):
                plot_weights(self.tst_log[f"sig_edges_score_{i}"], self.tst_log[f"bkg_edges_score_{i}"],
                             [f"NN_edges_{i}_decision", "sig", "bkg"], self.version,
                             model="DFEI", channel=self.signal, log_dir=self.log_dir)
                plot_roc_curve(self.tst_log[f"sig_edges_score_{i}"], self.tst_log[f"bkg_edges_score_{i}"],
                               [f"NN_edges_{i}_roc", "sig", "bkg"], self.version,
                               model="DFEI", channel=self.signal, log_dir=self.log_dir)
        if self.configs["plt_pvs"]:
            for i in range(len(self.model._blocks)):
                plot_weights(self.tst_log[f"sig_pv_asso_score_{i}"], self.tst_log[f"bkg_pv_asso_score_{i}"],
                             [f"NN_pv_asso_{i}_decision", "correct", "false"], self.version,
                             model="DFEI", channel=self.signal, log_dir=self.log_dir)
                plot_roc_curve(self.tst_log[f"sig_pv_asso_score_{i}"], self.tst_log[f"bkg_pv_asso_score_{i}"],
                               [f"NN_pv_asso_{i}_roc", "sig", "bkg"], self.version,
                               model="DFEI", channel=self.signal, log_dir=self.log_dir)
            # Get the PV association performance
            log = self.evt_reco.log
            pv_perf = {}
            pv_perf["all_tracks"] = plot_pv_missasso(log["pv_corr_ml"], log["pv_corr_ip"], log["pv_total"], log["npvs"],
                                                     self.version, self.signal, log_dir=self.log_dir)
            pv_sig_tracks = plot_sig_pv_missasso(sig_df, self.version, self.signal, log_dir=self.log_dir)
            pv_perf.update(pv_sig_tracks)
            pv_perf["sig_b_system"] = plot_sig_b_system_pv_missasso(sig_df, self.version, self.signal,
                                                                    log_dir=self.log_dir)
            acc_pv_asso(pv_perf, self.version, self.signal, self.log_dir, model="DFEI")