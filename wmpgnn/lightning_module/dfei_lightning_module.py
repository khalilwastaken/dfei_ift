import pytorch_lightning as L

from collections import defaultdict

import torch.nn as nn

from wmpgnn.lightning_module.lightning_helper import *
from wmpgnn.reconstruction.reconstruction import EventReconstruction
from wmpgnn.performance.plotter import *
from wmpgnn.performance.reco_accuracy import acc_four_class, obtain_reco_accuracy, acc_pv_asso
from wmpgnn.performance.plot_results import plot_sig_pv_missasso, plot_sig_b_system_pv_missasso


class DFEILightningModule(L.LightningModule):
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

        self.trn_log, self.val_log = init_logs(configs)
        self.tst_log = init_logs(configs, mode="test")
        # init event reconstruction class
        self.evt_reco = EventReconstruction(configs)

        # Pruning threshold for reco
        self.edge_prune = configs["inference"]["edge_prune_thr"]
        self.node_prune = configs["inference"]["node_prune_thr"]

        self.log_dir = configs["log_dir"]

        self.tst_mode = 'MC'

    def forward(self, batch):
        if self.use_pid:
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)
        return self.model(batch)

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_params)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min", factor=0.5, patience=5, min_lr=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, "monitor": "val_combined_loss",
                "interval": "epoch", "frequency": 1, "strict": True,
            },
        }

    def shared_step(self, batch, batch_idx, log):
        optimizers = self.optimizers()
        loss = init_loss(self.device)

        # Forward pass
        outputs = self.forward(batch)

        # Get the y score and LCA performance
        y_nodes, y_edges, pv_filter, y_pv_asso = None, None, None, None
        if self.configs["LCA"]:
            y_LCA = batch[('tracks', 'tracks')].y.to(torch.int64)
            outputs[('tracks', 'tracks')].lca = outputs[('tracks', 'tracks')].edges
            loss["LCA"] = self.lca_criterion(outputs[('tracks', 'tracks')].lca, y_LCA)
            log["LCA_loss"].append(loss["LCA"].item())
            acc_LCA = acc_four_class(outputs[('tracks', 'tracks')].lca, y_LCA)
            log.update({key: log[key] + [values] for key, values in acc_LCA.items()})
        if self.configs["node_prune"]:
            y_nodes = (batch["tracks"].ft != 1).to(torch.float32).unsqueeze(-1)
        if self.configs["edge_prune"]:
            y_edges = batch[('tracks', 'tracks')].y > 0
            y_edges = y_edges.to(torch.float32).unsqueeze(-1)
        if self.configs["pv_asso"]:
            y_pv_asso = batch[("tracks", "pvs")].y.to(torch.float32)
            pv_filter = batch[('tracks', 'pvs')].filter == 1

        # Get the loss of the remaining quantities
        for i, block in enumerate(self.model._blocks):
            if self.configs["node_prune"]:
                loss["t_nodes"] += self.node_criterion(block.node_logits['tracks'], y_nodes)
            if self.configs["edge_prune"]:
                loss["tt_edges"] += self.edge_criterion(block.edge_logits[('tracks', 'tracks')], y_edges)
            if self.configs["pv_asso"]:
                loss["pv_asso"] += self.pv_asso_criterion(block.edge_logits[("tracks", "pvs")][pv_filter],
                                                          y_pv_asso[pv_filter])

        # Combine the loss and log them
        combined_loss = loss["LCA"] + loss["t_nodes"] + 33 * loss["tt_edges"] + loss["pv_asso"]
        log = loss_logging(log, loss, self.configs, mode="DFEI")
        log["combined_loss"].append(combined_loss.item())
        return combined_loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, self.trn_log)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, self.val_log)
        return loss

    def test_step(self, batch, batch_idx):
        # Save minIP value before being passed to model and changed in place
        minip = None
        if self.configs["pv_asso"]:
            minip = batch[("tracks", "to", "pvs")].edges.flatten()

        # Forward pass
        outputs = self.forward(batch)

        # Get score plots on MC
        y_pv_asso, pv_filter = None, None
        if self.tst_mode == 'MC':
            y_nodes, y_edges = None, None
            if self.configs["node_prune"]:
                y_nodes = (batch["tracks"].ft != 1).to(torch.float32).unsqueeze(-1)
            if self.configs["edge_prune"]:
                y_edges = batch[('tracks', 'tracks')].y > 0
                y_edges = y_edges.to(torch.float32).unsqueeze(-1)
            if self.configs["pv_asso"]:
                y_pv_asso = batch[("tracks", "pvs")].y.to(torch.float32)
                pv_filter = batch[('tracks', 'pvs')].filter == 1

            for i, block in enumerate(self.model._blocks):
                if self.configs["node_prune"] and self.configs["plt_nodes"]:
                    get_block_score(self.tst_log, block.node_weights['tracks'].squeeze(), y_nodes, i, var="nodes")
                if self.configs["edge_prune"] and self.configs["plt_edges"]:
                    get_block_score(self.tst_log, block.edge_weights[('tracks', 'tracks')].squeeze(), y_edges, i,
                                    var="edges")
                if self.configs["pv_asso"] and self.configs["plt_pvs"]:
                    get_block_score(self.tst_log, block.edge_weights[("tracks", "pvs")].squeeze(), y_pv_asso, i,
                                    var="pv_asso")

        # Attaching pruning information to graph + PV decision
        if self.configs["LCA"]:
            outputs[('tracks', 'to', 'tracks')].lca = outputs[('tracks', 'tracks')].edges
        if self.configs["node_prune"]:
            outputs["node_weights"] = self.model._blocks[-1].node_weights["tracks"].squeeze()
        if self.configs["edge_prune"]:
            outputs["edge_weights"] = self.model._blocks[-1].edge_weights[('tracks', 'tracks')].squeeze()
        pv_asso_des = None
        if self.configs["pv_asso"]:
            pv_asso_des = {"pred": self.model._blocks[-1].edge_weights[('tracks', 'pvs')].squeeze(), "minIP": minip,
                           "true": y_pv_asso.squeeze(), "pv_filter": pv_filter}

        # Perform the reconstruction
        self.evt_reco.reconstruct_heavyhadrons(outputs, pv_des=pv_asso_des) # pass the mode on self.tst_mode
        return {}

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
        # Grab all the reconstructed events and save them to disk
        sig_df = self.evt_reco.collect_results()
        sig_df.to_csv(f'{self.log_dir}/DFEI/version_{self.version}/signal_reco_df_{self.signal}.csv', index=False)
        # If looking at MC add the additional performance scripts
        if self.tst_mode == 'MC':
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
