import pytorch_lightning as L

from collections import defaultdict

import torch.nn as nn

from wmpgnn.lightning_module.lightning_helper import *
from wmpgnn.util.pruners import edge_pruning, true_node_pruning
from wmpgnn.util.functions import acc_four_class
from wmpgnn.performance.plotter import *
from wmpgnn.performance.reco_accuracy import obtain_reco_accuracy
from wmpgnn.reconstruction.reconstruction import reco_event


class DFEILightningModule(L.LightningModule):
    def __init__(self, model, optimizer_class, optimizer_params, configs, pos_weights):
        super().__init__()
        self.version = None
        self.save_hyperparameters({
            **configs,
            "pos_weights": make_loggable(pos_weights)
        })

        self.signal = "_".join(configs["evaluate"]["sample"])
        if configs["evaluate"]["over_write"] != "None":
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
        self.sig_df, self.evt_df = None, None

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
        return self.optimizer_class(self.model.parameters(), **self.optimizer_params)

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

        if self.configs["LCA"]:
            y_LCA = batch[('tracks', 'to', 'tracks')].y.to(torch.int64)
            outputs[('tracks', 'to', 'tracks')].lca = outputs[('tracks', 'to', 'tracks')].edges
            loss["LCA"] = self.lca_criterion(outputs[('tracks', 'to', 'tracks')].lca, y_LCA)
            log["LCA_loss"].append(loss["LCA"].item())
            acc_LCA = acc_four_class(outputs[('tracks', 'to', 'tracks')].lca, y_LCA)
            for key, values in acc_LCA.items():
                log[key].append(values)
        if self.configs["node_prune"]:
            y_nodes = (batch["tracks"].ft != 1).to(torch.float32).unsqueeze(-1)
        if self.configs["edge_prune"]:
            y_edges = batch[('tracks', 'to', 'tracks')].y > 0
            y_edges = y_edges.to(torch.float32).unsqueeze(-1)
        if self.configs["pv_asso"]:
            y_pv_asso = batch[("tracks", "to", "pvs")].y.to(torch.float32)
            pv_filter = batch[('tracks', 'pvs')].filter == 1

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

        combined_loss = loss["LCA"] + loss["t_nodes"] + 33 * loss["tt_edges"] + loss["pv_asso"]

        # Apply reco
        if mode == "test":
            # check pv association
            if self.configs["pv_asso"]:
                ntracks = torch.unique(outputs[("tracks", "to", "pvs")]["edge_index"][0]).shape[0]
                npvs = torch.unique(outputs[("tracks", "to", "pvs")]["edge_index"][1]).shape[0]
                y_pv = torch.argmax(outputs[("tracks", "to", "pvs")].y.view(ntracks, npvs), dim=1)
                pred_pv = torch.argmax(block.edge_weights[('tracks', 'to', 'pvs')].view(ntracks, npvs), dim=1)
                min_ip_pv = torch.argmin(minip.view(ntracks, npvs), dim=1)
                if npvs not in log["pv_total"].keys():
                    log["pv_corr_ml"][npvs], log["pv_corr_ip"][npvs], log["pv_total"][npvs] = [], [], []
                log["pv_corr_ml"][npvs].append(torch.sum(y_pv == pred_pv).item())
                log["pv_corr_ip"][npvs].append(torch.sum(y_pv == min_ip_pv).item())
                log["pv_total"][npvs].append(ntracks)

            # reconstruction with cuts
            if self.configs["node_prune"]:
                node_selbool = block.node_weights["tracks"].squeeze() > self.node_prune
                edge_mask = true_node_pruning(node_selbool, outputs, "tracks", [('tracks', 'to', 'tracks')])
                if self.configs["pv_asso"]:
                    y_pv, pred_pv, min_ip_pv = y_pv[node_selbool], pred_pv[node_selbool], min_ip_pv[node_selbool]
                edge_selbool = block.edge_weights[('tracks', 'to', 'tracks')].squeeze()[edge_mask] > self.edge_prune
            else:
                edge_selbool = block.edge_weights[('tracks', 'to', 'tracks')].squeeze() > self.edge_prune

            if self.configs["edge_prune"]:
                edge_pruning(edge_selbool, outputs, ('tracks', 'to', 'tracks'))

            if self.configs["pv_asso"]:
                pv_asso_des = {"true": y_pv, "pred": pred_pv, "minIP": min_ip_pv, "npvs": npvs}
            else:
                pv_asso_des = None
            self.sig_df, self.evt_df = reco_event(outputs, batch_idx, self.configs, self.signal,
                                                  self.sig_df, self.evt_df, pv_des=pv_asso_des)

        """Logging"""
        log = loss_logging(log, loss, self.configs, mode="DFEI")

        log["combined_loss"].append(combined_loss.item())
        return combined_loss

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

    def on_test_epoch_end(self):
        if self.version is None:
            self.version = self.logger.version
        self.sig_df.to_csv(f'{self.log_dir}/DFEI/version_{self.version}/signal_reco_df_{self.signal}.csv', index=False)
        self.evt_df.to_csv(f'{self.log_dir}/DFEI/version_{self.version}/event_reco_df_{self.signal}.csv', index=False)
        if self.configs["LCA"]:
            obtain_reco_accuracy(self.sig_df, self.version, self.signal, self.log_dir, model="DFEI")
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
            plot_pv_missasso(self.tst_log, self.version, self.signal, log_dir=self.log_dir)
            plot_sig_pv_missasso(self.sig_df, self.version, self.signal, log_dir=self.log_dir)
