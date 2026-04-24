import pytorch_lightning as L

from collections import defaultdict
import copy

import torch.nn as nn

from wmpgnn.lightning_module.lightning_helper import *
from wmpgnn.util.pruners import edge_pruning, true_node_pruning
from wmpgnn.reconstruction.reconstruction import EventReconstruction
from wmpgnn.performance.reco_accuracy import obtain_reco_accuracy
from wmpgnn.performance.plotter import *
from wmpgnn.performance.tagging_power import analyze_tagging_power
from wmpgnn.performance.plot_results import process_ft
from wmpgnn.calibration.calibration import *


class IFTLightningModule(L.LightningModule):
    def __init__(self, model, dfei_model, optimizer_class, optimizer_params, configs, pos_weights):
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

        self.dfei_model = [dfei_model]  # doesnt get saved in ckpt
        if self.dfei_model[0] is not None:
            for param in self.dfei_model[0].parameters():
                param.requires_grad = False
        self.dfei_use_pid = configs['DFEI']["use_pid"] if "DFEI" in configs else "None"
        self.ift_use_pid = configs["IFT"]["use_pid"]

        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params

        if self.configs["frag"]:
            self.frag_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights["frag"])
        if self.configs["FT"]:
            self.ft_criterion = nn.CrossEntropyLoss(weight=pos_weights["FT"])

        self.trn_log, self.val_log = init_logs(configs)
        self.tst_log = init_logs(configs, mode="test")
        # init event reconstruction class
        self.evt_reco = EventReconstruction(configs)

        # Pruning threshold for reco
        self.edge_prune = configs["inference"]["edge_prune_thr"]
        self.node_prune = configs["inference"]["node_prune_thr"]

        self.log_dir = configs["log_dir"]

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        return self.optimizer_class(self.model.parameters(), **self.optimizer_params)

    def shared_step(self, batch, batch_idx, log, mode="train"):
        optimizers = self.optimizers()
        loss = init_loss(self.device)

        # Adding lca information to edges
        if self.dfei_model[0] is not None:  # lca from dfei model, overwrites from pv asso
            dfei_input = copy.deepcopy(batch)
            if self.dfei_use_pid == "realistic":
                dfei_input["tracks"].x = torch.cat([dfei_input["tracks"].x, dfei_input["tracks"].real_pid], dim=1)
            elif self.dfei_use_pid == "true":
                dfei_input["tracks"].x = torch.cat([dfei_input["tracks"].x, dfei_input["tracks"].pid], dim=1)
            self.dfei_model[0] = self.dfei_model[0].to(self.device)
            dfei_outputs = self.dfei_model[0](dfei_input)
            lca = dfei_outputs[("tracks", "to", "tracks")].edges
        elif "lca" in batch[("tracks", "to", "tracks")]:  # using the information from the pv asso model
            lca = batch[("tracks", "to", "tracks")].lca
        else:  # using truth information
            lca = torch.nn.functional.one_hot(batch[("tracks", "tracks")].y.to(torch.long), num_classes=4).to(
                torch.float32)
        lca_score = torch.argmax(lca, dim=1).unsqueeze(1)
        batch[("tracks", "tracks")].edges = torch.cat([batch[("tracks", "tracks")].edges, lca_score], dim=1)

        # Adding pid information to nodes, here again realistic or not
        if self.ift_use_pid == "realistic":
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].real_pid], dim=1)
        elif self.ift_use_pid == "true":
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)

        # Forward pass of IFt model
        outputs_ft = self.model(batch)
        outputs_ft[("tracks", "to", "tracks")].lca = lca

        y_ft = batch['tracks'].ft
        selbool = y_ft != 1  # this is actually a good question if one should use ft truth or the predicted from DFEI
        ift_loss = self.ft_criterion(outputs_ft["tracks"].x[selbool], y_ft[selbool])
        loss["ft_nodes"] += ift_loss

        # Starting evaluating the performance
        if mode == "test":
            ft_des = torch.softmax(outputs_ft["tracks"].x, dim=1)
            if "frag" in outputs_ft["tracks"]:
                frag_selbool = outputs_ft["tracks"].frag != -1  # this does not need to exist
                frag_in_evt = outputs_ft["tracks"].frag[frag_selbool]
                frag_pid = outputs_ft["part_ids"][frag_selbool]

            # Attach reco information to graph
            # Node weights
            if self.dfei_model[0] is not None:
                outputs_ft["node_weights"] = self.dfei_model[0]._blocks[-1].node_weights["tracks"].squeeze()
            elif "pred_y" in outputs_ft["tracks"]:
                outputs_ft["node_weights"] = outputs_ft["tracks"].pred_y
            else:
                outputs_ft["node_weights"] = (outputs_ft["tracks"].ft != 1).int()
            # Edge weights
            if self.dfei_model[0] is not None:
                outputs_ft["edge_weights"] = self.dfei_model[0]._blocks[-1].edge_weights[
                    ('tracks', 'to', 'tracks')].squeeze()
            elif "pred_y" in outputs_ft[("tracks", "tracks")]:
                outputs_ft["edge_weights"] = outputs_ft[("tracks", "tracks")].pred_y
            else:
                outputs_ft["edge_weights"] = (lca_score.squeeze() != 0).int()

            if "frag" in outputs_ft["tracks"]:  # how in the world does this make sense?
                outputs_ft["frag_y"] = frag_in_evt
                outputs_ft["frag_pid"] = frag_pid
            self.evt_reco.reconstruct_heavyhadrons(outputs_ft, ft_des=ft_des)

        log = loss_logging(log, loss, self.configs, mode="IFT")
        return ift_loss

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
            self.log(f"train_{key}", val, prog_bar=(key == "ft_loss"), on_epoch=True, on_step=False)
        self.trn_log = defaultdict(list)

    def on_validation_epoch_end(self):
        avg_losses = epoch_end_loggable(self.val_log)
        for key, val in avg_losses.items():
            self.log(f"val_{key}", val, prog_bar=(key == "ft_loss"), on_epoch=True, on_step=False)
        self.val_log = defaultdict(list)

    def on_test_epoch_end(self):
        if self.version is None:
            self.version = self.logger.version
        sig_df, evt_df = self.evt_reco.collect_results()
        sig_df.to_csv(f'{self.log_dir}/IFT/version_{self.version}/signal_reco_df_{self.signal}.csv', index=False)
        evt_df.to_csv(f'{self.log_dir}/IFT/version_{self.version}/event_reco_df_{self.signal}.csv', index=False)
        obtain_reco_accuracy(sig_df, self.version, self.signal, self.log_dir, model="IFT")
        """# Removing heavy hadron daughters of B since they are classified as signal (Ds in Bs->Dspi for example)
        if "Bs" in self.signal:
            sig_id = 531
        elif "Bd" in self.signal:
            sig_id = 511
        else:
            ValueError("Currently undefined")
        sig_selbool = sig_df["SigMatch"] == 1
        sig_id_selbool = np.abs(sig_df["B_id"]) != sig_id
        sig_df = sig_df[~(sig_selbool * sig_id_selbool)]"""
        if self.configs["FT"]:
            process_ft(self.tst_log, sig_df, self.version, self.signal, log_dir=self.log_dir)
            analyze_tagging_power(sig_df, self.version, self.signal, log_dir=self.log_dir)
            # Only consider event which are whitened
            if "is_whiten" in sig_df.keys():
                whiten = self.signal + "__is_whiten"
                whiten_df = sig_df[sig_df["is_whiten"] == 1]
                obtain_reco_accuracy(whiten_df, self.version, whiten, self.log_dir, model="IFT")
                process_ft(self.tst_log, whiten_df, self.version, whiten, log_dir=self.log_dir)
                analyze_tagging_power(whiten_df, self.version, whiten, log_dir=self.log_dir)

                # create calib root file
                create_calib_root(whiten_df, self.version, whiten, log_dir=self.log_dir)
