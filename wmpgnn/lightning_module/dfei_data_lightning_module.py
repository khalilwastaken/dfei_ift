import pytorch_lightning as L

import torch.nn as nn

from wmpgnn.lightning_module.lightning_helper import *
from wmpgnn.reconstruction_data.reconstruction import EventReconstruction


class DFEILightningModule(L.LightningModule):
    def __init__(self, model, configs):
        super().__init__()
        self.version = configs["settings"]["model"]

        self.signal = "_".join(configs["evaluate"]["sample"])
        if configs["evaluate"]["over_write"] != "":
            self.signal += "__" + configs["evaluate"]["over_write"]

        self.configs = configs["inference"]
        self.model = model

        # Those function are not used however in the ckpt information is saved, and thus it needs to be loaded in
        if self.configs["LCA"]:
            self.lca_criterion = nn.CrossEntropyLoss(weight=torch.ones(4))
        if self.configs["node_prune"]:
            self.node_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.ones(1))
        if self.configs["edge_prune"]:
            self.edge_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.ones(1))
        if self.configs["pv_asso"]:
            self.pv_asso_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.ones(1))

        # init event reconstruction class
        self.evt_reco = EventReconstruction(configs)

        # Pruning threshold for reco
        self.edge_prune = configs["inference"]["edge_prune_thr"]
        self.node_prune = configs["inference"]["node_prune_thr"]

        self.log_dir = configs["log_dir"]

        self.use_pid = configs["DFEI"]["use_pid"]

    def forward(self, batch):
        return self.model(batch)

    def test_step(self, batch, batch_idx):
        if self.use_pid == "realistic":  # only for pythia
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].real_pid], dim=1)
        elif self.use_pid == "true":  # mc response for lhcb or onehot for pythia
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)

        outputs = self.model(batch)

        # Getting the PV decisions
        if self.configs["pv_asso"]:
            minip = batch[("tracks", "to", "pvs")].edges.flatten()
            pv_asso_des = {"pred": self.model._blocks[-1].edge_weights[('tracks', 'to', 'pvs')].squeeze(), "minIP": minip}
        else:
            pv_asso_des = None
        self.evt_reco.reconstruct_heavyhadrons(outputs, pv_des=pv_asso_des)

    def on_test_epoch_end(self):
        sig_df = self.evt_reco.collect_results()
        sig_df.to_csv(f'{self.log_dir}/IFT/version_{self.version}/signal_reco_data_df_{self.signal}.csv', index=False)
