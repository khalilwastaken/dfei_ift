import pytorch_lightning as L


class IFTLightningModule(L.LightningModule):
    # here we need to add the initial dfei model to pass thorugh and then to ift
    def __init__(self, model, dfei_model, optimizer_class, configs, pos_weights, is_train=True):
        super().__init__()
        self.is_train = is_train
        if is_train:
            self.version = None
            self.save_hyperparameters({
                **config,
                "pos_weights": make_loggable(pos_weights)
            })
        else:
            self.version = config["IFT"]["cpt"].split("_")[0]
        self.signal = config["evaluate"]["sample"]

        self.configs = configs["IFT"]["inference"]
        self.model = model
        self.dfei_model = dfei_model
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

            if self.config["node_prune"] or self.config["edge_prune"]:
                node_selbool = block.node_weights["tracks"].squeeze() > self.node_prune
                edge_mask = true_node_pruning(node_selbool, outputs_ft, "tracks", [('tracks', 'to', 'tracks')])
                ft_des = ft_des[node_selbool]
                edge_selbool = block.edge_weights[('tracks', 'to', 'tracks')].squeeze()[edge_mask] > self.edge_prune
                edge_pruning(edge_selbool, outputs_ft, ('tracks', 'to', 'tracks'))
                outputs_ft[("tracks", "to", "tracks")].lca = outputs_ft[("tracks", "to", "tracks")].lca[edge_mask][edge_selbool]
            outputs_ft["frag_y"] = frag_in_evt
            outputs_ft["frag_pid"] = frag_pid
            self.sig_df, self.evt_df = reco_event(outputs_ft, batch_idx, self.config, self.signal, self.sig_df,
                                                  self.evt_df, ft_des)
        return ift_loss