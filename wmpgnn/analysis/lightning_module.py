import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from collections import defaultdict

import torch
from torch import nn

from lightning_module_helper import *

from wmpgnn.util.functions import acc_four_class
from wmpgnn.lightning.plot_helper import *


class HGNNLightningModule(L.LightningModule):
    def __init__(self, model, optimizer_class, optimizer_params, config, pos_weights):
        super().__init__()
        self.save_hyperparameters({
            **config,
            "pos_weights": make_loggable(pos_weights)
        })
        self.model = model
        self.config = config["training"]["infer"]
        self.nFT_layers = config["model"]["GNblocks"]["FTlayers"]
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params

        if self.config["LCA"]:
            self.LCA_criterion = nn.CrossEntropyLoss(weight=pos_weights["LCA"])
        if self.config["node_prune"]:
            self.nodes_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights["nodes"])
        if self.config["edge_prune"]:
            self.edges_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights["edges"])
        if self.config["frag"]:
            self.frag_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights["frag"])
        if self.config["FT"]:
            self.FT_criterion = nn.CrossEntropyLoss(weight=pos_weights["FT"])

        self.trn_log, self.val_log = init_logs(self.config)

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        return self.optimizer_class(self.model.parameters(), **self.optimizer_params)

    def shared_step(self, batch, batch_idx, log):
        loss = {"LCA": 0., "t_nodes": 0., "tt_edges": 0., "tPV_edges": 0., "frag_nodes": 0., "ft_nodes": 0.}
        outputs = self.model(batch)

        if self.config["LCA"]:
            y_LCA = batch[('tracks', 'to', 'tracks')].y.to(torch.int64)
            loss["LCA"] = self.LCA_criterion(outputs[('tracks', 'to', 'tracks')].edges, y_LCA)
            log["LCA_loss"].append(loss["LCA"].item())
            acc_LCA = acc_four_class(outputs[('tracks', 'to', 'tracks')].edges, y_LCA)
            for key, values in acc_LCA.items():
                log[key].append(values)

        if self.config["node_prune"]:
            y_nodes = (batch["tracks"].ft != 0).to(torch.float32).unsqueeze(-1)
        if self.config["edge_prune"]:
            y_edges = batch[('tracks', 'to', 'tracks')].y.to(torch.float32).unsqueeze(-1)
        if self.config["frag"]:
            y_frag = (batch['tracks'].frag != 0).unsqueeze(-1).to(torch.float32)
        if self.config["FT"]:
            y_ft = batch['tracks'].ft

        for i, block in enumerate(self.model._blocks):
            if self.config["node_prune"]:
                loss["t_nodes"] += self.nodes_criterion(block.node_logits['tracks'], y_nodes)
            if self.config["edge_prune"]:
                loss["tt_edges"] += self.edges_criterion(block.edge_logits[('tracks', 'to', 'tracks')], y_edges)
            if i >= len(self.model._blocks) - self.nFT_layers:
                if self.config["frag"]:
                    loss["frag_nodes"] += self.frag_criterion(block.node_logits['frag'], y_frag)
                if self.config["FT"]:
                    loss["ft_nodes"] += self.FT_criterion(block.node_logits['ft'], y_ft)

        combined_loss = loss["LCA"] + loss["t_nodes"] + loss["tt_edges"] + loss["frag_nodes"] + loss["ft_nodes"]

        """temp logging"""
        log_dict["t_nodes_loss"].append(loss["t_nodes"].item())
        log_dict["tt_edges_loss"].append(loss["tt_edges"].item())
        log["frag_loss"].append(loss["frag_nodes"].item())
        log["ft_loss"].append(loss["ft_nodes"].item())
        log_dict["combined_loss"].append(loss.item())
        return combined_loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, self.trn_log)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, self.val_log)
        return loss

    def on_train_epoch_end(self):
        avg_losses = {key: torch.tensor(vals).nanmean(dim=0) for key, vals in self.trn_log.items()}
        for key, val in avg_losses.items():
            self.log(f"train_{key}", val, prog_bar=(key == "combined_loss"), on_epoch=True, on_step=False)
        self.trn_log = defaultdict(list)

    def on_validation_epoch_end(self):
        avg_losses = {key: torch.tensor(vals).nanmean(dim=0) for key, vals in self.val_log.items()}
        for key, val in avg_losses.items():
            self.log(f"val_{key}", val, prog_bar=(key == "combined_loss"), on_epoch=True, on_step=False)
        self.val_log = defaultdict(list)


# Here we define a wrapper to do the training
def training(model, trn_loader, val_loader, config, pos_weights):
    module = HGNNLightningModule(
        model=model,
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
        config=config,
        pos_weights=pos_weights
    )

    early_stopping = EarlyStopping(
        monitor="val_combined_loss",
        verbose=True,
        mode="min",
        patience=10,
    )

    best_model_callback = ModelCheckpoint(
        filename="best-{epoch:02d}-{val_combined_loss:.2f}",
        monitor="val_combined_loss",
        mode="min",
        save_top_k=1
    )

    all_epochs_callback = ModelCheckpoint(
        filename="epoch-{epoch:02d}",
        save_top_k=-1,
        every_n_epochs=1
    )

    log_dir = "lightning_logs"
    experiment_name = None  # default name

    tb_logger = TensorBoardLogger(save_dir=log_dir, name=experiment_name)
    csv_logger = CSVLogger(save_dir=log_dir, name=experiment_name, version=tb_logger.version)

    config = config["training"]
    trainer = Trainer(
        logger=[csv_logger, tb_logger],
        max_epochs=config["epochs"],
        accelerator="gpu",
        devices=config["ngpu"],
        strategy="auto",
        callbacks=[early_stopping, best_model_callback, all_epochs_callback],
        precision="32",
        accumulate_grad_batches=config["gacc"],
        num_sanity_val_steps=1,
        gradient_clip_val=0.5,
        sync_batchnorm=True,
        reload_dataloaders_every_n_epochs=1
    )

    """Start training"""
    trainer.fit(module, trn_loader, val_loader)

    csv_path = os.path.join(log_dir, f"version_{version}", "metrics.csv")
    df = pd.read_csv(csv_path)
    df = df.groupby('epoch').agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else None).reset_index()
    plot_LCA_acc(df, version)
    plot_LCA_loss(df, version)
