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


class HGNNLightningModule(L.LightningModule):
    def __init__(self, model, optimizer_class, optimizer_params, config):
        super().__init__()
        self.save_hyperparameters({
            **config
        })
        self.model = model
        self.config = config["training"]["infer"]
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params

        if self.config["LCA"]:
            self.LCA_criterion = nn.CrossEntropyLoss()

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

        return loss["LCA"]

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
def training(model, trn_loader, val_loader, config):
    module = HGNNLightningModule(
        model=model,
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
        config=config
    )

    """Start training"""
    trainer = Trainer(
        max_epochs=5,
        accelerator="cpu",
        devices=1,
        precision="32",  # never do 16-mixed
    )

    trainer.fit(module, trn_loader, val_loader)
