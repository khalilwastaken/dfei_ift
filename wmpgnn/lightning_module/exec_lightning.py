from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

import torch
torch.set_float32_matmul_precision("high")

from wmpgnn.analysis.trainer_helper import *
from wmpgnn.model.model import DFEI_HGNN, FT_HGNN
from wmpgnn.lightning_module.dfei_lightning_module import DFEILightningModule
from wmpgnn.lightning_module.ift_lightning_module import IFTLightningModule

seed_everything(42, workers=True)

def load_module(configs, pos_weights, model, dfei_model=None, is_train=True):

    # Checking if need to load from cpt
    load_from_cpt = configs[model]["cpt"]
    if isinstance(load_from_cpt, int):
        bis_model = get_bis_model(load_from_cpt, model)
    elif isinstance(load_from_cpt, str):
        bis_model = load_from_cpt
    else:
       bis_model = "None"
    configs[model]["cpt"] = bis_model

    if model == "DFEI":
        model = DFEI_HGNN(configs["DFEI"])
        if load_from_cpt == "None":
            module = DFEILightningModule(
                model=model,
                optimizer_class=torch.optim.Adam,
                optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
                configs=configs,
                pos_weights=pos_weights,
                is_train=is_train
            )
        else:
            print("Loading from checkpoint")
            print(bis_model)
            print("=" * 30)
            module = DFEILightningModule.load_from_checkpoint(
                checkpoint_path=bis_model,
                model=model,
                pos_weights=pos_weights,
                optimizer_class=torch.optim.Adam,
                optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
                configs=configs,
                is_train=is_train
            )
    elif model == "IFT":
        model = FT_HGNN(configs["IFT"])
        if load_from_cpt == "None":
            module = IFTLightningModule(
                model=model,
                dfei_model=dfei_model,
                optimizer_class=torch.optim.Adam,
                optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
                configs=configs,
                pos_weights=pos_weights,
                is_train=is_train
            )
        else:
            print("Loading from checkpoint")
            print(bis_model)
            print("=" * 30)
            module = IFTLightningModule.load_from_checkpoint(
                checkpoint_path=bis_model,
                model=model,
                dfei_model=dfei_model,
                pos_weights=pos_weights,
                optimizer_class=torch.optim.Adam,
                optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
                configs=configs,
                is_train=is_train
            )
    else:
        raise ValueError("Invalid model")
    return module


def training(module, trn_loader, val_loader, configs, model="DFEI"):
    #module = torch.compile(module)

    monitoring_loss = "val_combined_loss" if model =="DFEI" else "val_ft_loss"

    early_stopping = EarlyStopping(
        monitor=monitoring_loss,
        verbose=True,
        mode="min",
        patience=10,
    )

    best_model_callback = ModelCheckpoint(
        filename=f"best-{{epoch:02d}}-{{{monitoring_loss}:.2f}}",
        monitor=monitoring_loss,
        mode="min",
        save_top_k=5
    )

    log_dir = "lightning_logs"

    tb_logger = TensorBoardLogger(save_dir=log_dir, name=model)
    csv_logger = CSVLogger(save_dir=log_dir, name=model, version=tb_logger.version)
    configs = configs[model]["settings"]
    trainer = Trainer(
        logger=[csv_logger, tb_logger],
        max_epochs=configs["epochs"],
        accelerator="auto",
        devices=configs["ngpu"],
        strategy="auto",
        callbacks=[early_stopping, best_model_callback],
        precision="32",
        accumulate_grad_batches=configs["gacc"],
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        benchmark=True,
        #limit_train_batches=1,
        #limit_val_batches=1,
        #limit_test_batches=1,
    )

    """Start training"""
    trainer.fit(module, trn_loader, val_loader)
    _ = trainer.logger.version
    return trainer

def evaluate(trainer, module, tst_loader):
    if trainer is None:
        trainer = Trainer(
            default_root_dir=f'lightning_logs/version_0',  # save the eval stuff in the dir of the model
        )
    trainer.test(module, dataloaders=tst_loader)


