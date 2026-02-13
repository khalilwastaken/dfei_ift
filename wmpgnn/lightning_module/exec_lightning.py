from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

import torch

torch.set_float32_matmul_precision("high")
seed_everything(42, workers=True)


def training(module, configs, trn_loader=None, val_loader=None, chunkloader=None):
    # module = torch.compile(module)
    model = configs["model"]

    monitoring_loss = "val_combined_loss" if model == "DFEI" else "val_ft_loss"

    early_stopping = EarlyStopping(
        monitor=monitoring_loss,
        verbose=True,
        mode="min",
        patience=5,
    )

    best_model_callback = ModelCheckpoint(
        filename=f"best-{{epoch:02d}}-{{{monitoring_loss}:.3f}}",
        monitor=monitoring_loss,
        mode="min",
        save_top_k=5
    )

    log_dir = configs["log_dir"]
    tb_logger = TensorBoardLogger(save_dir=log_dir, name=model)
    csv_logger = CSVLogger(save_dir=log_dir, name=model, version=tb_logger.version)

    configs = configs["settings"]
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
    )

    """Start training"""
    if trn_loader is not None and val_loader is not None:
        trainer.fit(module, trn_loader, val_loader)
    elif chunkloader is not None:
        trainer.fit(module, chunkloader)
    else:
        raise NotImplemented
    _ = trainer.logger.version
    return trainer


def evaluate(trainer, module, tst_loader=None, chunkloader=None):
    if trainer is None:
        trainer = Trainer(
            default_root_dir=f'lightning_logs/version_0',  # save the eval stuff in the dir of the model
        )
    if tst_loader is not None:
        trainer.test(module, dataloaders=tst_loader)
    else:
        trainer.test(module, dataloaders=chunkloader)
