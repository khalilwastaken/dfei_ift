import glob
import re
import yaml

import torch

from typing import Dict

from wmpgnn.data_loader.weights_calculator import transform_pos_weight
from wmpgnn.model.model import DFEI_HGNN, FT_HGNN
from wmpgnn.lightning_module.dfei_lightning_module import DFEILightningModule
from wmpgnn.lightning_module.ift_lightning_module import IFTLightningModule


def load_dfei_for_ift(configs):
    version = configs['IFT']['dfei_model']
    log_dir = configs['log_dir']
    if version != "None":
        with open(f"{log_dir}/DFEI/version_{version}/hparams.yaml", "r") as file:
            dfei_hparams = yaml.safe_load(file)
        if isinstance(version, int):
            dfei_bis_model = get_bis_model(version, dfei_hparams)
        elif isinstance(version, str):
            dfei_bis_model = version
        else:
            raise RuntimeError(f"Unsupported load_dfei: {type(version)}")
        pos_weights = transform_pos_weight(None, None, mode="eval")
        print("Using DFEI model:", dfei_bis_model)

        dfei_hparams["DFEI"]["cpt"] = dfei_bis_model
        configs["DFEI"] = dfei_hparams["DFEI"]
        module = load_module(dfei_hparams, pos_weights)
        model = module.model
    else:
        print("No DFEI model specified. Information from pv asso/truth is used as a replacement")
        model = None
    return configs, model


def get_bis_model(version: int, configs: Dict) -> str:
    # find the model with the best performance in the checkpoints
    model = configs["model"]
    log_dir = configs["log_dir"]
    files = glob.glob(f"{log_dir}/{model}/version_{version}/checkpoints/*.ckpt")
    if model == "DFEI":
        pattern = re.compile(r"val_combined_loss=([\d.]+)")
    elif model == "IFT":
        pattern = re.compile(r"val_ft_loss=([\d.]+)")
    else:
        raise ValueError(f"undefined model: {model}")
    bis = min(files, key=lambda s: float(pattern.search(s).group(1)[:-1]))
    return bis


def load_module(configs: Dict, pos_weights:Dict, dfei_model=None):
    model = configs["model"]
    # Checking if need to load from cpt
    load_from_cpt = configs[model]["cpt"]
    if isinstance(configs[model]["cpt"], int):  # adjusting if passed an int
        bis_model = get_bis_model(load_from_cpt, configs)
    else:
        bis_model = configs[model]["cpt"]
    lr = float(configs["settings"]["lr"])
    weight_decay = float(configs["settings"]["weight_decay"])
    if model == "DFEI":
        model = DFEI_HGNN(configs[model])
        if load_from_cpt == "None":
            module = DFEILightningModule(
                model=model,
                optimizer_class=torch.optim.Adam,
                optimizer_params={"lr": lr, "weight_decay": weight_decay},
                configs=configs,
                pos_weights=pos_weights,
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
                optimizer_params={"lr": lr, "weight_decay": weight_decay},
                configs=configs,
            )
    elif model == "IFT":
        model = FT_HGNN(configs["IFT"])
        if load_from_cpt == "None":
            module = IFTLightningModule(
                model=model,
                dfei_model=dfei_model,
                optimizer_class=torch.optim.Adam,
                optimizer_params={"lr": lr, "weight_decay": weight_decay},
                configs=configs,
                pos_weights=pos_weights,
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
                optimizer_params={"lr": lr, "weight_decay": weight_decay},
                configs=configs,
            )
    else:
        raise ValueError("Invalid model")
    return module
