import glob
import re
import yaml

import torch

from typing import Dict

from wmpgnn.data_loader.weights_calculator import transform_pos_weight
from wmpgnn.model.model import *
# standard
from wmpgnn.lightning_module.dfei_lightning_module import DFEILightningModule
from wmpgnn.lightning_module.ift_lightning_module import IFTLightningModule
"""# domain adapt
from wmpgnn.lightning_module.dfei_domain_adapt_lightning_module import DFEIADLightningModule
# data path
from wmpgnn.lightning_module.dfei_data_lightning_module import DFEILightningModuleData
from wmpgnn.lightning_module.ift_data_lightning_module import IFTLightningModuleData"""


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
        module, ckpt = load_module(dfei_hparams, pos_weights)
        if configs["settings"]["dfei_model_name"] != "None":
            checkpoint = torch.load(configs["settings"]["dfei_model_name"])
        else:
            checkpoint = torch.load(ckpt)
        module.load_state_dict(checkpoint["state_dict"])
        model = module.model
    else:
        print("No DFEI model specified. Information from pv asso/truth is used as a replacement")
        model = None
    return configs, model


def get_bis_model(version: int, configs: Dict, mode: str) -> str:
    # find the model with the best performance in the checkpoints
    model = configs["model"]
    log_dir = configs["log_dir"]
    files = glob.glob(f"{log_dir}/{model}/version_{version}/checkpoints/*best-epoch*.ckpt")
    if mode == "bis":
        if "DFEI" in model:
            pattern = re.compile(r"val_combined_loss=([\d.]+)")
        elif "IFT" in model:
            pattern = re.compile(r"val_ft_loss=([\d.]+)")
        else:
            raise ValueError(f"undefined model: {model}")
        model = min(files, key=lambda s: float(pattern.search(s).group(1)[:-1]))
    elif mode == "None":
        pattern = re.compile(r"best-epoch=([\d]+)")
        model = max(files, key=lambda s: float(pattern.search(s).group(1)))
    else:
        model = f"{log_dir}/{model}/version_{version}/checkpoints/{mode}"
    return model


def load_module(configs: Dict, pos_weights: Dict, dfei_model=None, mode="simulation"):
    model_name = configs["model"]
    if model_name == "DFEI":
        """if configs["settings"].get("domain_adapt"):
            raise NotImplementedError
            #model = DFEI_DA_HGNN(configs[model_name])
        else:"""
        model = DFEI_HGNN(configs[model_name])
    elif model_name == "IFT":
        model = FT_HGNN(configs[model_name])
    else:
        raise NotImplementedError

    # Checking if need to load from cpt
    load_from_cpt = configs[model_name]["cpt"]
    if isinstance(configs[model_name]["cpt"], int):  # adjusting if passed an int
        bis_model = get_bis_model(load_from_cpt, configs, configs[model_name]["cpt_name"])
    else:
        bis_model = None

    lr = float(configs["settings"]["lr"])
    weight_decay = float(configs["settings"]["weight_decay"])
    if model_name == "DFEI":
        """if configs["settings"].get("domain_adapt") and mode != 'eval':
            module = DFEIADLightningModule(
                model=model,
                optimizer_class=torch.optim.Adam,
                optimizer_params={"lr": lr, "weight_decay": weight_decay},
                configs=configs,
                pos_weights=pos_weights,
            )
        else:"""
        module = DFEILightningModule(
            model=model,
            optimizer_class=torch.optim.Adam,
            optimizer_params={"lr": lr, "weight_decay": weight_decay},
            configs=configs,
            pos_weights=pos_weights,
        )
    elif model_name == "IFT":
        module = IFTLightningModule(
            model=model,
            dfei_model=dfei_model,
            optimizer_class=torch.optim.Adam,
            optimizer_params={"lr": lr, "weight_decay": weight_decay},
            configs=configs,
            pos_weights=pos_weights,
        )
    else:
        raise NotImplementedError

    return module, bis_model
