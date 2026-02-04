import sys, os
import re

import pandas as pd

import yaml
from optparse import OptionParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from wmpgnn.analysis.trainer_helper import *
from wmpgnn.data_loader.data_loader_helper import load_tst_loader
from wmpgnn.data_loader.weights_calculator import transform_pos_weight
from wmpgnn.performance.plotter import metrics_eval
from wmpgnn.lightning_module.dfei_lightning_module import DFEILightningModule
from wmpgnn.lightning_module.exec_lightning import load_module, training, evaluate

if __name__ == "__main__":
    # python trainer.py  --config  to hparams.yaml
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option("", "--config", type=str, default=None,
                      dest="CONFIG", help="Config file path")
    (option, args) = parser.parse_args()
    if len(args) != 0:
        raise RuntimeError("Got undefined arguments", " ".join(args))

    # Load config file
    with open(option.CONFIG, "r") as file:
        configs = yaml.safe_load(file)

    model = "None"
    if "IFT" in option.CONFIG:  # grabbing information from the path
        model = "IFT"
    elif "DFEI" in option.CONFIG:
        model = "DFEI"
    else:
        model = configs["evaluate"]["model_arch"]
        # load in the hparams file from the model
        if 'pythia' in configs['evaluate']['data_dir']:
            log_dir = 'pythia_logs'
        elif 'LHCb' in configs['evaluate']['data_dir']:
            log_dir = 'LHCb_logs'
        else:
            raise ValueError("Invalid config")
        hparams_file = f"{log_dir}/{model}/version_{configs['evaluate']['model']}/hparams.yaml"
        with open(hparams_file, "r") as file:
            hparams = yaml.safe_load(file)
        configs[model] = hparams[model]

        # overwrite data_dir and ncpu
        configs[model]["settings"]["data_dir"] = configs["evaluate"]["data_dir"]
        configs[model]["settings"]["ncpu"] = configs["evaluate"]["ncpu"]
        configs[model]["cpt"] = configs["evaluate"]["model"]
        # for dfei evaluation necessary
        configs[model]["settings"]["node_prune_thr"] = configs["evaluate"]["node_prune_thr"]
        configs[model]["settings"]["edge_prune_thr"] = configs["evaluate"]["edge_prune_thr"]
    print(f"Evaluation script started of {model}")
    print("=" * 30)

    if model == "IFT" and configs["evaluate"]["dfei_model"] != "None":
        # loading in additional hparams from dfei
        if 'pythia' in configs['evaluate']['data_dir']:
            log_dir = 'pythia_logs'
        elif 'LHCb' in configs['evaluate']['data_dir']:
            log_dir = 'LHCb_logs'
        else:
            raise ValueError("Invalid config")
        hparams_file = f"{log_dir}/DFEI/version_{configs['evaluate']['dfei_model']}/hparams.yaml"
        with open(hparams_file, "r") as file:
            hparams = yaml.safe_load(file)
        configs["DFEI"] = hparams["DFEI"]

    # Loading data
    configs, tst_loader, chunkloader = load_tst_loader(configs, model=model)
    pos_weights = transform_pos_weight(None, None, mode="eval")
    version = configs['evaluate']['model']

    # Getting the DFEI model
    if model == "DFEI" or configs["evaluate"]["dfei_model"] != "None":
        print("DFEI module:")
        module = load_module(configs, pos_weights, model="DFEI", is_train=False)
        dfei_model = module.model
    else:
        dfei_model = None

    if model == "IFT":
        print("IFT module:")
        module = load_module(configs, pos_weights, model="IFT", dfei_model=dfei_model, is_train=False)
    evaluate(None, module, tst_loader=tst_loader, chunkloader=chunkloader)

    metric_path = f"{log_dir}/{model}/version_{version}/metrics.csv"
    df = pd.read_csv(metric_path)
    df = df.groupby('epoch').agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else None).reset_index()
    metrics_eval(df, configs[model]["inference"], version, mode=model, log_dir=log_dir)
