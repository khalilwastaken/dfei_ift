import sys, os
import re

import pandas as pd

import yaml
from optparse import OptionParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from wmpgnn.analysis.trainer_helper import *
from wmpgnn.data_loader.data_loader_helper import load_tst_loader
from wmpgnn.analysis.weights_calculator import transform_pos_weight
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
    if "IFT" in option.CONFIG:
        model = "IFT"
    elif "DFEI" in option.CONFIG:
        model = "DFEI"
    else:
        model = configs["evaluate"]["model_arch"]
        # load in the hparams file from the model
        hparams_file = f"lightning_logs/{model}/version_{configs["evaluate"]["model"]}/hparams.yaml"
        with open(hparams_file, "r") as file:
            hparams = yaml.safe_load(file)
        configs[model] = hparams[model]
        # overwrite data_dir and ncpu
        configs[model]["settings"]["data_dir"] = configs["evaluate"]["data_dir"]
        configs[model]["settings"]["ncpu"] = configs["evaluate"]["ncpu"]
    print(f"Evaluation script started of {model}")
    print("=" * 30)

    # Loading data
    configs, tst_loader, chunkloader = load_tst_loader(configs, model=model)

    # Getting the DFEI model
    pos_weights = transform_pos_weight(None, None, mode="eval")
    print("DFEI module:")
    configs[model]["cpt"] = configs["evaluate"]["model"]
    module = load_module(configs, pos_weights, model="DFEI", is_train=False)
    version = re.search(r'version_(\d+)', configs[model]["cpt"]).group(1)
    dfei_model = module.model

    if model == "IFT":
        print("IFT module:")
        module = load_module(configs, pos_weights, model="IFT", dfei_model=dfei_model, is_train=False)
    evaluate(None, module, tst_loader=tst_loader, chunkloader=chunkloader)

    metric_path = f"lightning_logs/{model}/version_{version}/metrics.csv"
    df = pd.read_csv(metric_path)
    df = df.groupby('epoch').agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else None).reset_index()
    sample = configs["evaluate"]["sample"]
    if configs["evaluate"]["over_write"] != "None":
        sample += "__" + configs["evaluate"]["over_write"]
    metrics_eval(df, configs[model]["inference"], version, sample, mode=model)
