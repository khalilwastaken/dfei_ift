import sys, os
import re

import pandas as pd

import yaml
from optparse import OptionParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from wmpgnn.analysis.trainer_helper import *
from wmpgnn.analysis.weights_calculator import transform_pos_weight
from wmpgnn.analysis.data_loader import get_tst_loaders
from wmpgnn.performance.plotter import metrics_eval
from wmpgnn.lightning_module.dfei_lightning_module import DFEILightningModule
from wmpgnn.lightning_module.exec_lightning import load_module, training, evaluate

if __name__ == "__main__":
    # python trainer.py  --config  ../../config_files/lightning.yaml
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option("", "--config", type=str, default=None,
                      dest="CONFIG", help="Config file path")
    (option, args) = parser.parse_args()
    if len(args) != 0:
        raise RuntimeError("Got undefined arguments", " ".join(args))

    model = "None"
    if "IFT" in option.CONFIG:
        model = "IFT"
    elif "DFEI" in option.CONFIG:
        model = "DFEI"
    else:
        raise RuntimeError("Config file path must contain 'IFT' or 'DFEI'")
    print(f"Evaluation script started of {model}")
    print("=" * 30)

    # Load config file
    with open(option.CONFIG, "r") as file:
        configs = yaml.safe_load(file)

    # Loading data
    tst_loader, nevts = get_tst_loaders(configs, model=model)
    configs[model].update({"num_events": nevts})
    

    # Getting the DFEI model
    pos_weights = transform_pos_weight(None, None, mode="eval")
    print("DFEI module:")
    module = load_module(configs, pos_weights, model="DFEI", is_train=False)
    import pdb; pdb.set_trace()
    version = re.search(r'version_(\d+)', configs[model]["cpt"]).group(1)
    dfei_model = module.model
    if model == "DFEI":
        evaluate(None, module, tst_loader)


    if model == "IFT":
        print("IFT module:")
        module = load_module(configs, pos_weights, model="IFT", dfei_model=dfei_model, is_train=False)
        evaluate(None, module, tst_loader)

    metric_path = f"lightning_logs/{model}/version_{version}/metrics.csv"
    df = pd.read_csv(metric_path)
    df = df.groupby('epoch').agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else None).reset_index()
    metrics_eval(df, configs[model]["inference"], version, configs["evaluate"]["sample"], mode=model)
