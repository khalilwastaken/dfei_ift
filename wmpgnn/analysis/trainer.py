import sys, os, glob

import pandas as pd

import yaml
from optparse import OptionParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from wmpgnn.analysis.trainer_helper import *
from wmpgnn.analysis.weights_calculator import transform_pos_weight
from wmpgnn.analysis.data_loader import get_trn_val_loaders, get_tst_loaders

from wmpgnn.lightning_module.dfei_lightning_module import DFEILightningModule
from wmpgnn.lightning_module.exec_lightning import load_module, training, evaluate

import pdb

if __name__ == "__main__":
    # python trainer.py  --config  ../../config_files/lightning.yaml
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option("", "--config", type=str, default=None,
                      dest="CONFIG", help="Config file path")
    (option, args) = parser.parse_args()
    if len(args) != 0:
        raise RuntimeError("Got undefined arguments", " ".join(args))

    print("Training script started")
    print("=" * 30)

    # Load config file
    with open(option.CONFIG, "r") as file:
        configs = adjust_config(yaml.safe_load(file))

    print(f"DFEI mode: {configs['DFEI']['mode']}")
    print(f"IFT mode : {configs['IFT']['mode']}")
    print("=" * 30)

    # Check if both use the same dataset
    data_loaded = False
    same_sample = configs["DFEI"]["settings"]["sample"] == configs["IFT"]["settings"]["sample"]
    same_nfiles = configs["DFEI"]["settings"]["nfiles"] == configs["IFT"]["settings"]["nfiles"]
    if same_sample and same_nfiles:
        # Since same, we pass the DFEI model
        trn_loader, val_loader, weights, nevts = get_trn_val_loaders(configs["DFEI"])
        configs.update({"num_events": nevts})
        pos_weights = transform_pos_weight(weights, configs["DFEI"]["inference"])
        data_loaded = True

    """Start DFEI training"""
    if configs['DFEI']['mode'] == "train":
        if not data_loaded:
            trn_loader, val_loader, weights, nevts = get_trn_val_loaders(configs["DFEI"])
            configs["DFEI"].update({"num_events": nevts})
            pos_weights = transform_pos_weight(weights, configs["DFEI"]["inference"])

        # Start training DFEI
        module = load_module(configs, pos_weights, model="DFEI")
        trainer = training(module, trn_loader, val_loader, configs, model="DFEI")

        # Start testing
        run_test = any(value for key, value in configs["DFEI"]["inference"].items() if not key.endswith("weights"))
        if run_test:
            print("=" * 30)
            print("Loading data")
            tst_loader, nevts = get_tst_loaders(configs, model="DFEI")
            configs["DFEI"].update({"num_events": nevts})
            print("=" * 30)
            evaluate(trainer, module, tst_loader)
            version = trainer.logger.version
            metric_path = f"lightning_logs/DFEI/version_{version}/metrics.csv"
            df = pd.read_csv(metric_path)
            df = df.groupby('epoch').agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else None).reset_index()

        dfei_model = module.model
    else:
        if configs['IFT']['dfei_model'] != "None":
            dfei_path = configs["IFT"]["dfei_model"]
            # here we need to load the model
            print("Using DFEI model:", dfei_path)
        else:
            raise RuntimeError("No dfei model specified, either load or train")


    """Start IFT training"""
    if not data_loaded:
        trn_loader, val_loader, weights, nevts = get_trn_val_loaders(configs["IFT"])
        configs["IFT"].update({"num_events": nevts})
        pos_weights = transform_pos_weight(weights, configs["IFT"]["inference"])


    # Load IFT model
    module = load_module(configs, pos_weights, model="IFT", dfei_model=dfei_model)
    trainer = training(module, trn_loader, val_loader, configs, model="IFT")
