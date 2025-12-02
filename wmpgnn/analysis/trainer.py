import sys, os

import pandas as pd

import yaml
from optparse import OptionParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from wmpgnn.analysis.trainer_helper import *
from wmpgnn.data_loader.data_loader_helper import load_trn_val_data
from wmpgnn.analysis.weights_calculator import transform_pos_weight
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

    print("Training script started")
    print("=" * 30)

    # Load config file
    with open(option.CONFIG, "r") as file:
        configs = adjust_config(yaml.safe_load(file))

    print(f"DFEI mode: {configs['DFEI']['mode']}")
    print(f"IFT mode : {configs['IFT']['mode']}")
    print("=" * 30)

    """Start DFEI training"""
    dfei_model = None
    if configs['DFEI']['mode'] == "train":
        # Obtaining train and validation dataloaders
        configs, weights, trn_loader, val_loader, chunkloader = load_trn_val_loader(configs, model="DFEI")
        pos_weights = transform_pos_weight(weights, configs["DFEI"]["inference"])

        # Start training DFEI
        module = load_module(configs, pos_weights, model="DFEI")
        trainer = training(module, configs, model="DFEI",
                           trn_loader=trn_loader, val_loader=val_loader, chunkloader=chunkloader)
        version = trainer.logger.version

        # Start testing
        configs, tst_loader, chunkloader = load_tst_loader(configs, model="DFEI")
        evaluate(trainer, module, tst_loader=tst_loader, chunkloader=chunkloader)
        metric_path = f"lightning_logs/DFEI/version_{version}/metrics.csv"
        df = pd.read_csv(metric_path)
        df = df.groupby('epoch').agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else None).reset_index()
        metrics_eval(df, configs["DFEI"]["inference"], version, configs["evaluate"]["sample"], mode="DFEI")

        dfei_bis_model = get_bis_model(version, "DFEI")
        print("Obtained best DFEI model:", dfei_bis_model)
    elif configs['DFEI']['mode'] == "usage":
        load_dfei = configs['IFT']['dfei_model']
        if isinstance(load_dfei, int):
            dfei_bis_model = get_bis_model(load_dfei, "DFEI")
        elif isinstance(load_dfei, str):
            dfei_bis_model = load_dfei
        else:
            raise RuntimeError(f"Unsupported load_dfei: {type(load_dfei)}")
        pos_weights = transform_pos_weight(None, None, mode="eval")
    else:
        raise RuntimeError(f"either train or pass DFEI model, not usage currently not possible for IFT")

    """Start IFT training"""
    if configs['IFT']['mode'] == "train":
        # Loading the DFEI model
        print("Using DFEI model:", dfei_bis_model)
        configs["DFEI"]["cpt"] = dfei_bis_model
        module = load_module(configs, pos_weights, model="DFEI")
        dfei_model = module.model

        # Loading in IFT model
        configs, weights, trn_loader, val_loader, chunkloader = load_trn_val_loader(configs, model="IFT")
        pos_weights = transform_pos_weight(weights, configs["IFT"]["inference"])

        # Load IFT model
        module = load_module(configs, pos_weights, model="IFT", dfei_model=dfei_model)
        trainer = training(module, configs, model="IFT",
                           trn_loader=trn_loader, val_loader=val_loader, chunkloader=chunkloader)
        version = trainer.logger.version

        configs, tst_loader, chunkloader = load_tst_loader(configs, model="DFEI")
        evaluate(trainer, module, tst_loader=tst_loader, chunkloader=chunkloader)
        metric_path = f"lightning_logs/IFT/version_{version}/metrics.csv"
        df = pd.read_csv(metric_path)
        df = df.groupby('epoch').agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else None).reset_index()
        metrics_eval(df, configs["IFT"]["inference"], version, configs["evaluate"]["sample"], mode="IFT")
