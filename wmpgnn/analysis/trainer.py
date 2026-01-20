import sys, os

import pandas as pd

import yaml
from optparse import OptionParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from wmpgnn.analysis.trainer_helper import *
from wmpgnn.data_loader.data_loader_helper import load_trn_val_loader, load_tst_loader
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

    for model in ["DFEI", "IFT"]:
        if model in configs.keys():
            print("Training model", model)
    print("=" * 30)

    """Start DFEI training"""
    dfei_model = None
    if "DFEI" in configs.keys():
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
        if 'pythia' in configs['settings']['data_dir']:
            log_dir = 'pythia_logs'
        elif 'LHCb' in configs['settings']['data_dir']:
            log_dir = 'LHCb_logs'
        else:
            raise ValueError("Invalid config")
        metric_path = f"{log_dir}/IFT/version_{version}/metrics.csv"
        df = pd.read_csv(metric_path)
        df = df.groupby('epoch').agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else None).reset_index()
        sample = configs["evaluate"]["sample"]
        if configs["evaluate"]["over_write"] != "None":
            sample += "__" + configs["evaluate"]["over_write"]
        metrics_eval(df, configs["DFEI"]["inference"], version, sample, mode="DFEI")

        dfei_bis_model = get_bis_model(version, "DFEI")
        print("Obtained best DFEI model:", dfei_bis_model)
    else:
        version = configs['IFT']['dfei_model']
        if isinstance(version, int):
            dfei_bis_model = get_bis_model(version, "DFEI")
        elif isinstance(version, str):
            dfei_bis_model = version
            version = re.search(r"version_(\d+)", dfei_bis_model).group(1)
        else:
            raise RuntimeError(f"Unsupported load_dfei: {type(version)}")
        pos_weights = transform_pos_weight(None, None, mode="eval")

    """Start IFT training"""
    if "IFT" in configs.keys():
        # Loading the DFEI model by loading the hparams of the used model
        print("Using DFEI model:", dfei_bis_model)
        with open(f"lightning_logs/DFEI/version_{version}/hparams.yaml", "r") as file:
            dfei_hparams = yaml.safe_load(file)
        configs["DFEI"] = dfei_hparams["DFEI"]
        configs["DFEI"]["cpt"] = dfei_bis_model
        module = load_module(configs, pos_weights, model="DFEI")
        dfei_model = module.model

        # Obtaining train and validation dataloaders
        configs, weights, trn_loader, val_loader, chunkloader = load_trn_val_loader(configs, model="IFT")
        pos_weights = transform_pos_weight(weights, configs["IFT"]["inference"])

        # Start training IFT
        module = load_module(configs, pos_weights, model="IFT", dfei_model=dfei_model)
        trainer = training(module, configs, model="IFT",
                           trn_loader=trn_loader, val_loader=val_loader, chunkloader=chunkloader)
        version = trainer.logger.version

        # Start testing
        configs, tst_loader, chunkloader = load_tst_loader(configs, model="IFT")
        evaluate(trainer, module, tst_loader=tst_loader, chunkloader=chunkloader)
        if 'pythia' in configs['settings']['data_dir']:
            log_dir = 'pythia_logs'
        elif 'LHCb' in configs['settings']['data_dir']:
            log_dir = 'LHCb_logs'
        else:
            raise ValueError("Invalid config")
        metric_path = f"{log_dir}/IFT/version_{version}/metrics.csv"
        df = pd.read_csv(metric_path)
        df = df.groupby('epoch').agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else None).reset_index()
        sample = configs["evaluate"]["sample"]
        if configs["evaluate"]["over_write"] != "None":
            sample += "__" + configs["evaluate"]["over_write"]
        metrics_eval(df, configs["IFT"]["inference"], version, sample, mode="IFT")
