import sys, os

import pandas as pd

import yaml
from optparse import OptionParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from wmpgnn.analysis.trainer_helper import *
from wmpgnn.data_loader.data_loader_helper import load_trn_val_loader, load_tst_loader
from wmpgnn.data_loader.weights_calculator import transform_pos_weight
from wmpgnn.performance.plotter import metrics_eval
from wmpgnn.lightning_module.dfei_lightning_module import DFEILightningModule
from wmpgnn.lightning_module.exec_lightning import load_module, training, evaluate

"""import sys
import io

class PrintTracker(io.TextIOBase):
    def write(self, s):
        if 'edges' in s.lower():
            import traceback
            traceback.print_stack()
        return sys.__stdout__.write(s)

sys.stdout = PrintTracker()"""

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
    if 'pythia' in configs['settings']['data_dir']:
        log_dir = 'pythia_logs'
    elif 'LHCb' in configs['settings']['data_dir']:
        log_dir = 'LHCb_logs'
    else:
        raise ValueError("Invalid config")
    print("=" * 30)

    dfei_model = None
    if "DFEI" in configs.keys():
        """Start DFEI training"""
        # Obtaining train and validation dataloaders
        configs, weights, trn_loader, val_loader, chunkloader = load_trn_val_loader(configs, model="DFEI")
        pos_weights = transform_pos_weight(weights, configs["DFEI"]["inference"])

        # Obtain the DFEI module
        model_name = "DFEI"
        module = load_module(configs, pos_weights, model="DFEI")
    elif "IFT" in configs.keys():
        """Start IFT training"""
        # Loading the DFEI model by loading the hparams of the used model
        version = configs['IFT']['dfei_model']
        if version != "None":
            if isinstance(version, int):
                dfei_bis_model = get_bis_model(version, "DFEI", configs)
            elif isinstance(version, str):
                dfei_bis_model = version
                version = re.search(r"version_(\d+)", dfei_bis_model).group(1)
            else:
                raise RuntimeError(f"Unsupported load_dfei: {type(version)}")
            pos_weights = transform_pos_weight(None, None, mode="eval")
            print("Using DFEI model:", dfei_bis_model)
            with open(f"{log_dir}/DFEI/version_{version}/hparams.yaml", "r") as file:
                dfei_hparams = yaml.safe_load(file)
            configs["DFEI"] = dfei_hparams["DFEI"]
            configs["DFEI"]["cpt"] = dfei_bis_model
            module = load_module(configs, pos_weights, model="DFEI")
            dfei_model = module.model
        else:
            print("No DFEI model specified. Information from pv association or truth information is used as a replacement")
            dfei_model = None
        print("=" * 30)

        # Obtaining train and validation dataloaders
        configs, weights, trn_loader, val_loader, chunkloader = load_trn_val_loader(configs, model="IFT")
        pos_weights = transform_pos_weight(weights, configs["IFT"]["inference"])

        # Obtain the IFT module
        model_name = "IFT"
        module = load_module(configs, pos_weights, model="IFT", dfei_model=dfei_model)
    else:
        raise RuntimeError("No configuration file specified")

    """Start the training"""
    trainer = training(module, configs, model=model_name,
                       trn_loader=trn_loader, val_loader=val_loader, chunkloader=chunkloader)
    version = trainer.logger.version

    # Start testing
    configs, tst_loader, chunkloader = load_tst_loader(configs, model=model_name)
    evaluate(trainer, module, tst_loader=tst_loader, chunkloader=chunkloader)
    metric_path = f"{log_dir}/{model_name}/version_{version}/metrics.csv"
    df = pd.read_csv(metric_path)
    df = df.groupby('epoch').agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else None).reset_index()
    metrics_eval(df, configs[model_name]["inference"], version, mode=model_name)
