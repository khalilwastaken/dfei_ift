from optparse import OptionParser
import sys, os
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from wmpgnn.analysis.config_adjusting import *
from wmpgnn.analysis.load_module import *
from wmpgnn.data_loader.get_data_loader import load_tst_loader
from wmpgnn.data_loader.weights_calculator import transform_pos_weight
from wmpgnn.lightning_module.exec_lightning import evaluate
from wmpgnn.performance.plot_results import metrics_eval

if __name__ == "__main__":
    # python trainer.py  --config  to hparams.yaml
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option("", "--config", type=str, default=None,
                      dest="CONFIG", help="Config file path")
    (option, args) = parser.parse_args()
    if len(args) != 0:
        raise RuntimeError("Got undefined arguments", " ".join(args))
    print("=" * 45)
    print("Starting the evaluation script")

    # Load config file
    with open(option.CONFIG, "r") as file:
        configs = yaml.safe_load(file)
    configs = adjust_config_evaluation(configs)
    configs, tst_loader, chunkloader = load_tst_loader(configs)
    pos_weights = transform_pos_weight(None, None, mode="eval")

    # Obtaining the lightning module for evaluation
    if configs["model"] == "DFEI":
        # Obtain the DFEI module
        module, ckpt = load_module(configs, pos_weights)
    elif configs["model"] == "IFT":
        # Loading the DFEI model by loading the hparams of the used model
        # load dfei module
        configs, dfei_model = load_dfei_for_ift(configs)
        module, ckpt = load_module(configs, pos_weights, dfei_model=dfei_model)
    else:
        raise RuntimeError("No configuration file specified")

    if configs["settings"]["model_name"] != "None":
        ckpt = configs["settings"]["model_name"]
    evaluate(None, module, tst_loader=tst_loader, chunkloader=chunkloader, ckpt=ckpt)
    # Creating loss plots and acc evaluation
    version = configs[configs["model"]]["cpt"]
    metric_path = f"{configs['log_dir']}/{configs['model']}/version_{version}/metrics.csv"
    metrics_eval(metric_path, configs, version)
    print("Done")
    print("=" * 45)
