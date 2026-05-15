import yaml

from typing import Dict


def training_model_name(_configs: Dict) -> Dict:
    for model in ["DFEI", "IFT"]:
        if model in _configs.keys():
            print("Training model", model)
            print("=" * 15)
            _configs["model"] = model
            return _configs
    raise ValueError("Invalid config")


def adjust_dfei_configs(_configs: Dict) -> Dict:
    if "plt_nodes" not in _configs["inference"].keys():
        _configs["inference"]["plt_nodes"] = True if _configs["inference"]["node_prune"] else False
    if "plt_edges" not in _configs["inference"].keys():
        _configs["inference"]["plt_edges"] = True if _configs["inference"]["edge_prune"] else False
    if "plt_pvs" not in _configs["inference"].keys():
        _configs["inference"]["plt_pvs"] = True if _configs["inference"]["pv_asso"] else False
    return _configs


def adjust_ift_configs(_configs: Dict) -> Dict:
    return _configs


def adjust_config_training(_configs: Dict) -> Dict:
    # Obtaining the model_to be trained
    _configs = training_model_name(_configs)
    # Obtaining log_dir
    _configs = _configs["log_dir"] = 'LHCb_logs'

    # Check if weights need to be calculated, to save a bit of performance later on
    _configs["inference"]["get_weights"] = any(
        _configs["inference"].get(key)
        for key in _configs["inference"]
        if key.endswith("weights")
    )

    if _configs["model"] == "DFEI":
        _configs = adjust_dfei_configs(_configs)
    elif _configs["model"] == "IFT":
        _configs = adjust_ift_configs(_configs)

    return _configs


# function to overwrite the config file with new settings
def update_nested_dict(target, source):
    keys_to_update = source.keys()
    for key in keys_to_update:
        if key not in source:
            continue
        if key not in target:
            target[key] = source[key]
        elif isinstance(source[key], dict) and isinstance(target[key], dict):
            update_nested_dict(target[key], source[key])
        else:
            target[key] = source[key]
    return target


def adjust_config_evaluation(_configs: Dict) -> Dict:
    # Obtaining the model_to be trained
    _configs["model"] = _configs["settings"]["model_arch"]
    print("Training model", _configs["model"])
    print("=" * 15)

    # Obtaining log_dir
    _configs = obtain_log_dir(_configs)

    # Obtaining the model information of the model to be evaluated
    hparams_file = f"{_configs['log_dir']}/{_configs['model']}/version_{_configs['settings']['model']}/input_config.yaml"
    with open(hparams_file, "r") as file:
        hparams = yaml.safe_load(file)
    # overwriting settings from evaluation
    hparams = update_nested_dict(hparams, _configs)
    hparams[_configs["model"]]["cpt"] = _configs["settings"]["model"]  # loading from cpt for eval
    if _configs["model"] == "DFEI":
        hparams = adjust_dfei_configs(hparams)
        # here we can overwrite the plot configs
    elif _configs["model"] == "IFT":
        hparams = adjust_ift_configs(hparams)

    return hparams
