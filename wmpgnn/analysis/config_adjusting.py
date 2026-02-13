from typing import Dict


def obtain_log_dir(_configs: Dict) -> Dict:
    if 'pythia' in _configs['settings']['data_dir']:
        _configs["log_dir"] = 'pythia_logs'
    elif 'LHCb' in _configs['settings']['data_dir']:
        _configs["log_dir"] = 'LHCb_logs'
    else:
        raise ValueError("Invalid config")
    return _configs


def training_model_name(_configs: Dict) -> Dict:
    for model in ["DFEI", "IFT"]:
        if model in _configs.keys():
            print("Training model", model)
            print("=" * 15)
            _configs["model"] = model
            return _configs
    raise ValueError("Invalid config")


def adjust_config(_configs: Dict) -> Dict:
    # Obtaining the model_to be trained
    _configs = training_model_name(_configs)
    # Obtaining log_dir
    _configs = obtain_log_dir(_configs)

    # Check if weights need to be calculated, to save a bit of performance later on
    _configs["inference"]["get_weights"] = any(
        _configs["inference"].get(key)
        for key in _configs["inference"]
        if key.endswith("weights")
    )

    if _configs["model"] == "DFEI":
        _configs = adjust_dfei_configs(_configs)
    elif _configs["model"]  == "IFT":
        _configs = adjust_ift_configs(_configs)

    return _configs


def adjust_dfei_configs(_configs: Dict) -> Dict:
    _configs["inference"]["plt_nodes"] = True if _configs["inference"]["node_prune"] else False
    _configs["inference"]["plt_edges"] = True if _configs["inference"]["edge_prune"] else False
    _configs["inference"]["plt_pvs"] = True if _configs["inference"]["pv_asso"] else False
    return _configs


def adjust_ift_configs(_configs: Dict) -> Dict:
    return _configs
"""
def adjust_ift_configs(configs: Dict, over_write_configs: Dict):
    return 0
    print("ok")
    # this adds soemthing like create this and this plot which i can later on overwrite during eval
"""