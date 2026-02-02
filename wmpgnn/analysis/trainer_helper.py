import glob
import re


def adjust_config(_configs):
    # Check if weights need to be calculated
    _configs["inference"]["get_weights"] = any(
        _configs["inference"].get(key)
        for key in _configs["inference"]
        if key.endswith("weights")
    )

    # Check if model has custom train settings if not copy the global one
    for model in ["DFEI", "IFT"]:
        if model in _configs.keys():
            for setting in ["settings", "inference"]:
                if setting in _configs[model].keys():
                    for key in config[setting].keys():
                        if key not in _configs[model].keys():
                            _configs[model][setting][key] = _configs[setting][key]
                else:
                    _configs[model][setting] = _configs[setting]

    return _configs


def get_bis_model(version, model="DFEI", configs=None):
    if model == "DFEI_pv_asso":
        model = "DFEI"
    if 'pythia' in configs['settings']['data_dir']:
            log_dir = 'pythia_logs'
    elif 'LHCb' in configs['settings']['data_dir']:
        log_dir = 'LHCb_logs'
    else:
        raise ValueError("Invalid config")
    files = glob.glob(f"{log_dir}/{model}/version_{version}/checkpoints/*.ckpt")
    if model == "DFEI":
        pattern = re.compile(r"val_combined_loss=([\d.]+)")
    elif model == "IFT":
        pattern = re.compile(r"val_ft_loss=([\d.]+)")
    else:
        raise ValueError(f"undefined model: {model}")
    bis = min(files, key=lambda s: float(pattern.search(s).group(1)[:-1]))
    return bis
