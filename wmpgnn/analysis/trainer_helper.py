def adjust_config(_configs):
    # Check if weights need to be calculated
    _configs["inference"]["get_weights"] = any(
        _configs["inference"].get(key)
        for key in _configs["inference"]
        if key.endswith("weights")
    )

    # Check if model has custom train settings if not copy the global one
    for model in ["DFEI", "IFT"]:
        for setting in ["settings", "inference"]:
            if setting in _configs[model].keys():
                for key in config[setting].keys():
                    if key not in _configs[model].keys():
                        _configs[model][setting][key] = _configs[setting][key]
            else:
                _configs[model][setting] = _configs[setting]

    del _configs["settings"], _configs["inference"]
    return _configs
