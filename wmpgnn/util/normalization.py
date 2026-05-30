@dataclass
class Normalization:  # hold information regarding normalization
    norm: pd.DataFrame
    node_features: list[str]


# this can be ported to utils
def obtain_normalization(configs):
    # this should be generalized  to just load in the var which is needed
    # This is a current problem, one needs to copy the config normalization file
    norm_file = "normalization_dict.pt"
    # Determining simulation data mode based on log dir
    if configs["log_dir"] == "LHCb_logs":
        key = "LHCbcollision"
    elif configs["log_dir"] == "pythia_logs":
        key = "nu7p6"
    else:
        raise NotImplementedError
    try:
        normalization_dict = torch.load(norm_file)[key]
    except:
        raise ValueError("norm file not found, pls copy manually")

    # Obtain the features of how the graph is setup to reverse normalization
    data_dir = configs["settings"]["data_dir"]
    # one of the two should exist
    if configs.get("evaluate", {}).get("sample"):
        channel = configs.get("evaluate", {}).get("sample")[0]
    elif configs.get("settings", {}).get("sample"):
        channel = configs.get("settings", {}).get("sample")[0]
    else:
        raise NotImplementedError("Something went wrong here pls check")
    with open(f"{data_dir}/{channel}/config.yaml", "r") as file:
        graph_configs = yaml.safe_load(file)["node_features"]

    # variables where we need to revert the normalization
    normalization = pd.DataFrame(normalization_dict).T.loc[graph_configs]

    selected = pd.DataFrame(index=normalization.index, columns=['value'])
    skew_thresh = 1.0
    for feature in normalization.index:
        if abs(normalization.loc[feature, 'skew']) > skew_thresh:
            selected.loc[feature, 'value'] = (normalization.loc[feature, 'median'],
                                              normalization.loc[feature, 'iqr'])
        else:
            selected.loc[feature, 'value'] = (normalization.loc[feature, 'mean'],
                                              normalization.loc[feature, 'std'])
    selected[['val1', 'val2']] = pd.DataFrame(selected['value'].tolist(), index=selected.index)
    selected = selected.drop(columns='value')
    selected = torch.tensor(selected.values).T

    return Normalization(selected, graph_configs)