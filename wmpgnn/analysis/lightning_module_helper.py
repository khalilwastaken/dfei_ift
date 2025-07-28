import torch


def init_logs(configs):
    log = {"combined_loss": []}
    if configs["LCA"]:
        log["LCA_loss"] = []
        for i in range(4):  # something like num classes in config file
            log[f"LCA_class{i}_num"] = []
            for j in range(4):
                log[f"LCA_class{i}_pred_class{j}"] = []

    if configs["node_prune"]:
        log["t_nodes_loss"] = []
    if configs["edge_prune"]:
        log["tt_edges_loss"] = []
    if configs["FT"]:
        log["ft_loss"] = []
    if configs["frag"]:
        log["frag_loss"] = []
    return log, log


def make_loggable(hparams_dict):
    loggable = {}
    for k, v in hparams_dict.items():
        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                loggable[k] = v.item()  # convert scalar tensor to float
            else:
                loggable[k] = v.tolist()  # convert vector/matrix to list
        else:
            loggable[k] = str(v)  # fallback to string
    return loggable
