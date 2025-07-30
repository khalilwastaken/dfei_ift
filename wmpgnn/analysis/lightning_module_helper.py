import torch


def init_logs(configs, mode="train"):
    loss_config = configs["training"]["infer"]
    gn_blocks = configs["model"]["GNblocks"]["nBlocks"]
    ft_layers = configs["model"]["GNblocks"]["FTlayers"]

    log = {"combined_loss": []}
    if loss_config["LCA"]:
        log["LCA_loss"] = []
        for i in range(4):  # something like num classes in config file
            log[f"LCA_class{i}_num"] = []
            for j in range(4):
                log[f"LCA_class{i}_pred_class{j}"] = []

    if loss_config["node_prune"]:
        log["t_nodes_loss"] = []
        if mode == "test":
            for i in range(gn_blocks):
                log[f"sig_nodes_score_{i}"] = torch.tensor([])
                log[f"bkg_nodes_score_{i}"] = torch.tensor([])

    if loss_config["edge_prune"]:
        log["tt_edges_loss"] = []
        if mode == "test":
            for i in range(gn_blocks):
                log[f"sig_edges_score_{i}"] = torch.tensor([])
                log[f"bkg_edges_score_{i}"] = torch.tensor([])

    if loss_config["FT"]:
        log["ft_loss"] = []
        if mode == "test":
            for i in range(ft_layers):
                log[f"b_ft_score_{gn_blocks - i - 1}"] = torch.tensor([])
                log[f"bbar_ft_score_{gn_blocks - i - 1}"] = torch.tensor([])

    if loss_config["frag"]:
        log["frag_loss"] = []
        if mode == "test":
            for i in range(ft_layers):
                log[f"sig_frag_score_{gn_blocks - i - 1}"] = torch.tensor([])
                log[f"bkg_frag_score_{gn_blocks - i - 1}"] = torch.tensor([])

    if mode == "train":
        return log, log
    elif mode == "test":
        return log


def loss_logging(log, loss, configs):
    if configs["node_prune"]:
        log["t_nodes_loss"].append(loss["t_nodes"].item())
    if configs["edge_prune"]:
        log["tt_edges_loss"].append(loss["tt_edges"].item())
    if configs["frag"]:
        log["frag_loss"].append(loss["frag_nodes"].item())
    if configs["FT"]:
        log["ft_loss"].append(loss["ft_nodes"].item())
    return log


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
