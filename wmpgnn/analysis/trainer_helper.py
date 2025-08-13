import re

import torch

from plotter import *


def node_pruning(valid_mask, graph, node_type, edge_types):
    # Mask with true nodes as input
    # automatically updates graph
    edge_node_indices = {}
    for edge_type in edge_types:
        if edge_type[0] == node_type and edge_type[2] == node_type:
            mask = valid_mask[graph[edge_type].edge_index[0]] & valid_mask[graph[edge_type].edge_index[1]]
        elif edge_type[0] == node_type:
            mask = valid_mask[graph[edge_type].edge_index[0]]
        else:
            mask = valid_mask[graph[edge_type].edge_index[1]]

        graph[edge_type].edge_index = graph[edge_type].edge_index[:, mask]
        graph[edge_type].edges = graph[edge_type].edges[mask, :]
        graph[edge_type].y = graph[edge_type].y[mask]
        edge_node_indices[edge_type] = mask
    return edge_node_indices


def load_dataset(path, config, mode):
    with open(path, "rb") as f:
        data = torch.load(f, weights_only=False)
    if config["graph_mode"] == "true":
        data_selbool = torch.ones(len(data))
        for i, evt in enumerate(data):
            if config["data"] == "LHCb":
                edge_data = evt[('tracks', 'to', 'tracks')]
                selbool = edge_data.y != 0
                sig_nodes = torch.unique(edge_data.edge_index[:, selbool])
                y_nodes = torch.zeros(evt['tracks'].x.shape[0], dtype=torch.bool)
                y_nodes[sig_nodes] = True
            elif config["data"] == "pythia":
                y_nodes = evt["tracks"].ft != 1
            else:
                raise "Mode not implemented"
            _ = node_pruning(y_nodes, evt, "tracks", [('tracks', 'to', 'tracks')])
            if evt[("tracks", "to", "tracks")].y.shape[0] == 0:
                data_selbool[i] = 0
    filtered_data = [d for d, sel in zip(data, data_selbool) if sel]
    if config["weights"]["weights"] and mode == "train":
        weights = get_hetero_weight(filtered_data, config)
    else:
        weights = {}
    return filtered_data, weights


def get_hetero_weight(data, configs):
    config = configs["weights"]
    raw_weights = {}
    if config["LCA"]:
        raw_weights["LCA"] = torch.zeros(4)
    if config["node"]:
        raw_weights["pos_nodes"] = 0
        raw_weights["neg_nodes"] = 0
    if config["edge"]:
        raw_weights["pos_edges"] = 0
        raw_weights["neg_edges"] = 0
    if config["frag"]:
        raw_weights["pos_frag"] = 0
        raw_weights["neg_frag"] = 0
    if config["FT"]:
        raw_weights["FT"] = torch.zeros(3)

    for evt in data:
        if config["LCA"]:
            y = evt[('tracks', 'to', 'tracks')].y
            raw_weights["LCA"] += torch.bincount(y, minlength=4)

        if config["node"]:
            selbool = evt["tracks"].ft == 0
            raw_weights["pos_nodes"] = torch.sum(~selbool).item()
            raw_weights["neg_nodes"] = torch.sum(selbool).item()

        if config["edge"]:
            selbool = evt[('tracks', 'to', 'tracks')].y == 0
            raw_weights["pos_edges"] += torch.sum(~selbool).item()
            raw_weights["neg_edges"] += torch.sum(selbool).item()

        if config["frag"]:
            selbool = evt["tracks"].frag == 0
            raw_weights["pos_frag"] += torch.sum(~selbool).item()
            raw_weights["neg_frag"] += torch.sum(selbool).item()

        if config["FT"]:
            y = evt['tracks'].ft
            if configs["graph_mode"] == "true":  # Consider frag nodes later on
                selbool = evt["tracks"].ft != 1
                y = y[selbool]
            raw_weights["FT"] += torch.bincount(y, minlength=3)

    return raw_weights


def transform_pos_weight(weights, config, mode="train"):
    if mode == "eval":
        pos_weight = {}
        pos_weight["LCA"] = torch.ones(4)
        pos_weight["nodes"] = torch.ones(1)
        pos_weight["edges"] = torch.ones(1)
        pos_weight["FT"] = torch.ones(3)
        return pos_weight

    summed = {}
    for d in weights:
        for key, value in d.items():
            if key not in summed:
                summed[key] = value.clone() if isinstance(value, torch.Tensor) else value
            else:
                summed[key] += value

    pos_weight = {}

    if config["LCA"]:
        pos_weight["LCA"] = torch.sum(summed["LCA"]) / (4 * summed["LCA"])
    else:
        pos_weight["LCA"] = torch.ones(4)

    if config["node"]:
        pos_weight["nodes"] = torch.tensor(summed["neg_nodes"] / summed["pos_nodes"])
    else:
        pos_weight["nodes"] = torch.ones(1)

    if config["edge"]:
        pos_weight["edges"] = torch.tensor(summed["neg_edges"] / summed["pos_edges"])
    else:
        pos_weight["edges"] = torch.ones(1)

    if config["frag"]:
        pos_weight["frag"] = torch.tensor(summed["neg_frag"] / summed["pos_frag"])
    else:
        pos_weight["frag"] = torch.ones(1)

    if config["FT"]:
        ft_weights = torch.sum(summed["FT"]) / (3 * summed["FT"])
        ft_weights[torch.isinf(ft_weights)] = 1.0
        pos_weight["FT"] = ft_weights
    else:
        pos_weight["FT"] = torch.ones(3)

    return pos_weight


def adjust_config(configs):
    if configs["data_dir"].split("_")[2] in ["LHCb", "pythia"]:
        configs["training"]["data"] = configs["data_dir"].split("_")[2]
        configs["training"]["infer"]["sim"] = configs["data_dir"].split("_")[2]
    else:
        raise ValueError("Data type cannot be inferred. Please check.")
    configs["training"]["weights"]["weights"] = any(configs["training"]["weights"].values())

    return configs


def metrics_eval(metrics, configs, version, channel):
    if configs["LCA"]:
        plot_LCA_acc(metrics, version, channel=channel)

    loss_val = [
        match.group(1)
        for key in metrics.keys()
        if (match := re.fullmatch(r"train_(.+?)_loss", key))
    ]

    for loss in loss_val:
        plot_loss(metrics, version, loss)
