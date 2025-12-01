import torch
from collections import defaultdict


def get_hetero_weight(data, _configs):
    config = _configs["inference"]

    raw_weights = {
        "LCA": torch.zeros(4, dtype=torch.int64),
        "FT": torch.zeros(3, dtype=torch.int64),
        "pos_nodes": torch.zeros(1, dtype=torch.int64), "neg_nodes": torch.zeros(1, dtype=torch.int64),
        "pos_edges": torch.zeros(1, dtype=torch.int64), "neg_edges": torch.zeros(1, dtype=torch.int64),
        "pos_frag": torch.zeros(1, dtype=torch.int64), "neg_frag": torch.zeros(1, dtype=torch.int64),
    }

    for evt in data:
        if config["LCA_weights"]:
            y = evt[('tracks', 'to', 'tracks')].y
            raw_weights["LCA"] += torch.bincount(y, minlength=4).to(torch.int64)

        if config["node_prune_weights"]:
            selbool = evt["tracks"].ft == 0
            raw_weights["pos_nodes"] += torch.sum(~selbool).to(torch.int64)
            raw_weights["neg_nodes"] += torch.sum(selbool).to(torch.int64)

        if config["edge_prune_weights"]:
            selbool = evt[('tracks', 'to', 'tracks')].y == 0
            raw_weights["pos_edges"] += torch.sum(~selbool).to(torch.int64)
            raw_weights["neg_edges"] += torch.sum(selbool).to(torch.int64)

        if config["frag_weights"]:
            selbool = evt["tracks"].frag == 0
            raw_weights["pos_frag"] += torch.sum(~selbool).to(torch.int64)
            raw_weights["neg_frag"] += torch.sum(selbool).to(torch.int64)

        if config["FT_weights"]:
            y = evt['tracks'].ft.to(torch.int64)
            if _configs["settings"]["graph_mode"] == "true":  # Consider frag nodes later on
                selbool = evt["tracks"].ft != 1
                y = y[selbool].to(torch.int64)
            raw_weights["FT"] += torch.bincount(y, minlength=3)

        if config["pv_asso_weights"]:
            selbool = evt[("tracks", "to", "pvs")].y.squeeze() == 0
            raw_weights["pos_pv_asso"] = torch.sum(~selbool, ).to(torch.int64)
            raw_weights["neg_pv_asso"] = torch.sum(selbool).to(torch.int64)

    return raw_weights


def transform_pos_weight(weights, config, mode="train"):
    pos_weight = {
        "LCA": torch.ones(4),
        "nodes": torch.ones(1),
        "edges": torch.ones(1),
        "FT": torch.ones(3),
        "frag": torch.ones(1),
        "pv_asso": torch.ones(1)
    }

    if mode == "eval":
        return pos_weight

    if "LCA" not in weights.keys():
        combined = defaultdict(int)
        for sample_name, sample_weights in weights.items():
            for key, value in sample_weights.items():
                if key not in combined:
                    combined[key] = value
                else:
                    combined[key] += value
        summed = dict(combined)
    else:
        summed = weights

    if config["LCA_weights"]:
        pos_weight["LCA"] = torch.sum(summed["LCA"]) / (4 * summed["LCA"])

    if config["node_prune_weights"]:
        pos_weight["nodes"] = torch.tensor([summed["neg_nodes"] / summed["pos_nodes"]])

    if config["edge_prune_weights"]:
        pos_weight["edges"] = torch.tensor([summed["neg_edges"] / summed["pos_edges"]])

    if config["frag_weights"]:
        pos_weight["frag"] = torch.tensor(summed["neg_frag"] / summed["pos_frag"])

    if config["FT_weights"]:
        ft_weights = torch.sum(summed["FT"]) / (3 * summed["FT"])
        ft_weights[torch.isinf(ft_weights)] = 1.0
        pos_weight["FT"] = ft_weights

    if config["pv_asso_weights"] and summed["neg_pv_asso"] != 0:
        pos_weight["pv_asso"] = torch.tensor([summed["neg_pv_asso"] / summed["pos_pv_asso"]])

    return pos_weight
