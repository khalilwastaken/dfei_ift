import torch
from collections import defaultdict


def get_hetero_weight(_data, _configs):
    config = _configs["inference"]

    raw_weights = {
        "LCA": torch.zeros(4, dtype=torch.int64),
        "FT": torch.zeros(3, dtype=torch.int64),
        "pos_nodes": torch.zeros(1, dtype=torch.int64), "neg_nodes": torch.zeros(1, dtype=torch.int64),
        "pos_edges": torch.zeros(1, dtype=torch.int64), "neg_edges": torch.zeros(1, dtype=torch.int64),
        "pos_frag": torch.zeros(1, dtype=torch.int64), "neg_frag": torch.zeros(1, dtype=torch.int64),
    }

    for evt in _data:
        if config.get("LCA_weights"):
            y = evt[('tracks', 'to', 'tracks')].y
            raw_weights["LCA"] += torch.bincount(y, minlength=4).to(torch.int64)

        if config.get("node_prune_weights"):
            selbool = evt["tracks"].ft == 1  # 0 = bbar, 1 = background, 2 = b
            raw_weights["pos_nodes"] += torch.sum(~selbool).to(torch.int64)
            raw_weights["neg_nodes"] += torch.sum(selbool).to(torch.int64)

        if config.get("edge_prune_weights"):
            selbool = evt[('tracks', 'to', 'tracks')].y == 0
            raw_weights["pos_edges"] += torch.sum(~selbool).to(torch.int64)
            raw_weights["neg_edges"] += torch.sum(selbool).to(torch.int64)

        if config.get("frag_weights"):
            selbool = evt["tracks"].frag == 0
            raw_weights["pos_frag"] += torch.sum(~selbool).to(torch.int64)
            raw_weights["neg_frag"] += torch.sum(selbool).to(torch.int64)

        if config.get("FT_weights"):
            y = evt['tracks'].ft.to(torch.int64)
            if _configs["settings"]["graph_mode"] == "true":  # Consider frag nodes later on
                selbool = evt["tracks"].ft != 1
                y = y[selbool].to(torch.int64)
            raw_weights["FT"] += torch.bincount(y, minlength=3)

        if config.get("pv_asso_weights"):
            selbool = evt[("tracks", "to", "pvs")].y.squeeze() == 0
            raw_weights["pos_pv_asso"] = torch.sum(~selbool, ).to(torch.int64)
            raw_weights["neg_pv_asso"] = torch.sum(selbool).to(torch.int64)

    return raw_weights


def transform_pos_weight(_weights, _configs, mode="train"):
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

    if "LCA" not in _weights.keys():
        combined = defaultdict(int)
        for sample_name, sample__weights in _weights.items():
            for key, value in sample__weights.items():
                if key not in combined:
                    combined[key] = value
                else:
                    combined[key] += value
        summed = dict(combined)
    else:
        summed = _weights

    if _configs.get("LCA_weights"):
        pos_weight["LCA"] = torch.sum(summed["LCA"]) / (4 * summed["LCA"])

    if _configs.get("node_prune_weights"):
        pos_weight["nodes"] = torch.tensor([summed["neg_nodes"] / summed["pos_nodes"]])

    if _configs.get("edge_prune_weights"):
        pos_weight["edges"] = torch.tensor([summed["neg_edges"] / summed["pos_edges"]])

    if _configs.get("frag_weights"):
        pos_weight["frag"] = torch.tensor(summed["neg_frag"] / summed["pos_frag"])

    if _configs.get("FT_weights"):
        ft__weights = torch.sum(summed["FT"]) / (3 * summed["FT"])
        ft__weights[torch.isinf(ft__weights)] = 1.0
        pos_weight["FT"] = ft__weights

    if _configs.get("pv_asso_weights") and summed["neg_pv_asso"] != 0:
        pos_weight["pv_asso"] = torch.tensor([summed["neg_pv_asso"] / summed["pos_pv_asso"]])

    return pos_weight
