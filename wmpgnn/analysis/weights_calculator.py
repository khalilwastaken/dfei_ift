import torch


def get_hetero_weight(data, _configs):
    config = _configs["inference"]

    raw_weights = {
        "LCA": torch.zeros(4),
        "FT": torch.zeros(3),
        "pos_nodes": 0, "neg_nodes": 0,
        "pos_edges": 0, "neg_edges": 0,
        "pos_frag": 0, "neg_frag": 0,
    }

    for evt in data:
        if config["LCA_weights"]:
            y = evt[('tracks', 'to', 'tracks')].y
            raw_weights["LCA"] += torch.bincount(y, minlength=4)

        if config["node_prune_weights"]:
            selbool = evt["tracks"].ft == 0
            raw_weights["pos_nodes"] += torch.sum(~selbool).item()
            raw_weights["neg_nodes"] += torch.sum(selbool).item()

        if config["edge_prune_weights"]:
            selbool = evt[('tracks', 'to', 'tracks')].y == 0
            raw_weights["pos_edges"] += torch.sum(~selbool).item()
            raw_weights["neg_edges"] += torch.sum(selbool).item()

        if config["frag_weights"]:
            selbool = evt["tracks"].frag == 0
            raw_weights["pos_frag"] += torch.sum(~selbool).item()
            raw_weights["neg_frag"] += torch.sum(selbool).item()

        if config["FT_weights"]:
            y = evt['tracks'].ft
            if _configs["settings"]["graph_mode"] == "true":  # Consider frag nodes later on
                selbool = evt["tracks"].ft != 1
                y = y[selbool]
            raw_weights["FT"] += torch.bincount(y, minlength=3)

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

    summed = {}
    for d in weights:
        for key, value in d.items():
            if key not in summed:
                summed[key] = value.clone() if isinstance(value, torch.Tensor) else value
            else:
                summed[key] += value

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

    return pos_weight
