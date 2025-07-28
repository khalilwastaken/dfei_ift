import torch


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
        for evt in data:
            _ = node_pruning((evt["tracks"].ft != 1), evt, "tracks", [('tracks', 'to', 'tracks')])
    if config["weights"]["weights"] and mode == "train":
        weights = get_hetero_weight(data, config["weights"])
    else:
        weights = {}
    return data, weights


def get_hetero_weight(data, config):
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
            raw_weights["FT"] += torch.bincount(y, minlength=3)

    return raw_weights


def transform_pos_weight(weights, config):
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
        pos_weight["FT"] = torch.sum(summed["FT"]) / (3 * summed["FT"])
    else:
        pos_weight["FT"] = torch.ones(3)

    return pos_weight
