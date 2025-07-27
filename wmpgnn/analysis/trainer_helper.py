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


def load_dataset(path, mode):
    with open(path, "rb") as f:
        data = torch.load(f, weights_only=False)
    if mode == "true":
        for evt in data:
            _ = node_pruning((evt["tracks"].ft != 1), evt, "tracks", [('tracks', 'to', 'tracks')])

    return data
