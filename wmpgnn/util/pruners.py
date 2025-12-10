import torch


def edge_pruning(edge_indices, graph, edge_type):
    graph[edge_type].edges = graph[edge_type].edges[edge_indices]
    graph[edge_type].edge_index = torch.vstack(
        [graph[edge_type].edge_index[0][edge_indices],
         graph[edge_type].edge_index[1][edge_indices]])
    graph[edge_type].y = graph[edge_type].y[edge_indices]


def node_pruning(valid_mask, graph, node_type, edge_types):
    # Removes the edges, but keep the nodes and features
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


def test_node_pruning(valid_mask, graph, node_type, edge_types):
    # Removes edges and replace the nodes feature with a tensor of 0
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
    graph["tracks"].x[~valid_mask] = torch.zeros(torch.sum(~valid_mask), 12)


def true_node_pruning(node_mask, graph, node_type, edge_types):
    # Removes the nodes and associated edges from the graph
    device = node_mask.device
    node_indx = torch.arange(len(node_mask), device=device)[node_mask]
    indx_proj = torch.arange(len(node_indx), device=device)
    lookup = torch.full((len(node_mask),), -1, device=device, dtype=torch.long)
    lookup[node_indx] = indx_proj

    # Removing the edges nodes
    node_keys = list(graph[node_type].keys())
    for node_key in node_keys:
        if node_key == "ptr":
            continue
        graph[node_type][node_key] = graph[node_type][node_key][node_mask]
    if hasattr(graph, "final_keys") and node_type == "tracks":
        graph.final_keys = graph.final_keys[node_mask]
        graph.part_ids = graph.part_ids[node_mask]

    # Adjusting the number of tracks in the global feature
    graph["globals"].x[0][0] = torch.sum(node_mask)

    # Removing the pruned edges
    for edge_type in edge_types:
        if edge_type[0] == node_type and edge_type[2] == node_type:
            mask = node_mask[graph[edge_type].edge_index[0]] & node_mask[graph[edge_type].edge_index[1]]
            graph[edge_type].edge_index = lookup[graph[edge_type].edge_index[:, mask]]
        elif edge_type[0] == node_type:
            mask = node_mask[graph[edge_type].edge_index[0]]
            graph[edge_type].edge_index = graph[edge_type].edge_index[:, mask]
            graph[edge_type].edge_index[0] = lookup[graph[edge_type].edge_index[0]]
        else:
            mask = node_mask[graph[edge_type].edge_index[1]]
            graph[edge_type].edge_index = graph[edge_type].edge_index[:, mask]
            graph[edge_type].edge_index[1] = lookup[graph[edge_type].edge_index[1]]
        graph[edge_type].edges = graph[edge_type].edges[mask, :]
        if "lca" in graph[edge_type]:
            graph[edge_type].lca = graph[edge_type].lca[mask]
        graph[edge_type].y = graph[edge_type].y[mask]
    # here we need to be careful since it only return 1 mask even though it creates multiple
    return mask
