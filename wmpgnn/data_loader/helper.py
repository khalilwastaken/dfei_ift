from itertools import chain

import torch
from torch_geometric.loader import DataLoader

from wmpgnn.util.pruners import *
from wmpgnn.data_loader.weights_calculator import get_hetero_weight


def load_dataset(path, configs, mode="train", pv_asso_model=None):
    with open(path, "rb") as f:
        data = torch.load(f, weights_only=False)

    """Applying pruning for different using truth pruning initially"""
    filtered_data = None
    if "true" in configs["settings"]["graph_mode"]: # here we need to apply the pv association as well
        data_selbool = torch.ones(len(data))
        edge_types = [("tracks", "to", "tracks"), ("tracks", "to", "pvs")]
        for i, evt in enumerate(data):
            y_nodes = evt["tracks"].ft != 1  # 0 bbar 2 b
            if "frag" in configs["settings"]["graph_mode"]:
                frag_selbool = evt["tracks"].frag != 0
                y_nodes = y_nodes | frag_selbool
            if configs["settings"]["node_sel"] == "true":
                true_node_pruning(y_nodes, evt, "tracks", edge_types)
            elif configs["settings"]["node_sel"] == "default":
                node_pruning(y_nodes, evt, "tracks", edge_types)
            elif configs["settings"]["node_sel"] == "zeros":
                test_node_pruning(y_nodes, evt, "tracks", edge_types)
            if evt[("tracks", "to", "tracks")].y.shape[0] == 0 or torch.all(evt[("tracks", "to", "tracks")].y == 0):
                data_selbool[i] = 0
        filtered_data = [d for d, sel in zip(data, data_selbool) if sel]
    elif pv_asso_model is not None:
        pv_data = DataLoader(filtered_data if filtered_data is not None else data)

        filtered_data = []
        for evt in pv_data:
            res = pv_asso_model.forward(evt)
            filtered_data.append(res)
        filtered_data = list(chain.from_iterable(filtered_data))
    else:
        filtered_data = data

    """Making the graph bidirectional"""
    for data in filtered_data:
        edge_type = ('tracks', 'to', 'tracks')
        store = data[edge_type]

        store.edge_index = torch.cat([store.edge_index, store.edge_index.flip(0)], dim=1)
        store.edges = store.edges.repeat(2, 1)  # More efficient than cat([x]*2)
        store.y = store.y.repeat(2)

    if mode == "weights_only":
        weights = get_hetero_weight(filtered_data, configs)
        return weights
    elif "weights" in mode:
        weights = get_hetero_weight(filtered_data, configs)
        return filtered_data, weights
    else:
        return filtered_data
