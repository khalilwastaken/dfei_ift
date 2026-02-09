from itertools import chain
import copy

import torch
from torch_geometric.loader import DataLoader

from wmpgnn.util.pruners import *
from wmpgnn.data_loader.weights_calculator import get_hetero_weight



def pv_asso(data, metrics, node_thrs):
    def check_data(evt):
        for key in evt["tracks"].keys():
            if evt["tracks"][key].shape[0] == 0:
                return False
        for key in evt["pvs"].keys():
            if evt["pvs"][key].shape[0] == 0:
                return False
        for key in evt[("tracks", "to", "tracks")].keys():
            if evt[("tracks", "to", "tracks")][key].shape[0] == 0:
                return False
        for key in evt[("tracks", "to", "pvs")].keys():
            if evt[("tracks", "to", "pvs")][key].shape[0] == 0:
                return False
        return True

    pv_asso_data = []
    lca_score, tracks_pred_y, tr_tr_pred_y, tr_pv_pred_y = metrics[0], metrics[1], metrics[2], metrics[3]

    # batch information to add later on to batch
    track_batch = data["tracks"].batch
    pv_batch = data['pvs'].batch
    tr_tr_edge_idx = data[('tracks', 'tracks')].edge_index
    tr_pv_edge_idx = data[('tracks', 'pvs')].edge_index

    graphs = data.to_data_list()
    for i, graph in enumerate(graphs):
        track_selbool = track_batch == i
        graph["tracks"].pred_y = tracks_pred_y[track_selbool]

        tr_tr_selbool = (track_batch[tr_tr_edge_idx[0]] == i) & (track_batch[tr_tr_edge_idx[1]] == i)
        graph[('tracks', 'tracks')].lca = lca_score[tr_tr_selbool]
        graph[('tracks', 'tracks')].pred_y = tr_tr_pred_y[tr_tr_selbool]

        # add here something linke num_pvs
        graph["num_pvs"] = torch.tensor([graph["pvs"].x.shape[0]])

        tr_pv_selbool = (track_batch[tr_pv_edge_idx[0]] == i) & (pv_batch[tr_pv_edge_idx[1]] == i)
        pv_desc = tr_pv_pred_y[tr_pv_selbool]

        # getting the association per pv for every track
        ntracks = torch.unique(graph[("tracks", "to", "pvs")]["edge_index"][0]).shape[0]
        npvs = torch.unique(graph[("tracks", "to", "pvs")]["edge_index"][1]).shape[0]
        pred_pv = torch.argmax(pv_desc.view(ntracks, npvs), dim=1)

        # finding the pv which are interesting
        node_selbool = tracks_pred_y[track_selbool] >= node_thrs
        pv_oi, counts = torch.unique(pred_pv[node_selbool], return_counts=True)
        # Here we can do some fancy stuff to boost the pv asso, later on which should also be time efficient
        pv_oi = pv_oi[counts > 2]  # require at least two identified signal track to point to the same pv
        for pv in pv_oi:
            if pv_oi.shape[0] == 1:
                pv_oi_data = graph
            else:
                pv_oi_data = copy.deepcopy(graph)

            # removes all the nodes associated to a different pv
            nodes_asso_pv_selbool = pred_pv == pv
            true_node_pruning(nodes_asso_pv_selbool, pv_oi_data, "tracks",
                              [('tracks', 'to', 'tracks'), ('tracks', 'to', 'pvs')])

            # Lastly remove the pv which are not used anymore
            pv_selbool = torch.zeros(pv_oi_data["pvs"].x.shape[0], dtype=torch.bool)
            pv_selbool[pv] = True

            true_node_pruning(pv_selbool, pv_oi_data, "pvs", [('tracks', 'to', 'pvs')])
            del pv_oi_data["last_chunk"]
            if check_data(pv_oi_data):  # some data safety checks
                pv_asso_data.append(pv_oi_data.to('cpu'))
    return pv_asso_data


def load_dataset(path, configs, mode="train", pv_asso_model=None):
    with open(path, "rb") as f:
        data = torch.load(f, weights_only=False)

    """Applying pruning for different using truth pruning initially"""
    filtered_data = None
    if "true" in configs["settings"]["graph_mode"]:  # here we need to apply the pv association as well
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
        pv_data = DataLoader(filtered_data if filtered_data is not None else data, batch_size=512)
        filtered_data = []
        for evt in pv_data:
            if pv_asso_model.name == "pv_asso_module":
                original_data = copy.deepcopy(evt)
                metrics = pv_asso_model.forward(evt)
                res = pv_asso(original_data, metrics, pv_asso_model.node_thrs)
            else:
                res = pv_asso_model.forward(evt)

            # if name is pv_asso_module do pv asso here
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
        if hasattr(store, 'lca') and store.lca is not None:
            store.lca = store.lca.repeat(2, 1)
        if hasattr(store, 'pred_y') and store.lca is not None:
            store.pred_y = store.pred_y.repeat(2)
        # if exists pred y and lca on edges as well

    if mode == "weights_only":
        weights = get_hetero_weight(filtered_data, configs)
        return weights
    elif "weights" in mode:
        weights = get_hetero_weight(filtered_data, configs)
        return filtered_data, weights
    else:
        return filtered_data
