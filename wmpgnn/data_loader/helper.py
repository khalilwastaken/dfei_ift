from itertools import chain
import copy

from multiprocessing.pool import ThreadPool
import torch
from torch_geometric.loader import DataLoader

from wmpgnn.util.pruners import *
from wmpgnn.data_loader.weights_calculator import get_hetero_weight
from wmpgnn.util.pv_association import pv_association




def create_pv_assoed_data(data, metrics, node_thrs, n_cores=4): # this is happening on 4
    def process_single_graph(args):
        """Process one graph independently"""
        graph, metric = args
        graph_results = []

        # adding the pred information
        graph["tracks"].pred_y = metric["tracks_pred_y"]
        graph[('tracks', 'tracks')].lca = metric["lca_score"]
        graph[('tracks', 'tracks')].pred_y = metric["tr_tr_pred_y"]

        # Getting the ntracks npvs for pv asso
        edge_index = graph[("tracks", "to", "pvs")]["edge_index"]
        graph["num_pvs"] = edge_index[1].max().item() + 1
        pred_pv = pv_association(edge_index, metric["pv_desc"])



        # finding the pv which are interesting
        node_selbool = metric["tracks_pred_y"] >= node_thrs
        pv_oi, counts = torch.unique(pred_pv[node_selbool], return_counts=True)
        pv_oi = pv_oi[counts > 2]

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
            graph_results.append(pv_oi_data) # check data dont need
        return graph_results


    graphs = data.to_data_list()
    metrics = metrics
    args_list = [(graph, metric) for graph, metric in zip(graphs, metrics)]

    # Parallel processing of graphs
    with ThreadPool(processes=n_cores) as pool:
        results_nested = pool.map(process_single_graph, args_list)

    # Flatten results
    pv_asso_data = []
    for graph_results in results_nested:
        pv_asso_data.extend(graph_results)

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
    elif pv_asso_model is not None: # this is happening on 2 cpus
        pv_data = DataLoader(filtered_data if filtered_data is not None else data, batch_size=1024)
        filtered_data = []
        for evt in pv_data:
            if pv_asso_model.name == "pv_asso_module":
                original_data = copy.deepcopy(evt)
                metrics = pv_asso_model.forward(evt)
                res = create_pv_assoed_data(original_data, metrics, pv_asso_model.node_thrs)
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
