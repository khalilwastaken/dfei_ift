import copy
import torch

from multiprocessing.pool import ThreadPool

from wmpgnn.util.pruners import *


def find_components_bfs(edge_index):
    # breadth first search
    edge_index = edge_index.numpy()
    all_nodes = set(edge_index[0]).union(set(edge_index[1]))

    adj_list = {node: set() for node in all_nodes}
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        adj_list[src].add(dst)
        adj_list[dst].add(src)

    visited = set()
    components = []
    for start_node in all_nodes:
        if start_node in visited:
            continue

        component = set()
        queue = [start_node]

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue

            visited.add(node)
            component.add(node)

            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append([int(node) for node in component])

    return components


def pv_association(edge_index, pv_desc, edge_selbool=None):
    ntracks = edge_index[0].max().item() + 1
    npvs = edge_index[1].max().item() + 1
    pv_desc = pv_desc.view(ntracks, npvs)
    pred_pv = torch.argmax(pv_desc, dim=1)

    # Here we implement different models

    """b_edge_idx = edge_index[:, edge_selbool]
    b_systems = find_components_bfs(b_edge_idx)
    for b in b_systems:
        print(torch.argmax(torch.sum(pred_desc[b], dim=0)))

    print("="*30)"""
    return pred_pv


def pv_associate_graph(args):
    """Process one graph independently"""
    graph, metric, node_thr = args
    graph_results = []

    # Adding information from the pv association DFEI model if used
    if "tracks_pred_y" in metric.keys():
        graph["tracks"].pred_y = metric["tracks_pred_y"]
    if "lca_score" in metric.keys():
        graph[('tracks', 'tracks')].lca = metric["lca_score"]
    if "tr_tr_pred_y" in metric.keys():
        graph[('tracks', 'tracks')].pred_y = metric["tr_tr_pred_y"]

    # Adding num pvs to original graph
    edge_index = graph[("tracks", "pvs")]["edge_index"]
    graph["num_pvs"] = edge_index[1].max().item() + 1
    pred_pv = pv_association(edge_index, metric["pv_desc"],metric["tr_tr_edge_selbool"])

    # finding the pv which are interesting
    if node_thr is not None:
        node_selbool = metric["tracks_pred_y"] >= node_thr
    else:
        node_selbool = graph["tracks"].ft != 1
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
        graph_results.append(pv_oi_data)  # check data dont need
    return graph_results


def pv_associate_data(data, metrics, node_thr=None, n_cores=4):
    graphs = data.to_data_list()
    metrics = metrics
    args_list = [(graph, metric, node_thr) for graph, metric in zip(graphs, metrics)]

    # Parallel processing of graphs
    with ThreadPool(processes=n_cores) as pool: # adjust to 1 during develop
        results_nested = pool.map(pv_associate_graph, args_list)

    # Flatten results
    pv_asso_data = []
    for graph_results in results_nested:
        pv_asso_data.extend(graph_results)

    return pv_asso_data
