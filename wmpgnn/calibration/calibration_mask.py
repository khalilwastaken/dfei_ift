import os
import glob
import yaml

import torch
import pandas as pd

from multiprocessing.pool import ThreadPool

from wmpgnn.util.bfs import find_components_bfs
from wmpgnn.reconstruction.signal_dict import get_ref_signal, sig_matching
from wmpgnn.reconstruction.reconstruction_helper import *


class CalibrationClass:
    def __init__(self, configs):
        self.configs = configs

        cacheing_config = glob.glob(f"{configs['settings']['data_dir']}/{configs['evaluate']['sample'][0]}/*.yaml")[0]
        with open(cacheing_config, "r") as file:
            self.cacheing_config = yaml.safe_load(file)

        # Obtain the normalization dictionary
        self.norm = False  # obtain_normalization(configs)

        # Threshold for edge and node pruning
        self.node_thrs = configs["inference"]["node_prune_thr"]
        self.edge_thrs = configs["inference"]["edge_prune_thr"]

        to_whiten = configs["settings"]["to_whiten"]
        track_x = self.cacheing_config['graphs']['tracks_nodes']
        track_pid = self.cacheing_config['graphs']['tracks_pid']
        if isinstance(to_whiten, list):
            self.track_x_whiten = torch.tensor([f in to_whiten for f in track_x])
            self.track_pid_whiten = torch.tensor([f in to_whiten for f in track_pid])
        elif isinstance(configs["settings"]["to_whiten"], int):
            self.track_x_whiten = torch.ones(len(track_x)).to(bool)
            self.track_pid_whiten = torch.ones(len(track_pid)).to(bool)

        self.n_cores = configs['ncpus']['whiten']

    def remove_sig(self, path, graph_data):
        base_dir = os.path.basename(os.path.dirname(path))

        channel = base_dir.split('_')[0]

        if channel not in self.configs["settings"]["calibrate_mode"]:
            return graph_data
        else:
            channel_prop = get_ref_signal(channel)

            args_list = [(graph, channel_prop, path) for graph in graph_data]

            with ThreadPool(processes=1) as pool:
                results_nested = pool.map(self.whitening, args_list)
            return list(results_nested)

    def whitening(self, args):
        graph, channel_prop, path = args
        # Denorm data
        graph["tracks"].org_x = graph["tracks"].x.clone()
        graph["tracks"].org_pid = graph["tracks"].pid.clone()
        is_whiten = torch.tensor([0])  # flag to add if the event is whiten or not

        # Pruning needs to be applied manually, since it is done in place (alternatively deep copy it)
        """Manual pruning"""
        # Node pruning
        node_selbool = graph["tracks"].pred_y > self.node_thrs
        node_indices = torch.arange(0, graph["tracks"].pred_y.shape[0])[node_selbool]
        # Edge pruning
        edge_mask_1 = torch.isin(graph[("tracks", "tracks")].edge_index[0], node_indices)
        edge_mask_2 = torch.isin(graph[("tracks", "tracks")].edge_index[1], node_indices)
        edge_selbool = graph[("tracks", "tracks")].pred_y > self.edge_thrs
        lca_selbool = torch.argmax(graph[("tracks", "tracks")].lca, dim=1) != 0
        edge_bool = edge_mask_1 & edge_mask_2 & edge_selbool & lca_selbool

        # Get lca and edge score to find the B systems
        lca = torch.argmax(graph[("tracks", "tracks")].lca, dim=1)[edge_bool]
        edges = graph[("tracks", "tracks")].edge_index[:, edge_bool]

        reco_components = add_lca(find_components_bfs(edges), lca)
        reco_components = add_base_quant(graph, reco_components, self.cacheing_config)

        for reco_component in reco_components:
            if sig_matching(reco_component, channel_prop, "reco"):
                is_whiten = torch.tensor([1])

                # Set vars which should be whitened to 0
                # Didn't found a cleaner solution...
                nodes = reco_component['nodes']
                tmp_x = graph["tracks"].x[nodes].clone()
                tmp_x[:, self.track_x_whiten] = 0
                graph['tracks'].x[nodes] = tmp_x
                tmp_pid = graph["tracks"].pid[nodes].clone()
                tmp_pid[:, self.track_pid_whiten] = 0
                graph['tracks'].pid[nodes] = tmp_pid
        if "tst_data" in path:
            graph["is_whiten"] = is_whiten
        del graph['tracks'].org_x, graph['tracks'].org_pid
        return graph
