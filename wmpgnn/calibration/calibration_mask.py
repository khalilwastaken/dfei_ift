import os
import re
import yaml

from dataclasses import dataclass

import torch
import numpy as np
import pandas as pd

from multiprocessing.pool import ThreadPool

# this can be put to utils and add LCA the true reco info inference during reco
from wmpgnn.calibration.decay_dict import decay_prop, pid_dict
from wmpgnn.util.bfs import *

# module to create calibration mask
# in data loader helper add if statement if to use calibration mode or not to call the function
# Add a list of channels in config file for decay channels which one wants to use (primarily for training)
# -> need to change something in adjust config -> always do in evaluation if calibration is on

"""What does this function do"""


# Takes a given channel with the DFEI output
# using BFS to isolate B particles
# Do an "unfolding" of the normalization to get the pid information
# Based on the pid and the LCA do a matching to obtain signal decay channel
# (
# This has the problem that there can be overlap, as a priory it is not 100% confident in match
# it only requires correct final pid and matching LCA score -> could lead to ambiguity
# )
# Whiten the features -> add mode for all / only pid + charge

@dataclass
class Normalization:  # hold information regarding normalization
    norm: pd.DataFrame
    node_features: list[str]


# this can be ported to utils
def obtain_normalization(configs):
    # this should be generalized  to just load in the var which is needed
    # This is a current problem, one needs to copy the config normalization file
    norm_file = "normalization_dict.pt"
    # Determining simulation data mode based on log dir
    if configs["log_dir"] == "LHCb_logs":
        key = "LHCbcollision"
    elif configs["log_dir"] == "pythia_logs":
        key = "nu7p6"
    else:
        raise NotImplementedError
    try:
        normalization_dict = torch.load(norm_file)[key]
    except:
        raise ValueError("norm file not found, pls copy manually")

    # Obtain the features of how the graph is setup to reverse normalization
    data_dir = configs["settings"]["data_dir"]
    # one of the two should exist
    if configs.get("evaluate", {}).get("sample"):
        channel = configs.get("evaluate", {}).get("sample")[0]
    elif configs.get("settings", {}).get("sample"):
        channel = configs.get("settings", {}).get("sample")[0]
    else:
        raise NotImplementedError("Something went wrong here pls check")
    with open(f"{data_dir}/{channel}/config.yaml", "r") as file:
        graph_configs = yaml.safe_load(file)["node_features"]

    # variables where we need to revert the normalization
    normalization = pd.DataFrame(normalization_dict).T.loc[graph_configs]

    selected = pd.DataFrame(index=normalization.index, columns=['value'])
    skew_thresh = 1.0
    for feature in normalization.index:
        if abs(normalization.loc[feature, 'skew']) > skew_thresh:
            selected.loc[feature, 'value'] = (normalization.loc[feature, 'median'],
                                              normalization.loc[feature, 'iqr'])
        else:
            selected.loc[feature, 'value'] = (normalization.loc[feature, 'mean'],
                                              normalization.loc[feature, 'std'])
    selected[['val1', 'val2']] = pd.DataFrame(selected['value'].tolist(), index=selected.index)
    selected = selected.drop(columns='value')
    selected = torch.tensor(selected.values).T

    return Normalization(selected, graph_configs)


def whitening(args):
    graph, prune_thr, to_whiten, channel_prop, norm = args

    is_whiten =  torch.tensor([0]) # flag to add if the event is whiten or not
    graph["tracks"].org_x = graph["tracks"].x.clone()
    graph["tracks"].org_pid = graph["tracks"].pid.clone()

    # Need to apply both node and edge pruning to increase the high purity, edge prune alone can be not sufficient
    node_selbool = graph["tracks"].pred_y > prune_thr[0]
    node_indices = torch.arange(0, graph["tracks"].pred_y.shape[0])[node_selbool]

    edge_mask_1 = torch.isin(graph[("tracks", "tracks")].edge_index[0], node_indices)
    edge_mask_2 = torch.isin(graph[("tracks", "tracks")].edge_index[1], node_indices)
    edge_selbool = graph[("tracks", "tracks")].pred_y > prune_thr[1]
    lca_selbool = torch.argmax(graph[("tracks", "tracks")].lca, dim=1) != 0
    edge_bool = edge_mask_1 & edge_mask_2 & edge_selbool & lca_selbool

    # Get lca and edge score to find the B systems
    lca = torch.argmax(graph[("tracks", "tracks")].lca, dim=1)[edge_bool]
    edges = graph[("tracks", "tracks")].edge_index[:, edge_bool]
    components = find_components_bfs(edges)

    for comp in components:
        nodes_indx = comp["nodes"]
        edges_indx = comp["edge_indices"]
        if not len(nodes_indx) == channel_prop[0].shape[0] and not len(edges_indx) == channel_prop[1].shape[0] * 2:
            continue

        # Here we are combining the features of tracks and pid to unfold the normalization
        try:
            combined_features = torch.cat((graph["tracks"].x[nodes_indx], graph["tracks"].pid[nodes_indx]), dim=1)
        except:
            raise NotImplementedError("Probably realistic pid issue")

        # Revert the normalization
        denorm_data = combined_features * norm.norm[1] + norm.norm[0]
        prob_nn_mask = np.array([c.lower().startswith('prob') for c in norm.node_features])
        pid_features = np.array(norm.node_features)[prob_nn_mask]
        pid_response = denorm_data[:, prob_nn_mask]
        pred_pid = pid_features[torch.argmax(pid_response, dim=1)]

        charge = denorm_data[:, np.array(norm.node_features) == "charge"].squeeze()

        # Expanded for potential debug
        sel_pid = []
        for i in pred_pid:
            try:
                sel_pid.append(pid_dict[i])
            except KeyError:
                raise KeyError("Missing key", i)
        sel_pid = torch.sort((torch.tensor(sel_pid) * charge).round().to(torch.int32)).values

        true_pos_pid, true_neg_pid = torch.sort(channel_prop[0]).values, torch.sort(-1 * channel_prop[0]).values,
        matching_final = torch.equal(sel_pid, true_pos_pid) or torch.equal(sel_pid, true_neg_pid)
        matching_lca = torch.equal(lca[edges_indx], torch.cat([channel_prop[1], channel_prop[1]]))
        if matching_final and matching_lca:
            # what do we want to whiten
            combined_features[:, to_whiten] = 0

            pid_features = [x for x in norm.node_features if x.startswith("Prob")]
            pid_mask = [f in pid_features for f in norm.node_features]
            node_features = [x for x in norm.node_features if x not in pid_features]
            node_mask = [f in node_features for f in norm.node_features]

            # org contain the raw information which is uses to evaluate
            graph["tracks"].x[nodes_indx] = combined_features[:, node_mask]
            graph["tracks"].pid[nodes_indx] = combined_features[:, pid_mask]
            is_whiten = torch.tensor([1])
    graph["is_whiten"] = is_whiten
    return graph


def adjust_for_calibration(configs, path, graph_data, n_cores=1):
    base_dir = os.path.basename(os.path.dirname(path))
    if "LHCb" in configs["log_dir"]:
        channel = re.search(r'\d{8}_(.*)', base_dir).group(1)
    else:
        raise NotImplementedError

    if channel not in configs["settings"]["calibrate_mode"]:
        # for inclusive or if we want to exclude some channel
        return graph_data
    else:
        # Obtain the normalization
        norm = obtain_normalization(configs)

        # perform calibration, the graph is bidrectional, allow for multi processing
        if channel not in decay_prop.keys():
            raise NotImplementedError("Channel {} not implemented".format(channel))

        # thrs
        node_thrs = configs["inference"]["node_prune_thr"]
        edge_thrs = configs["inference"]["edge_prune_thr"]
        prune_thrs = [node_thrs, edge_thrs]

        channel_prop = decay_prop[channel]
        if isinstance(configs["settings"]["to_whiten"], list):
            to_whiten = np.array([f in configs["settings"]["to_whiten"] for f in norm.node_features])
        elif isinstance(configs["settings"]["to_whiten"], int):
            to_whiten = torch.ones(len(norm.node_features)).to(bool)
        else:
            raise NotImplementedError

        args_list = [(graph, prune_thrs, to_whiten, channel_prop, norm) for graph in graph_data]

        with ThreadPool(processes=1) as pool:
            results_nested = pool.map(whitening, args_list)
        # Flatten results
        whiten_data = []
        for graph_results in results_nested:
            whiten_data.append(graph_results)

        return whiten_data
