import torch
import numpy as np


def get_asso_frag(sig_dict, graph, cluster):
    # check if the frag exists if not just return it problem solved. yey
    if "frag_y" not in graph:
        return sig_dict
    res_dict = sig_dict

    cluster_keys = cluster['node_keys']
    keys = graph['final_keys']
    b_daugthers_mask = np.isin(keys, cluster_keys)
    b_idx = str(graph["tracks"].asso_hh[b_daugthers_mask][0].item())

    int_list = graph["frag_y"].tolist()
    first_two = np.array([str(num)[:2] for num in int_list])
    rest_digits = np.array([str(num)[2:] if len(str(num)) > 2 else '' for num in int_list])

    frag_selbool = rest_digits == b_idx
    res_dict["frags"] = "_".join(first_two[frag_selbool])
    res_dict["frags_pid"] = "_".join(str(x.item()) for x in graph["frag_pid"][rest_digits == b_idx])
    return res_dict


def get_pred_ft(sig_dict, graph, cluster, ft_score):
    res_dict = sig_dict
    # Save combined b bbar score, save individual scores, save pid of final
    cluster_keys = cluster['node_keys']
    keys = graph['final_keys']
    b_daugthers_mask = np.isin(keys, cluster_keys)

    # Get the pid of the particles
    res_dict["final_pid"] = ','.join(str(x.item()) for x in graph["part_ids"][b_daugthers_mask])

    # Get the individual scores stored as strings
    ft_score = ft_score[b_daugthers_mask].cpu()
    res_dict["final_b_score"] = ','.join(str(x.item()) for x in ft_score[:, :1].squeeze())
    res_dict["final_bbar_score"] = ','.join(str(x.item()) for x in ft_score[:, 2:].squeeze())

    res_dict["ft_b_score"], _, res_dict["ft_bbar_score"] = ft_score.mean(dim=0).tolist()

    return res_dict


def get_pv_asso(sig_dict, graph, cluster, pv_des):
    res_dict = sig_dict
    # Get the key information of the cluster which is looked at
    cluster_keys = cluster['node_keys']
    keys = graph['final_keys']
    b_daugthers_mask = np.isin(keys, cluster_keys)

    # Get the individual scores stored as strings
    true_pv = pv_des["true"][b_daugthers_mask]
    minIP_pv = pv_des["ip"][b_daugthers_mask]

    # Either per track level or on the full B system
    pred_pv = pv_des["pred"][b_daugthers_mask]
    pred_pv_track = torch.argmax(pred_pv, dim=1)
    pred_pv_system = torch.argmax(torch.sum(pred_pv, dim=0))

    res_dict["npvs"] = pv_des["npvs"]
    res_dict["true_pv"] = '_'.join(str(x.item()) for x in true_pv)
    res_dict["pred_pv"] = '_'.join(str(x.item()) for x in pred_pv_track)
    res_dict["pred_pv_b_lvl"] = int(pred_pv_system.item())
    res_dict["minIP_pv"] = '_'.join(str(x.item()) for x in minIP_pv)
    return res_dict
