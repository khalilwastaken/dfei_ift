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
    res_dict["part_ids"] = '_'.join(str(x.item()) for x in graph["pid_holder"])
    return res_dict


def get_pv_asso(sig_dict, cluster, pv_des):
    if cluster is None:
        sig_dict["npvs"] = pv_des["npvs"]
        sig_dict["true_pv"] = str(-1)
        sig_dict["pred_pv"] = str(-1)
        sig_dict["pred_pv_b_lvl"] = -1
        sig_dict["minIP_pv"] = str(-1)
        return sig_dict

    nodes = cluster['nodes']

    # Individual track scores
    true_pv = pv_des["true"][nodes]
    minIP_pv = pv_des["ip"][nodes]
    pred_pv = pv_des["pred"][cluster['nodes']]
    pred_pv_track = torch.argmax(pred_pv, dim=1)
    pred_pv_system = torch.argmax(torch.sum(pred_pv, dim=0))

    # Storing the information
    sig_dict["npvs"] = pv_des["npvs"]
    sig_dict["true_pv"] = '_'.join(str(x.item()) for x in true_pv)
    sig_dict["pred_pv"] = '_'.join(str(x.item()) for x in pred_pv_track)
    sig_dict["pred_pv_b_lvl"] = int(pred_pv_system.item())
    sig_dict["minIP_pv"] = '_'.join(str(x.item()) for x in minIP_pv)
    return sig_dict


def get_sig_lvl_info(sig_dict, graph):
    keys = [
        "minIP_pv_b_lvl", "true_pv_b_lvl",
        "B_M", "B_PT", "B_ETA",

        "B_v0_OSKaon", "B_v1_OSKaon",
        "B_v0_OSMuon", "B_v1_OSMuon",
        "B_v0_OSElectron", "B_v1_OSElectron",
        "B_v0_SSKaon", "B_v1_SSKaon",
        "B_v0_SSPion", "B_v1_SSPion",
        "B_v0_SSProton", "B_v1_SSProton",
        "B_v1_IFT_Bs", "B_v1_IFT_Bd",
    ]

    for k in keys:
        if k in graph:
            sig_dict[k] = graph[k].item()
    return sig_dict