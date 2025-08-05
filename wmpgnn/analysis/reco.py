import torch

import pandas as pd
import numpy as np

from particle import Particle

from wmpgnn.performance.reconstruction import reconstruct_decay, flatten, match_decays


def get_ref_signal(ref_signal):  # Here we can define them all
    if ref_signal == 'Bs_JpsiPhi':
        signal_decay = {'daughters': ['mu+', 'mu-', 'K+', 'K-'], 'mothers': ['B(s)0']}
        cc_signal_decay = {'daughters': ['mu+', 'mu-', 'K+', 'K-'], 'mothers': ['B(s)~0']}
        return (signal_decay, cc_signal_decay)
    return {}


def particle_name(id_):
    if id_ == 0:
        return 'ghost'
    elif id_ == 10413:
        return 'D1(2420)+'
    elif id_ == -10413:
        return 'D1(2420)-'
    elif id_ == 4422:
        return 'Chi_cc++'
    elif id_ == -4422:
        return 'Chi_cc--'
    elif id_ == 4432:
        return 'Omega_cc++'
    elif id_ == -4432:
        return 'Omega_cc--'
    else:
        return Particle.from_pdgid(id_).name


def lca_reco_matrix(graph, mode="reco"):
    edge_index = graph[('tracks', 'to', 'tracks')].edge_index.cpu()
    edges = graph[('tracks', 'to', 'tracks')].edges.cpu()

    pd_matrix = pd.DataFrame(edge_index.T, columns=['senders', 'receivers'])
    if mode == "reco":
        pd_matrix["LCA_dec"] = torch.argmax(edges, axis=-1).tolist()  # LCA decision
    else:
        pd_matrix["LCA_dec"] = graph[('tracks', 'to', 'tracks')].y.tolist()  # LCA decision
    pd_matrix.set_index(['senders', 'receivers'], inplace=True)
    pd_matrix = pd_matrix.reset_index()
    pd_matrix = pd_matrix[pd_matrix['senders'] < pd_matrix['receivers']]
    return pd_matrix


def lca_truth_matrix(graph):
    senders = graph.truth_senders.cpu()
    receivers = graph.truth_receivers.cpu()
    init_y = graph["truth_y"].cpu()

    truth_lca = pd.DataFrame(np.column_stack((senders, receivers)), columns=['senders', 'receivers'])
    truth_lca['LCA_dec'] = np.reshape(
        np.argmax(
            np.reshape(init_y, (init_y.shape[0], 4)), axis=-1),
        (-1,))
    truth_lca = truth_lca[truth_lca['senders'] < truth_lca['receivers']]
    truth_lca['LCA_id_label'] = list(map(particle_name, graph['truth_moth_ids'].cpu().numpy()))
    truth_lca['LCA_id'] = graph['truth_moth_ids'].cpu().numpy()
    truth_lca['TrueFullChainLCA'] = graph['lca_chain'].cpu()
    return truth_lca


def reco_event(graph, event, config, signal, sig_df, evt_df):
    ref_signal = get_ref_signal(signal)
    graph = graph.cpu()

    """Check if reco B exist and true B exist"""
    if torch.sum(graph['tracks', 'to', 'tracks'].y) == 0:
        print("no true B exist")
    if config["LCA"] and torch.sum(torch.argmax(graph['tracks', 'to', 'tracks'].edges, dim=-1)) == 0:
        print("no reco B candidate found")

    """Obtain the LCA scores of both true and reco graphs"""
    if config["LCA"]:
        reco_LCA = lca_reco_matrix(graph, mode="reco")
    else:
        reco_LCA = lca_reco_matrix(graph, mode="true")
    true_LCA = lca_truth_matrix(graph)

    particle_keys = graph["final_keys"].tolist()
    n_part = len(particle_keys)
    rc_dict, r_nclust_order, _ = reconstruct_decay(reco_LCA, particle_keys)

    particle_keys = graph["truth_part_keys"].tolist()
    particle_ids = list(map(particle_name, graph['truth_part_ids'].numpy()))
    tc_dict, t_nclust_order, max_chain_depth = reconstruct_decay(true_LCA, particle_keys,
                                                                 particle_ids=particle_ids, truth_level_simulation=1)
    if tc_dict != {}:
        part_heavy_h = flatten([tc_dict[tc_firstkey]['node_keys'] for tc_firstkey in tc_dict.keys()])
        n_part_heavy_h = len(part_heavy_h)
        n_bkg_part = n_part - n_part_heavy_h

        if rc_dict != {}:
            sel_part = flatten(
                [rc_dict[tc_firstkey]['node_keys'] for tc_firstkey in rc_dict.keys()])
            n_sel_part = len(sel_part)
            n_sel_heavy_h = len(list(set(sel_part).intersection(part_heavy_h)))
            n_sel_bkg_part = n_sel_part - n_sel_heavy_h
        else:
            n_sel_part = 0
            n_sel_heavy_h = 0
            n_sel_bkg_part = 0

        perfect_evt_reco = 1  # Flag for perfect event reco
        if n_sel_bkg_part > 0:
            perfect_evt_reco = 0

        # Looping over reco candidates
        for tc_key in tc_dict.keys():
            tc = tc_dict[tc_key]
            sig_match = 0
            if ref_signal:
                labels = tc['labels']
                mothers = [label[3:] for label in labels if 'c' == label[0]]
                node_keys = tc['node_keys']
                daughters = [label.split(':')[1] for label in labels if int(label.split(':')[0][1:]) in node_keys]
                if match_decays(daughters, ref_signal[0]['daughters']) or match_decays(daughters,
                                                                                       ref_signal[1]['daughters']):
                    check_mothers1 = True
                    check_mothers2 = True
                    for i in range(len(ref_signal[0]['mothers'])):
                        if ref_signal[0]['mothers'][i] not in mothers:
                            check_mothers1 = False
                        if ref_signal[1]['mothers'][i] not in mothers:
                            check_mothers2 = False
                    sig_match = int(check_mothers1 or check_mothers2)

            n_sig_part = len(tc['node_keys'])

            if tc_key in rc_dict.keys():
                per_sig_reco = int(
                    rc_dict[tc_key]['node_keys'] == tc['node_keys']
                    and rc_dict[tc_key]['LCA_values'] == tc['LCA_values']
                )
            else:
                per_sig_reco = 0
            perfect_evt_reco *= per_sig_reco

            sig_dict = {"perfect_reco": 0, "all_particles": 0, "none_iso": 0, "part_reco": 0, "none_associated": 0,
                        "none_iso_n_bkg": -999}
            for rc in rc_dict.values():
                true_in_reco = np.sum(np.isin(tc['node_keys'], rc['node_keys'])) / len(tc['node_keys'])
                if rc['node_keys'] == tc['node_keys']:
                    sig_dict["all_particles"] = 1
                    if rc['LCA_values'] == tc['LCA_values']:
                        sig_dict["perfect_reco"] = 1
                    break
                elif true_in_reco == 1 and len(rc['node_keys']) > len(tc['node_keys']):
                    sig_dict["none_iso"] = 1  # background tracks in signal
                    break
                elif 0.2 <= true_in_reco < 1:
                    sig_dict["part_reco"] = 1  # FT decision can not be trusted
                    sig_dict["none_iso_n_bkg"] = len(rc['node_keys']) - len(tc['node_keys'])
            if sig_dict["all_particles"] == 1:
                sig_dict["none_iso"] = sig_dict["part_reco"] = 0
            if sig_dict["none_iso"] == 1:
                sig_dict["part_reco"] = 0
            if sig_dict["all_particles"] == 0 and sig_dict["none_iso"] == 0 and sig_dict["part_reco"] == 0:
                sig_dict["none_associated"] = 1

            # Get origin B id
            indices = [particle_keys.index(x) for x in tc['node_keys']]
            signal_LCA_id = true_LCA[true_LCA['senders'].isin(indices) | true_LCA['receivers'].isin(indices)][
                "LCA_id"]
            values, counts = np.unique(signal_LCA_id, return_counts=True)
            origin_B_id = values[np.argmax(counts)]

            sig_df = sig_df._append({'EventNumber': event,
                                     'NumParticlesInEvent': n_part,
                                     'NumSignalParticles': n_sig_part,
                                     'PerfectSignalReconstruction': per_sig_reco,
                                     'AllParticles': sig_dict["all_particles"],
                                     'PerfectReco': sig_dict["perfect_reco"],
                                     'NoneIso': sig_dict["none_iso"],
                                     'PartReco': sig_dict["part_reco"],
                                     'NotFound': sig_dict["none_associated"],
                                     'SigMatch': sig_match,
                                     'B_id': origin_B_id,
                                     'NumBkgParticles_noniso': sig_dict["none_iso_n_bkg"],
                                     # 'Pred_FT_b_score': ft_bbar_score,
                                     # 'Pred_FT_no_scrore': ft_no_score,
                                     # 'Pred_FT_bbar_score': ft_b_score,
                                     # 'reco_pv_idx': reco_pv_idx,
                                     # 'true_pv_idx': true_pv_idx
                                     }, ignore_index=True)
        evt_df = evt_df._append({'EventNumber': event,
                                 'NumParticlesInEvent': n_part,
                                 'NumParticlesFromHeavyHadronInEvent': n_part_heavy_h,
                                 'NumBackgroundParticlesInEvent': n_bkg_part,
                                 'NumSelectedParticlesInEvent': n_sel_part,
                                 'NumSelectedParticlesFromHeavyHadronInEvent': n_sel_heavy_h,
                                 'NumSelectedBackgroundParticlesInEvent': n_sel_bkg_part,
                                 'NumTruthClustersGen1': t_nclust_order[0],
                                 'NumTruthClustersGen2': t_nclust_order[1],
                                 'NumTruthClustersGen3': t_nclust_order[2],
                                 'NumTruthClustersGen4': t_nclust_order[3],
                                 'NumRecoClustersGen1': r_nclust_order[0],
                                 'NumRecoClustersGen2': r_nclust_order[1],
                                 'NumRecoClustersGen3': r_nclust_order[2],
                                 'NumRecoClustersGen4': r_nclust_order[3],
                                 'MaxTruthFullChainDepthInEvent': max_chain_depth,
                                 'EfficiencyParticlesFromHeavyHadronInEvent': float(
                                     n_sel_heavy_h) / n_part_heavy_h,
                                 'EfficiencyBackgroundParticlesInEvent': float(
                                     n_sel_bkg_part) / n_bkg_part,
                                 'BackgroundRejectionPowerInEvent': 1. - float(
                                     n_sel_bkg_part) / n_bkg_part,
                                 'PerfectEventReconstruction': perfect_evt_reco,
                                 'NumTrueSignalsInEvent': len(tc_dict.keys()),
                                 'NumRecoSignalsInEvent': len(rc_dict.keys()),
                                 },
                                ignore_index=True)
    return sig_df, evt_df
