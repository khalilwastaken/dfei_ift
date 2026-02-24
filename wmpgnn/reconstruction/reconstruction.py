from tqdm import tqdm

import pandas as pd
import torch

from multiprocessing.pool import ThreadPool

from wmpgnn.util.pruners import edge_pruning, true_node_pruning
from wmpgnn.reconstruction.signal_dict import get_ref_signal
from wmpgnn.reconstruction.reco_helper import *
from wmpgnn.reconstruction.quantity_adder import *


class EventReconstruction:
    def __init__(self, configs):
        # boolean whether to use true reconstruction or predicted reconstruction
        self.configs = configs["inference"]
        self.use_lca = True
        if "LCA" in self.configs:
            self.use_lca = self.configs["LCA"]

        self.signal = get_ref_signal(configs["evaluate"]["sample"][0])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.evt_counter = 0
        self.sig_df, self.evt_df = [], []

        # we look at independent tracks so we can directly sum it which shouldn't be a problem
        self.log = {"pv_corr_ml": {}, "pv_corr_ip": {}, "pv_total": {}, "npvs": {}}

    def collect_results(self):
        sig_dfs = []
        evt_dfs = []

        for event_number, (sig_df, evt_df) in enumerate(zip(self.sig_df, self.evt_df), start=1):
            sig_df = sig_df.copy()
            sig_df.insert(0, 'EventNumber', event_number)
            sig_dfs.append(sig_df)

            evt_df = evt_df.copy()
            evt_df.insert(0, 'EventNumber', event_number)
            evt_dfs.append(evt_df)

        self.sig_df = pd.concat(sig_dfs, ignore_index=True)
        self.evt_df = pd.concat(evt_dfs, ignore_index=True)
        return self.sig_df, self.evt_df

    def reconstruct_heavyhadrons(self, outputs, ft_des=None, pv_des=None):
        # debatching
        graphs = outputs.to_data_list()

        # Output weights
        node_weights = outputs["node_weights"]
        edge_weights = outputs["edge_weights"]
        lca = outputs[("tracks", "tracks")].lca

        track_batch = outputs["tracks"].batch
        pv_batch = outputs['pvs'].batch
        tr_tr_edge_idx = outputs[('tracks', 'tracks')].edge_index
        tr_pv_edge_idx = outputs[('tracks', 'pvs')].edge_index

        n_graphs = track_batch.max().item() + 1
        graph_ids = torch.arange(n_graphs, device=self.device)
        track_masks = track_batch.unsqueeze(1) == graph_ids.unsqueeze(0)  # Shape: [n_tracks, n_graphs]

        precomputed_pv_desc = []
        precomputed_ft_desc = []
        for i in range(n_graphs):
            # Create boolean mask to find per event information
            track_mask = track_masks[:, i]
            tr_tr_mask = track_masks[tr_tr_edge_idx[0], i] & track_masks[tr_tr_edge_idx[1], i]
            pv_mask = pv_batch == i
            tr_pv_mask = track_masks[tr_pv_edge_idx[0], i] & pv_mask[tr_pv_edge_idx[1]]

            graphs[i][("tracks", "tracks")].lca = lca[tr_tr_mask]

            evt_tr_pv_edge_idx = tr_pv_edge_idx[:, tr_pv_mask]
            ntracks = torch.unique(evt_tr_pv_edge_idx[0]).shape[0]
            npvs = torch.unique(evt_tr_pv_edge_idx[1]).shape[0]


            # log pv performance for all tracks
            if pv_des is not None:
                pv_filter = torch.prod(pv_des["pv_filter"][tr_pv_mask].view(ntracks, npvs), dim=1, dtype=bool)

                y_pv = torch.argmax(pv_des["true"][tr_pv_mask].view(ntracks, npvs), dim=1)
                y_pv[~pv_filter] = -1
                min_ip_pv = torch.argmin(pv_des["minIP"][tr_pv_mask].view(ntracks, npvs), dim=1)
                pred_pv = pv_des["pred"][tr_pv_mask].view(ntracks, npvs)
                pred_pv_track_level = torch.argmax(pred_pv, dim=1)
                # Here we remove ghost/tracks with true pv not being recoed
                if npvs not in self.log["pv_total"].keys():
                    self.log["pv_corr_ml"][npvs], self.log["pv_corr_ip"][npvs], self.log["pv_total"][npvs] = 0, 0, 0
                    self.log["npvs"][npvs] = 0
                self.log["pv_corr_ml"][npvs] += torch.sum(y_pv[pv_filter] == pred_pv_track_level[pv_filter]).item()
                self.log["pv_corr_ip"][npvs] += torch.sum(y_pv[pv_filter] == min_ip_pv[pv_filter]).item()
                self.log["pv_total"][npvs] += int(torch.sum(pv_filter).item())
                self.log["npvs"][npvs] += 1

            # apply the pruning
            if self.configs.get("node_prune", True):
                node_selbool = node_weights[track_mask] > self.configs["node_prune_thr"]
                edge_mask = true_node_pruning(node_selbool, graphs[i], "tracks", [('tracks', 'to', 'tracks')])
                edge_selbool = edge_weights[tr_tr_mask][edge_mask] > self.configs["edge_prune_thr"]
            else:
                edge_selbool = edge_weights[tr_tr_mask] > self.configs["edge_prune_thr"]
            if self.configs.get("edge_prune", True):
                edge_pruning(edge_selbool, graphs[i], ('tracks', 'to', 'tracks'))

            # Apply pruning on pv prediction
            if pv_des is not None:
                evt_pv_des = {"true": y_pv[node_selbool].cpu(), "pred": pred_pv[node_selbool].cpu(),
                              "ip": min_ip_pv[node_selbool].cpu(), "npvs": npvs}
                precomputed_pv_desc.append(evt_pv_des)
            else:
                precomputed_pv_desc.append(None)
            # Apply pruning on ft bool
            if ft_des is not None:
                evt_ft_des = ft_des[track_mask][node_selbool]
                precomputed_ft_desc.append(evt_ft_des.cpu())
            else:
                precomputed_ft_desc.append(None)

        # now multiprocess the reco
        args_list = [(graph.cpu(), pv_desc, ft_desc) for graph, pv_desc, ft_desc in
                     zip(graphs, precomputed_pv_desc, precomputed_ft_desc)]
        with ThreadPool(processes=4) as pool:
            res = list(tqdm(pool.imap(self.reconstruct_single_evt, args_list), total=len(args_list),
                            desc="Reconstructing events", leave=False))

        for r in res:
            self.sig_df.append(pd.DataFrame(r[0]))
            self.evt_df.append(pd.DataFrame([r[1]]))

    def reconstruct_single_evt(self, args):
        graph, pv_des, ft_des = args

        """Obtain the LCA scores of both true and reco graphs"""
        true_LCA = lca_truth_matrix(graph)
        if self.use_lca:
            reco_LCA = lca_reco_matrix(graph, mode="reco")
        else:
            reco_LCA = lca_reco_matrix(graph, mode="true")

        particle_keys = graph["final_keys"].tolist()
        n_part = len(particle_keys)
        rc_dict, r_nclust_order, _ = reconstruct_decay(reco_LCA, particle_keys)

        particle_keys = graph["truth_part_keys"].tolist()

        particle_ids = list(map(particle_name, graph['truth_part_ids'].numpy()))

        tc_dict, t_nclust_order, max_chain_depth = reconstruct_decay(true_LCA, particle_keys,
                                                                     particle_ids=particle_ids,
                                                                     truth_level_simulation=1)
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
                n_sel_part, n_sel_heavy_h, n_sel_bkg_part = 0, 0, 0

            perfect_evt_reco = 1  # Flag for perfect event reco
            if n_sel_bkg_part > 0:
                perfect_evt_reco = 0

            # Looping over reco candidates
            sig_dict_holder = []
            for tc_key in tc_dict.keys():
                sig_dict = {'NumParticlesInEvent': n_part,
                            "PerfectReco": 0, "AllParticles": 0, "NoneIso": 0, "PartReco": 0, "NotFound": 0,
                            "NumBkgParticles_noniso": -999}
                tc = tc_dict[tc_key]
                sig_dict["SigMatch"] = 0
                if self.signal:
                    labels = tc['labels']
                    mothers = [label[3:] for label in labels if 'c' == label[0]]
                    node_keys = tc['node_keys']
                    daughters = [label.split(':')[1] for label in labels if
                                 int(float(label.split(':')[0][1:])) in node_keys]
                    if match_decays(daughters, self.signal[0]['daughters']) or match_decays(daughters,
                                                                                            self.signal[1][
                                                                                                'daughters']):
                        check_mothers1 = True
                        check_mothers2 = True
                        for i in range(len(self.signal[0]['mothers'])):
                            if self.signal[0]['mothers'][i] not in mothers:
                                check_mothers1 = False
                            if self.signal[1]['mothers'][i] not in mothers:
                                check_mothers2 = False
                        sig_dict["SigMatch"] = int(check_mothers1 or check_mothers2)

                sig_dict["NumSignalParticles"] = len(tc['node_keys'])

                if tc_key in rc_dict.keys():
                    perfect_sig_reco = int(
                        rc_dict[tc_key]['node_keys'] == tc['node_keys']
                        and rc_dict[tc_key]['LCA_values'] == tc['LCA_values']
                    )
                else:
                    perfect_sig_reco = 0
                perfect_evt_reco *= perfect_sig_reco

                for rc in rc_dict.values():
                    true_in_reco = np.sum(np.isin(tc['node_keys'], rc['node_keys'])) / len(tc['node_keys'])
                    if rc['node_keys'] == tc['node_keys']:
                        sig_dict["AllParticles"] = 1
                        if rc['LCA_values'] == tc['LCA_values']:
                            sig_dict["PerfectReco"] = 1
                        sig_dict["NoneIso"] = sig_dict["PartReco"] = 0
                        if ft_des is not None:
                            sig_dict = get_pred_ft(sig_dict, graph, rc, ft_des)
                            sig_dict = get_asso_frag(sig_dict, graph, rc)
                        if pv_des is not None:
                            get_pv_asso(sig_dict, graph, rc, pv_des)
                        break
                    elif true_in_reco == 1 and len(rc['node_keys']) > len(tc['node_keys']):
                        sig_dict["NoneIso"] = 1  # background tracks in signal
                        sig_dict["PartReco"] = 0
                        if ft_des is not None:
                            sig_dict = get_pred_ft(sig_dict, graph, rc, ft_des)
                            sig_dict = get_asso_frag(sig_dict, graph, rc)
                        if pv_des is not None:
                            get_pv_asso(sig_dict, graph, rc, pv_des)
                        break
                    elif 0.2 <= true_in_reco < 1:
                        sig_dict["PartReco"] = 1  # FT decision can not be trusted
                        sig_dict["NumBkgParticles_noniso"] = len(rc['node_keys']) - len(tc['node_keys'])
                        if ft_des is not None:
                            sig_dict = get_pred_ft(sig_dict, graph, rc, ft_des)
                            sig_dict = get_asso_frag(sig_dict, graph, rc)
                        if pv_des is not None:
                            get_pv_asso(sig_dict, graph, rc, pv_des)
                        break
                    """else:
                        sig_dict["final_pid"] = sig_dict["final_b_score"] = sig_dict["final_bbar_score"] = ""
                        sig_dict["ft_b_score"] = sig_dict["ft_bbar_score"] = 0"""

                if sig_dict["AllParticles"] == 0 and sig_dict["NoneIso"] == 0 and sig_dict["PartReco"] == 0:
                    sig_dict["NotFound"] = 1

                # Get origin B id
                indices = [particle_keys.index(x) for x in tc['node_keys']]
                signal_LCA_id = true_LCA[true_LCA['senders'].isin(indices) | true_LCA['receivers'].isin(indices)][
                    "LCA_id"]
                values, counts = np.unique(signal_LCA_id, return_counts=True)
                max_indices = np.where(counts == counts.max())[0]
                if len(max_indices) == 1:
                    sig_dict["B_id"] = values[max_indices[0]]
                else:
                    candidate_lca_ids = values[max_indices]
                    candidates_df = true_LCA[true_LCA['LCA_id'].isin(candidate_lca_ids)]
                    max_chain_per_lca = candidates_df.groupby('LCA_id')['TrueFullChainLCA'].max()
                    sig_dict["B_id"] = max_chain_per_lca.idxmax()
                if "EVENTNUMBER" in graph.keys():
                    sig_dict["EVENTNUMBER"] = graph["EVENTNUMBER"].item()
                    sig_dict["RUNNUMBER"] = graph["RUNNUMBER"].item()
                if "num_pvs" in graph.keys():
                    sig_dict["num_pvs"] = graph["num_pvs"].item()
                else:
                    sig_dict["num_pvs"] = graph["pvs"].x.shape[0]
                sig_dict_holder.append(sig_dict)

            evt_dict = {'NumParticlesInEvent': n_part,
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
                        'PerfectEventReconstruction': perfect_evt_reco,
                        'NumTrueSignalsInEvent': len(tc_dict.keys()),
                        'NumRecoSignalsInEvent': len(rc_dict.keys()),
                        }
            return sig_dict_holder, evt_dict
