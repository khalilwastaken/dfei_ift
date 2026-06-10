from tqdm import tqdm
import glob
import yaml

import pandas as pd
import torch

from multiprocessing.pool import ThreadPool

from wmpgnn.util.pruners import edge_pruning, true_node_pruning
from wmpgnn.reconstruction.signal_dict import get_ref_signal, sig_matching
from wmpgnn.reconstruction.reconstruction_helper import *
from wmpgnn.reconstruction.quantity_adder import *
from wmpgnn.util.bfs import find_components_bfs
from wmpgnn.util.normalization import get_normalization, denorm_data


class EventReconstruction:
    def __init__(self, configs):
        self.configs = configs["inference"]

        cacheing_config = glob.glob(f"{configs['settings']['data_dir']}/{configs['evaluate']['sample'][0]}/*.yaml")[0]
        with open(cacheing_config, "r") as file:
            self.cacheing_config = yaml.safe_load(file)['graphs']

        self.signal = get_ref_signal(configs["evaluate"]["sample"][0])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get the normalization
        self.norm = get_normalization(configs)

        self.evt_counter = 0
        self.sig_df = []

        # we look at independent tracks so we can directly sum it which shouldn't be a problem
        self.log = {"pv_corr_ml": {}, "pv_corr_ip": {}, "pv_total": {}, "npvs": {}}

        self.mode = 'MC'

    def collect_results(self):
        sig_dfs = []

        for event_number, sig_df in enumerate(self.sig_df):
            sig_df = sig_df.copy()
            sig_df.insert(0, 'EventNumber', event_number)
            sig_dfs.append(sig_df)

        self.sig_df = pd.concat(sig_dfs, ignore_index=True)
        return self.sig_df

    def reconstruct_heavyhadrons(self, outputs, mode='MC', ft_des=None, pv_des=None):
        # Debatching the batched graph to perform the reconstruction
        graphs = outputs.to_data_list()

        # Output weights
        node_weights = outputs["node_weights"]
        edge_weights = outputs["edge_weights"]
        lca = outputs[("tracks", "tracks")].lca

        # Unfold normalization here
        if self.norm is not None:
            track_org_x = denorm_data(outputs["tracks"].org_x, self.norm, self.cacheing_config['tracks_nodes'])
            track_pid = denorm_data(outputs["tracks"].org_pid, self.norm, self.cacheing_config['tracks_pid'])
        else:
            track_org_x = outputs["tracks"].org_x
            track_pid = outputs["tracks"].org_pid

        # Extract origin_flag for evaluation diagnostics (not normalized)
        track_origin_flag = getattr(outputs["tracks"], "origin_flag", None)


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
                min_ip_pv = torch.argmin(pv_des["minIP"][tr_pv_mask].view(ntracks, npvs), dim=1)
                pred_pv = pv_des["pred"][tr_pv_mask].view(ntracks, npvs)
                if mode == 'MC':
                    pv_filter = torch.prod(pv_des["pv_filter"][tr_pv_mask].view(ntracks, npvs), dim=1, dtype=bool)
                    y_pv = torch.argmax(pv_des["true"][tr_pv_mask].view(ntracks, npvs), dim=1)
                    y_pv[~pv_filter] = -1
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
            edge_selbool = None
            if self.configs.get("node_prune", True):
                node_selbool = node_weights[track_mask] > self.configs["node_prune_thr"]
                edge_mask = true_node_pruning(node_selbool, graphs[i], "tracks", [('tracks', 'tracks')])
                edge_selbool = edge_weights[tr_tr_mask][edge_mask] > self.configs["edge_prune_thr"]
            elif self.configs.get("edge_prune", True):
                edge_selbool = edge_weights[tr_tr_mask] > self.configs["edge_prune_thr"]
            if self.configs.get("edge_prune", True):
                edge_pruning(edge_selbool, graphs[i], ('tracks', 'tracks'))

            graphs[i]["tracks"].org_x = track_org_x[track_mask][node_selbool]
            graphs[i]["tracks"].org_pid = track_pid[track_mask][node_selbool]
            if track_origin_flag is not None:
                graphs[i]["tracks"].origin_flag = track_origin_flag[track_mask][node_selbool]

            # Apply pruning on pv prediction
            if pv_des is not None:
                evt_pv_des = {"pred": pred_pv[node_selbool].cpu(), "ip": min_ip_pv[node_selbool].cpu(), "npvs": npvs}
                if mode == 'MC':
                    evt_pv_des["true"] = y_pv[node_selbool].cpu()
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
        with ThreadPool(processes=1) as pool:
            res = list(tqdm(pool.imap(self.reconstruct_single_evt, args_list), total=len(args_list),
                            desc="Reconstructing events", leave=False))
        for r in res:
            try:
                self.sig_df.append(pd.DataFrame(r))
            except:
                continue

    def reconstruct_single_evt(self, args):
        graph, pv_des, ft_des = args

        if self.mode == 'MC':
            res = self.true_reco(graph, pv_des, ft_des)
        else:
            res = self.reco(graph, pv_des, ft_des)

        return res

    def true_reco(self, graph, pv_des, ft_des):
        # Reconstructed components
        lca = torch.argmax(graph[("tracks", "tracks")].lca, dim=1)
        lca_bool = lca != 0  # Remove edges which survived but have predicted score of 0
        reco_edges = graph[("tracks", "tracks")].edge_index[:, lca_bool]
        reco_lca = lca[lca_bool]
        reco_components = add_lca(find_components_bfs(reco_edges), reco_lca)
        reco_components = apply_reco_mapping(graph, reco_components, reco_edges)
        reco_components = add_base_quant(graph, reco_components, self.cacheing_config)

        # Truth components
        true_edges = torch.stack([graph[('tracks', 'tracks')].senders, graph[('tracks', 'tracks')].receivers])
        true_lca = graph[('tracks', 'tracks')].sig_y
        true_components = add_lca(find_components_bfs(true_edges), true_lca)
        true_components = apply_true_mapping(graph, true_components, true_edges)

        # Loop over tru components
        sig_dict_holder = []
        for true_component in true_components:
            sig_dict = {"PerfectReco": 0, "AllParticles": 0, "NoneIso": 0, "PartReco": 0, "NotFound": 0, 'SigMatch': 0,
                        "NumBkgParticles": -999}
            # Identify if the true_component hold the signal decay mode
            if self.signal:
                if sig_matching(true_component, self.signal, "true"):
                    sig_dict['SigMatch'] = 1

            # Find the matching reconstructed cluster
            reco_component = None
            for reco_component in reco_components:
                sig_dict, flag = get_reco_type(true_component, reco_component, sig_dict)
                if flag:
                    break
                if sig_dict["AllParticles"] == 0 and sig_dict["NoneIso"] == 0 and sig_dict["PartReco"] == 0:
                    sig_dict["NotFound"] = 1
                    reco_component = None
            if pv_des is not None:
                sig_dict = get_pv_asso(sig_dict, reco_component, pv_des)
            if ft_des is not None:
                sig_dict = get_pred_ft(sig_dict, reco_component, ft_des)
                sig_dict["part_ids"] = '_'.join(str(x.item()) for x in true_component['part_id'])  # true

            # Adding event level info, currently just slammed do it in a function
            sig_dict['true_part_ids'] = '_'.join(str(x.item()) for x in true_component['part_id'])
            sig_dict["B_id"] = true_component['head_id']
            sig_dict['EVENTNUMBER'] = graph['EVENTNUMBER'].item()
            sig_dict['RUNNUMBER'] = graph['RUNNUMBER'].item()
            if 'num_pvs' in graph and 'npvs' not in sig_dict:
                sig_dict['npvs'] = graph['num_pvs'].item()
            if 'is_whiten' in graph:
                sig_dict['is_whiten'] = graph['is_whiten'].item()
            # Adding base quant info
            if reco_component:
                sig_dict = get_track_info(sig_dict, reco_component)
                if "origin_flag" in reco_component:
                    sig_dict["origin_flags"] = "_".join(str(x.item()) for x in reco_component["origin_flag"])
            if sig_dict['SigMatch']:  # Adding tupled signal B information
                sig_dict = get_sig_lvl_info(sig_dict, graph, pv_des, ft_des)
            sig_dict_holder.append(sig_dict)
        return sig_dict_holder

    def reco(self, graph, pv_des, ft_des):
        # Reconstructed components
        lca = torch.argmax(graph[("tracks", "tracks")].lca, dim=1)
        lca_bool = lca != 0  # Remove edges which survived but have predicted score of 0
        reco_edges = graph[("tracks", "tracks")].edge_index[:, lca_bool]
        reco_lca = lca[lca_bool]
        reco_components = add_lca(find_components_bfs(reco_edges), reco_lca)
        reco_components = apply_reco_mapping(graph, reco_components, reco_edges)
        reco_components = add_base_quant(graph, reco_components, self.cacheing_config)

        # Loop over reconstructed components
        sig_dict_holder = []
        for reco_component in reco_components:
            sig_dict = {'SigLike': 0}
            if self.signal:
                if sig_matching(reco_component, self.signal, "reco"):
                    sig_dict['SigLike'] = int(1)

            if pv_des is not None:
                sig_dict = get_pv_asso(sig_dict, reco_component, pv_des)
            if ft_des is not None:
                sig_dict = get_pred_ft(sig_dict, reco_component, ft_des)

            # Adding event level info, currently just slammed do it in a function
            sig_dict['EVENTNUMBER'] = graph['EVENTNUMBER'].item()
            sig_dict['RUNNUMBER'] = graph['RUNNUMBER'].item()
            if 'num_pvs' in graph and 'npvs' not in sig_dict:
                sig_dict['npvs'] = graph['num_pvs'].item()
            if 'is_whiten' in graph:
                sig_dict['is_whiten'] = graph['is_whiten'].item()
            # Adding base quant info
            sig_dict = get_track_info(sig_dict, reco_component)
            if sig_dict['SigLike']:  # Adding tupled signal B information
                sig_dict = get_sig_lvl_info(sig_dict, graph, pv_des, ft_des)
            sig_dict_holder.append(sig_dict)
        return sig_dict_holder
