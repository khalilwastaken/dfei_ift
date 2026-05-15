from tqdm import tqdm

import pandas as pd
import torch

from multiprocessing.pool import ThreadPool

from wmpgnn.util.pruners import edge_pruning, true_node_pruning
from wmpgnn.reconstruction.signal_dict import get_ref_signal
from wmpgnn.reconstruction.reconstruction_helper import *

from wmpgnn.reconstruction.reco_helper import * # old will be removed
from wmpgnn.reconstruction.quantity_adder import *



class EventReconstruction:
    def __init__(self, configs):
        # add flag if it is true or reco reco :}

        # this needs a general clean up

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

        for event_number, sig_df in enumerate(self.sig_df):
            sig_df = sig_df.copy()
            sig_df.insert(0, 'EventNumber', event_number)
            sig_dfs.append(sig_df)

        self.sig_df = pd.concat(sig_dfs, ignore_index=True)
        return self.sig_df

    def reconstruct_heavyhadrons(self, outputs, ft_des=None, pv_des=None):
        # Debatching the batched graph to perform the reconstruction
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
            edge_selbool = None
            if self.configs.get("node_prune", True):
                node_selbool = node_weights[track_mask] > self.configs["node_prune_thr"]
                edge_mask = true_node_pruning(node_selbool, graphs[i], "tracks", [('tracks', 'to', 'tracks')])
                edge_selbool = edge_weights[tr_tr_mask][edge_mask] > self.configs["edge_prune_thr"]
            elif self.configs.get("edge_prune", True):
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

        args = (graphs[0].cpu(), precomputed_pv_desc[0], precomputed_ft_desc[0])
        self.reconstruct_single_evt(args)


        # now multiprocess the reco
        args_list = [(graph.cpu(), pv_desc, ft_desc) for graph, pv_desc, ft_desc in
                     zip(graphs, precomputed_pv_desc, precomputed_ft_desc)]
        with ThreadPool(processes=4) as pool:
            res = list(tqdm(pool.imap(self.reconstruct_single_evt, args_list), total=len(args_list),
                            desc="Reconstructing events", leave=False))

        for r in res:
            try:
                self.sig_df.append(pd.DataFrame(r))
            except:
                continue
            

    def reconstruct_single_evt(self, args):
        graph, pv_des, ft_des = args


        res = true_reco(graph, pv_des, ft_des, self.signal)
        return res

