from tqdm import tqdm

import pandas as pd
import numpy as np
from multiprocessing.pool import ThreadPool

from wmpgnn.util.pruners import edge_pruning, true_node_pruning
from wmpgnn.reconstruction.signal_dict import get_ref_signal
from wmpgnn.reconstruction.reco_helper import *
from wmpgnn.reconstruction.quantity_adder import *
from wmpgnn.calibration.calibration_mask import *
from wmpgnn.calibration.decay_dict import decay_prop, pid_dict
from wmpgnn.util.bfs import *


class EventReconstruction:
    def __init__(self, configs):
        # boolean whether to use true reconstruction or predicted reconstruction
        self.configs = configs["inference"]
        self.norm = obtain_normalization(configs)
        self.prob_nn_mask = torch.tensor([c.lower().startswith('prob') for c in self.norm.node_features])
        self.momentum_mask = torch.tensor([x in ["px_reco", "py_reco", "pz_reco"] for x in self.norm.node_features])
        self.pid_features = [x for x, y in zip(self.norm.node_features, self.prob_nn_mask) if y]
        self.charge_mask = torch.tensor([x == "charge" for x in self.norm.node_features])
        if torch.sum(self.charge_mask) == 0:
            raise ValueError("Charge does not exist thus no pid inference possible")

        self.evt_counter = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.norm.norm = self.norm.norm.to(device=self.device)

        self.ref_signal = None
        splitted = configs["evaluate"]["sample"][0].split("_")
        if splitted[-1] != "inclusive":
            self.ref_signal = decay_prop[f"{splitted[-2]}_{splitted[-1]}"]

        self.sig_df = []

    def collect_results(self):
        sig_dfs = []

        for event_number, sig_df in enumerate(self.sig_df, start=1):
            sig_df = sig_df.copy()
            sig_df.insert(0, 'EventNumber', event_number)
            sig_dfs.append(sig_df)

        self.sig_df = pd.concat(sig_dfs, ignore_index=True)
        return self.sig_df

    def reconstruct_heavyhadrons(self, outputs, ft_des=None, pv_des=None):
        graphs = outputs.to_data_list()

        # Output weights
        node_weights = outputs["node_weights"]
        edge_weights = outputs["edge_weights"]
        lca = outputs[("tracks", "tracks")].lca

        # Unfold normalization
        track_org_x = outputs["tracks"].org_x
        track_pid = outputs["tracks"].org_pid

        # indexing + per graph mask creation
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

            # Apply the pruning
            if self.configs.get("node_prune", True):
                node_selbool = node_weights[track_mask] > self.configs["node_prune_thr"]
                edge_mask = true_node_pruning(node_selbool, graphs[i], "tracks", [('tracks', 'to', 'tracks')])
                edge_selbool = edge_weights[tr_tr_mask][edge_mask] > self.configs["edge_prune_thr"]
            else:
                node_selbool = True
                edge_selbool = edge_weights[tr_tr_mask] > self.configs["edge_prune_thr"]
            if self.configs.get("edge_prune", True):
                edge_pruning(edge_selbool, graphs[i], ('tracks', 'to', 'tracks'))

            # denorm information
            graph_track_x = track_org_x[track_mask]
            graph_track_pid = track_pid[track_mask]
            combined_features = torch.cat((graph_track_x, graph_track_pid), dim=1)

            denorm_data = combined_features * self.norm.norm[1] + self.norm.norm[0]
            pid_response = denorm_data[:, self.prob_nn_mask][node_selbool]
            features = denorm_data[:, ~self.prob_nn_mask][node_selbool]
            charge = denorm_data[:, self.charge_mask][node_selbool].squeeze()

            pred_pid = [self.pid_features[i] for i in torch.argmax(pid_response, dim=1).tolist()]
            sel_pid = torch.tensor([pid_dict[k] for k in pred_pid]).to(self.device)
            graph_track_pid = (sel_pid * charge).round().to(torch.int32)

            # Add the original feature to the graph
            graphs[i]["tracks"].x = features
            graphs[i]["tracks"].pid = graph_track_pid

            # Get the PV association of all tracks
            evt_tr_pv_edge_idx = tr_pv_edge_idx[:, tr_pv_mask]
            ntracks = torch.unique(evt_tr_pv_edge_idx[0]).shape[0]
            npvs = torch.unique(evt_tr_pv_edge_idx[1]).shape[0]
            if pv_des is not None:
                min_ip_pv = torch.argmin(pv_des["minIP"][tr_pv_mask].view(ntracks, npvs), dim=1)
                pred_pv = pv_des["pred"][tr_pv_mask].view(ntracks, npvs)

                evt_pv_des = {"pred": pred_pv[node_selbool].cpu(), "ip": min_ip_pv[node_selbool].cpu(), "npvs": npvs}
                precomputed_pv_desc.append(evt_pv_des)
            else:
                precomputed_pv_desc.append(None)

            # get the ft decision of the selected B particles
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

        # Remove the edges which remain but are bkg due to lca score
        lca = torch.argmax(graph[("tracks", "tracks")].lca, dim=1)
        lca_bool = lca != 0
        lca = lca[lca_bool]
        edges = graph[("tracks", "tracks")].edge_index[:, lca_bool]

        # Looping over B
        components = find_components_bfs(edges)
        res = []
        for comp in components:
            sig_df = {"SigLike": 0, "EVENTNUMBER": graph["EVENTNUMBER"].item(), "RUNNUMBER": graph["RUNNUMBER"].item()}
            nodes_indx = comp["nodes"]
            edges_indx = comp["edge_indices"]

            sel_pid = torch.sort(graph["tracks"].pid[nodes_indx]).values
            true_pos_pid, true_neg_pid = torch.sort(self.ref_signal[0]).values, torch.sort(
                -1 * self.ref_signal[0]).values
            matching_final = torch.equal(sel_pid, true_pos_pid) or torch.equal(sel_pid, true_neg_pid)
            matching_lca = torch.equal(lca[edges_indx], torch.cat([self.ref_signal[1], self.ref_signal[1]]))
            if matching_final and matching_lca:
                sig_df["SigLike"] = 1

            # Get PV association
            # Get the pid
            sig_df["final_pid"] = ','.join(str(x.item()) for x in graph["tracks"].pid[nodes_indx])
            
            # Get the individual scores stored as strings
            ft_score = ft_des[nodes_indx]
            sig_df["final_b_score"] = ','.join(str(x.item()) for x in ft_score[:, :1].squeeze())
            sig_df["final_bbar_score"] = ','.join(str(x.item()) for x in ft_score[:, 2:].squeeze())
            sig_df["ft_b_score"], _, sig_df["ft_bbar_score"] = ft_score.mean(dim=0).tolist()

            p = graph["tracks"].x[nodes_indx][:, self.momentum_mask[:graph["tracks"].x.shape[1]]]
            sig_df["final_px"] = ','.join(str(x.item()) for x in p[:, 0:1])
            sig_df["final_py"] = ','.join(str(x.item()) for x in p[:, 1:2])
            sig_df["final_pz"] = ','.join(str(x.item()) for x in p[:, 2:3])

            res.append(sig_df)
        return res
