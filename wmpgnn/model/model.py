import torch
import torch.nn as nn
from torch_geometric.data import Batch

from wmpgnn.model.model_helper import *
from wmpgnn.model.gnn.hetero_graphcoder import HeteroGraphCoder
from wmpgnn.model.gnn.hetero_output_trafo import HeteroGraphTrafo
from wmpgnn.model.gnn.hetero_domain_adapt import HeteroDomainAdapt

import pytorch_lightning as pl


def hetero_graph_concat(g1, g2):
    graph = g1.clone()
    for edge_type in g1.edge_types:
        graph[edge_type].edges = torch.cat([g1[edge_type].edges, g2[edge_type].edges], -1)
    for node_type in g1.node_types:
        graph[node_type].x = torch.cat([g1[node_type].x, g2[node_type].x], -1)
    graph['globals'].x = torch.cat([g1['globals'].x, g2['globals'].x], -1)
    return graph


"""
Normally used models
"""


class DFEI_HGNN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        node_types = config["node_types"]
        edge_types = [(edge.split('_')[0], edge.split('_')[1]) for edge in config["edge_types"]]

        # Start setting up the DFEI model
        self.encode = config["encoder"]["usage"]
        if self.encode:
            self._encoder = HeteroGraphCoder(config["encoder"], node_types, edge_types)

        self.GN_block = config["GNblocks"]["nBlocks"] > 0
        if self.GN_block:
            self._blocks = get_blocks(config["GNblocks"], node_types, edge_types)

        self.decode = config["decoder"]["usage"]
        if self.decode:
            self._decoder = HeteroGraphCoder(config["decoder"], node_types, edge_types)

        self.out_trafo = config["op_trafo"]["usage"]
        if self.out_trafo:
            self._op_trafo = HeteroGraphTrafo(config["op_trafo"])

    def forward(self, data):
        # this init_graph_pid is not used anymore, was used during GNblock FT inference
        init_graph_pid = data['tracks'].x
        # Latent graph
        if self.encode:
            data = self._encoder(data)
        latent = data.clone()

        for b, core in enumerate(self._blocks):
            data = core(data, init_graph_pid)
            if b < (len(self._blocks) - 1):
                data = hetero_graph_concat(latent, data)

        if self.decode:
            data = self._decoder(data)

        data = self._op_trafo(data)

        return data


class FT_HGNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        node_types = config["node_types"]
        edge_types = [(edge.split('_')[0], edge.split('_')[1]) for edge in config["edge_types"]]

        # Start setting up the DFEI model
        self.encode = config["encoder"]["usage"]
        if self.encode:
            self._encoder = HeteroGraphCoder(config["encoder"], node_types, edge_types)

        self.GN_block = config["GNblocks"]["nBlocks"] > 0
        if self.GN_block:
            self._blocks = get_blocks(config["GNblocks"], node_types, edge_types)

        self.decode = config["decoder"]["usage"]
        if self.decode:
            self._decoder = HeteroGraphCoder(config["decoder"], node_types, edge_types)

        self.out_trafo = config["op_trafo"]["usage"]
        if self.out_trafo:
            self._op_trafo = HeteroGraphTrafo(config["op_trafo"])

    def forward(self, data):
        # this init_graph_pid is not used anymore, was used during GNblock FT inference
        init_graph_pid = data['tracks'].x

        # Latent graph
        if self.encode:
            data = self._encoder(data)
        latent = data.clone()

        for b, core in enumerate(self._blocks):
            data = core(data, init_graph_pid)
            if b < (len(self._blocks) - 1):
                data = hetero_graph_concat(latent, data)

        if self.decode:
            data = self._decoder(data)

        data = self._op_trafo(data)

        return data


"""
Combined training of DFEI and FT model
"""


class DFEIFT(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.dfei_usage = config["DFEI"]["usage"]
        if self.dfei_usage:
            self.dfei_model = DFEI_HGNN(config["DFEI"])

        self.ft_usage = config["FT_inferer"]["usage"]
        if self.ft_usage:
            self.ft_model = FT_HGNN(config["FT_inferer"])

    def forward(self, data):
        init_graph = data["tracks"].x
        lca = None

        if self.dfei_usage:
            data = self.dfei_model(data)
            lca = data[("tracks", "to", "tracks")].edges
            lca_score = torch.argmax(lca, dim=1).unsqueeze(1)
        else:
            # Add a sampling based on LCA prediction of bis?
            lca_score = data[("tracks", "to", "tracks")].y.unsqueeze(1)
            # lca_score = F.one_hot(data[("tracks", "to", "tracks")].y.to(torch.long), num_classes=4).to(torch.float)

        if self.ft_usage:
            data["tracks"].x = torch.cat([data["tracks"].x, init_graph], dim=1)
            data[("tracks", "to", "tracks")].edges = torch.cat([data[("tracks", "to", "tracks")].edges, lca_score],
                                                               dim=1)

            data = self.ft_model(data)

        if self.dfei_usage:
            data[("tracks", "to", "tracks")].lca = lca
        return data


"""
Model including Domain adaptation training strategies
"""


class DFEI_DA_HGNN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        node_types = config["node_types"]
        edge_types = [(edge.split('_')[0], 'to', edge.split('_')[1]) for edge in config["edge_types"]]

        # Start setting up the DFEI model
        self.encode = config["encoder"]["usage"]
        if self.encode:
            self._encoder = HeteroGraphCoder(config["encoder"], node_types, edge_types)


        self._domain_adapt = HeteroDomainAdapt(config["domain_adapt"], node_types, edge_types)

        self.GN_block = config["GNblocks"]["nBlocks"] > 0
        if self.GN_block:
            self._blocks = get_blocks(config["GNblocks"], node_types, edge_types)

        self.decode = config["decoder"]["usage"]
        if self.decode:
            self._decoder = HeteroGraphCoder(config["decoder"], node_types, edge_types)

        self.out_trafo = config["op_trafo"]["usage"]
        if self.out_trafo:
            self._op_trafo = HeteroGraphTrafo(config["op_trafo"])

    def forward(self, data):
        # this init_graph_pid is not used anymore, was used during GNblock FT inference
        init_graph_pid = data['tracks'].x[:, -6:]  # charge + 5 pid, hard coded be careful
        # Latent graph
        if self.encode:
            data = self._encoder(data)

        # get the domain adapt score
        self._domain_adapt(data)

        # split the graph apart as rest runs on MC only
        mc_selbool = data["da_label"] == 0
        indices = mc_selbool.nonzero(as_tuple=True)[0].tolist()
        data = Batch.from_data_list([data.get_example(i) for i in indices])
        latent = data.clone()

        for b, core in enumerate(self._blocks):
            data = core(data, init_graph_pid)
            if b < (len(self._blocks) - 1):
                data = hetero_graph_concat(latent, data)

        if self.decode:
            data = self._decoder(data)

        data = self._op_trafo(data)

        return data
