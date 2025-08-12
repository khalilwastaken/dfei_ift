import torch
import torch.nn as nn
from torch_geometric.nn.models import MLP

from model_helper import *

from wmpgnn.gnn.hetero_graph_network import edge_pruning, node_pruning, faster_node_pruning


def hetero_graph_concat(g1, g2):
    """
    Concatenate two heterogeneous graphs along their edge and node feature dimensions.

    This function clones the first graph, then for each edge and node type,
    concatenates the corresponding features from the second graph along the last dimension.
    Global features are also concatenated similarly.

    Args:
        g1 (HeteroData): The first heterogeneous graph data.
        g2 (HeteroData): The second heterogeneous graph data, to be concatenated with g1.

    Returns:
        HeteroData: A new heterogeneous graph with concatenated features.
    """
    graph = g1.clone()
    for edge_type in g1.edge_types:
        graph[edge_type].edges = torch.cat([g1[edge_type].edges, g2[edge_type].edges], -1)
    for node_type in g1.node_types:
        graph[node_type].x = torch.cat([g1[node_type].x, g2[node_type].x], -1)
    graph['globals'].x = torch.cat([g1['globals'].x, g2['globals'].x], -1)
    return graph


class DFEI_HGNN(nn.Module):
    def __init__(self, config):
        super(DFEI_HGNN, self).__init__()
        self.node_types = config["node_types"]
        self.edge_types = [(edge.split('_')[0], 'to', edge.split('_')[1]) for edge in config["edge_types"]]

        self.encode = config["encoder"]["usage"]
        if self.encode:
            self._encoder = get_encoder(config["encoder"], self.node_types, self.edge_types)

        self.GN_block = config["GNblocks"]["nBlocks"] > 0
        if self.GN_block:
            self._blocks = get_blocks(config["GNblocks"], self.node_types, self.edge_types)

        self.decode = config["decoder"]["usage"]
        if self.decode:
            self._decoder = get_encoder(config["decoder"], self.node_types, self.edge_types)

        self.out_trafo = config["op_trafo"]["usage"]
        if self.out_trafo:
            self._op_trafo = get_op_trafo(config["op_trafo"], self.node_types, self.edge_types)

        self.ft_model = config["FT_inferer"]["usage"]
        if self.ft_model:
            self._ftblocks = get_IFT_model(config["FT_inferer"], self.node_types, self.edge_types)
            self._ftlayer = nn.LazyLinear(3)

    def forward(self, data):
        init_graph_pid = data['tracks'].x[:, -6:]  # charge + 5 pid, hard coded be careful

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
        LCA_score = data[("tracks", "to", "tracks")].edges
        if self.ft_model:
            data["tracks"].x = torch.cat([data["tracks"].x, init_graph_pid], dim=1)
            for b, core in enumerate(self._ftblocks):
                data = core(data, torch.ones(1))
                if b < (len(self._ftblocks) - 1):
                    data = hetero_graph_concat(latent, data)

            data["tracks"].x = self._ftlayer(data["tracks"].x)
        return data, LCA_score


class FT_HGNN(nn.Module):
    def __init__(self, config):
        super(FT_HGNN, self).__init__()
        self.node_types = config["node_types"]
        self.edge_types = [(edge.split('_')[0], 'to', edge.split('_')[1]) for edge in config["edge_types"]]

        self._blocks = get_IFT_model(config["FT_inferer"], self.node_types, self.edge_types)
        self._ftlayer = nn.LazyLinear(3)

    def forward(self, data):
        for b, core in enumerate(self._blocks):
            data = core(data, torch.ones(1))
            if b < (len(self._blocks) - 1):
                data = hetero_graph_concat(latent, data)

        data["tracks"].x = self._ftlayer(data["tracks"].x)
        return data
