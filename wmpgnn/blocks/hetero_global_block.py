from wmpgnn.blocks.abstract_module import AbstractModule
from wmpgnn.blocks.hetero_aggregators import HeteroEdgesToGlobalsAggregator
from wmpgnn.blocks.hetero_aggregators import HeteroNodesToGlobalsAggregator
import torch
from torch_geometric.nn.models import MLP
from wmpgnn.util.helper import create_mlp
import pytorch_lightning as pl


class HeteroGlobalBlock(pl.LightningModule):
    def __init__(self, config, node_types, edge_types, use_edges=True,
                 use_nodes=True, use_globals=True, weighted_mp=False):
        super(HeteroGlobalBlock, self).__init__()

        self._use_edges = use_edges
        self._use_nodes = use_nodes
        self._use_globals = use_globals
        self._node_types = node_types
        self._edge_types = edge_types

        self._global_model = create_mlp(config)
        if self._use_edges:
            self._edges_aggregator = HeteroEdgesToGlobalsAggregator(weighted=weighted_mp)

        if self._use_nodes:
            self._nodes_aggregator = HeteroNodesToGlobalsAggregator(weighted=weighted_mp)

    def forward(self, graph, edge_weights, node_weights, global_model_kwargs=None):
        globals_to_collect = []

        if self._use_edges:
            for edge_type in self._edge_types:
                globals_to_collect.append(self._edges_aggregator(graph, edge_type, edge_weights[edge_type]))

        if self._use_nodes:
            for node_type in self._node_types:
                globals_to_collect.append(self._nodes_aggregator(graph, node_type, node_weights[node_type]))

        if self._use_globals:
            globals_to_collect.append(graph['globals'].x)

        collected_globals = torch.cat(globals_to_collect, axis=-1)

        updated_globals = self._global_model(collected_globals)

        graph['globals'].x = updated_globals

        return graph
