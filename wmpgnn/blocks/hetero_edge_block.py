import pytorch_lightning as pl

import torch

from wmpgnn.util.helper import create_mlp


class HeteroEdgeBlock(pl.LightningModule):
    def __init__(self, config, edge_types,
                 use_edges=True,
                 use_receiver_nodes=True,
                 use_sender_nodes=True,
                 use_globals=True):
        super(HeteroEdgeBlock, self).__init__()
        self._edge_types = edge_types
        self._use_edges = use_edges
        self._use_receiver_nodes = use_receiver_nodes
        self._use_sender_nodes = use_sender_nodes
        self._use_globals = use_globals
        self._edge_models = {}
        for edge_type in edge_types:
            self._edge_models[edge_type] = create_mlp(config)
        self._edge_models_model_dict = torch.nn.ModuleDict({str(i): j for i, j in self._edge_models.items()})

    def forward(self, graph):
        for edge_type in self._edge_types:
            edges_to_collect = []
            edges = graph[edge_type]

            if self._use_edges:
                edges_to_collect.append(edges.edges)
            if self._use_receiver_nodes:
                edges_to_collect.append(graph[edge_type[2]].x[edges.edge_index[1], :])
            node_0 = graph[edge_type[0]]
            if self._use_sender_nodes:
                edges_to_collect.append(node_0.x[edges.edge_index[0], :])

            if self._use_globals:
                edges_to_collect.append(graph['globals'].x[node_0.batch[edges.edge_index[0]]])

            collected_edges = torch.cat(edges_to_collect, axis=-1)
            updated_edges = self._edge_models[edge_type](collected_edges,
                                                         graph[edge_type[0]].batch[graph[edge_type].edge_index[0]])
            graph[edge_type].edges = updated_edges

        return graph
