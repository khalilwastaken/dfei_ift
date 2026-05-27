import pytorch_lightning as pl

import torch
from torch_scatter import scatter_add


class HeteroEdgesToNodesAggregator(pl.LightningModule):
    def __init__(self, use_sent_edges=False, weighted=True, scatter_func=scatter_add):
        super(HeteroEdgesToNodesAggregator, self).__init__()
        self._use_sent_edges = use_sent_edges
        self._weighted = weighted

        self._scatter_func = scatter_func

    def forward(self, graph, edge_type, weight):
        indices = graph[edge_type].edge_index[0] if self._use_sent_edges else graph[edge_type].edge_index[1]
        num_nodes = graph[edge_type[0]].x.shape[0] if self._use_sent_edges else graph[edge_type[1]].x.shape[0]

        out = graph[edge_type].edges.new_zeros(num_nodes, graph[edge_type].edges.shape[1])
        if self._weighted:
            output = scatter_add(graph[edge_type].edges * weight, indices.to(torch.int64), out=out, dim=0)
        else:
            output = scatter_add(graph[edge_type].edges, indices.to(torch.int64), out=out, dim=0)
        return output

class HeteroEdgesToGlobalsAggregator(pl.LightningModule):
    def __init__(self, num_graphs=None, scatter_func = scatter_add, weighted = True):
        super(HeteroEdgesToGlobalsAggregator, self).__init__()


        self._scatter_func = scatter_func
        self._weighted = weighted

    def forward(self, graph, edge_type, weights):
        out = graph[edge_type].edges.new_zeros(graph['globals'].x.shape[0], graph[edge_type].edges.shape[1])
        if self._weighted:
            output = self._scatter_func(graph[edge_type].edges*weights, graph[edge_type[0]].batch[ graph[edge_type].edge_index[0] ] ,out=out, dim=0)
        else:
            output = self._scatter_func(graph[edge_type].edges, graph[edge_type[0]].batch[ graph[edge_type].edge_index[0] ] ,out=out, dim=0)
        return output


class HeteroNodesToGlobalsAggregator(pl.LightningModule):
    def __init__(self, num_graphs=None, scatter_func = scatter_add, weighted = True):
        super(HeteroNodesToGlobalsAggregator, self).__init__()
        self._weighted = weighted
        self._scatter_func = scatter_func
    def forward(self, graph, node_type, weights):
        out = graph[node_type].x.new_zeros(graph['globals'].x.shape[0], graph[node_type].x.shape[1])
        if self._weighted:
            output = self._scatter_func(graph[node_type].x * weights, graph[node_type].batch,out=out, dim=0)
        else:
            output = self._scatter_func(graph[node_type].x, graph[node_type].batch,out=out, dim=0)
        return output