from wmpgnn.blocks.abstract_module import AbstractModule
from wmpgnn.blocks.hetero_edge_block import HeteroEdgeBlock
from wmpgnn.blocks.hetero_global_block import HeteroGlobalBlock
from wmpgnn.blocks.hetero_node_block import HeteroNodeBlock
from wmpgnn.util.helper import create_mlp
import torch
import torch.nn as nn
from torch.nn import Sigmoid
import pytorch_lightning as pl


def edge_pruning(edge_indices, graph, edge_type):
    graph[edge_type].edges = graph[edge_type].edges[edge_indices]
    graph[edge_type].edge_index = torch.vstack(
        [graph[edge_type].edge_index[0][edge_indices],
         graph[edge_type].edge_index[1][edge_indices]])
    graph[edge_type].y = graph[edge_type].y[edge_indices]


def node_pruning(node_indices, graph, node_type, edge_types):
    # Does not remove the nodes, only the edges
    num_nodes = graph[node_type].x.shape[0]
    valid_mask = torch.zeros(num_nodes, dtype=torch.bool)
    valid_mask[node_indices] = True

    edge_node_indices = {}
    for edge_type in edge_types:
        if edge_type[0] == node_type and edge_type[2] == node_type:
            # Use the valid mask directly for both source and target.
            mask = valid_mask[graph[edge_type].edge_index[0]] & valid_mask[graph[edge_type].edge_index[1]]
        elif edge_type[0] == node_type:
            mask = valid_mask[graph[edge_type].edge_index[0]]
        else:
            mask = valid_mask[graph[edge_type].edge_index[1]]

        graph[edge_type].edge_index = graph[edge_type].edge_index[:, mask]
        graph[edge_type].edges = graph[edge_type].edges[mask, :]
        graph[edge_type].y = graph[edge_type].y[mask]
        edge_node_indices[edge_type] = mask
    return edge_node_indices


class HeteroGraphNetwork(pl.LightningModule):
    def __init__(self, config, node_types, edge_types, FT_layer=False):
        super().__init__()
        self.edge_types = edge_types
        self.node_types = node_types
        self.FT = FT_layer
        self._use_globals = config["use_globals"]
        self._use_node_weights = config["use_node_weights"]
        self._use_edge_weights = config["use_edge_weights"]
        self._weighted_pass = False
        if any([config["use_node_weights"], config["use_edge_weights"]]) and config["weighted_pass"]:
            self._weighted_pass = config["weighted_pass"]

        # Edge, Node, Global block
        self._edge_block = HeteroEdgeBlock(config["MLP_forward"], edge_types)
        self._node_block = HeteroNodeBlock(config["MLP_forward"], node_types, edge_types)
        if self._use_globals:
            self._global_block = HeteroGlobalBlock(config["MLP_forward"], node_types, edge_types,
                                                   weighted_mp=self._weighted_pass)
        # Inference layers
        self._node_mlps = {}
        self._edge_mlps = {}
        if config["use_node_weights"]:
            for edge_type in edge_types:
                self._edge_mlps[edge_type] = create_mlp(config["MLP_infer"])

        if config["use_edge_weights"]:
            self._node_mlps['tracks'] = create_mlp(config["MLP_infer"])

        if self.FT:
            self._node_mlps['ft'] = create_mlp(config["MLP_infer"], outdim=3)

        self._edge_models_model_dict = torch.nn.ModuleDict({str(i): j for i, j in self._edge_mlps.items()})
        self._node_models_model_dict = torch.nn.ModuleDict({str(i): j for i, j in self._node_mlps.items()})

        self._sigmoid = Sigmoid()
        # nodes
        self.node_weights = {}
        self.node_logits = {}
        # edges
        self.edge_weights = {}
        self.edge_logits = {}

        # Pruning cuts for evaluate
        self.edge_prune = False
        self.node_prune = False
        self.prune_by_cut = False
        self.k_edges = 20
        self.k_nodes = 70
        self.edge_weight_cut = 0.001
        self.node_weight_cut = 0.001

        self.edge_indices = {}
        self.node_indices = {}
        self.edge_node_pruning_indices = {}

    def forward(self, graph, pid_nodes):
        # Applying edge update
        node_input = self._edge_block(graph)

        # Infer edges
        for edge_type in self.edge_types:
            if self._use_edge_weights:
                graph_batch = node_input[edge_type[0]].batch[node_input[edge_type].edge_index[0]]
                self.edge_logits[edge_type] = self._edge_mlps[edge_type](node_input[edge_type].edges, graph_batch)
                self.edge_weights[edge_type] = self._sigmoid(self.edge_logits[edge_type])
            else:
                self.edge_weights[edge_type] = torch.ones((graph[edge_type].edges.shape[0], 1)).to(self.device)

        if self.edge_prune:
            for edge_type in self.edge_types:
                if edge_type == ('tracks', 'to', 'tracks'):
                    mask = self.edge_weights[edge_type] > self.edge_weight_cut
                    edge_indices = torch.nonzero(mask, as_tuple=True)[0]
                    self.edge_indices[edge_type] = edge_indices
                    self.edge_weights[edge_type] = self.edge_weights[edge_type][edge_indices, :]
                    edge_pruning(edge_indices, node_input, edge_type)

        # Node update
        global_input = self._node_block(node_input, self.edge_weights)

        # Node infer
        for node_type in self.node_types:
            if self._use_node_weights and node_type != "pvs":
                self.node_logits[node_type] = self._node_mlps[node_type](global_input[node_type].x,
                                                                         global_input[node_type].batch)
                self.node_weights[node_type] = self._sigmoid(self.node_logits[node_type])
            else:
                self.node_weights[node_type] = torch.ones((graph[node_type].x.shape[0], 1)).to(self.device)

        if self.FT:
            # self.node_logits["frag"] = self._node_mlps["frag"](global_input["tracks"].x, global_input["tracks"].batch)
            # self.node_weights["frag"] = self._sigmoid(self.node_logits["frag"])
            # FT, catting pid information before pass, as well as the nodes itself
            combined_graph = torch.cat([global_input["tracks"].x, pid_nodes], dim=1)
            combined_graph = torch.cat([combined_graph, self.node_weights['tracks']], dim=1)
            self.node_logits["ft"] = self._node_mlps["ft"](combined_graph, global_input["tracks"].batch)
            self.node_weights["ft"] = torch.softmax(self.node_logits["ft"], dim=1)

        if self.node_prune:
            for node_type in self.node_types:
                if node_type == "tracks":
                    mask = self.node_weights[node_type] > self.node_weight_cut
                    node_indices = torch.nonzero(mask, as_tuple=True)[0]
                    self.node_indices[node_type] = node_indices
                    edge_index = faster_node_pruning(node_indices, global_input, node_type,
                                                     [('tracks', 'to', 'tracks')],
                                                     device=self.device)
                    self.edge_node_pruning_indices[node_type] = edge_index
                    for key in edge_index.keys():
                        self.edge_weights[key] = self.edge_weights[key][edge_index[key]]

        # Global update
        if self._use_globals:
            return self._global_block(global_input, self.edge_weights, self.node_weights)
        else:
            return global_input
