import pytorch_lightning as pl

import torch
import torch.nn as nn

from wmpgnn.model.mlp_class import create_mlp


class HeteroGraphTrafo(pl.LightningModule):
    def __init__(self, config, endecoder=True):
        super(HeteroGraphTrafo, self).__init__()

        self._node_models = {}
        self._edge_models = {}
        self._endecoder = endecoder

        # edges
        if "tr_tr" in config and config["tr_tr"] != "None":
            self._edge_models[('tracks', 'tracks')] = create_mlp(config["MLP"], outdim=config["tr_tr"])

        if "tr_pv" in config and config["tr_pv"] != "None":
            self._edge_models[('tracks', 'pvs')] = create_mlp(config["MLP"], outdim=config["tr_pv"])

        if "tr" in config and config["tr"] != "None":
            self._node_models["tracks"] = create_mlp(config["MLP"], outdim=config["tr"])

        if "pv" in config and config["pv"] != "None":
            self._node_models["pvs"] = create_mlp(config["MLP"], outdim=config["pv"])

        if "global" in config and config["global"] != "None":
            self._global_model = create_mlp(config["MLP"], outdim=config["global"])
        else:
            self._global_model = nn.Identity()

        self._edge_models_model_dict = torch.nn.ModuleDict({str(i): j for i, j in self._edge_models.items()})
        self._node_models_model_dict = torch.nn.ModuleDict({str(i): j for i, j in self._node_models.items()})

    def forward(self, graph):
        if self._endecoder:
            # Functions as an decoder
            for node_type in self._node_models.keys():
                graph[node_type].x = self._node_models[node_type](graph[node_type].x, graph[node_type].batch)
            for edge_type in self._edge_models.keys():
                graph[edge_type].edges = self._edge_models[edge_type](graph[edge_type].edges, graph[edge_type[0]].batch[
                    graph[edge_type].edge_index[0]])
        else:
            for node_type in self._node_models.keys():
                graph[node_type].x = self._node_models[node_type](graph[node_type].x)
            for edge_type in self._edge_models.keys():
                graph[edge_type].edges = self._edge_models[edge_type](graph[edge_type].edges)
        graph['globals'].x = self._global_model(graph['globals'].x)
        return graph
