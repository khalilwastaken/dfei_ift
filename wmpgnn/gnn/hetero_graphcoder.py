import pytorch_lightning as pl

import torch

from wmpgnn.util.helper import create_mlp


class HeteroGraphCoder(pl.LightningModule):
    def __init__(self, config, node_types, edge_types, endecoder=True):
        super(HeteroGraphCoder, self).__init__()

        self._node_types = node_types
        self._edge_types = edge_types
        self._edge_models = {}
        self._node_models = {}
        self._endecoder = endecoder

        for node_type in self._node_types:
            if node_type in config["vars"]:
                self._node_models[node_type] = create_mlp(config["MLP"])
            else:
                self._node_models[node_type] = nn.Identity()

        for edge_type in self._edge_types:
            edge_config_def = f"{edge_type[0]}_{edge_type[2]}"
            if edge_config_def in config["vars"]:
                self._edge_models[edge_type] = create_mlp(config["MLP"])
            else:
                self._edge_models[edge_type] = nn.Identity()

        if "global" in config["vars"]:
            self._global_model = create_mlp(config["MLP"])
        else:
            self._global_model = nn.Identity()

        self._edge_models_model_dict = torch.nn.ModuleDict({str(i): j for i, j in self._edge_models.items()})
        self._node_models_model_dict = torch.nn.ModuleDict({str(i): j for i, j in self._node_models.items()})

    def forward(self, graph):
        if self._endecoder:
            # Functions as an decoder
            for node_type in self._node_types:
                graph[node_type].x = self._node_models[node_type](graph[node_type].x, graph[node_type].batch)
            for edge_type in self._edge_types:
                graph[edge_type].edges = self._edge_models[edge_type](graph[edge_type].edges, graph[edge_type[0]].batch[
                    graph[edge_type].edge_index[0]])
        else:
            for node_type in self._node_types:
                graph[node_type].x = self._node_models[node_type](graph[node_type].x)
            for edge_type in self._edge_types:
                graph[edge_type].edges = self._edge_models[edge_type](graph[edge_type].edges)
        graph['globals'].x = self._global_model(graph['globals'].x)
        return graph
