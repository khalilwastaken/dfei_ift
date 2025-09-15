import os, sys

import torch.nn as nn
from torch_geometric.nn.models import MLP

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from wmpgnn.gnn.hetero_graphcoder import HeteroGraphCoder
from wmpgnn.gnn.hetero_graph_network import HeteroGraphNetwork


def get_blocks(config, node_types, edge_types):
    n_gn_blocks = config["nBlocks"]
    n_ft_layers = config["FTlayers"]

    add_ft_layer = False
    blocks = []
    for i in range(n_gn_blocks):
        if i >= n_gn_blocks - n_ft_layers:
            add_ft_layer = True
        blocks.append(HeteroGraphNetwork(config, node_types, edge_types, add_ft_layer))
    return nn.ModuleList(blocks)


def get_op_trafo(config, node_types, edge_types):
    edge_models = {}
    node_models = {}

    # Edge trafo
    if config["tr_tr_edge_op"] is not None:
        edge_models[('tracks', 'to', 'tracks')] = lambda: nn.Linear(16, config["tr_tr_edge_op"])
    else:
        edge_models[('tracks', 'to', 'tracks')] = lambda: nn.Identity()
    if config["tr_pv_edge_op"] is not None:
        edge_models[('tracks', 'to', 'pvs')] = lambda: nn.Linear(16, config["tr_pv_edge_op"])
    else:
        edge_models[('tracks', 'to', 'pvs')] = lambda: nn.Identity()

    # Node trafo
    if config["tr_node_op"] != "None":
        node_models["tracks"] = lambda: nn.Linear(16, config["tr_node_op"])
    else:
        node_models["tracks"] = lambda: nn.Identity()
    if config["pv_node_op"] != "None":
        node_models["pvs"] = lambda: nn.Linear(16, config["pv_node_op"])
    else:
        node_models["pvs"] = lambda: nn.Identity()

    # Global trafo
    if config["global_node_op"] != "None":
        global_fn = lambda: nn.Linear(16, config["global_node_op"])
    else:
        global_fn = lambda: nn.Identity()

    _output_transform = HeteroGraphCoder(node_types, edge_types, edge_models=edge_models,
                                         node_models=node_models, global_model=global_fn, endecoder=False)

    return _output_transform


def get_IFT_model(config, node_types, edge_types):
    n_blocks = config["nBlocks"]
    mlp_layer = config["MLP"]
    dropout = config["dropout"]
    norm = config["norm"]

    use_node_weights = config["node_infer"]
    use_edge_weights = config["edge_infer"]

    # temporary stuff
    weight_mlp_channels = 1
    weight_mlp_layers = 1
    weighted_mp = False

    mlp = lambda: MLP(mlp_layer, norm=norm, drop_out=dropout)  # mlp for GN blocks update

    blocks = []
    for i in range(n_blocks):
        blocks.append(
            HeteroGraphNetwork(node_types, edge_types, edge_model=mlp, node_model=mlp, global_model=mlp,
                               use_node_weights=use_node_weights, use_edge_weights=use_edge_weights,
                               weight_mlp_channels=weight_mlp_channels, weight_mlp_layers=weight_mlp_layers,
                               weighted_mp=weighted_mp, norm=norm, drop_out=dropout, nFT_layers=False))

    return nn.ModuleList(blocks)
