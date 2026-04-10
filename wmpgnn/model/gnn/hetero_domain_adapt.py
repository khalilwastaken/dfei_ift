import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter_min
from torch.autograd import Function

from wmpgnn.model.mlp_class import create_mlp


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


def get_scatter_func(name: str):
    scatter_funcs = {
        "mean": scatter_mean,
        "add": scatter_add,
        "max": scatter_max,
        "min": scatter_min,
    }
    if name not in scatter_funcs:
        raise ValueError(f"Unknown scatter function: {name}. Choose from {list(scatter_funcs.keys())}")
    return scatter_funcs[name]


class HeteroDomainAdapt(pl.LightningModule):
    def __init__(self, config, node_types, edge_types):
        super().__init__()
        self.edge_types = edge_types
        self.node_types = node_types
        self.scatter_func = get_scatter_func(config["scatter_func"])
        self.grl = GRL(alpha=1.0)
        self.mlp = create_mlp(config["MLP"])

        self.da_score = None

    def forward(self, data):
        aggregated = []

        # Aggregate node features
        for node_type in self.node_types:
            out = data[node_type].x.new_zeros(data['globals'].x.shape[0], data[node_type].x.shape[1])
            node_agg = self.scatter_func(data[node_type].x, data[node_type].batch, out=out, dim=0)  # [B, D]
            aggregated.append(node_agg)

        # Aggregate global features
        global_agg = self.scatter_func(data['globals'].x, data['globals'].batch,
                                       out=data['globals'].x.new_zeros(data['globals'].x.shape[0],
                                                                       data['globals'].x.shape[1]), dim=0)  # [B, D]
        aggregated.append(global_agg)

        # Aggregate edge features
        for edge_type in self.edge_types:
            src_type = edge_type[0]
            out = data[edge_type].edges.new_zeros(data['globals'].x.shape[0], data[edge_type].edges.shape[1])
            batch_idx = data[src_type].batch[data[edge_type].edge_index[0]]
            edge_agg = self.scatter_func(data[edge_type].edges, batch_idx, out=out, dim=0)  # [B, D]
            aggregated.append(edge_agg)

        # Concat all and pass to MLP
        x = torch.cat(aggregated, dim=-1)  # [B, D * (n_nodes + 1 + n_edges)]
        x = self.grl(x)  # <-- GRL applied
        self.da_score = self.mlp(x)
