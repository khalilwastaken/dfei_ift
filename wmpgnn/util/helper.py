from typing import Dict, Any

import torch.nn as nn
from wmpgnn.model.mlp_class import MLP


def create_mlp(config: Dict[str, Any], outdim: int = -1) -> nn.Module:
    layers = config["layers"].copy()
    if outdim > 0:
        layers[-1] = outdim
    norm = config["norm"]
    dropout = config["dropout"]
    return MLP(layers, norm=norm, dropout=dropout)
