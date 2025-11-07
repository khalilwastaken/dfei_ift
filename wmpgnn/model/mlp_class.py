import inspect
import warnings
from typing import Any, Callable, Dict, Final, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import NoneType


class MLP(torch.nn.Module):
    supports_norm_batch: Final[List[bool]]

    def __init__(
            self,
            channel_list: Optional[Union[List[int], int]] = None,
            *,
            in_channels: Optional[int] = None,
            hidden_channels: Optional[int] = None,
            out_channels: Optional[int] = None,
            dropout: Union[float, List[float]] = 0.,
            act: Union[str, Callable, None] = "relu",
            act_first: bool = False,
            act_kwargs: Optional[Dict[str, Any]] = None,
            norm: Union[str, List[str]] = "batch_norm",
            norm_kwargs: Optional[Dict[str, Any]] = None,
            plain_last: bool = True,
            bias: Union[bool, List[bool]] = True,
    ):
        super().__init__()

        if isinstance(channel_list, int):
            in_channels = channel_list

        if in_channels is not None:
            if num_layers is None:
                raise ValueError("Argument `num_layers` must be given")
            if num_layers > 1 and hidden_channels is None:
                raise ValueError(f"Argument `hidden_channels` must be given "
                                 f"for `num_layers={num_layers}`")
            if out_channels is None:
                raise ValueError("Argument `out_channels` must be given")

            channel_list = [hidden_channels] * (num_layers - 1)
            channel_list = [in_channels] + channel_list + [out_channels]

        # Only allow passing of
        assert isinstance(channel_list, (tuple, list))
        assert len(channel_list) >= 2
        self.channel_list = channel_list

        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.act_first = act_first
        self.plain_last = plain_last

        if isinstance(dropout, float):
            dropout = [dropout] * (len(channel_list) - 1)
        if len(dropout) != len(channel_list) - 1:
            raise ValueError(
                f"Number of dropout values provided ({len(dropout)} does not "
                f"match the number of layers specified "
                f"({len(channel_list) - 1})")
        self.dropout = dropout

        if isinstance(bias, bool):
            bias = [bias] * (len(channel_list) - 1)
        if len(bias) != len(channel_list) - 1:
            raise ValueError(
                f"Number of bias values provided ({len(bias)}) does not match "
                f"the number of layers specified ({len(channel_list) - 1})")

        self.lins = torch.nn.ModuleList()
        iterator = zip(channel_list[:-1], channel_list[1:], bias)
        for in_channels, out_channels, _bias in iterator:
            self.lins.append(Linear(in_channels, out_channels, bias=_bias))

        self.norms = torch.nn.ModuleList()
        self.supports_norm_batch = [False] * (len(channel_list) - 2)
        if isinstance(norm, str):
            norm = [norm] * (len(channel_list) - 1)

        iterator = channel_list[1:-1] if plain_last else channel_list[1:]
        for i, hidden_channels in enumerate(iterator):
            if norm[i] != "None":
                norm_layer = normalization_resolver(
                    norm[i],
                    hidden_channels,
                    **(norm_kwargs or {}),
                )
                norm_params = inspect.signature(norm_layer.forward).parameters
                self.supports_norm_batch[i] = 'batch' in norm_params
            else:
                norm_layer = nn.Identity()
                self.supports_norm_batch[i] = False
            self.norms.append(norm_layer)

        self.reset_parameters()

    @property
    def in_channels(self) -> int:
        r"""Size of each input sample."""
        return self.channel_list[0]

    @property
    def out_channels(self) -> int:
        r"""Size of each output sample."""
        return self.channel_list[-1]

    @property
    def num_layers(self) -> int:
        r"""The number of layers."""
        return len(self.channel_list) - 1

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()

    def forward(
            self,
            x: Tensor,
            batch: Optional[Tensor] = None,
            batch_size: Optional[int] = None,
            return_emb: NoneType = None,
    ) -> Tensor:
        emb: Optional[Tensor] = None

        for i, (lin, norm) in enumerate(zip(self.lins, self.norms)):
            x = lin(x)
            if self.act is not None and self.act_first:
                x = self.act(x)
            if self.supports_norm_batch[i]:
                x = norm(x, batch, batch_size)
            else:
                x = norm(x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout[i], training=self.training)
            if isinstance(return_emb, bool) and return_emb is True:
                emb = x

        if self.plain_last:
            x = self.lins[-1](x)
            x = F.dropout(x, p=self.dropout[-1], training=self.training)

        return (x, emb) if isinstance(return_emb, bool) else x

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self.channel_list)[1:-1]})'


def create_mlp(config: Dict[str, Any], outdim: int = -1) -> nn.Module:
    layers = config["layers"].copy()
    if outdim > 0:
        layers[-1] = outdim
    norm = config["norm"]
    dropout = config["dropout"]
    return MLP(channel_list=layers, norm=norm, dropout=dropout)
