import torch
import torch.nn as nn
from torch.nn import functional as F

import math
from collections.abc import Sequence

class MLP(nn.Module):
    """
    Multi-layer Perceptron.
    Note there is no batch normalization, activation or dropout in the last layer.

    Parameters:
        input_dim (int): input dimension
        hidden_dim (list of int): hidden dimensions
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        dropout (float, optional): dropout rate
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        short_cut=False,
        batch_norm=False,
        activation="relu",
        dropout=0,
    ):
        super(MLP, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.dims = [input_dim] + hidden_dims
        self.short_cut = short_cut

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 2):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))
        else:
            self.batch_norms = None

    def forward(self, input):
        """"""
        layer_input = input

        for i, layer in enumerate(self.layers):
            hidden = layer(layer_input)
            if i < len(self.layers) - 1:
                if self.batch_norms:
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                hidden = self.activation(hidden)
                if self.dropout:
                    hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden

        return hidden


class SinusoidalPositionEmbedding(nn.Module):
    """
    Positional embedding based on sine and cosine functions, proposed in `Attention Is All You Need`_.

    .. _Attention Is All You Need:
       https://arxiv.org/pdf/1706.03762.pdf

    Parameters:
       output_dim (int): output dimension
    """

    def __init__(self, output_dim):
        super(SinusoidalPositionEmbedding, self).__init__()
        inverse_frequency = 1 / (
            10000 ** (torch.arange(0.0, output_dim, 2.0) / output_dim)
        )
        self.register_buffer("inverse_frequency", inverse_frequency)

    def forward(self, input):
        """"""
        # input: [B, L, ...]
        positions = torch.arange(
            input.shape[1] - 1, -1, -1.0, dtype=input.dtype, device=input.device
        )
        sinusoidal_input = torch.outer(positions, self.inverse_frequency)
        position_embedding = torch.cat(
            [sinusoidal_input.sin(), sinusoidal_input.cos()], -1
        )
        return position_embedding


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15.0, min_res=15.0 / 2000.0, div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = (
            2 * math.pi * div_factor ** torch.arange(self.n_frequencies) / max_res
        )
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()

