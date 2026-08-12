"""GVP layer norm. Verbatim port of LigandDiff's ``src/layer_norm.py``."""

import math
from typing import Tuple, Union

import torch
import torch.nn as nn

s_V = Tuple[torch.Tensor, torch.Tensor]


class GVPLayerNorm(nn.Module):
    """LayerNorm on the scalar channel, RMS rescaling on the vector channel."""

    def __init__(self, dims: Tuple[int, int], eps: float = 0.00001) -> None:
        super().__init__()
        self.eps = math.sqrt(eps)
        self.scalar_size, self.vector_size = dims
        self.feature_layer_norm = nn.LayerNorm(self.scalar_size, eps=eps)

    def forward(
        self, x: Union[torch.Tensor, s_V]
    ) -> Union[torch.Tensor, s_V]:
        if self.vector_size == 0:
            return self.feature_layer_norm(x)

        s, V = x
        if self.scalar_size != 1:
            s = self.feature_layer_norm(s)
        norm = torch.clip(
            torch.linalg.vector_norm(V, dim=(-1, -2), keepdim=True)
            / math.sqrt(self.vector_size),
            min=self.eps,
        )

        V = V / norm
        return s, V
