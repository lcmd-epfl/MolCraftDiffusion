"""GVP dropout. Verbatim port of LigandDiff's ``src/dropout.py``."""

from typing import Tuple, Union

import torch
import torch.nn as nn

s_V = Tuple[torch.Tensor, torch.Tensor]


class GVPDropout(nn.Module):
    """Dropout on scalars, channel dropout on vectors."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.dropout_features = nn.Dropout(p)
        self.dropout_vector = nn.Dropout1d(p)

    def forward(
        self, x: Union[torch.Tensor, s_V]
    ) -> Union[torch.Tensor, s_V]:
        if isinstance(x, torch.Tensor):
            return self.dropout_features(x)

        s, V = x
        s = self.dropout_features(s)
        V = self.dropout_vector(V)
        return s, V
