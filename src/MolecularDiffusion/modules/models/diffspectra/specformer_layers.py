"""Building blocks for :mod:`specformer` -- ported verbatim from
``others/DiffSpectra/models/specformer_layers.py`` (itself PatchTST's
building blocks, https://github.com/yuqinie98/PatchTST). No behaviour
changes; only the unused ``__all__`` re-export list and a couple of
docstrings were trimmed.

``moving_avg``/``series_decomp``/``Coord1dPosEncoding``/``Coord2dPosEncoding``
are dead in DMT's own use of :class:`~.specformer.SpecFormer` (it always
constructs with ``pe='zeros'``), but are kept because
:func:`positional_encoding` dispatches to them for other ``pe`` values and
the file is small enough that a partial port would only be a trap for
whoever changes ``pe`` later.
"""

import math

import torch
from torch import nn

__all__ = [
    "Coord1dPosEncoding",
    "Coord2dPosEncoding",
    "PositionalEncoding",
    "SinCosPosEncoding",
    "Transpose",
    "get_activation_fn",
    "moving_avg",
    "positional_encoding",
    "series_decomp",
]


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


def get_activation_fn(activation):
    if callable(activation):
        return activation()
    if activation.lower() == "relu":
        return nn.ReLU()
    if activation.lower() == "gelu":
        return nn.GELU()
    msg = f'{activation} is not available. You can use "relu", "gelu", or a callable'
    raise ValueError(msg)


class moving_avg(nn.Module):  # noqa: N801 - upstream name, kept for diffability
    """Moving average block to highlight the trend of a time series."""

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size, stride=stride, padding=0
        )

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class series_decomp(nn.Module):  # noqa: N801 - upstream name, kept for diffability
    """Series decomposition block."""

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


def PositionalEncoding(q_len, d_model, normalize=True):  # noqa: N802 - upstream name
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


SinCosPosEncoding = PositionalEncoding


def Coord2dPosEncoding(  # noqa: N802 - upstream name
    q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False
):
    x = 0.5 if exponential else 1
    cpe = None
    for _ in range(100):
        cpe = (
            2
            * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x)
            * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x)
            - 1
        )
        if verbose:
            print(f"{x:5.3f}  {cpe.mean():+6.3f}")  # noqa: T201
        if abs(cpe.mean()) <= eps:
            break
        if cpe.mean() > eps:
            x += 0.001
        else:
            x -= 0.001
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def Coord1dPosEncoding(q_len, exponential=False, normalize=True):  # noqa: N802
    cpe = (
        2
        * (
            torch.linspace(0, 1, q_len).reshape(-1, 1)
            ** (0.5 if exponential else 1)
        )
        - 1
    )
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def positional_encoding(pe, learn_pe, q_len, d_model):
    if pe is None:
        w_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(w_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == "zero":
        w_pos = torch.empty((q_len, 1))
        nn.init.uniform_(w_pos, -0.02, 0.02)
    elif pe == "zeros":
        w_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(w_pos, -0.02, 0.02)
    elif pe in ("normal", "gauss"):
        w_pos = torch.zeros((q_len, 1))
        nn.init.normal_(w_pos, mean=0.0, std=0.1)
    elif pe == "uniform":
        w_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(w_pos, a=0.0, b=0.1)
    elif pe == "lin1d":
        w_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == "exp1d":
        w_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == "lin2d":
        w_pos = Coord2dPosEncoding(
            q_len, d_model, exponential=False, normalize=True
        )
    elif pe == "exp2d":
        w_pos = Coord2dPosEncoding(
            q_len, d_model, exponential=True, normalize=True
        )
    elif pe == "sincos":
        w_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else:
        msg = (
            f"{pe} is not a valid pe (positional encoder). Available types: "
            "'gauss'=='normal', 'zeros', 'zero', 'uniform', 'lin1d', 'exp1d', "
            "'lin2d', 'exp2d', 'sincos', None."
        )
        raise ValueError(msg)
    return nn.Parameter(w_pos, requires_grad=learn_pe)
