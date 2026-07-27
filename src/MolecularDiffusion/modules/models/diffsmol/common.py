"""Shared primitives for the DiffSMol backbone.

Ported from DiffSMol ``source/models/common.py`` (only the parts the
bond-free score model actually reaches).

``GVP``/``GVPLayerNorm`` here are DiffSMol's own variants and are NOT
interchangeable with ``modules/layers/gvp`` (FlowMol's): that one takes a
four-dim ``(dim_vectors_in, dim_vectors_out, dim_feats_in, dim_feats_out)``
signature with SiLU/Sigmoid activations and a different tensor layout, while
this one takes ``(in_dims, h_dim, out_dims)`` tuples with ReLU/sigmoid. They
are kept separate rather than unified because silently swapping activations
inside a ported architecture is how you get a model that trains but is not
the model you ported.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "GVP",
    "GVPLayerNorm",
    "GaussianSmearing",
    "MLP",
    "ShiftedSoftplus",
    "SinusoidalPosEmb",
    "norm_no_nan",
    "outer_product",
]


def norm_no_nan(
    x: torch.Tensor,
    axis: int = -1,
    keepdims: bool = False,
    eps: float = 1e-8,
    sqrt: bool = True,
) -> torch.Tensor:
    """L2 norm clamped below at ``eps`` so the gradient never sees 0/0."""
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def outer_product(*vectors: torch.Tensor) -> torch.Tensor:
    """Flattened outer product of a sequence of per-edge feature tensors."""
    out = None
    for index, vector in enumerate(vectors):
        if index == 0:
            out = vector.unsqueeze(-1)
        else:
            out = out * vector.unsqueeze(1)
            out = out.view(out.shape[0], -1).unsqueeze(-1)
    return out.squeeze(-1)


class GaussianSmearing(nn.Module):
    """Distance expansion on a fixed, hand-chosen offset grid.

    Note the offsets are hardcoded upstream (20 of them), so ``num_gaussians``
    is advisory only -- the true feature width is always 20. It is kept in the
    signature because the config sets ``num_r_gaussian: 20`` to match.
    """

    _OFFSETS = [
        0, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3,
        3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10,
    ]

    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
    ) -> None:
        super().__init__()
        self.start = start
        self.stop = stop
        offset = torch.tensor(self._OFFSETS)
        self.num_gaussians = offset.numel()
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) - self.shift


NONLINEARITIES = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "softplus": nn.Softplus,
    "elu": nn.ELU,
    "silu": nn.SiLU,
}


class MLP(nn.Module):
    """MLP with the same hidden dim across all layers."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layer: int = 2,
        norm: bool = True,
        act_fn: str = "relu",
        act_last: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for layer_idx in range(num_layer):
            if layer_idx == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif layer_idx == num_layer - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_idx < num_layer - 1 or act_last:
                if norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(NONLINEARITIES[act_fn]())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GVP(nn.Module):
    """Geometric Vector Perceptron (Jing et al. 2021), vector-gated.

    Takes and returns ``(scalars [N, s], vectors [N, v, 3])``.
    """

    def __init__(
        self,
        in_dims: tuple[int, int],
        h_dim: int,
        out_dims: tuple[int, int],
        activations=(F.relu, torch.sigmoid),
        vector_gate: bool = True,
    ) -> None:
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate:
                    self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)
        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)
            vn = norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo:
                v = self.wv(vh)
                v = torch.transpose(v, -1, -2)
                if self.vector_gate:
                    gate = self.wsv(
                        self.vector_act(s) if self.vector_act else s
                    )
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(
                        norm_no_nan(v, axis=-1, keepdims=True)
                    )
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(
                    s.shape[0], self.vo, 3, device=self.dummy_param.device
                )
        if self.scalar_act:
            s = self.scalar_act(s)
        return (s, v) if self.vo else s


class GVPLayerNorm(nn.Module):
    """LayerNorm on scalars; RMS rescale (no learned params) on vectors."""

    def __init__(self, dims: tuple[int, int]) -> None:
        super().__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)

    def forward(self, x):
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)
