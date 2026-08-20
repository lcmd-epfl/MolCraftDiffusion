"""Shared building blocks for DiTMC. Port of ``dit_mc/backbones/utils.py``.

Parity traps carried in here from the plan, all load-bearing:

* Flax ``nn.LayerNorm`` defaults to ``epsilon=1e-6``; PyTorch's default is
  ``1e-5``. Every LayerNorm below pins ``1e-6``.
* Flax ``nn.Dense``'s default kernel init is ``lecun_normal`` -- a **truncated**
  normal, not PyTorch's uniform default.
* ``GaussianRandomFourierFeatures`` interleaves ``cos, sin`` (stack-then-reshape),
  it does **not** concatenate halves.
* adaLN-Zero ``Dense`` layers stay zero-initialized.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from MolecularDiffusion.modules.layers.e3x import (
    Dense as E3Dense,
)
from MolecularDiffusion.modules.layers.e3x import (
    add as e3x_add,
)
from MolecularDiffusion.modules.layers.e3x import (
    broadcast_equivariant_multiplication,
    extract_max_degree,
    get_activation_fn,
    get_e3x_activation_fn,
)
from MolecularDiffusion.modules.layers.e3x.initializers import lecun_normal_

#: Flax ``nn.LayerNorm`` epsilon.
LAYERNORM_EPS = 1e-6


def flax_layer_norm(
    num_features: int, *, use_scale: bool = True, use_bias: bool = True
) -> nn.LayerNorm:
    """``nn.LayerNorm`` with Flax's defaults (eps 1e-6, last axis only)."""
    ln = nn.LayerNorm(
        num_features, eps=LAYERNORM_EPS, elementwise_affine=use_scale or use_bias
    )
    if ln.elementwise_affine:
        if not use_scale:
            ln.weight.requires_grad_(False)
            nn.init.ones_(ln.weight)
        if not use_bias:
            ln.bias.requires_grad_(False)
            nn.init.zeros_(ln.bias)
    return ln


def flax_dense(
    in_features: int, out_features: int, *, bias: bool = True, zero_init: bool = False
) -> nn.Linear:
    """``flax.linen.Dense`` equivalent: lecun-normal kernel, zero bias."""
    layer = nn.Linear(in_features, out_features, bias=bias)
    if zero_init:
        nn.init.zeros_(layer.weight)
    else:
        lecun_normal_(layer.weight, in_features)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def modulate_adaLN(  # noqa: N802
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    if not x.shape == scale.shape == shift.shape:
        msg = (
            f"shape of features, scale and shift must be identical. Received "
            f"{tuple(x.shape)}, {tuple(scale.shape)} and {tuple(shift.shape)}"
        )
        raise ValueError(msg)
    return x * (1 + scale) + shift


def modulate_E3adaLN(  # noqa: N802
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    x_scaled = broadcast_equivariant_multiplication(factor=1 + scale, tensor=x)
    return e3x_add(x_scaled, shift)


class GaussianRandomFourierFeatures(nn.Module):
    """``gamma(x) = [cos(2π bᵀx), sin(2π bᵀx)]`` **interleaved**.

    ``b`` has shape ``(d, features//2)`` and is initialized ``normal(sigma)``.
    """

    def __init__(self, in_features: int, features: int, sigma: float = 1.0) -> None:
        super().__init__()
        if features % 2 != 0:
            msg = f"features must be even, received {features}"
            raise ValueError(msg)
        self.b = nn.Parameter(torch.empty(in_features, features // 2))
        nn.init.normal_(self.b, mean=0.0, std=sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bt_x = torch.einsum("...d,dh->...h", x, self.b)
        cos = torch.cos(2 * math.pi * bt_x)
        sin = torch.sin(2 * math.pi * bt_x)
        return torch.stack([cos, sin], dim=-1).reshape(*cos.shape[:-1], -1)


class MLP(nn.Module):
    """Plain (elementwise-activation) MLP. Port of ``backbones/utils.MLP``.

    Uses ``get_activation_fn`` (the ``jax.nn.<name>`` family), **not** the e3x
    gated ones -- ``DiTLayer`` uses this and ``SO3DiTLayer`` uses
    :class:`E3MLP`, and they are different functions with the same names.
    """

    def __init__(
        self,
        in_features: int,
        num_features: int | Sequence[int],
        num_layers: int = 2,
        activation_fn: str = "identity",
        use_bias: bool = True,
        output_is_zero_at_init: bool = False,
    ) -> None:
        super().__init__()
        feats = (
            list(num_features)
            if isinstance(num_features, (list, tuple))
            else [num_features] * num_layers
        )
        self.activation_fn = get_activation_fn(activation_fn)
        self.num_layers = num_layers
        layers = []
        prev = in_features
        for n in range(num_layers):
            layers.append(
                flax_dense(
                    prev,
                    feats[n],
                    bias=use_bias,
                    zero_init=output_is_zero_at_init and n == num_layers - 1,
                )
            )
            prev = feats[n]
        self.layers = nn.ModuleList(layers)
        self.out_features = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for n, layer in enumerate(self.layers):
            x = layer(x)
            if n < self.num_layers - 1:
                x = self.activation_fn(x)
        return x


class E3MLP(nn.Module):
    """Equivariant MLP: e3x ``Dense`` layers with a **gated** activation."""

    def __init__(
        self,
        in_features: int,
        num_features: int | Sequence[int],
        max_degree: int,
        num_parity: int,
        num_layers: int = 2,
        activation_fn: str = "identity",
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        feats = (
            list(num_features)
            if isinstance(num_features, (list, tuple))
            else [num_features] * num_layers
        )
        self.activation_fn = get_e3x_activation_fn(activation_fn)
        self.num_layers = num_layers
        layers = []
        prev = in_features
        for n in range(num_layers):
            layers.append(
                E3Dense(prev, feats[n], max_degree, num_parity, use_bias=use_bias)
            )
            prev = feats[n]
        self.layers = nn.ModuleList(layers)
        self.out_features = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for n, layer in enumerate(self.layers):
            x = layer(x)
            if n < self.num_layers - 1:
                x = self.activation_fn(x)
        return x


def _degree_repeat_ids(degrees: Sequence[int], values) -> torch.Tensor:
    out = []
    for d, v in zip(degrees, values, strict=True):
        out.extend([v] * (2 * d + 1))
    return torch.tensor(out, dtype=torch.long)


class EquivariantLayerNorm(nn.Module):
    r"""Layer norm that respects degree/parity structure.

    The ``l=0`` block goes through an ordinary ``LayerNorm``. The ``l>0``
    channels are **not** mean-subtracted: the per-``(parity, degree)`` norm over
    the order index is computed, its variance over the *feature* axis is taken,
    and the block is multiplied by ``rsqrt(var + eps)`` (times an optional
    learnable ``scales_lm``). ``epsilon = 1e-6``.

    DiTMC instantiates this only with ``use_scale=False, use_bias=False``, so in
    the shipped models it carries no parameters at all.
    """

    def __init__(
        self,
        num_features: int,
        max_degree: int,
        num_parity: int,
        *,
        use_scale: bool = True,
        use_bias: bool = True,
        epsilon: float = LAYERNORM_EPS,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.max_degree = max_degree
        self.num_parity = num_parity
        self.use_scale = use_scale
        self.use_bias = use_bias
        self.epsilon = epsilon

        self.has_pseudotensors = num_parity == 2
        self.has_ylms = (max_degree + 1) ** 2 > 1
        self.norm00 = flax_layer_norm(
            num_features, use_scale=use_scale, use_bias=use_bias
        )

        if self.has_pseudotensors or self.has_ylms:
            even = _degree_repeat_ids(
                list(range(1, max_degree + 1)), list(range(max_degree))
            )
            if self.has_pseudotensors:
                odd = _degree_repeat_ids(
                    list(range(max_degree + 1)),
                    list(range(max_degree, 2 * max_degree + 1)),
                )
            else:
                odd = torch.empty(0, dtype=torch.long)
            self.register_buffer("sum_idx", torch.cat([even, odd]), persistent=False)
            self.num_segments = (
                2 * max_degree + 1 if self.has_pseudotensors else max_degree
            )
            if use_scale:
                self.scales_lm = nn.Parameter(torch.ones(self.num_segments))
            else:
                self.scales_lm = None
        else:
            self.sum_idx = None
            self.num_segments = 0
            self.scales_lm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            msg = f"expected a rank-4 tensor, received {tuple(x.shape)}"
            raise ValueError(msg)
        if not (self.has_pseudotensors or self.has_ylms):
            return self.norm00(x)

        num_atoms = x.shape[0]
        plm_axes = x.shape[-3:-1]
        y = x.reshape(num_atoms, -1, self.num_features)
        y00, ylm = y[:, :1], y[:, 1:]

        sq = ylm.square()
        summed = sq.new_zeros(num_atoms, self.num_segments, self.num_features)
        summed = summed.index_add(1, self.sum_idx, sq)
        # safe_mask: sqrt only where the sum exceeds eps, else 0 (gradient-safe).
        big = summed > self.epsilon
        ylm_inv = torch.where(big, torch.sqrt(torch.where(big, summed, torch.zeros_like(summed))), torch.zeros_like(summed))

        # flax nn.normalization._compute_stats: var = E[x^2] - E[x]^2 over the
        # feature axis. Only the variance is used -- no mean subtraction.
        mean = ylm_inv.mean(dim=-1)
        var = (ylm_inv * ylm_inv).mean(dim=-1) - mean * mean
        mul_lm = torch.rsqrt(var + self.epsilon)
        if self.scales_lm is not None:
            mul_lm = mul_lm * self.scales_lm
        mul_lm = mul_lm.unsqueeze(-1)

        ylm = ylm * mul_lm.index_select(1, self.sum_idx)
        y00 = self.norm00(y00)
        y = torch.cat([y00, ylm], dim=1)
        return y.reshape(num_atoms, *plm_axes, self.num_features)


def get_max_degree_from_tensor_e3x(x: torch.Tensor) -> int:
    return extract_max_degree(x.shape)
