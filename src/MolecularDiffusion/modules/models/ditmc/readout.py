"""Readouts. Port of ``dit_mc/backbones/readout.py``.

Note a quirk that is reproduced deliberately: neither readout applies
``cond_scaling``. The conditioner's features enter the readout **unscaled**,
so classifier-free guidance does not zero them here even when it zeroes them in
every DiT block. That is upstream behaviour.

Neither readout branches on ``act_dense_correct_bool`` either -- both always
use ``act_fn(Dense(c))``.
"""

from __future__ import annotations

import torch
from torch import nn

from MolecularDiffusion.modules.layers.e3x import (
    add as e3x_add,
)
from MolecularDiffusion.modules.layers.e3x import (
    change_max_degree_or_type,
    get_activation_fn,
)

from .layers import (
    EquivariantLayerNorm,
    flax_dense,
    flax_layer_norm,
    modulate_adaLN,
    modulate_E3adaLN,
)

_ALLOWED_OUTPUTS = ("noise", "drift", "drift_and_noise")


def _check_output(output: str) -> None:
    if output not in _ALLOWED_OUTPUTS:
        msg = (
            f"`output` must be one of {_ALLOWED_OUTPUTS}. Received "
            f"output={output!r}"
        )
        raise ValueError(msg)


class SimpleReadout(nn.Module):
    """Non-equivariant readout (``dit_ape`` / ``dit_rpe``)."""

    def __init__(
        self, num_features: int, activation_fn: str, output: str = "drift_and_noise"
    ) -> None:
        super().__init__()
        _check_output(output)
        self.output = output
        self.num_features = num_features
        self.act_fn = get_activation_fn(activation_fn)
        self.ada_dense = flax_dense(
            num_features, 2 * num_features, bias=True, zero_init=True
        )
        self.norm = flax_layer_norm(num_features, use_scale=False, use_bias=False)
        self.head = flax_dense(
            num_features, 3 if output != "drift_and_noise" else 6, bias=True
        )

    def forward(
        self,
        features_nodes: torch.Tensor,
        features_time: torch.Tensor,
        features_cond: torch.Tensor | None = None,
    ):
        if features_nodes.dim() != 4:
            msg = "Features are assumed to be in the e3x convention."
            raise ValueError(msg)
        if features_nodes.shape[1] != 1 or features_nodes.shape[2] != 1:
            msg = "Parity must be 1 and maximal degree must be 0."
            raise ValueError(msg)
        h = features_nodes.squeeze(1).squeeze(1)
        t = features_time.squeeze(1).squeeze(1)
        cond = (
            features_cond.squeeze(1).squeeze(1)
            if features_cond is not None
            else torch.zeros_like(t)
        )

        shift, scale = self.act_fn(self.ada_dense(t + cond)).chunk(2, dim=-1)
        y = modulate_adaLN(self.norm(h), shift=shift, scale=scale)
        out = self.head(y)
        if self.output == "drift_and_noise":
            drift, noise = out.chunk(2, dim=-1)
            return drift, noise
        return out


class EquivariantReadout(nn.Module):
    """SO(3)-equivariant readout (``dit_so3``).

    ``y[:, 0] + y[:, 1]`` deliberately mixes the even and odd parity blocks --
    it breaks O(3) but leaves the output SO(3)-equivariant, which is what a
    coordinate model needs. The ``l=1`` block (indices 1:4) is the answer;
    index 0 (the scalar) is discarded.
    """

    def __init__(
        self,
        num_features: int,
        activation_fn: str,
        output: str = "drift_and_noise",
        in_max_degree: int = 1,
        in_num_parity: int = 2,
    ) -> None:
        super().__init__()
        _check_output(output)
        self.output = output
        self.num_features = num_features
        self.in_max_degree = in_max_degree
        self.in_num_parity = in_num_parity
        self.act_fn = get_activation_fn(activation_fn)
        self.cond_norm = flax_layer_norm(num_features)
        self.ada_dense = flax_dense(
            num_features, 3 * num_features, bias=True, zero_init=True
        )
        self.norm = EquivariantLayerNorm(
            num_features, 1, 1, use_scale=False, use_bias=False
        )
        self.head = flax_dense(
            num_features, 1 if output != "drift_and_noise" else 2, bias=False
        )

    def forward(
        self,
        features_nodes: torch.Tensor,
        features_time: torch.Tensor,
        features_cond: torch.Tensor | None = None,
    ):
        if features_nodes.dim() != 4:
            msg = "Features are assumed to be in the e3x convention."
            raise ValueError(msg)
        num_nodes = features_nodes.shape[0]
        f = self.num_features

        if features_cond is not None:
            if features_cond.shape[1:3] != (1, 1):
                msg = (
                    "Node features for conditioning must be invariant, i.e. "
                    f"max_degree = 0 and parity = 1. Received "
                    f"{tuple(features_cond.shape)}."
                )
                raise ValueError(msg)
            cond = features_cond
        else:
            cond = torch.zeros_like(features_time)

        c = self.cond_norm(e3x_add(cond, features_time))

        y = change_max_degree_or_type(features_nodes, max_degree=1)
        # Upstream indexes y[:, 1] unconditionally (dit_mc/backbones/
        # readout.py:151). Under JAX an out-of-bounds index CLAMPS, so
        # with include_pseudotensors=False (parity axis size 1) it
        # silently computes 2*y[:, 0]. PyTorch raises instead, so clamp
        # explicitly to reproduce upstream exactly. The factor 2 is then
        # cancelled by the scale-invariant EquivariantLayerNorm below.
        p1 = 1 if y.shape[1] > 1 else 0
        y = (y[:, 0, :, :] + y[:, p1, :, :]).unsqueeze(-3)  # (N,1,4,F)

        mod = self.act_fn(self.ada_dense(c)).squeeze(1).squeeze(1)
        scale, shift = torch.split(mod, [2 * f, f], dim=-1)
        scale = scale.reshape(num_nodes, 1, 2, f)
        shift = shift.reshape(num_nodes, 1, 1, f)

        y = modulate_E3adaLN(self.norm(y), shift=shift, scale=scale)
        out = self.head(y)  # (N, 1, 4, 1 or 2)

        if self.output == "drift_and_noise":
            out = out.squeeze(-3)  # (N, 4, 2)
            drift, noise = out[..., 0], out[..., 1]
            return drift[:, 1:], noise[:, 1:]
        return out.squeeze(-3).squeeze(-1)[:, 1:]
