"""e3x gated-linear activations.

**Every** activation exported from ``e3x.nn`` is a gated linear::

    _gated_linear(g, x) = g(x[..., 0:1, 0:1, :]) * x

The gate is computed from the **even-parity scalar channel only** (``0+``) and
broadcast over the whole ``(P, (L+1)**2, F)`` tensor, including the ``l=0``
block itself. The odd-parity scalar is deliberately excluded -- gating on a
pseudoscalar would break parity. There is no norm pooling over ``m`` and no
unit-variance correction constant.

Do **not** conflate these with ``jax.nn.<name>``: DiTMC's plain ``MLP`` uses the
elementwise ``jax.nn`` versions, and ``E3MLP`` uses these. Same names, different
functions.

``gelu`` here (and ``jax.nn.gelu``) defaults to ``approximate=True``, the tanh
form -- ``torch.nn.GELU()`` defaults to the exact erf form.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812


def _gated_linear(g, x: torch.Tensor) -> torch.Tensor:
    if x.dim() < 3:
        msg = f"shape of x must have at least three dimensions, received {tuple(x.shape)}"
        raise ValueError(msg)
    return g(x[..., 0:1, 0:1, :]) * x


def silu(x: torch.Tensor) -> torch.Tensor:
    """Gated linear with a sigmoid gate (``jax.scipy.special.expit``)."""
    return _gated_linear(torch.sigmoid, x)


swish = silu


def gelu(x: torch.Tensor, *, approximate: bool = True) -> torch.Tensor:
    """Gated linear with a GELU-CDF gate; tanh approximation by default."""

    def g(z: torch.Tensor) -> torch.Tensor:
        if approximate:
            sqrt_2_over_pi = (2.0 / torch.pi) ** 0.5
            return 0.5 * (1.0 + torch.tanh(sqrt_2_over_pi * (z + 0.044715 * z**3)))
        return (torch.erf(z / 2**0.5) + 1) / 2

    return _gated_linear(g, x)


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


_E3X_ACTIVATIONS = {"silu": silu, "swish": swish, "gelu": gelu, "identity": identity}

#: elementwise counterparts, matching ``jax.nn.<name>``
_PLAIN_ACTIVATIONS = {
    "silu": F.silu,
    "swish": F.silu,
    # jax.nn.gelu defaults to approximate=True -> the tanh form.
    "gelu": lambda x: F.gelu(x, approximate="tanh"),
    "relu": F.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "identity": identity,
}


def get_e3x_activation_fn(name: str):
    """Port of ``backbones/utils.get_e3x_activation_fn``."""
    if name not in _E3X_ACTIVATIONS:
        msg = f"unknown e3x activation '{name}'"
        raise ValueError(msg)
    return _E3X_ACTIVATIONS[name]


def get_activation_fn(name: str):
    """Port of ``backbones/utils.get_activation_fn`` (elementwise ``jax.nn``)."""
    if name not in _PLAIN_ACTIVATIONS:
        msg = f"unknown activation '{name}'"
        raise ValueError(msg)
    return _PLAIN_ACTIVATIONS[name]
