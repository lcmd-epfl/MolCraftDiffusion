"""Radial basis functions and the radial-angular ``basis`` wrapper.

Only the families DiTMC's shipped configs reach are ported:
``reciprocal_bernstein`` (``dit_so3``) and ``basic_fourier`` (``dit_rpe``'s
optional ``rpe_radial_basis_bool``, off in every shipped config but kept for
completeness). Port of ``e3x/nn/functions/{bernstein,trigonometric,mappings}.py``
and ``e3x/nn/wrappers.basis``.
"""

from __future__ import annotations

import functools
import math

import torch

from . import so3


def reciprocal_mapping(x: torch.Tensor, kind: str = "shifted") -> torch.Tensor:
    """Map ``[0, inf)`` to ``(0, 1]``. ``'shifted'`` is ``1/(x+1)``."""
    if kind == "shifted":
        return 1 / (x + 1)
    if kind == "damped":
        eps = torch.finfo(x.dtype).eps
        small = x < eps
        safe = torch.where(small, torch.ones_like(x), x)
        return torch.where(small, 1 - x / 2 + x * x / 6, -torch.expm1(-safe) / safe)
    if kind == "cuspless":
        return 1 / (x + torch.exp(-x))
    msg = f"unknown reciprocal mapping kind '{kind}'"
    raise ValueError(msg)


@functools.lru_cache(maxsize=8)
def _binomln(num: int) -> torch.Tensor:
    n = num - 1
    v = torch.arange(n + 1, dtype=torch.float64)
    # betaln(a, b) = gammaln(a) + gammaln(b) - gammaln(a + b)
    a, b = 1 + n - v, 1 + v
    betaln = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
    return -betaln - math.log(n + 1)


def _bernstein(x: torch.Tensor, num: int) -> torch.Tensor:
    """First ``num`` Bernstein polynomials, evaluated in log space.

    ``B_v(t) = C(n-1, v) t^v (1-t)^(n-1-v)``, ``v = 0..num-1``, with explicit
    endpoint fix-ups (``B_0(0) = 1``, ``B_{n-1}(1) = 1``) so the logs never see
    a zero.
    """
    x = x.unsqueeze(-1)
    if num < 1:
        msg = f"num must be greater or equal to 1, received {num}"
        raise ValueError(msg)
    if num == 1:
        return torch.ones_like(x)
    n = num - 1
    v = torch.arange(n + 1, dtype=x.dtype, device=x.device)
    binomln = _binomln(num).to(dtype=x.dtype, device=x.device)

    # jnp uses finfo.epsneg here; torch.finfo has no epsneg, but for IEEE
    # binary formats epsneg == eps / 2.
    eps = torch.finfo(x.dtype).eps / 2
    mask0 = x < eps
    mask1 = x > 1 - eps
    mask = mask0 | mask1
    safe_x = torch.where(mask, torch.full_like(x, 0.5), x)
    y = torch.where(
        mask,
        torch.zeros_like(x),
        torch.exp(binomln + v * torch.log(safe_x) + (n - v) * torch.log1p(-safe_x)),
    )
    y = torch.where(mask0 & (v == 0), torch.ones_like(y), y)
    y = torch.where(mask1 & (v == n), torch.ones_like(y), y)
    return y


def reciprocal_bernstein(
    x: torch.Tensor,
    num: int,
    kind: str = "shifted",
    use_reciprocal_weighting: bool = False,
) -> torch.Tensor:
    """``_bernstein(1 - reciprocal_mapping(x), num)``.

    DiTMC's ``dit_so3`` calls this with ``num=64`` and both defaults.
    """
    mapping = reciprocal_mapping(x, kind=kind)
    bernstein = _bernstein(1 - mapping, num=num)
    if use_reciprocal_weighting:
        bernstein = bernstein * mapping.unsqueeze(-1)
    return bernstein


def basic_fourier(x: torch.Tensor, num: int, limit: float = 1.0) -> torch.Tensor:
    """``cos(k·π·x/limit)`` for ``k = 0..num-1``.

    ``k`` starts at **0**, so channel 0 is the constant 1. No normalization
    constant.
    """
    if num < 1:
        msg = f"num must be greater or equal to 1, received {num}"
        raise ValueError(msg)
    frequency = math.pi * torch.arange(0, num, dtype=x.dtype, device=x.device)
    return torch.cos(frequency * (x / limit).unsqueeze(-1))


_RADIAL_FNS = {
    "reciprocal_bernstein": reciprocal_bernstein,
    "basic_fourier": basic_fourier,
}


def get_radial_fn(name: str):
    if name not in _RADIAL_FNS:
        msg = (
            f"unknown radial basis '{name}'. Only the families DiTMC's shipped "
            f"configs use are ported: {sorted(_RADIAL_FNS)}"
        )
        raise ValueError(msg)
    return _RADIAL_FNS[name]


def basis(
    r: torch.Tensor,
    *,
    max_degree: int,
    num: int,
    radial_fn,
    cutoff_fn=None,
) -> torch.Tensor:
    """Radial-angular basis, ``(..., 1, (L+1)**2, num)``.

    **Angular on axis -2, radial on axis -1** -- transposing these two is the
    classic silent bug. A parity axis of size 1 is appended at -3; ``basis``
    never emits ``P = 2``.

    ``dit_so3`` sets ``cutoff_fn: null`` and the factory raises if a cutoff is
    given, so no cutoff function is ever applied in practice.
    """
    if r.shape[-1] != 3:
        msg = f"r must have shape (..., 3), received {tuple(r.shape)}"
        raise ValueError(msg)

    norm = torch.linalg.vector_norm(r, dim=-1, keepdim=True)
    u = r / torch.where(norm > 0, norm, torch.ones_like(norm))
    norm = norm.squeeze(-1)

    rbf = radial_fn(norm, num)
    if cutoff_fn is not None:
        rbf = rbf * cutoff_fn(norm).unsqueeze(-1)

    ylm = so3.spherical_harmonics(u, max_degree, r_is_normalized=True)
    out = ylm.unsqueeze(-1) * rbf.unsqueeze(-2)
    return out.unsqueeze(-3)
