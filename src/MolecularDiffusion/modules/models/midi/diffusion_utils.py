"""The four diffusion helpers MiDi's forward and reverse processes need.

Upstream ``midi/diffusion/diffusion_utils.py`` also carries the KL/NLL
machinery (``mask_distributions``, ``posterior_distributions``,
``gaussian_KL``, ``SNR`` ...). That is validation-only -- it feeds the
variational NLL, which is out of scope for this port -- so it is not ported.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812

from .placeholder import PlaceHolder


def assert_correctly_masked(
    variable: torch.Tensor, node_mask: torch.Tensor
) -> None:
    """Raise if ``variable`` is NaN or nonzero outside ``node_mask``."""
    if torch.isnan(variable).any():
        msg = f"NaN in masked tensor of shape {variable.shape}"
        raise ValueError(msg)
    residual = (variable * (1 - node_mask.long())).abs().max().item()
    if residual >= 1e-4:
        msg = f"variable not masked properly (max residual {residual})"
        raise ValueError(msg)


def cosine_beta_schedule_discrete(
    timesteps: int, nu_arr: list[float], s: float = 0.008
) -> np.ndarray:
    """MiDi's adaptive cosine schedule: one exponent ``nu`` per modality.

    Returns ``(timesteps + 1, n_modalities)`` betas, ordered as
    ``['p', 'x', 'c', 'e', 'y']``.
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    x = np.expand_dims(x, 0)  # (1, steps)

    nu = np.expand_dims(np.array(nu_arr), 1)  # (components, 1)

    alphas_cumprod = (
        np.cos(0.5 * np.pi * (((x / steps) ** nu) + s) / (1 + s)) ** 2
    )
    alphas_cumprod_new = alphas_cumprod / np.expand_dims(
        alphas_cumprod[:, 0], 1
    )
    alphas = alphas_cumprod_new[:, 1:] / alphas_cumprod_new[:, :-1]
    betas = 1 - alphas
    return np.swapaxes(betas, 0, 1)


def sample_discrete_features(
    probX: torch.Tensor,  # noqa: N803
    probE: torch.Tensor,  # noqa: N803
    prob_charges: torch.Tensor,
    node_mask: torch.Tensor,
) -> PlaceHolder:
    """Multinomial-sample node types, charges and (symmetric) bonds.

    Args:
        probX: ``(B,N,dx)`` node-type probabilities.
        probE: ``(B,N,N,de)`` edge probabilities.
        prob_charges: ``(B,N,dc)`` charge probabilities.
        node_mask: ``(B,N)`` bool.

    Returns:
        A ``PlaceHolder`` with integer class ids; ``E`` is upper-triangular
        sampled then mirrored, so symmetry holds by construction.
    """
    bs, n = node_mask.shape
    # Masked rows still have to be valid distributions for multinomial.
    probX[~node_mask] = 1 / probX.shape[-1]
    prob_charges[~node_mask] = 1 / prob_charges.shape[-1]

    probX = probX.reshape(bs * n, -1)
    prob_charges = prob_charges.reshape(bs * n, -1)

    X_t = probX.multinomial(1).reshape(bs, n)  # noqa: N806
    charges_t = prob_charges.multinomial(1).reshape(bs, n)

    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n, device=probE.device).unsqueeze(0).expand(bs, -1, -1)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]
    probE = probE.reshape(bs * n * n, -1)

    E_t = probE.multinomial(1).reshape(bs, n, n)  # noqa: N806
    E_t = torch.triu(E_t, diagonal=1)  # noqa: N806
    E_t = E_t + torch.transpose(E_t, 1, 2)  # noqa: N806

    return PlaceHolder(
        X=X_t,
        charges=charges_t,
        E=E_t,
        y=torch.zeros(bs, 0, device=X_t.device),
        pos=None,
    )


def compute_batched_over0_posterior_distribution(
    X_t: torch.Tensor,  # noqa: N803
    Qt: torch.Tensor,  # noqa: N803
    Qsb: torch.Tensor,  # noqa: N803
    Qtb: torch.Tensor,  # noqa: N803
) -> torch.Tensor:
    """``q(z_s | z_t, x_0)`` for every possible ``x_0``.

    Args:
        X_t: ``(B,N,dt)`` or ``(B,N,N,dt)``.
        Qt: ``(B,d_{t-1},dt)`` one-step transition matrix.
        Qsb: ``(B,d0,d_{t-1})`` cumulative to ``s``.
        Qtb: ``(B,d0,dt)`` cumulative to ``t``.

    Returns:
        ``(B, N, d0, d_{t-1})``.
    """
    X_t = X_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)  # noqa: N806

    left_term = X_t @ Qt.transpose(-1, -2)  # bs, N, d_t-1
    left_term = left_term.unsqueeze(dim=2)  # bs, N, 1, d_t-1
    numerator = left_term * Qsb.unsqueeze(1)  # bs, N, d0, d_t-1

    prod = Qtb @ X_t.transpose(-1, -2)  # bs, d0, N
    denominator = prod.transpose(-1, -2).unsqueeze(-1)  # bs, N, d0, 1
    denominator[denominator == 0] = 1e-6
    return numerator / denominator


def onehot_float(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    """``F.one_hot`` that returns float, since every consumer here wants it."""
    return F.one_hot(x, num_classes=num_classes).float()
