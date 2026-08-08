"""Shared building blocks for the KGDiff backbone.

Ported from KGDiff's ``models/common.py`` (commit ``ad893fc``). Only the
pieces ``uni_transformer.py`` / ``score_model.py`` actually reach are kept:
the custom-offset :class:`GaussianSmearing`, the :class:`MLP` used inside
every attention head, :func:`outer_product`, :func:`compose_context`, and
:class:`ShiftedSoftplus`.

Dropped on purpose (dead for this integration, verified by call site):
``AngleExpansion``/``Swish``/``get_h_dist``/``get_r_feat`` are referenced
nowhere in the ported path, and ``hybrid_edge_connection`` /
``batch_hybrid_edge_connection`` only serve ``cutoff_mode='hybrid'``, which
neither the released config nor the released checkpoint uses (both are
``cutoff_mode: knn``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

#: ``act_fn`` lookup used by :class:`MLP`. The released configs only ever
#: say ``relu``; the rest are kept so an override does not KeyError.
NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "silu": nn.SiLU(),
}


class GaussianSmearing(nn.Module):
    """Distance expansion on a hand-picked, non-uniform offset grid.

    ``start``/``stop`` are recorded for ``__repr__`` only -- upstream
    overrides the linspace with a fixed 20-point table biased towards bond
    distances, and ``coeff`` is derived from the *first* gap (1.0 A), so the
    kernel width is constant even though the grid is not.
    """

    def __init__(
        self, start: float = 0.0, stop: float = 5.0, num_gaussians: int = 50
    ) -> None:
        super().__init__()
        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians
        offset = torch.tensor(
            [
                0, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3,
                3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10,
            ]
        )
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def __repr__(self) -> str:
        return (
            f"GaussianSmearing(start={self.start}, stop={self.stop}, "
            f"num_gaussians={self.num_gaussians})"
        )

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class MLP(nn.Module):
    """MLP with the same hidden dim across all layers.

    The ``net`` attribute name and the exact ``nn.Sequential`` ordering
    (Linear, LayerNorm, activation) are load-bearing: the released
    checkpoint's keys are ``*.net.0/1/3.*``.
    """

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
                layers.append(NONLINEARITIES[act_fn])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def outer_product(*vectors: torch.Tensor) -> torch.Tensor:
    """Flattened outer product, folded left to right."""
    out = None
    for index, vector in enumerate(vectors):
        if index == 0:
            out = vector.unsqueeze(-1)
        else:
            out = out * vector.unsqueeze(1)
            out = out.view(out.shape[0], -1).unsqueeze(-1)
    return out.squeeze()


def compose_context(
    h_protein: torch.Tensor,
    h_ligand: torch.Tensor,
    pos_protein: torch.Tensor,
    pos_ligand: torch.Tensor,
    batch_protein: torch.Tensor,
    batch_ligand: torch.Tensor,
):
    """Interleave pocket and ligand tokens into one graph-ordered node set.

    The sort is **stable**, so ligand atoms keep their relative order within
    each complex -- upstream's comment records that an unstable sort broke
    fixed-atom-type runs.

    Returns:
        ``(h_ctx, pos_ctx, batch_ctx, mask_ligand)`` where ``mask_ligand``
        is ``True`` on ligand rows.
    """
    batch_ctx = torch.cat([batch_protein, batch_ligand], dim=0)
    sort_idx = torch.sort(batch_ctx, stable=True).indices

    mask_ligand = torch.cat(
        [
            torch.zeros(
                [batch_protein.size(0)], device=batch_protein.device
            ).bool(),
            torch.ones(
                [batch_ligand.size(0)], device=batch_ligand.device
            ).bool(),
        ],
        dim=0,
    )[sort_idx]

    batch_ctx = batch_ctx[sort_idx]
    h_ctx = torch.cat([h_protein, h_ligand], dim=0)[sort_idx]
    pos_ctx = torch.cat([pos_protein, pos_ligand], dim=0)[sort_idx]

    return h_ctx, pos_ctx, batch_ctx, mask_ligand


class ShiftedSoftplus(nn.Module):
    """``softplus(x) - log(2)``, so ``f(0) == 0``."""

    def __init__(self) -> None:
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) - self.shift
