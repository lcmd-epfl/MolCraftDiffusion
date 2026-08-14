"""Shared building blocks for Apo2Mol's ``uni_o2`` backbone.

Ported verbatim from ``others/Apo2Mol/models/common.py`` (only the pieces the
backbone actually touches). Kept as its own module rather than imported from
``modules/models/kgdiff/common.py`` because ``GaussianSmearing`` registers a
buffer that lands in the released checkpoint, so the class identity has to
stay under this package's key prefix.

Dead upstream helpers not ported: ``AngleExpansion``, ``get_h_dist``,
``get_r_feat`` (no call sites reachable from ``uni_o2``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import knn_graph


class GaussianSmearing(nn.Module):
    """Radial basis over a FIXED 20-entry offset table.

    Note the upstream quirk, preserved deliberately: ``start`` / ``stop`` /
    ``num_gaussians`` are recorded but **ignored** -- the offsets are a
    hardcoded, non-uniform table (``models/common.py:14``). The released
    checkpoint stores it as a buffer, so changing it would silently change
    every edge feature.
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


class Swish(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    "silu": nn.SiLU(),
}


class ShiftedSoftplus(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) - self.shift


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
        layers = []
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
    """Flattened outer product, used to cross edge type with distance RBF.

    ``outer_product(edge_attr (E,5), dist_feat (E,20)) -> (E, 100)``, which is
    the ``r_feat_dim = num_r_gaussian * 5`` the attention layers expect.
    """
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
    hbap_protein=None,
    hbap_ligand=None,
):
    """Interleave the pocket and ligand token sets into one graph-sorted list.

    The stable sort by graph id is what keeps each complex's tokens
    contiguous; ``mask_ligand`` records which of them came from the ligand.
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
    mask_protein = ~mask_ligand

    batch_ctx = batch_ctx[sort_idx]
    h_ctx = torch.cat([h_protein, h_ligand], dim=0)[sort_idx]
    pos_ctx = torch.cat([pos_protein, pos_ligand], dim=0)[sort_idx]

    hbap_ctx = None
    if hbap_protein is not None and hbap_ligand is not None:
        hbap_ctx = torch.cat([hbap_protein, hbap_ligand], dim=0)[sort_idx]

    return h_ctx, pos_ctx, batch_ctx, mask_ligand, mask_protein, hbap_ctx


def hybrid_edge_connection(ligand_pos, protein_pos, k, ligand_index, protein_index):
    dst = torch.repeat_interleave(ligand_index, len(ligand_index))
    src = ligand_index.repeat(len(ligand_index))
    mask = dst != src
    dst, src = dst[mask], src[mask]
    ll_edge_index = torch.stack([src, dst])

    dist = torch.unsqueeze(ligand_pos, 1) - torch.unsqueeze(protein_pos, 0)
    dist = torch.norm(dist, p=2, dim=-1)
    knn_p_idx = torch.topk(dist, k=k, largest=False, dim=1).indices
    knn_p_idx = protein_index[knn_p_idx]
    knn_l_idx = torch.unsqueeze(ligand_index, 1).repeat(1, k)
    pl_edge_index = torch.stack([knn_p_idx, knn_l_idx], dim=0).view(2, -1)
    return ll_edge_index, pl_edge_index


def batch_hybrid_edge_connection(x, k, mask_ligand, batch, add_p_index=False):
    """Only reachable via ``cutoff_mode='hybrid'``; the release uses ``knn``."""
    batch_size = batch.max().item() + 1
    batch_ll, batch_pl, batch_p = [], [], []
    with torch.no_grad():
        for i in range(batch_size):
            ligand_index = ((batch == i) & (mask_ligand == 1)).nonzero()[:, 0]
            protein_index = ((batch == i) & (mask_ligand == 0)).nonzero()[:, 0]
            ligand_pos, protein_pos = x[ligand_index], x[protein_index]
            ll_edge_index, pl_edge_index = hybrid_edge_connection(
                ligand_pos, protein_pos, k, ligand_index, protein_index
            )
            batch_ll.append(ll_edge_index)
            batch_pl.append(pl_edge_index)
            if add_p_index:
                all_pos = torch.cat([protein_pos, ligand_pos], 0)
                p_edge_index = knn_graph(all_pos, k=k, flow="source_to_target")
                p_edge_index = p_edge_index[
                    :, p_edge_index[1] < len(protein_pos)
                ]
                p_src, p_dst = p_edge_index
                all_index = torch.cat([protein_index, ligand_index], 0)
                batch_p.append(
                    torch.stack([all_index[p_src], all_index[p_dst]], 0)
                )

    if add_p_index:
        edge_index = [
            torch.cat([ll, pl, p], -1)
            for ll, pl, p in zip(batch_ll, batch_pl, batch_p)
        ]
    else:
        edge_index = [
            torch.cat([ll, pl], -1) for ll, pl in zip(batch_ll, batch_pl)
        ]
    return torch.cat(edge_index, -1)
