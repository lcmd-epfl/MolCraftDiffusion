"""Model-specific compatibility tests for GFMDiff integration."""

from __future__ import annotations

import torch

from MolecularDiffusion.modules.layers.gfmdiff.blocks import get_angle
from MolecularDiffusion.modules.models.gfmdiff.dynamics import GFMDiffDynamics


def test_gfmdiff_get_angle_clamps_dot_product_past_unity():
    # Float rounding can push a normalized dot product marginally above 1.0,
    # which sends unclamped acos to NaN (regression: generation produced
    # NaN coordinates on real, non-padded atom triples).
    vec = torch.tensor([[1.0, 0.0, 0.0]])
    angle = get_angle(vec, vec * (1.0 + 1e-6))

    assert torch.isfinite(angle).all()
    assert angle.item() < 0.1


def test_gfmdiff_dynamics_zeros_padded_node_rows():
    torch.manual_seed(0)
    in_node_nf = 6  # atom-type one-hot (5) + atomic number (1)
    dynamics = GFMDiffDynamics(
        in_node_nf=in_node_nf,
        num_layers=1,
        emb_dim=8,
        hidden_dim=16,
        num_heads=2,
    ).eval()

    bs, n_nodes = 1, 4
    xh = torch.randn(bs, n_nodes, 3 + in_node_nf)
    node_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]]).unsqueeze(-1)
    edge_mask = (node_mask * node_mask.transpose(1, 2)).unsqueeze(-1)
    t = torch.zeros(bs, 1)

    with torch.no_grad():
        out = dynamics._forward(t, xh, node_mask, edge_mask)

    # Regression: EquiGNN's remove_mean_with_mask only recenters the mean,
    # it doesn't zero individual masked rows, so padded atoms could drift
    # nonzero and fail EnVariationalDiffusion's masked-zero assertion.
    assert torch.equal(out[0, -1, :], torch.zeros(3 + in_node_nf))
