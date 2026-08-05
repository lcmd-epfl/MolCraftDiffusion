"""Shape + E(3)-equivariance check for the ported PaiNN dynamics.

The one thing worth guarding: PaiNNDynamics packs the diffusion engine's
dense padded batch into EquivNet's flat graph and back, and a silent bug
there (wrong compact index, un-removed CoM, leaking padded rows) produces
plausible-looking output that trains to nothing.
"""

import torch

from MolecularDiffusion.modules.models.painn_dynamics import PaiNNDynamics


def test_painn_dynamics_shapes_and_equivariance():
    torch.manual_seed(0)
    b, n, f = 3, 7, 6
    dyn = PaiNNDynamics(
        in_node_nf=f,
        num_interactions=2,
        node_size=16,
        edge_size=8,
        embedding_dim=16,
        rbf_features=8,
        time_features=4,
    ).double()

    node_mask = torch.zeros(b, n, 1, dtype=torch.float64)
    for i, k in enumerate([7, 4, 1]):  # incl. a single-atom graph
        node_mask[i, :k] = 1.0
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    edge_mask *= ~torch.eye(n, dtype=torch.bool)[None, :, :, None]
    edge_mask = edge_mask.reshape(b * n * n, 1)

    xh = torch.randn(b, n, 3 + f, dtype=torch.float64) * node_mask
    t = torch.rand(b, 1, dtype=torch.float64)

    out = dyn._forward(t, xh, node_mask, edge_mask, None)
    assert out.shape == (b, n, 3 + f)
    assert torch.allclose(
        out * (1 - node_mask), torch.zeros_like(out)
    ), "padded rows must be exactly zero"
    com = (out[:, :, :3] * node_mask).sum(1)
    assert com.abs().max() < 1e-8, f"CoM not removed: {com.abs().max()}"

    # rotate + translate: positions rotate, features stay invariant
    rot, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    rot = rot * torch.det(rot).sign()
    xh_r = (
        torch.cat([xh[:, :, :3] @ rot.T + 5.0, xh[:, :, 3:]], 2) * node_mask
    )
    out_r = dyn._forward(t, xh_r, node_mask, edge_mask, None)
    assert torch.allclose(out_r[:, :, :3], out[:, :, :3] @ rot.T, atol=1e-8)
    assert torch.allclose(out_r[:, :, 3:], out[:, :, 3:], atol=1e-8)

    # the (B, N*N) edge_mask layout (modules/tasks/diffusion.py:475) must
    # give the same answer as the (B*N*N, 1) one
    out2 = dyn._forward(t, xh, node_mask, edge_mask.reshape(b, n * n), None)
    assert torch.allclose(out, out2)

    # cutoff path: fewer edges, isolated nodes, still equivariant
    dyn.cutoff = 1.0
    out_c = dyn._forward(t, xh, node_mask, edge_mask, None)
    out_cr = dyn._forward(t, xh_r, node_mask, edge_mask, None)
    assert torch.allclose(out_cr[:, :, :3], out_c[:, :, :3] @ rot.T, atol=1e-8)
