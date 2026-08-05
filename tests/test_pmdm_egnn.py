"""Equivalence + equivariance checks for the rewritten PMDM EGNN layer.

Upstream's ``EGNN_Sparse`` overrides ``MessagePassing.propagate`` and calls
``self.inspector.distribute`` / ``_collect``, private API that moved in PyG
2.5 (this env has 2.7), so the port writes message/aggregate directly with
``scatter_add``. That rewrite is the largest deviation from upstream, and the
easiest way to get it subtly wrong is to aggregate over the wrong end of each
edge -- PyG's ``x_i`` is the *target* (``edge_index[1]``) and ``x_j`` the
*source* (``edge_index[0]``), with ``aggr="add"`` summing into the target.
A swap still runs, still trains, and still produces molecule-shaped output;
it just breaks local geometry.

``reference_layer_forward`` re-derives one layer from PyG's documented
semantics with an explicit per-edge Python loop -- no scatter, no PyG -- so
it shares no code with the implementation under test.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_scatter")

from MolecularDiffusion.modules.models.pmdm.encoders import (  # noqa: E402
    EGNNSparseLayer,
)


def reference_layer_forward(layer, x, edge_index, edge_attr, batch, n_ligand):
    """One layer, computed edge-by-edge straight from PyG's semantics."""
    coors, feats = x[:, : layer.pos_dim], x[:, layer.pos_dim :]
    n = feats.size(0)

    m_i = torch.zeros(n, layer.m_dim)
    mhat_i = torch.zeros(n, layer.pos_dim)

    for e in range(edge_index.size(1)):
        source = int(edge_index[0, e])
        target = int(edge_index[1, e])

        rel = coors[source] - coors[target]
        rel_dist = (rel**2).sum().reshape(1)
        efeat = (
            rel_dist if edge_attr is None else torch.cat([edge_attr[e], rel_dist])
        )

        # PyG message(x_i, x_j, edge_attr): x_i is the TARGET, x_j the SOURCE.
        m_ij = layer.edge_mlp(torch.cat([feats[target], feats[source], efeat]))
        if layer.soft_edge:
            m_ij = m_ij * layer.edge_weight(m_ij)

        # aggr="add" sums each message into its TARGET node.
        m_i[target] = m_i[target] + m_ij
        mhat_i[target] = mhat_i[target] + layer.coors_mlp(m_ij) * layer.coors_norm(
            rel.unsqueeze(0)
        ).squeeze(0)

    hidden = layer.node_norm(feats, batch) if layer.node_norm is not None else feats
    hidden_out = feats + layer.node_mlp(torch.cat([hidden, m_i], dim=-1))

    # Only ligand rows move; the pocket is fixed conditioning.
    coors_out = torch.cat(
        [coors[:n_ligand] + mhat_i[:n_ligand], coors[n_ligand:]], dim=0
    )
    return torch.cat([coors_out, hidden_out], dim=-1)


def _fixture(seed=0, n_ligand=5, n_pocket=4, feats_dim=8, edge_attr_dim=3):
    torch.manual_seed(seed)
    layer = EGNNSparseLayer(
        feats_dim=feats_dim,
        edge_attr_dim=edge_attr_dim,
        m_dim=6,
        soft_edge=1,
        norm_coors=True,
    ).eval()
    n = n_ligand + n_pocket
    x = torch.randn(n, 3 + feats_dim)
    # directed, no self-loops, deliberately asymmetric so a src/dst swap shows
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j and (i + j) % 3]
    edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    edge_attr = torch.randn(edge_index.size(1), edge_attr_dim)
    batch = torch.zeros(n, dtype=torch.long)
    return layer, x, edge_index, edge_attr, batch, n_ligand


def test_matches_reference_message_passing():
    layer, x, edge_index, edge_attr, batch, n_ligand = _fixture()
    with torch.no_grad():
        got = layer(x, edge_index, edge_attr, batch, n_ligand)
        want = reference_layer_forward(
            layer, x, edge_index, edge_attr, batch, n_ligand
        )
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()


def test_swapped_aggregation_would_be_caught():
    """The reference must actually discriminate -- guard against a vacuous test."""
    layer, x, edge_index, edge_attr, batch, n_ligand = _fixture()
    flipped = edge_index.flip(0)
    with torch.no_grad():
        got = layer(x, edge_index, edge_attr, batch, n_ligand)
        want_flipped = reference_layer_forward(
            layer, x, flipped, edge_attr, batch, n_ligand
        )
    assert not torch.allclose(got, want_flipped, atol=1e-5)


def test_pocket_coordinates_are_frozen():
    layer, x, edge_index, edge_attr, batch, n_ligand = _fixture()
    with torch.no_grad():
        out = layer(x, edge_index, edge_attr, batch, n_ligand)
    assert torch.equal(out[n_ligand:, :3], x[n_ligand:, :3])
    assert not torch.allclose(out[:n_ligand, :3], x[:n_ligand, :3])


def test_equivariance_under_rotation_and_translation():
    """Coordinate output rotates with the input; features are invariant."""
    layer, x, edge_index, edge_attr, batch, n_ligand = _fixture()
    torch.manual_seed(7)
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    shift = torch.randn(3)

    moved = x.clone()
    moved[:, :3] = x[:, :3] @ q.T + shift

    with torch.no_grad():
        out = layer(x, edge_index, edge_attr, batch, n_ligand)
        out_moved = layer(moved, edge_index, edge_attr, batch, n_ligand)

    expected = out[:, :3] @ q.T + shift
    assert torch.allclose(out_moved[:, :3], expected, atol=1e-4)
    assert torch.allclose(out_moved[:, 3:], out[:, 3:], atol=1e-4)
