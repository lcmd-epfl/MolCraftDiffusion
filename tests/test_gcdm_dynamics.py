"""Checks for the GCDM/GCPNet dense<->flat adapter.

The adapter in ``modules/models/gcdm/dynamics.py`` is the one place where the
port could be wrong in a way that still runs: it compacts the platform's
padded ``(B, N, ...)`` batch down to real atoms before calling the ported
network. These assert the two properties that would break if it were not.
"""

import torch

from MolecularDiffusion.modules.models.gcdm.dynamics import GCDMDynamics


def _tiny_dynamics(in_node_nf=6):
    torch.manual_seed(0)
    return GCDMDynamics(
        in_node_nf=in_node_nf,
        context_node_nf=0,
        num_encoder_layers=1,
        h_hidden_dim=16,
        chi_hidden_dim=4,
        e_hidden_dim=8,
        xi_hidden_dim=4,
    ).eval()


def _batch(n_atoms, n_pad, in_node_nf=6, seed=1):
    """One molecule of ``n_atoms`` real atoms padded out to ``n_pad`` rows."""
    g = torch.Generator().manual_seed(seed)
    xh = torch.zeros(1, n_pad, 3 + in_node_nf)
    xh[0, :n_atoms] = torch.randn(n_atoms, 3 + in_node_nf, generator=g)
    xh[0, :n_atoms, :3] -= xh[0, :n_atoms, :3].mean(0)  # zero CoG, as EDM does
    node_mask = torch.zeros(1, n_pad, 1)
    node_mask[0, :n_atoms] = 1.0
    return xh * node_mask, node_mask


def test_padding_does_not_change_the_prediction():
    """Same molecule, different pad width -> identical prediction.

    GCPNet's equivariant node channel is built from displacements between
    *consecutive rows of the flat node list*, so a naive ``reshape(B*N, ...)``
    would let padding rows leak into it and make the answer depend on
    ``max_atom``.
    """
    net = _tiny_dynamics()
    t = torch.full((1, 1), 0.3)

    xh_a, mask_a = _batch(n_atoms=7, n_pad=9)
    xh_b, mask_b = _batch(n_atoms=7, n_pad=20)

    with torch.no_grad():
        out_a = net._forward(t, xh_a, mask_a, None)
        out_b = net._forward(t, xh_b, mask_b, None)

    assert torch.allclose(out_a[0, :7], out_b[0, :7], atol=1e-6)
    # padded rows must come back exactly zero -- EnVariationalDiffusion
    # asserts this downstream
    assert torch.equal(out_b[0, 7:], torch.zeros_like(out_b[0, 7:]))


def test_rotation_equivariance():
    """Rotating the input rotates the coordinate prediction and leaves the
    scalar prediction alone."""
    net = _tiny_dynamics()
    t = torch.full((1, 1), 0.5)
    xh, mask = _batch(n_atoms=6, n_pad=6, seed=3)

    # a proper rotation about z
    theta = torch.tensor(0.7)
    c, s = torch.cos(theta), torch.sin(theta)
    rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    xh_rot = xh.clone()
    xh_rot[..., :3] = xh[..., :3] @ rot.T

    with torch.no_grad():
        out = net._forward(t, xh, mask, None)
        out_rot = net._forward(t, xh_rot, mask, None)

    assert torch.allclose(out[..., :3] @ rot.T, out_rot[..., :3], atol=1e-4)
    assert torch.allclose(out[..., 3:], out_rot[..., 3:], atol=1e-4)


def test_output_is_zero_centre_of_gravity():
    net = _tiny_dynamics()
    t = torch.full((1, 1), 0.9)
    xh, mask = _batch(n_atoms=8, n_pad=12, seed=5)
    with torch.no_grad():
        out = net._forward(t, xh, mask, None)
    assert out[0, :8, :3].mean(0).abs().max() < 1e-5


# -- GCDMOptimizeFactory context normalization --------------------------------
# Regression guard: `_build_context_row` used to have a silent `else` that
# passed the RAW target value through, so `normalize_condition: value_10` fed
# the network a 10x out-of-distribution context and every optimized molecule
# collapsed to ~97% carbon. It ran, exit 0, plausible-looking output.


def _factory(method):
    """A `GCDMOptimizeFactory` with only what `_build_context_row` reads."""
    from types import SimpleNamespace

    from MolecularDiffusion.runmodes.generate.tasks_gcdm_optimize import (
        GCDMOptimizeFactory,
    )

    factory = object.__new__(GCDMOptimizeFactory)
    factory.target_values = [80.0]
    factory.device = torch.device("cpu")
    factory.task = SimpleNamespace(
        normalize_condition=method,
        prop_dist_model=SimpleNamespace(
            distributions={"alpha": None},
            normalizer={"alpha": {"mean": 75.0, "mad": 5.0,
                                  "min": 30.0, "max": 130.0}},
        ),
    )
    return factory


def test_context_normalization():
    assert _factory("value_10")._build_context_row().item() == 8.0
    assert _factory("mad")._build_context_row().item() == 1.0
    assert _factory("maxmin")._build_context_row().item() == 0.0
    assert _factory(None)._build_context_row().item() == 80.0


def test_unknown_normalization_raises():
    try:
        _factory("nonsense")._build_context_row()
    except ValueError:
        return
    raise AssertionError("unknown normalize_condition must raise, not pass through")


if __name__ == "__main__":
    test_padding_does_not_change_the_prediction()
    test_rotation_equivariance()
    test_output_is_zero_centre_of_gravity()
    test_context_normalization()
    test_unknown_normalization_raises()
    print("all GCDM dynamics checks passed")
