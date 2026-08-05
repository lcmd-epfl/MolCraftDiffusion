"""SILVR reverse step vs. a literal transcription of the published sampler.

Ground truth: ``docs/model_integrations/silvr/reference_source/
silvr_fork_en_diffusion.py`` L926-976. The transcription below is that
block copied line for line, with the source's hardcoded ``181`` replaced
by the node count and its ``zero_padding_tensor`` by ``F.pad``.
"""

import torch
from torch.nn import functional as F  # noqa: N812

from MolecularDiffusion.modules.models.en_diffusion import (
    EnVariationalDiffusion,
)


class _Stub(EnVariationalDiffusion):
    """Bare EnVariationalDiffusion: only what the SILVR step touches."""

    def __init__(self, n_dims: int) -> None:
        self.n_dims = n_dims
        self.in_node_nf = 4
        self.T = 10

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        # Monotone stand-in for the learned/polynomial schedule.
        return 6.0 * t - 3.0

    def sample_p_zs_given_zt(  # noqa: PLR0913
        self,
        s: torch.Tensor,  # noqa: ARG002
        t: torch.Tensor,  # noqa: ARG002
        zt: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: None = None,  # noqa: ARG002
        context: None = None,  # noqa: ARG002
        fix_noise: bool = False,  # noqa: ARG002, FBT001, FBT002
    ) -> torch.Tensor:
        # Deterministic stand-in for the EDM reverse step; must keep the
        # coordinate block mean-zero, like the real one does.
        zs = zt * 0.9 + 0.05
        x = zs[:, :, : self.n_dims]
        x = x - x.sum(1, keepdim=True) / node_mask.sum(1, keepdim=True)
        feats = zs[:, :, self.n_dims :]
        return torch.cat([x * node_mask, feats * node_mask], dim=2)

    def sample_combined_position_feature_noise(
        self,
        n_samples: int,  # noqa: ARG002
        n_nodes: int,  # noqa: ARG002
        node_mask: torch.Tensor,  # noqa: ARG002
        std: float = 1.0,  # noqa: ARG002
    ) -> torch.Tensor:
        return self._eps


def _reference_step(a):  # noqa: ANN001, ANN202
    """Fork L926-976, transcribed."""
    m, z, xh, total_shift = a["m"], a["z"], a["xh"], a["total_shift"]
    node_mask, ref_node_mask = a["node_mask"], a["ref_node_mask"]
    n_pad = z.size(2) - m.n_dims

    gamma_t = m.inflate_batch_array(m.gamma(a["t"]), z)
    alpha_t = m.alpha(gamma_t, z)
    sigma_t = m.sigma(gamma_t, z)

    n = node_mask.sum(1, keepdims=True)
    mean = torch.sum(z[:, :, : m.n_dims], dim=1, keepdim=True) / n
    z = z - F.pad(mean, (0, n_pad)) * node_mask

    xh = xh * ref_node_mask + z * (ref_node_mask - node_mask)
    mean = torch.sum(xh[:, :, : m.n_dims], dim=1, keepdim=True) / n
    mean = F.pad(mean, (0, n_pad))
    xh = xh - mean * ref_node_mask
    total_shift = total_shift + mean

    z = m.sample_p_zs_given_zt(a["s"], a["t"], z, node_mask, None, None)

    eps = m.sample_combined_position_feature_noise(
        z.size(0), z.size(1), node_mask
    )
    z_t = alpha_t * xh + sigma_t * eps
    rate = a["silvr_rate"]
    z = z - (z * alpha_t * ref_node_mask) * rate + (z_t * ref_node_mask) * rate
    return z, xh, total_shift


_N_NODES = 6
_N_REF = 4
_N_LIVE = 5


def _setup():  # noqa: ANN202
    torch.manual_seed(0)
    m = _Stub(3)
    c = m.n_dims + m.in_node_nf
    node_mask = torch.zeros(1, _N_NODES, 1)
    node_mask[:, :_N_LIVE, :] = 1.0
    ref_node_mask = torch.zeros(1, _N_NODES, 1)
    ref_node_mask[:, :_N_REF, :] = 1.0
    silvr_rate = torch.zeros(1, _N_NODES, 1)
    silvr_rate[0, :_N_REF, 0] = 0.01
    m._eps = torch.randn(1, _N_NODES, c) * node_mask  # noqa: SLF001
    return {
        "m": m,
        "s": torch.full((1, 1), 0.4),
        "t": torch.full((1, 1), 0.5),
        "z": torch.randn(1, _N_NODES, c) * node_mask,
        "node_mask": node_mask,
        "xh": torch.randn(1, _N_NODES, c) * ref_node_mask,
        "silvr_rate": silvr_rate,
        "ref_node_mask": ref_node_mask,
        "total_shift": torch.zeros(1, 1, c),
    }


def _call(a, component):  # noqa: ANN001, ANN202
    return a["m"].sample_p_zs_given_zt_silvr(
        a["s"],
        a["t"],
        a["z"],
        a["node_mask"],
        None,
        None,
        a["xh"],
        a["silvr_rate"],
        a["ref_node_mask"],
        a["total_shift"],
        component,
    )


def _plain_step(a):  # noqa: ANN001, ANN202
    """The unguided EDM step, on the same pre-centred ``z``."""
    m, z, node_mask = a["m"], a["z"], a["node_mask"]
    mean = z[:, :, : m.n_dims].sum(1, keepdim=True) / node_mask.sum(
        1, keepdim=True
    )
    z = z - F.pad(mean, (0, m.in_node_nf)) * node_mask
    return m.sample_p_zs_given_zt(a["s"], a["t"], z, node_mask, None, None)


def test_silvr_step_matches_published_sampler() -> None:
    """``condition_component="xh"`` must equal the source, bit for bit."""
    a = _setup()
    for got, want in zip(_call(a, "xh"), _reference_step(a), strict=True):
        assert torch.allclose(got, want, atol=1e-6), (got - want).abs().max()


def test_silvr_step_x_only_leaves_features_to_the_edm() -> None:
    """``condition_component="x"`` guides coordinates only."""
    a = _setup()
    n_dims = a["m"].n_dims
    zs_x = _call(a, "x")[0]
    zs_xh = _call(a, "xh")[0]
    # Coordinates: identical guidance.
    assert torch.allclose(zs_x[:, :, :n_dims], zs_xh[:, :, :n_dims], atol=1e-6)
    # Features: untouched by SILVR under "x", i.e. the plain EDM step.
    assert not torch.allclose(
        zs_x[:, :, n_dims:], zs_xh[:, :, n_dims:], atol=1e-6
    )
    plain = _plain_step(a)
    assert torch.allclose(zs_x[:, :, n_dims:], plain[:, :, n_dims:], atol=1e-6)


def test_silvr_step_never_touches_the_dummy_or_pad_rows() -> None:
    """Guidance is confined to the reference rows by ref_node_mask."""
    a = _setup()
    zs = _call(a, "xh")[0]
    plain = _plain_step(a)
    assert torch.allclose(zs[:, _N_REF:, :], plain[:, _N_REF:, :], atol=1e-6)
    assert not torch.allclose(
        zs[:, :_N_REF, :], plain[:, :_N_REF, :], atol=1e-6
    )
