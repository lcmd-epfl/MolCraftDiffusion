"""Priors for the FlowMol3 / CTMC port.

Ported from FlowMol (``flowmol/data_processing/priors.py``). Only the two the
released FlowMol3 config reaches are here:

- ``centered_normal_prior_batched_graph`` -- zero-COM Gaussian for ``x``.
- ``ctmc_masked_prior`` / ``ctmc_masked_edge_prior`` -- the all-mask categorical
  prior for ``a``/``c``/``e``.

The ``marginal``, ``c-given-a``, ``biased``, ``uniform-sample`` and
``barycenter`` priors are deliberately not ported: FlowMol3's own
``configure_prior`` (``flowmol/models/flowmol.py:190-193``) raises
``NotImplementedError`` unless all three categorical priors are ``ctmc``.

Per-item OT / rigid prior alignment (``prior_config.x.align: true``) is also not
ported -- it is a dataloader-``__getitem__`` operation upstream and the platform
has no per-item prior hook. It has no state-dict impact, and upstream does no
alignment at inference either (``flowmol.py:533``), so sampling from the
released weights is unaffected; only training efficiency is traded.
"""

import dgl
import torch
from torch.nn.functional import one_hot

__all__ = [
    "centered_normal_prior_batched_graph",
    "ctmc_masked_edge_prior",
    "ctmc_masked_prior",
]


def centered_normal_prior_batched_graph(
    g: dgl.DGLGraph, node_batch_idx: torch.Tensor, std: float = 1.0
) -> torch.Tensor:
    """Per-molecule zero-COM Gaussian prior for atom positions."""
    prior_sample = torch.randn(g.num_nodes(), 3, device=g.device) * std
    with g.local_scope():
        g.ndata["prior_sample"] = prior_sample
        com = dgl.readout_nodes(g, feat="prior_sample", op="mean")
        prior_sample = prior_sample - com[node_batch_idx]
    return prior_sample


def ctmc_masked_prior(n: int, d: int) -> torch.Tensor:
    """All-mask categorical prior: every one of ``n`` rows is the mask token.

    Returns a ``(n, d + 1)`` one-hot whose last column (index ``d``, the mask
    index) is set. ``d`` is the number of real classes -- the mask token is an
    internal noise state, never a chemistry class, and never appears in an
    output head.
    """
    p = torch.full((n,), fill_value=d)
    return one_hot(p, num_classes=d + 1).float()


def ctmc_masked_edge_prior(
    upper_edge_mask: torch.Tensor, n_bond_types: int
) -> torch.Tensor:
    """All-mask bond prior, mirrored across the two edge directions.

    The upper-triangle sample is written to both the upper and the mirrored
    lower triangle so the two directions of each bond always agree -- FlowMol
    enforces bond symmetry structurally, by this mirroring, not by assertion.
    """
    n_upper = int(upper_edge_mask.sum().item())
    upper = ctmc_masked_prior(n_upper, n_bond_types)
    prior = torch.zeros(
        upper_edge_mask.shape[0], upper.shape[1], device=upper_edge_mask.device
    )
    upper = upper.to(prior.device)
    prior[upper_edge_mask] = upper
    prior[~upper_edge_mask] = upper
    return prior
