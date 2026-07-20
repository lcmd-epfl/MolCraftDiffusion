"""Prior distributions for the coordinate-only FlowMol port.

Only the priors the ``x``/``a``/``c`` modalities need are kept (ported from
FlowMol ``flowmol/data_processing/priors.py``):

- ``centered_normal_prior_batched_graph`` — a zero-COM Gaussian prior for the
  continuous coordinate modality ``x`` (COM removed per molecule).
- ``uniform_simplex_prior`` — a uniform-on-the-simplex prior for the
  categorical modalities ``a`` (atom types) and ``c`` (formal charge). This
  avoids needing a dataset ``marginal_dists_file``.

Edge / bond priors, CTMC masked priors, marginal / biased priors, and OT
alignment are intentionally dropped (out of scope — coordinate-only port).
"""

import dgl
import torch
from torch.distributions import Exponential


def centered_normal_prior_batched_graph(
    g: dgl.DGLGraph, node_batch_idx: torch.Tensor, std: float = 1.0
) -> torch.Tensor:
    """Per-molecule zero-COM Gaussian prior for atom positions."""
    n = g.num_nodes()
    prior_sample = torch.randn(n, 3, device=g.device) * std
    with g.local_scope():
        g.ndata["prior_sample"] = prior_sample
        com = dgl.readout_nodes(g, feat="prior_sample", op="mean")
        prior_sample = prior_sample - com[node_batch_idx]
    return prior_sample


def uniform_simplex_prior(n: int, d: int) -> torch.Tensor:
    """Uniform distribution on the (d-1)-simplex, shape (n, d)."""
    exp_dist = Exponential(torch.tensor(1.0))
    sample = exp_dist.sample((n, d))
    sample = sample / sample.sum(dim=1, keepdim=True)
    return sample
