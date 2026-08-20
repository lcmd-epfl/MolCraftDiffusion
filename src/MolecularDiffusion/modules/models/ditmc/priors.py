"""Priors for the flow-matching interpolant. Port of ``generative_process/priors.py``.

The harmonic prior is the default and the interesting one: it draws
:math:`x_0 = P\\,\\mathrm{diag}(1/\\sqrt\\lambda)\\,z` where :math:`P, \\lambda` are the
eigenvectors/values of the **bond-graph Laplacian**, so the prior sample is
already a plausible-looking chain rather than a Gaussian blob.

Upstream evaluates this as a segment sum over a complete per-molecule index grid
(``prior_senders = j``, ``prior_receivers = i``, ``edge_attr = P.flatten()``):

    sample[i] = sum_j P[i][j] * D[j] * z[j]

which is exactly the matrix product above. The sparse form is kept here so the
batching is identical to upstream's, including the ``nan_to_num`` that zeroes
the :math:`\\lambda = 0` modes -- **that** is what removes the centre of mass.
"""

from __future__ import annotations

import torch

from .graphs import PriorGraph


class GaussianPrior:
    """``mu + sigma * N(0, 1)``. Ignores the graph entirely."""

    name = "GaussianPrior"

    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        self.mu = mu
        self.sigma = sigma

    def sample(
        self,
        shape,
        graph_prior: PriorGraph | None = None,  # noqa: ARG002
        *,
        device=None,
        dtype=torch.float32,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return self.mu + self.sigma * noise


class HarmonicPrior:
    """Gaussian shaped by the bond-graph Laplacian's pseudo-inverse."""

    name = "HarmonicPrior"

    def sample(
        self,
        shape,
        graph_prior: PriorGraph,
        *,
        device=None,
        dtype=torch.float32,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if graph_prior is None:
            msg = "HarmonicPrior.sample requires a PriorGraph"
            raise ValueError(msg)
        num_nodes = shape[0]
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        scaled = graph_prior.node_attr.to(dtype).unsqueeze(-1) * noise
        scaled = torch.nan_to_num(scaled)
        messages = graph_prior.edge_attr.to(dtype).unsqueeze(-1) * scaled.index_select(
            0, graph_prior.senders
        )
        out = messages.new_zeros((num_nodes, *shape[1:]))
        return out.index_add(0, graph_prior.receivers, messages)


PRIORS = {"harmonic": HarmonicPrior, "gaussian": GaussianPrior}


def build_prior(name: str, **kwargs):
    if name not in PRIORS:
        msg = f"prior must be one of {sorted(PRIORS)}, received {name!r}"
        raise ValueError(msg)
    return PRIORS[name](**kwargs)


def _self_check() -> None:  # pragma: no cover
    """Sample covariance of the harmonic prior ~ pinv of the Laplacian."""
    import numpy as np

    from .graph_features import laplacian_eigen

    adj = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=np.int64
    )
    d, p = laplacian_eigen(adj, num_components=1)
    n = 4
    i, j = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    graph = PriorGraph(
        node_attr=torch.as_tensor(d),
        senders=j.reshape(-1),
        receivers=i.reshape(-1),
        edge_attr=torch.as_tensor(p).reshape(-1),
    )
    g = torch.Generator().manual_seed(0)
    samples = torch.stack(
        [HarmonicPrior().sample((n, 3), graph, generator=g) for _ in range(40000)]
    )
    cov = torch.einsum("sid,sjd->ij", samples, samples) / (samples.shape[0] * 3)
    lap = np.diag(adj.sum(1)) - adj
    expect = torch.as_tensor(np.linalg.pinv(lap.astype(np.float64)), dtype=torch.float32)
    assert torch.allclose(cov, expect, atol=0.05), (cov - expect).abs().max()
    # The lambda=0 mode is zeroed, which removes the centre of mass.
    assert samples.mean(dim=1).abs().max() < 1e-4, samples.mean(dim=1).abs().max()
    print("ditmc.priors self-check OK")


if __name__ == "__main__":
    _self_check()
