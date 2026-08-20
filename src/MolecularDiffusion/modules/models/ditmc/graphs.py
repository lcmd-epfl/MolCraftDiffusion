"""The three graphs DiTMC consumes, as plain dataclasses.

Upstream splits every sample into three ``jraph.GraphsTuple``s
(``data_loader/utils.py:504-531``). Nothing about jraph is needed here beyond
the field layout, so these are dataclasses of flat tensors instead.

Two conventions inherited from upstream that silently produce wrong models if
reversed:

* **``senders = j``, ``receivers = i``.** ``create_graph_latent`` does
  ``receivers=centers, senders=others`` where ``centers`` is the row index.
  ``DiTEdgeEmbed`` and ``RadialSphericalEdgeEmbedding`` then compute
  ``displacements = positions[senders] - positions[receivers]``, i.e.
  :math:`\\vec r_j - \\vec r_i`. Reversing this flips the sign of every odd-``l``
  spherical harmonic.
* **The all-pairs edge list is in C-order over ``(i, j)``**, which is the only
  reason ``shortest_hops`` (extracted as ``M[~eye]``) lines up with it
  element-for-element.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass
class LatentGraph:
    """The fully connected graph the DiT attention runs over."""

    atomic_numbers: torch.Tensor  # (N,) long
    node_attr: torch.Tensor  # (N, A) float
    positions: torch.Tensor  # (N, 3) -- x_tau, the current latent state
    senders: torch.Tensor  # (E,) long, = j
    receivers: torch.Tensor  # (E,) long, = i
    shortest_hops: torch.Tensor  # (E,) long, 510 for unreachable pairs
    batch_segments: torch.Tensor  # (N,) long
    num_graphs: int
    cond_scaling_nodes: torch.Tensor  # (N,) -- classifier-free guidance switch
    cond_scaling_edges: torch.Tensor  # (E,)
    self_cond: torch.Tensor | None = None  # (N, 3)
    x1: torch.Tensor | None = None  # (N, 3) ground truth, training only

    @property
    def num_nodes(self) -> int:
        return int(self.atomic_numbers.shape[0])

    def replace(self, **kwargs) -> "LatentGraph":
        return replace(self, **kwargs)


@dataclass
class CondGraph:
    """The covalent-bond graph the MeshGraphNet conditioner runs over."""

    node_attr: torch.Tensor  # (N, A)
    senders: torch.Tensor  # (Ec,) long -- both directions, symmetric
    receivers: torch.Tensor  # (Ec,) long
    edge_attr: torch.Tensor  # (Ec, 4) one-hot SINGLE/DOUBLE/TRIPLE/AROMATIC


@dataclass
class PriorGraph:
    """Bond-graph Laplacian eigendecomposition, for the harmonic prior.

    Stored in upstream's sparse form: ``node_attr`` is :math:`1/\\sqrt{\\lambda}`
    per eigen-index (with the zero modes set to 0, which is what removes the
    centre of mass), and ``edge_attr`` is the flattened eigenvector matrix over
    a complete per-molecule index grid, so a segment sum evaluates
    :math:`P\\,\\mathrm{diag}(1/\\sqrt\\lambda)\\,z`.
    """

    node_attr: torch.Tensor  # (N,)
    senders: torch.Tensor  # (Ep,) long, = eigen index
    receivers: torch.Tensor  # (Ep,) long, = atom index
    edge_attr: torch.Tensor  # (Ep,)
