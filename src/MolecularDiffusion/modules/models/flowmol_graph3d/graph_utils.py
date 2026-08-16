"""Graph helpers for the bond-carrying FlowMol3 port.

Ported from FlowMol (``flowmol/data_processing/utils.py``). ``build_edge_idxs``
and ``get_node_batch_idxs`` are **imported** from the existing coordinate-only
port rather than duplicated -- ``build_edge_idxs`` in particular is the single
source of the ``[upper-triangle | mirrored-lower-triangle]`` edge ordering that
:func:`get_upper_edge_mask` below *infers* rather than stores. Re-deriving it
elsewhere with a different order would silently produce wrong masks (no error).
"""

import dgl
import torch

from MolecularDiffusion.modules.models.flowmol.graph_utils import (
    build_edge_idxs,
    get_node_batch_idxs,
)

__all__ = [
    "build_edge_idxs",
    "get_batch_idxs",
    "get_edge_batch_idxs",
    "get_node_batch_idxs",
    "get_upper_edge_mask",
]


def get_upper_edge_mask(g: dgl.DGLGraph) -> torch.Tensor:
    """Boolean mask selecting the upper-triangle edges of every batched graph.

    Derived purely from the edge *ordering* laid down by ``build_edge_idxs``
    (upper triangle first, then the mirrored lower triangle, per graph, then
    concatenated by ``dgl.batch``). There is no stored flag to fall back on, so
    graphs must always be built with ``build_edge_idxs``.
    """
    edges_per_mol = g.batch_num_edges()
    ul_pattern = torch.tensor([1, 0], device=g.device).repeat(g.batch_size)
    n_edges_pattern = (edges_per_mol / 2).int().repeat_interleave(2)
    return ul_pattern.repeat_interleave(n_edges_pattern).bool()


def get_edge_batch_idxs(g: dgl.DGLGraph) -> torch.Tensor:
    """Tensor mapping each edge to the molecule (graph) it belongs to."""
    edge_batch_idx = torch.arange(g.batch_size, device=g.device)
    return edge_batch_idx.repeat_interleave(g.batch_num_edges())


def get_batch_idxs(g: dgl.DGLGraph):
    """``(node_batch_idx, edge_batch_idx)`` for a batched graph."""
    return get_node_batch_idxs(g), get_edge_batch_idxs(g)
