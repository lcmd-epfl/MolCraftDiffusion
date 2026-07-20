"""Small graph-construction helpers for the FlowMol port.

Ported from FlowMol (``flowmol/data_processing/utils.py``). Only the
fully-connected edge builder and the node batch-index helper are needed —
the upper/lower-triangle edge masks are dropped since bonds (the ``e``
modality) are not generated in the coordinate-only port; edges are a pure
message-passing convenience here.
"""

import dgl
import torch


def build_edge_idxs(n_atoms: int) -> torch.Tensor:
    """Fully-connected (directed, no self-loops) edge index for n_atoms."""
    upper_edge_idxs = torch.triu_indices(n_atoms, n_atoms, offset=1)
    lower_edge_idxs = torch.stack(
        (upper_edge_idxs[1], upper_edge_idxs[0])
    )
    return torch.cat((upper_edge_idxs, lower_edge_idxs), dim=1)


def get_node_batch_idxs(g: dgl.DGLGraph) -> torch.Tensor:
    """Tensor mapping each node to the molecule (graph) it belongs to."""
    node_batch_idx = torch.arange(g.batch_size, device=g.device)
    return node_batch_idx.repeat_interleave(g.batch_num_nodes())
