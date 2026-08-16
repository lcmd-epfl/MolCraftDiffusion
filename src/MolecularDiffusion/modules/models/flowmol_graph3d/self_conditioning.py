"""Self-conditioning residual block for FlowMol3.

Ported from FlowMol (``flowmol/models/self_conditioning.py``). ``_rbf`` and
``_norm_no_nan`` are re-sourced to ``modules/layers/gvp``, which already exports
both. Upstream also imports ``rbf_twoscale`` here but never calls it (verified
by grep over the whole target repo: the only other occurrence is its
definition), so it is not ported.

Both MLP widths are checkpoint-verified against the released FlowMol3 weights:
``node_residual_mlp.0.weight`` is ``(256, 305)`` = ``256 + 11 + 6 + 32`` and
``edge_residual_mlp.0.weight`` is ``(128, 164)`` = ``128 + 4 + 32``.
"""

import dgl
import dgl.function as fn
import torch
from torch import nn

from MolecularDiffusion.modules.layers.gvp import _norm_no_nan, _rbf

__all__ = ["SelfConditioningResidualLayer"]


class SelfConditioningResidualLayer(nn.Module):
    """Fold a previously predicted endpoint back into the current features."""

    def __init__(  # noqa: PLR0913
        self,
        n_atom_types: int,
        n_charges: int,
        n_bond_types: int,
        node_embedding_dim: int,
        edge_embedding_dim: int,
        rbf_dim: int,
        rbf_dmax: float,
    ) -> None:
        super().__init__()

        self.rbf_dim = rbf_dim
        self.rbf_dmax = rbf_dmax

        self.node_residual_mlp = nn.Sequential(
            nn.Linear(
                node_embedding_dim + n_atom_types + n_charges + rbf_dim,
                node_embedding_dim,
            ),
            nn.SiLU(),
            nn.Linear(node_embedding_dim, node_embedding_dim),
            nn.SiLU(),
        )

        self.edge_residual_mlp = nn.Sequential(
            nn.Linear(
                edge_embedding_dim + n_bond_types + rbf_dim, edge_embedding_dim
            ),
            nn.SiLU(),
            nn.Linear(edge_embedding_dim, edge_embedding_dim),
            nn.SiLU(),
        )

    def forward(  # noqa: PLR0913
        self,
        g: dgl.DGLGraph,
        s_t: torch.Tensor,
        x_t: torch.Tensor,
        v_t: torch.Tensor,
        e_t: torch.Tensor,
        dst_dict: dict,
        node_batch_idx: torch.Tensor,  # noqa: ARG002 - upstream signature parity
        upper_edge_mask: torch.Tensor,
    ):
        # how far each node has to travel to reach its predicted endpoint
        d_node = _norm_no_nan(x_t - dst_dict["x"])
        d_node = _rbf(d_node, D_max=self.rbf_dmax, D_count=self.rbf_dim)

        node_residual = self.node_residual_mlp(
            torch.cat([s_t, dst_dict["a"], dst_dict["c"], d_node], dim=-1)
        )

        # change in every edge's length between now and the predicted endpoint
        d_edge_t = self.edge_distances(g, node_positions=x_t)
        d_edge_1 = self.edge_distances(g, node_positions=dst_dict["x"])
        d_edge_t = d_edge_t[upper_edge_mask]
        d_edge_1 = d_edge_1[upper_edge_mask]

        edge_residual = self.edge_residual_mlp(
            torch.cat(
                [e_t[upper_edge_mask], dst_dict["e"], d_edge_1 - d_edge_t],
                dim=-1,
            )
        )

        node_feats_out = s_t + node_residual
        positions_out = x_t
        vectors_out = v_t

        # mirror the upper-triangle update onto both edge directions
        edge_feats_out = torch.zeros_like(e_t)
        one_triangle_output = e_t[upper_edge_mask] + edge_residual
        edge_feats_out[upper_edge_mask] = one_triangle_output
        edge_feats_out[~upper_edge_mask] = one_triangle_output

        return node_feats_out, positions_out, vectors_out, edge_feats_out

    def edge_distances(
        self, g: dgl.DGLGraph, node_positions: torch.Tensor = None
    ) -> torch.Tensor:
        """RBF-embedded length of every edge in ``g``."""
        with g.local_scope():
            if node_positions is None:
                g.ndata["x_d"] = g.ndata["x_t"]
            else:
                g.ndata["x_d"] = node_positions

            g.apply_edges(fn.u_sub_v("x_d", "x_d", "x_diff"))
            dij = _norm_no_nan(g.edata["x_diff"], keepdims=True) + 1e-8
            d = _rbf(dij.squeeze(1), D_max=self.rbf_dmax, D_count=self.rbf_dim)

        return d
