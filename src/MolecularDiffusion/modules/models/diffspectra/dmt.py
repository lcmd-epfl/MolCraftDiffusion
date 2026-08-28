"""DMT: DiffSpectra's Diffusion Molecule Transformer.

Ported from ``others/DiffSpectra/models/dmt.py``. DMT is JODO's own
Graph-DiT-style backbone (``configs/base_qm9.py`` and
``configs/diffspectra_qm9s.py`` both open with the literal leftover module
docstring 'Training Conditional JODO with single property on QM9', and
``model.edge_ch = 2`` is JODO's exact QM9 value) with two changes: a
:class:`~.specformer.SpecFormer` conditioning branch in place of JODO's
scalar-property MLP, and self-conditioning always on.

Because of that shared ancestry, ``EquivariantMixBlock``/``MultiCondEquiUpdate``
(the per-layer equivariant message-passing block) and every GBF/attention
primitive they use are **byte-identical** to what
``modules/models/jodo/mol_gnn.py`` and ``modules/models/jodo/layers.py``
already ported for JODO -- verified line-for-line against
``others/DiffSpectra/models/layers.py`` and ``others/DiffSpectra/models/dmt.py``
before writing this file. They are imported here rather than re-ported, per
the integration plan's "cite, don't re-derive" instruction (the plan cites
this exact fact in its Repo Inspection section). Only the parts that
DIFFER from ``Cond_DGT_concat`` are new: the top-level ``node_emb`` /
``edge_emb`` / ``dist_layer`` / ``e_block_i`` stack construction (kept here,
not reused, because it has to end up with UPSTREAM'S OWN ATTRIBUTE NAMES --
``node_emb``, ``edge_emb``, ``e_block_%d``, ``node_pred_mlp``,
``edge_type_mlp``, ``edge_exist_mlp``, ``time_mlp``, ``cond_encoder``,
``cond_lin`` -- so that ``scripts/convert_checkpoint.py`` can load upstream's
released weights with a plain ``module.`` prefix strip and NO key remap;
see the Hyperparameter Provenance table's warning about a wrong remap
loading nothing under ``strict=False``), and ``cond_encoder``/``forward``'s
context branch, which is genuinely new (SpecFormer, not a scalar MLP).

Same reasoning, same reuse: :class:`~MolecularDiffusion.modules.models.jodo.NoiseScheduleVP`
is used as-is by the task file rather than re-ported -- its continuous
``cosine``/``linear`` branches (the only ones either model's shipped configs
use) are byte-identical to ``others/DiffSpectra/diffusion/noise_schedule.py``'s
(same ``beta_0=0.1``, ``beta_1=20.``, ``cosine_s=0.008``, ``T=0.9946``).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch_geometric.utils import dense_to_sparse

from MolecularDiffusion.modules.models.jodo.layers import (
    CondGaussianLayer,
    GaussianLayer,
    LearnedSinusodialposEmb,
)
from MolecularDiffusion.modules.models.jodo.mol_gnn import EquivariantMixBlock
from MolecularDiffusion.modules.models.jodo.utils import (
    coord2diff_adj,
    remove_mean_with_mask,
    to_dense_edge_attr,
)

from .specformer import SpecFormer

_GBF = {"GaussianLayer": GaussianLayer, "CondGaussianLayer": CondGaussianLayer}


class DMT(nn.Module):
    """SE(3)-equivariant dense graph transformer with SpecFormer conditioning.

    ``forward`` keeps upstream's raw positional/kwarg contract (``t, xh,
    node_mask, edge_mask, context=None, *, edge_x, cond_x, cond_edge_x,
    noise_level``) unchanged, because that is exactly the call shape
    ``modules/tasks/diffusion_diffspectra.py`` mechanically ports from
    ``losses.py``/``sampling.py``.
    """

    def __init__(  # noqa: PLR0913
        self,
        atom_types: int,
        include_fc_charge: bool,
        nf: int = 256,
        n_layers: int = 8,
        n_heads: int = 16,
        n_extra_heads: int = 2,
        dropout: float = 0.1,
        mlp_ratio: int = 2,
        spatial_cut_off: float = 2.0,
        edge_ch: int = 2,
        cond_time: bool = True,
        dist_gbf: bool = True,
        gbf_name: str = "CondGaussianLayer",
        trans_name: str = "TransMixLayer",
        softmax_inf: bool = True,
        edge_quan_th: float = 0.0,
        com: bool = True,
        pred_data: bool = True,
        patch_len: list | None = None,
        stride: list | None = None,
        spectra_version: str = "allspectra",
        specformer_kwargs: dict | None = None,
    ) -> None:
        super().__init__()

        in_node_dim = atom_types + int(include_fc_charge)
        hidden_dim = nf
        edge_hidden_dim = nf // 4
        self.dist_gbf = dist_gbf
        self.edge_th = edge_quan_th
        self.CoM = com
        self.spatial_cut_off = spatial_cut_off
        dist_dim = edge_hidden_dim if dist_gbf else 1
        in_edge_dim = edge_ch * 2 + dist_dim
        self.cond_time = cond_time
        self.n_layers = n_layers
        self.pred_data = pred_data
        time_dim = hidden_dim * 4
        self.dist_dim = dist_dim

        self.node_emb = nn.Linear(in_node_dim * 2, hidden_dim)
        self.edge_emb = nn.Linear(in_edge_dim, edge_hidden_dim)

        if self.dist_gbf:
            self.dist_layer = _GBF[gbf_name](dist_dim, time_dim)

        cat_node_dim = (hidden_dim * 2) // n_layers
        cat_edge_dim = (edge_hidden_dim * 2) // n_layers

        for i in range(n_layers):
            self.add_module(
                "e_block_%d" % i,
                EquivariantMixBlock(
                    hidden_dim,
                    edge_hidden_dim,
                    time_dim,
                    n_extra_heads,
                    n_heads,
                    cond_time,
                    dist_gbf,
                    softmax_inf,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    gbf_name=gbf_name,
                    trans_name=trans_name,
                ),
            )
            self.add_module("node_%d" % i, nn.Linear(hidden_dim, cat_node_dim))
            self.add_module(
                "edge_%d" % i, nn.Linear(edge_hidden_dim, cat_edge_dim)
            )

        self.node_pred_mlp = nn.Sequential(
            nn.Linear(cat_node_dim * n_layers + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, in_node_dim),
        )
        self.edge_type_mlp = nn.Sequential(
            nn.Linear(
                cat_edge_dim * n_layers + edge_hidden_dim, edge_hidden_dim
            ),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim // 2, edge_ch - 1),
        )
        self.edge_exist_mlp = nn.Sequential(
            nn.Linear(
                cat_edge_dim * n_layers + edge_hidden_dim, edge_hidden_dim
            ),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim // 2, 1),
        )

        if cond_time:
            learned_dim = 16
            self.time_mlp = nn.Sequential(
                LearnedSinusodialposEmb(learned_dim),
                nn.Linear(learned_dim + 1, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )

        # The genuinely new part: a spectral encoder instead of JODO's
        # scalar-property MLP, added straight into the timestep embedding
        # (see forward()). specformer_kwargs left {} rides SpecFormer's own
        # defaults (d_model=128, n_layers=3, n_heads=16, d_ff=256) --
        # NOT overridden in either of upstream's shipped configs (see the
        # integration plan's Hyperparameter Provenance table).
        self.spectra_version = spectra_version
        self.cond_encoder = SpecFormer(
            patch_len=patch_len or [20, 50, 50],
            stride=stride or [10, 25, 25],
            output_dim=hidden_dim,
            spectra_version=spectra_version,
            **(specformer_kwargs or {}),
        )
        self.cond_lin = nn.Linear(hidden_dim, time_dim)

    def forward(
        self,
        t,
        xh,
        node_mask,
        edge_mask,
        context=None,
        *args: Any,
        **kwargs: Any,
    ):
        """Mechanical port of ``others/DiffSpectra/models/dmt.py:306-412``.

        Args:
            t: ``(B,)`` diffusion time in ``[0, 1]`` (unused directly --
                ``noise_level`` in kwargs carries the actual conditioning
                signal, exactly as upstream).
            xh: ``(B, N, 3 + in_node_dim)`` positions concatenated with atom
                features (types [+ formal charge]).
            node_mask: ``(B, N, 1)`` float/bool.
            edge_mask: ``(B*N*N, 1)`` float -- flattened dense adjacency.
            context: raw spectra -- one ``(B, L)`` tensor, or a 3-list
                ``[uv, ir, raman]`` for ``allspectra``. ``None`` is not a
                supported call (DiffSpectra has no unconditional branch,
                see the integration plan) but is not asserted against here,
                matching upstream's own lack of a guard.
            kwargs: ``edge_x`` ``(B, N, N, edge_ch)``, ``cond_x``/``cond_edge_x``
                (self-conditioning feedback, ``None`` on the first pass),
                ``noise_level`` ``(B,)``.
        """
        edge_x, cond_x, cond_edge_x = (
            kwargs["edge_x"],
            kwargs["cond_x"],
            kwargs["cond_edge_x"],
        )

        bs, n_nodes, _dims = xh.shape
        pos = xh[:, :, 0:3].clone().reshape(bs * n_nodes, -1)
        h = xh[:, :, 3:].clone().reshape(bs * n_nodes, -1)

        adj_mask = edge_mask.reshape(bs, n_nodes, n_nodes)
        dense_index = adj_mask.nonzero(as_tuple=True)
        edge_index, _ = dense_to_sparse(adj_mask)

        if cond_x is None:
            cond_x = torch.zeros_like(xh)
            cond_edge_x = torch.zeros_like(edge_x)
            cond_adj_2d = torch.ones(
                (edge_index.size(1), 1), device=edge_x.device
            )
        else:
            with torch.no_grad():
                cond_adj_2d = cond_edge_x[dense_index][:, 0:1].clone()
                cond_adj_2d[cond_adj_2d >= self.edge_th] = 1.0
                cond_adj_2d[cond_adj_2d < self.edge_th] = 0.0

        cond_pos = cond_x[:, :, 0:3].clone().reshape(bs * n_nodes, -1)
        cond_h = cond_x[:, :, 3:].clone().reshape(bs * n_nodes, -1)
        h = torch.cat([h, cond_h], dim=-1)

        if context is not None:
            context = self.cond_encoder(context)
            context = self.cond_lin(context)

        if self.cond_time:
            noise_level = kwargs["noise_level"]
            time_emb = self.time_mlp(noise_level) + context
            node_time_emb = (
                time_emb.unsqueeze(1)
                .expand(-1, n_nodes, -1)
                .reshape(bs * n_nodes, -1)
            )
            edge_batch_id = torch.div(
                edge_index[0], n_nodes, rounding_mode="floor"
            )
            edge_time_emb = time_emb[edge_batch_id]
        else:
            node_time_emb = None
            edge_time_emb = None

        distances, cond_adj_spatial = coord2diff_adj(
            cond_pos, edge_index, self.spatial_cut_off
        )
        if distances.sum() == 0:
            distances = distances.repeat(1, self.dist_dim)
        elif self.dist_gbf:
            distances = self.dist_layer(distances, edge_time_emb)
        cur_edge_attr = edge_x[dense_index]
        cond_edge_attr = cond_edge_x[dense_index]

        extra_adj = torch.cat([cond_adj_2d, cond_adj_spatial], dim=-1)
        edge_attr = torch.cat(
            [cur_edge_attr, cond_edge_attr, distances], dim=-1
        )

        h = self.node_emb(h)
        edge_attr = self.edge_emb(edge_attr)

        atom_hids = [h]
        edge_hids = [edge_attr]
        for i in range(self.n_layers):
            h, edge_attr, pos = self._modules["e_block_%d" % i](
                pos,
                h,
                edge_attr,
                edge_index,
                node_mask.reshape(-1, 1),
                extra_adj,
                node_time_emb,
                edge_time_emb,
            )
            if self.CoM:
                pos = remove_mean_with_mask(
                    pos.reshape(bs, n_nodes, -1), node_mask
                ).reshape(bs * n_nodes, -1)
            atom_hids.append(self._modules["node_%d" % i](h))
            edge_hids.append(self._modules["edge_%d" % i](edge_attr))

        atom_hids = torch.cat(atom_hids, dim=-1)
        edge_hids = torch.cat(edge_hids, dim=-1)
        atom_pred = (
            self.node_pred_mlp(atom_hids).reshape(bs, n_nodes, -1) * node_mask
        )
        edge_pred = torch.cat(
            [self.edge_exist_mlp(edge_hids), self.edge_type_mlp(edge_hids)],
            dim=-1,
        )

        edge_final = torch.zeros_like(edge_x).reshape(
            bs * n_nodes * n_nodes, -1
        )
        edge_final = to_dense_edge_attr(
            edge_index, edge_pred, edge_final, bs, n_nodes
        )
        edge_final = 0.5 * (edge_final + edge_final.permute(0, 2, 1, 3))

        if self.pred_data:
            pos = pos * node_mask.reshape(-1, 1)
        else:
            pos_init = xh[:, :, 0:3].clone().reshape(bs * n_nodes, -1)
            pos = (pos - pos_init) * node_mask.reshape(-1, 1)

        if torch.any(torch.isnan(pos)):
            pos = torch.zeros_like(pos)

        pos = pos.reshape(bs, n_nodes, -1)
        pos = remove_mean_with_mask(pos, node_mask)

        return torch.cat([pos, atom_pred], dim=2), edge_final
