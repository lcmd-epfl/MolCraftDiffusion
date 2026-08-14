"""Apo2Mol's ``uni_o2`` backbone: TargetDiff's SE(3) attention transformer,
plus the two things that make it Apo2Mol's.

Ported from ``others/Apo2Mol/models/uni_transformer.py``. It needs its own
module rather than an import of ``modules/models/kgdiff/uni_transformer.py``
because of two additions:

1. **A residue head.** ``forward`` takes ``protein_atom_to_aa_group`` and
   returns ``residue_h`` (:meth:`_aggregate_atom_to_residue`,
   ``uni_transformer.py:608-653``): pocket token features are concatenated
   with their updated coordinates and pooled per residue, giving the
   ``hidden_dim + 3`` vector that ``ScorePosNet3D.res_inference`` decodes into
   ``(3 translation, 4 quaternion, 5 chi)``. This is the pocket half of the
   generative process.
2. **A 5-wide edge type** instead of KGDiff's 4 (:meth:`_build_edge_type`).
   KGDiff distinguishes ligand-ligand / ligand-protein / protein-ligand /
   protein-protein; Apo2Mol splits the last into *same residue* and *across
   residues*. Still purely topological -- no bond or chemistry information
   reaches the network in either model.

Also note ``h2x`` updates **both** pocket and ligand coordinates
(``uni_transformer.py:193``, where KGDiff masks the update to ligand atoms
only). That is deliberate upstream and is what makes the pocket mobile.

Not ported (dead in the released configuration, verified by reading call
sites): ``GVPLayer`` -- ``self.prot_gvp_layer`` is constructed upstream but
its only call site, ``uni_transformer.py:549-551``, is commented out, and
``h_protein_update`` at :554 is built from the raw ``protein_h``/``protein_pos``
instead. Its 6 checkpoint tensors are dropped by
``scripts/convert_checkpoint.py`` with that justification. ``SAGPoolNet`` by
contrast is **live** (called at :647) and is ported below.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import (
    GCNConv,
    SAGPooling,
    global_mean_pool,
    knn_graph,
    radius_graph,
)
from torch_scatter import scatter_softmax, scatter_sum

from .common import (
    GaussianSmearing,
    MLP,
    batch_hybrid_edge_connection,
    compose_context,
    outer_product,
)


class BaseX2HAttLayer(nn.Module):
    """Scalar-feature attention update (``uni_transformer.py:12-75``)."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        n_heads,
        edge_feat_dim,
        r_feat_dim,
        act_fn="relu",
        norm=True,
        ew_net_type="r",
        out_fc=True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.act_fn = act_fn
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.ew_net_type = ew_net_type
        self.out_fc = out_fc

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim
        self.hk_func = MLP(
            kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn
        )
        self.hv_func = MLP(
            kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn
        )
        self.hq_func = MLP(
            input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn
        )
        if ew_net_type == "r":
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())
        elif ew_net_type == "m":
            self.ew_net = nn.Sequential(nn.Linear(output_dim, 1), nn.Sigmoid())

        if self.out_fc:
            self.node_output = MLP(
                2 * hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn
            )

    def forward(self, h, r_feat, edge_feat, edge_index, e_w=None):
        n = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        k = self.hk_func(kv_input).view(
            -1, self.n_heads, self.output_dim // self.n_heads
        )
        v = self.hv_func(kv_input)

        if self.ew_net_type == "r":
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == "m":
            e_w = self.ew_net(v[..., : self.hidden_dim])
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.0
        v = v * e_w
        v = v.view(-1, self.n_heads, self.output_dim // self.n_heads)

        q = self.hq_func(h).view(
            -1, self.n_heads, self.output_dim // self.n_heads
        )

        alpha = scatter_softmax(
            (q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0, dim_size=n
        )

        m = alpha.unsqueeze(-1) * v
        output = scatter_sum(m, dst, dim=0, dim_size=n)
        output = output.view(-1, self.output_dim)
        if self.out_fc:
            output = self.node_output(torch.cat([output, h], -1))

        return output + h


class BaseH2XAttLayer(nn.Module):
    """Equivariant coordinate update (``uni_transformer.py:78-127``)."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        n_heads,
        edge_feat_dim,
        r_feat_dim,
        act_fn="relu",
        norm=True,
        ew_net_type="r",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.act_fn = act_fn
        self.ew_net_type = ew_net_type

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim

        self.xk_func = MLP(
            kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn
        )
        self.xv_func = MLP(
            kv_input_dim, self.n_heads, hidden_dim, norm=norm, act_fn=act_fn
        )
        self.xq_func = MLP(
            input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn
        )
        if ew_net_type == "r":
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())

    def forward(self, h, rel_x, r_feat, edge_feat, edge_index, e_w=None):
        n = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        k = self.xk_func(kv_input).view(
            -1, self.n_heads, self.output_dim // self.n_heads
        )
        v = self.xv_func(kv_input)
        if self.ew_net_type == "r":
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == "m":
            e_w = 1.0
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.0
        v = v * e_w

        v = v.unsqueeze(-1) * rel_x.unsqueeze(1)  # (E, heads, 3)
        q = self.xq_func(h).view(
            -1, self.n_heads, self.output_dim // self.n_heads
        )

        alpha = scatter_softmax(
            (q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0, dim_size=n
        )

        m = alpha.unsqueeze(-1) * v
        return scatter_sum(m, dst, dim=0, dim_size=n).mean(1)


class AttentionLayerO2TwoUpdateNodeGeneral(nn.Module):
    """One block: ``num_x2h`` feature updates then ``num_h2x`` coord updates."""

    def __init__(
        self,
        hidden_dim,
        n_heads,
        num_r_gaussian,
        edge_feat_dim,
        act_fn="relu",
        norm=True,
        num_x2h=1,
        num_h2x=1,
        r_min=0.0,
        r_max=10.0,
        num_node_types=8,
        ew_net_type="r",
        x2h_out_fc=True,
        sync_twoup=False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.norm = norm
        self.act_fn = act_fn
        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.r_min, self.r_max = r_min, r_max
        self.num_node_types = num_node_types
        self.ew_net_type = ew_net_type
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup

        self.distance_expansion = GaussianSmearing(
            self.r_min, self.r_max, num_gaussians=num_r_gaussian
        )

        # r_feat_dim = num_r_gaussian * edge_feat_dim: the attention layers see
        # the outer product of the 5-way edge type with the 20-bin distance RBF.
        r_feat_dim = num_r_gaussian * edge_feat_dim

        self.x2h_layers = nn.ModuleList(
            [
                BaseX2HAttLayer(
                    hidden_dim,
                    hidden_dim,
                    hidden_dim,
                    n_heads,
                    edge_feat_dim,
                    r_feat_dim=r_feat_dim,
                    act_fn=act_fn,
                    norm=norm,
                    ew_net_type=self.ew_net_type,
                    out_fc=self.x2h_out_fc,
                )
                for _ in range(self.num_x2h)
            ]
        )
        self.h2x_layers = nn.ModuleList(
            [
                BaseH2XAttLayer(
                    hidden_dim,
                    hidden_dim,
                    hidden_dim,
                    n_heads,
                    edge_feat_dim,
                    r_feat_dim=r_feat_dim,
                    act_fn=act_fn,
                    norm=norm,
                    ew_net_type=self.ew_net_type,
                )
                for _ in range(self.num_h2x)
            ]
        )

    def forward(self, h, x, edge_attr, edge_index, mask_ligand, e_w=None, fix_x=False):  # noqa: ARG002
        src, dst = edge_index
        edge_feat = edge_attr if self.edge_feat_dim > 0 else None

        rel_x = x[dst] - x[src]
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        h_in = h
        for i in range(self.num_x2h):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            h_in = self.x2h_layers[i](h_in, dist_feat, edge_feat, edge_index, e_w=e_w)
        x2h_out = h_in

        new_h = h if self.sync_twoup else x2h_out
        for i in range(self.num_h2x):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            delta_x = self.h2x_layers[i](
                new_h, rel_x, dist_feat, edge_feat, edge_index, e_w=e_w
            )
            if not fix_x:
                # NB: no mask_ligand here -- Apo2Mol moves the POCKET too.
                # (uni_transformer.py:193; KGDiff masks to ligand atoms.)
                x = x + delta_x
            rel_x = x[dst] - x[src]
            dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        return x2h_out, x


class SAGPoolNet(nn.Module):
    """Per-residue pooling of pocket atom features (``uni_transformer.py:274``).

    LIVE, despite sitting next to a lot of dead code: it is what turns
    ``(n_pocket_atoms, hidden_dim + 3)`` into one vector per residue for the
    pocket head.

    Built from ``torch_geometric``'s own ``GCNConv`` / ``SAGPooling`` rather
    than a reimplementation, and the parameter names line up with the released
    checkpoint (``conv1.lin.weight``, ``pool1.gnn.lin_rel.*``,
    ``pool1.gnn.lin_root.weight``, ``lin.*``).

    **One version-skew fixup.** PyG >= 2.4 factors ``SAGPooling``'s top-k
    selection into a ``SelectTopK`` submodule carrying its own projection
    ``select.weight`` of shape ``(1, 1)``; the PyG that trained the released
    checkpoint had no such parameter and scored nodes with a plain
    ``tanh(gnn(x))``. With ``in_channels == 1`` the new code computes
    ``tanh(score * w / |w|) == tanh(score * sign(w))``, so **any positive
    ``w`` reproduces the old behaviour exactly**. It is therefore pinned to
    +1.0 here at construction, and ``scripts/convert_checkpoint.py`` supplies
    the same value as a documented sidecar.
    """

    def __init__(self, in_dim: int, hidden_dim: int, ratio: float = 0.5) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.pool1 = SAGPooling(hidden_dim, ratio=ratio)
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        with torch.no_grad():
            self.pool1.select.weight.fill_(1.0)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x = global_mean_pool(x, batch)
        return self.lin(x)


class UniTransformerO2TwoUpdateGeneral(nn.Module):
    """The joint pocket+ligand SE(3) transformer (``uni_transformer.py:289``)."""

    def __init__(
        self,
        num_blocks,
        num_layers,
        hidden_dim,
        n_heads=1,
        k=32,
        num_r_gaussian=50,
        edge_feat_dim=0,
        num_node_types=8,
        act_fn="relu",
        norm=True,
        cutoff_mode="radius",
        ew_net_type="r",
        num_init_x2h=1,
        num_init_h2x=0,
        num_x2h=1,
        num_h2x=1,
        r_max=10.0,
        x2h_out_fc=True,
        sync_twoup=False,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.act_fn = act_fn
        self.norm = norm
        self.num_node_types = num_node_types
        self.cutoff_mode = cutoff_mode
        self.k = k
        self.ew_net_type = ew_net_type

        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.num_init_x2h = num_init_x2h
        self.num_init_h2x = num_init_h2x
        self.r_max = r_max
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup
        self.distance_expansion = GaussianSmearing(
            0.0, r_max, num_gaussians=num_r_gaussian
        )
        if self.ew_net_type == "global":
            self.edge_pred_layer = MLP(num_r_gaussian, 1, hidden_dim)

        self.init_h_emb_layer = self._build_init_h_layer()
        self.base_block = self._build_share_blocks()
        self.prot_sag_layer = SAGPoolNet(
            in_dim=self.hidden_dim + 3,  # +3 for the updated atom position
            hidden_dim=self.hidden_dim + 3,
            ratio=0.5,
        )

    def __repr__(self) -> str:
        return (
            f"UniTransformerO2(num_blocks={self.num_blocks}, "
            f"num_layers={self.num_layers}, n_heads={self.n_heads}, "
            f"act_fn={self.act_fn}, norm={self.norm}, "
            f"cutoff_mode={self.cutoff_mode}, ew_net_type={self.ew_net_type})"
        )

    def _build_init_h_layer(self):
        return AttentionLayerO2TwoUpdateNodeGeneral(
            self.hidden_dim,
            self.n_heads,
            self.num_r_gaussian,
            self.edge_feat_dim,
            act_fn=self.act_fn,
            norm=self.norm,
            num_x2h=self.num_init_x2h,
            num_h2x=self.num_init_h2x,
            r_max=self.r_max,
            num_node_types=self.num_node_types,
            ew_net_type=self.ew_net_type,
            x2h_out_fc=self.x2h_out_fc,
            sync_twoup=self.sync_twoup,
        )

    def _build_share_blocks(self):
        return nn.ModuleList(
            [
                AttentionLayerO2TwoUpdateNodeGeneral(
                    self.hidden_dim,
                    self.n_heads,
                    self.num_r_gaussian,
                    self.edge_feat_dim,
                    act_fn=self.act_fn,
                    norm=self.norm,
                    num_x2h=self.num_x2h,
                    num_h2x=self.num_h2x,
                    r_max=self.r_max,
                    num_node_types=self.num_node_types,
                    ew_net_type=self.ew_net_type,
                    x2h_out_fc=self.x2h_out_fc,
                    sync_twoup=self.sync_twoup,
                )
                for _ in range(self.num_layers)
            ]
        )

    def _connect_edge(self, x, mask_ligand, batch):
        if self.cutoff_mode == "radius":
            return radius_graph(x, r=self.r_max, batch=batch, flow="source_to_target")
        if self.cutoff_mode == "knn":
            return knn_graph(x, k=self.k, batch=batch, flow="source_to_target")
        if self.cutoff_mode == "hybrid":
            return batch_hybrid_edge_connection(
                x, k=self.k, mask_ligand=mask_ligand, batch=batch, add_p_index=True
            )
        raise ValueError(f"Not supported cutoff mode: {self.cutoff_mode}")

    @staticmethod
    def _global_residue_index(atom_to_residue, batch, mask=None):
        """LOCAL (per-complex) residue ids -> globally unique ids.

        ``protein_atom_to_aa_group`` deliberately restarts at 0 in every
        complex (see ``data/component/apo2mol_data.py``); upstream re-derives
        the global ids in two places (``uni_transformer.py:623-639`` and
        ``:416-434``) and this is the shared implementation of both.

        Returns ``(global_ids, total_num_residues)``.
        """
        global_idx = atom_to_residue.clone()
        offset = 0
        for batch_idx in torch.unique(batch):
            sel = batch == batch_idx
            if mask is not None:
                sel = sel & mask
            if sel.sum() == 0:
                continue
            global_idx[sel] += offset
            offset += int(atom_to_residue[sel].max().item()) + 1
        return global_idx, offset

    @classmethod
    def _build_edge_type(cls, edge_index, mask_ligand, atom_to_residue, batch_all):
        """5-way one-hot node-incidence edge feature (NOT a bond order).

        ``0`` ligand-ligand, ``1`` ligand->protein, ``2`` protein->ligand,
        ``3`` protein-protein within one residue, ``4`` protein-protein
        across residues. The 5th class is Apo2Mol's addition over KGDiff's 4.
        """
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)

        mask_protein = ~mask_ligand
        full_atom_to_residue = torch.full_like(batch_all, -1)
        full_atom_to_residue[mask_protein] = atom_to_residue
        global_residue_idx, _ = cls._global_residue_index(
            full_atom_to_residue, batch_all, mask=mask_protein
        )

        src_lig = mask_ligand[src] == 1
        dst_lig = mask_ligand[dst] == 1
        both_prot = mask_protein[src] & mask_protein[dst]

        edge_type[src_lig & dst_lig] = 0
        edge_type[src_lig & ~dst_lig] = 1
        edge_type[~src_lig & dst_lig] = 2

        same_res = global_residue_idx[src] == global_residue_idx[dst]
        edge_type[both_prot & same_res] = 3
        edge_type[both_prot & ~same_res] = 4

        return F.one_hot(edge_type, num_classes=5)

    @staticmethod
    def _build_prot_edge_index(prot_pos, k=16):
        """kNN over pocket atoms, later filtered to intra-residue edges."""
        return knn_graph(prot_pos, k=k, loop=False)

    def forward(
        self,
        h_protein,
        h_ligand,
        protein_pos,
        ligand_pos,
        batch_protein,
        batch_ligand,
        protein_atom_to_aa_group,
        return_all=False,
        fix_x=False,
    ):
        h_all, pos_all, batch_all, mask_ligand, mask_protein, _ = compose_context(
            h_protein=h_protein,
            h_ligand=h_ligand,
            pos_protein=protein_pos,
            pos_ligand=ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        all_pos_list = [pos_all]
        all_h_list = [h_all]

        for _b_idx in range(self.num_blocks):
            edge_index = self._connect_edge(pos_all, mask_ligand, batch_all)
            src, dst = edge_index

            edge_type = self._build_edge_type(
                edge_index=edge_index,
                mask_ligand=mask_ligand,
                atom_to_residue=protein_atom_to_aa_group,
                batch_all=batch_all,
            )
            if self.ew_net_type == "global":
                dist = torch.norm(
                    pos_all[dst] - pos_all[src], p=2, dim=-1, keepdim=True
                )
                dist_feat = self.distance_expansion(dist)
                e_w = torch.sigmoid(self.edge_pred_layer(dist_feat))
            else:
                e_w = None

            for layer in self.base_block:
                h_all, pos_all = layer(
                    h_all, pos_all, edge_type, edge_index, mask_ligand,
                    e_w=e_w, fix_x=fix_x,
                )
            all_pos_list.append(pos_all)
            all_h_list.append(h_all)

        final_ligand_pos, final_ligand_h = pos_all[mask_ligand], h_all[mask_ligand]
        prot_pos_out, protein_h = pos_all[mask_protein], h_all[mask_protein]

        h_protein_update = torch.concat([protein_h, prot_pos_out], dim=-1)
        h_residue_update = self._aggregate_atom_to_residue(
            atom_features=h_protein_update,
            atom_to_residue=protein_atom_to_aa_group,
            batch_protein=batch_protein,
            atom_pos=prot_pos_out,
        )

        outputs = {
            "ligand_pos": final_ligand_pos,
            "ligand_h": final_ligand_h,
            "residue_h": h_residue_update,
        }
        if return_all:
            outputs.update({"all_pos": all_pos_list, "all_h": all_h_list})
        return outputs

    def _aggregate_atom_to_residue(
        self, atom_features, atom_to_residue, batch_protein, atom_pos, knn_k=16
    ):
        """Pocket atom features -> one vector per residue.

        Builds a kNN graph over pocket atoms, keeps only the intra-residue
        edges, and runs the self-attention pooling net with the global residue
        id as the pooling "batch". Returns ``(n_residues_total, hidden_dim+3)``
        in ascending global residue order -- which is the order the collate
        emits ``protein_translations`` / ``protein_rotations`` /
        ``protein_chi_*`` in, so the pocket losses line up row for row.
        """
        prot_edge_index = self._build_prot_edge_index(prot_pos=atom_pos, k=knn_k)
        global_residue_idx, _total = self._global_residue_index(
            atom_to_residue, batch_protein
        )

        src, dst = prot_edge_index
        same_residue = global_residue_idx[src] == global_residue_idx[dst]

        return self.prot_sag_layer(
            x=atom_features,
            edge_index=prot_edge_index[:, same_residue],
            batch=global_residue_idx,
        )
