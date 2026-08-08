"""KGDiff's SE(3)-equivariant attention transformer (``uni_o2``).

Ported from KGDiff ``models/uni_transformer.py`` (commit ``ad893fc``), which
is itself TargetDiff's backbone. One block of ``num_layers`` alternating
x2h / h2x attention updates over a kNN graph rebuilt once per block.

The graph is **geometric, not chemical**: :meth:`_connect_edge` is a plain
``knn_graph(x, k=32)`` over the joint pocket+ligand point cloud, and the
``edge_feat_dim: 4`` channel is a one-hot of the four ligand/protein
incidence combinations (ll / lp / pl / pp), not a bond order. No bond
information reaches this network.

Only ``cutoff_mode='knn'`` is ported; the released config and the released
checkpoint both use it, and the ``hybrid`` branch pulled in a
``batch_hybrid_edge_connection`` helper nothing else needs.

Module and parameter names are kept byte-for-byte compatible with the
released checkpoint (``refine_net.base_block.<i>.x2h_layers.0.hk_func.net.*``
and friends), so conversion is a pure prefix add.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch_geometric.nn import knn_graph
from torch_scatter import scatter_softmax, scatter_sum

from MolecularDiffusion.modules.models.kgdiff.common import (
    MLP,
    GaussianSmearing,
    outer_product,
)


class BaseX2HAttLayer(nn.Module):
    """Multi-head attention that updates scalar node features from geometry."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_heads: int,
        edge_feat_dim: int,
        r_feat_dim: int,
        act_fn: str = "relu",
        norm: bool = True,
        ew_net_type: str = "r",
        out_fc: bool = True,
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
        n_nodes = h.size(0)
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
            (q[dst] * k / np.sqrt(k.shape[-1])).sum(-1),
            dst,
            dim=0,
            dim_size=n_nodes,
        )

        m = alpha.unsqueeze(-1) * v
        output = scatter_sum(m, dst, dim=0, dim_size=n_nodes)
        output = output.view(-1, self.output_dim)
        if self.out_fc:
            output = self.node_output(torch.cat([output, h], -1))

        return output + h


class BaseH2XAttLayer(nn.Module):
    """Multi-head attention that emits an equivariant coordinate update."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_heads: int,
        edge_feat_dim: int,
        r_feat_dim: int,
        act_fn: str = "relu",
        norm: bool = True,
        ew_net_type: str = "r",
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
        n_nodes = h.size(0)
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

        # (xi - xj) weighted per head -> [n_edges, n_heads, 3]
        v = v.unsqueeze(-1) * rel_x.unsqueeze(1)
        q = self.xq_func(h).view(
            -1, self.n_heads, self.output_dim // self.n_heads
        )

        alpha = scatter_softmax(
            (q[dst] * k / np.sqrt(k.shape[-1])).sum(-1),
            dst,
            dim=0,
            dim_size=n_nodes,
        )

        m = alpha.unsqueeze(-1) * v
        output = scatter_sum(m, dst, dim=0, dim_size=n_nodes)
        return output.mean(1)


class AttentionLayerO2TwoUpdateNodeGeneral(nn.Module):
    """One layer: ``num_x2h`` feature updates then ``num_h2x`` coord updates."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        num_r_gaussian: int,
        edge_feat_dim: int,
        act_fn: str = "relu",
        norm: bool = True,
        num_x2h: int = 1,
        num_h2x: int = 1,
        r_min: float = 0.0,
        r_max: float = 10.0,
        num_node_types: int = 8,
        ew_net_type: str = "r",
        x2h_out_fc: bool = True,
        sync_twoup: bool = False,
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

        self.x2h_layers = nn.ModuleList()
        for _ in range(self.num_x2h):
            self.x2h_layers.append(
                BaseX2HAttLayer(
                    hidden_dim,
                    hidden_dim,
                    hidden_dim,
                    n_heads,
                    edge_feat_dim,
                    r_feat_dim=num_r_gaussian * 4,
                    act_fn=act_fn,
                    norm=norm,
                    ew_net_type=self.ew_net_type,
                    out_fc=self.x2h_out_fc,
                )
            )
        self.h2x_layers = nn.ModuleList()
        for _ in range(self.num_h2x):
            self.h2x_layers.append(
                BaseH2XAttLayer(
                    hidden_dim,
                    hidden_dim,
                    hidden_dim,
                    n_heads,
                    edge_feat_dim,
                    r_feat_dim=num_r_gaussian * 4,
                    act_fn=act_fn,
                    norm=norm,
                    ew_net_type=self.ew_net_type,
                )
            )

    def forward(
        self, h, x, edge_attr, edge_index, mask_ligand, e_w=None, fix_x=False
    ):
        src, dst = edge_index
        edge_feat = edge_attr if self.edge_feat_dim > 0 else None

        rel_x = x[dst] - x[src]
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        h_in = h
        for i in range(self.num_x2h):
            dist_feat = self.distance_expansion(dist)
            # 4 separate distance embeddings, one per p-p / p-l / l-p / l-l
            dist_feat = outer_product(edge_attr, dist_feat)
            h_in = self.x2h_layers[i](
                h_in, dist_feat, edge_feat, edge_index, e_w=e_w
            )
        x2h_out = h_in

        new_h = h if self.sync_twoup else x2h_out
        for i in range(self.num_h2x):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            delta_x = self.h2x_layers[i](
                new_h, rel_x, dist_feat, edge_feat, edge_index, e_w=e_w
            )
            if not fix_x:
                # only ligand positions move; the pocket is fixed context
                x = x + delta_x * mask_ligand[:, None]
            rel_x = x[dst] - x[src]
            dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        return x2h_out, x


class UniTransformerO2TwoUpdateGeneral(nn.Module):
    """The ``uni_o2`` refine net: ``num_blocks`` x ``num_layers`` attention.

    ``init_h_emb_layer`` is built (and therefore present in the released
    checkpoint) but **never called** -- :meth:`forward` only iterates
    ``self.base_block``. It is kept so the checkpoint's key set matches
    exactly rather than needing a justified drop.
    """

    def __init__(
        self,
        num_blocks: int,
        num_layers: int,
        hidden_dim: int,
        n_heads: int = 1,
        k: int = 32,
        num_r_gaussian: int = 50,
        edge_feat_dim: int = 0,
        num_node_types: int = 8,
        act_fn: str = "relu",
        norm: bool = True,
        cutoff_mode: str = "knn",
        ew_net_type: str = "r",
        num_init_x2h: int = 1,
        num_init_h2x: int = 0,
        num_x2h: int = 1,
        num_h2x: int = 1,
        r_max: float = 10.0,
        x2h_out_fc: bool = True,
        sync_twoup: bool = False,
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

    def _build_init_h_layer(self) -> AttentionLayerO2TwoUpdateNodeGeneral:
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

    def _build_share_blocks(self) -> nn.ModuleList:
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

    def _connect_edge(self, x, mask_ligand, batch):  # noqa: ARG002
        if self.cutoff_mode != "knn":
            raise ValueError(
                f"Only cutoff_mode='knn' is ported for KGDiff, got "
                f"{self.cutoff_mode!r}. The released config and checkpoint "
                "both use knn."
            )
        return knn_graph(x, k=self.k, batch=batch, flow="source_to_target")

    @staticmethod
    def _build_edge_type(edge_index, mask_ligand) -> torch.Tensor:
        """One-hot of ll / lp / pl / pp incidence -- NOT a bond order."""
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1
        n_dst = mask_ligand[dst] == 1
        edge_type[n_src & n_dst] = 0
        edge_type[n_src & ~n_dst] = 1
        edge_type[~n_src & n_dst] = 2
        edge_type[~n_src & ~n_dst] = 3
        return F.one_hot(edge_type, num_classes=4)

    def forward(self, h, x, mask_ligand, batch, return_all=False, fix_x=False):
        all_x = [x]
        all_h = [h]

        for _ in range(self.num_blocks):
            edge_index = self._connect_edge(x, mask_ligand, batch)
            src, dst = edge_index

            edge_type = self._build_edge_type(edge_index, mask_ligand)
            if self.ew_net_type == "global":
                dist = torch.norm(
                    x[dst] - x[src], p=2, dim=-1, keepdim=True
                )
                dist_feat = self.distance_expansion(dist)
                e_w = torch.sigmoid(self.edge_pred_layer(dist_feat))
            else:
                e_w = None

            for layer in self.base_block:
                h, x = layer(
                    h, x, edge_type, edge_index, mask_ligand, e_w=e_w,
                    fix_x=fix_x,
                )
            all_x.append(x)
            all_h.append(h)

        outputs = {"x": x, "h": h}
        if return_all:
            outputs.update({"all_x": all_x, "all_h": all_h})
        return outputs
