"""Encoders used by the PMDM epsilon network.

Ported from PMDM's ``models/encoders/{attention,edge,egnn,egnn_pytorch,schnet}.py``,
keeping only what ``MDM_full_pocket_coor_shared`` reaches:

* :class:`CrossAttentionBlock` -- upstream ``BasicTransformerBlock``, the
  ligand<->pocket token mixer.
* :class:`SchNetProteinEncoder` -- upstream ``SchNetEncoder_protein``, used for
  both the pocket and the ligand tower.
* :class:`MLPEdgeEncoder` -- upstream ``MLPEdgeEncoder`` (``edge_encoder: mlp``).
* :class:`EGNNSparseNetwork` -- upstream ``EGNN_Sparse_Network``.

Two deliberate deviations from upstream:

* **No einops.** Upstream's attention uses ``rearrange`` for three reshapes;
  those are plain ``view``/``permute`` here, so PMDM needs no new dependency.
* **No ``MessagePassing``.** Upstream's ``EGNN_Sparse`` overrides
  ``propagate`` and reaches into PyG privates (``self.inspector.distribute``,
  ``_collect``) that moved in PyG 2.5. The message/aggregate pair is written
  directly with ``scatter_add``, which is what ``aggr="add"`` did anyway.
  Every graph fed to these layers is symmetric, so the source/target
  convention cannot flip a result.

Dropped as unreachable under this integration's scope: the fourier-feature
branch (``fourier_features=0``), global linear attention
(``global_linear_attn_every=0``), the node/edge embedding-table branch
(empty dims), ``linker_mask`` (linker sampling is out of scope), and
upstream's unused ``ligandemb``/``proteinemb``/inner ``atten_layer``
parameters in ``EGNN_Sparse_Network`` (the live code path is ``h = z``).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch_geometric
from torch import Tensor, nn
from torch_geometric.nn import radius_graph
from torch_scatter import scatter_add

from .common import GaussianSmearing, MultiLayerPerceptron, ShiftedSoftplus

SiLU = nn.SiLU


# --------------------------------------------------------------------------- #
# cross attention (encoders/attention.py)
# --------------------------------------------------------------------------- #
class _GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: Tensor) -> Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        inner = int(dim * mult)
        self.net = nn.Sequential(
            _GEGLU(dim, inner), nn.Dropout(dropout), nn.Linear(inner, dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _CrossAttention(nn.Module):
    """Multi-head attention over a flat ``(n, dim)`` token sequence.

    There is no batch dimension and no per-complex mask: upstream attends
    across every ligand and pocket token in the mini-batch at once. That is
    reproduced here rather than "fixed", since it is what the objective was
    defined against.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def _split(self, t: Tensor) -> Tensor:
        # (n, h*d) -> (h, n, d)
        return t.view(t.size(0), self.heads, self.dim_head).permute(1, 0, 2)

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        context = x if context is None else context
        q = self._split(self.to_q(x))
        k = self._split(self.to_k(context))
        v = self._split(self.to_v(context))

        sim = torch.einsum("hid,hjd->hij", q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum("hij,hjd->hid", attn, v)
        # (h, n, d) -> (n, h*d)
        out = out.permute(1, 0, 2).reshape(x.size(0), self.heads * self.dim_head)
        return self.to_out(out)


class CrossAttentionBlock(nn.Module):
    """Upstream ``BasicTransformerBlock``: self-attend both towers, then cross."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.attn1 = _CrossAttention(dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.attn_p = _CrossAttention(dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.attn2 = _CrossAttention(
            dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )
        self.attn2p = _CrossAttention(
            dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )
        self.ff = _FeedForward(dim, dropout=dropout)
        self.ffp = _FeedForward(dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm1p = nn.LayerNorm(dim)
        self.norm2p = nn.LayerNorm(dim)
        self.norm3p = nn.LayerNorm(dim)

    def forward(self, x: Tensor, context: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.attn1(self.norm1(x)) + x
        context = self.attn_p(self.norm1p(context)) + context
        x = self.attn2(self.norm2(x), context=context) + x
        context = self.attn2p(self.norm2p(context), context=x) + context
        x = self.ff(self.norm3(x)) + x
        context = self.ffp(self.norm3p(context)) + context
        return x, context


# --------------------------------------------------------------------------- #
# edge encoder (encoders/edge.py)
# --------------------------------------------------------------------------- #
class MLPEdgeEncoder(nn.Module):
    """Embed an edge length into ``hidden_dim``.

    Upstream also multiplies in a learned bond-type embedding, but both call
    sites in ``MDM_full_pocket_coor_shared.net`` pass ``edge_type=None`` (the
    joined ligand+pocket graphs are purely geometric), leaving that embedding
    permanently untrained. It is dropped here rather than shipped as dead
    weight that DDP would then flag as an unused parameter.
    """

    def __init__(self, hidden_dim: int = 100, activation: str = "relu") -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = MultiLayerPerceptron(
            1, [hidden_dim, hidden_dim], activation=activation
        )

    @property
    def out_channels(self) -> int:
        return self.hidden_dim

    def forward(self, edge_length: Tensor) -> Tensor:
        return self.mlp(edge_length)


# --------------------------------------------------------------------------- #
# SchNet tower (encoders/schnet.py)
# --------------------------------------------------------------------------- #
class _CFConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        edge_channels: int,
        cutoff: float = 10.0,
        smooth: bool = False,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, out_channels)
        self.nn = nn.Sequential(
            nn.Linear(edge_channels, num_filters),
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_filters),
        )
        self.cutoff = cutoff
        self.smooth = smooth

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_length: Tensor, edge_attr: Tensor
    ) -> Tensor:
        w = self.nn(edge_attr)
        if self.smooth:
            c = 0.5 * (torch.cos(edge_length * math.pi / self.cutoff) + 1.0)
            c = c * (edge_length <= self.cutoff) * (edge_length >= 0.0)
        else:
            c = (edge_length <= self.cutoff).float()
        w = w * c.view(-1, 1)

        x = self.lin1(x)
        # aggr="add" over source_to_target: message x_j * W scattered onto i
        x = scatter_add(
            x[edge_index[0]] * w, edge_index[1], dim=0, dim_size=x.size(0)
        )
        return self.lin2(x)


class _InteractionBlock(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_gaussians: int,
        num_filters: int,
        cutoff: float,
        smooth: bool = False,
    ) -> None:
        super().__init__()
        self.conv = _CFConv(
            hidden_channels, hidden_channels, num_filters, num_gaussians, cutoff, smooth
        )
        self.act = ShiftedSoftplus()
        self.lin = nn.Linear(hidden_channels, hidden_channels)

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_length: Tensor, edge_attr: Tensor
    ) -> Tensor:
        return self.lin(self.act(self.conv(x, edge_index, edge_length, edge_attr)))


class SchNetProteinEncoder(nn.Module):
    """Continuous-filter tower over a raw point cloud (its own radius graph)."""

    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        edge_channels: int = 64,
        cutoff: float = 10.0,
        input_dim: int = 31,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.distance_expansion = GaussianSmearing(
            stop=cutoff, num_gaussians=edge_channels
        )
        self.emblin = nn.Linear(input_dim, hidden_channels)
        self.interactions = nn.ModuleList(
            _InteractionBlock(
                hidden_channels, edge_channels, num_filters, cutoff, smooth=True
            )
            for _ in range(num_interactions)
        )

    @property
    def out_channels(self) -> int:
        return self.hidden_channels

    def forward(self, node_attr: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        edge_index = radius_graph(pos, self.cutoff, batch=batch, loop=False)
        edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        edge_attr = self.distance_expansion(edge_length)
        h = self.emblin(node_attr)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_length, edge_attr)
        return h


# --------------------------------------------------------------------------- #
# EGNN (encoders/egnn.py + encoders/egnn_pytorch.py)
# --------------------------------------------------------------------------- #
class _CoorsNorm(nn.Module):
    def __init__(self, eps: float = 1e-8, scale_init: float = 1.0) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(1).fill_(scale_init))

    def forward(self, coors: Tensor) -> Tensor:
        norm = coors.norm(dim=-1, keepdim=True)
        return coors / norm.clamp(min=self.eps) * self.scale


class EGNNSparseLayer(nn.Module):
    """One E(n)-equivariant message-passing layer.

    Only the *ligand* block of the coordinate update is applied: the pocket
    is a fixed conditioning cloud and must not move. Ligand nodes are the
    first ``n_ligand`` rows of every tensor (the caller concatenates
    ligand-then-pocket), which is exactly upstream's convention.
    """

    def __init__(
        self,
        feats_dim: int,
        pos_dim: int = 3,
        edge_attr_dim: int = 0,
        m_dim: int = 16,
        soft_edge: int = 0,
        norm_feats: bool = False,
        norm_coors: bool = False,
        norm_coors_scale_init: float = 1e-2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.m_dim = m_dim
        self.soft_edge = soft_edge

        edge_input_dim = edge_attr_dim + 1 + feats_dim * 2
        drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim * 2),
            drop,
            SiLU(),
            nn.Linear(edge_input_dim * 2, m_dim),
            SiLU(),
        )
        self.edge_weight = (
            nn.Sequential(nn.Linear(m_dim, 1), nn.Sigmoid()) if soft_edge else None
        )
        self.node_norm = (
            torch_geometric.nn.norm.LayerNorm(feats_dim) if norm_feats else None
        )
        self.coors_norm = (
            _CoorsNorm(scale_init=norm_coors_scale_init) if norm_coors else nn.Identity()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(feats_dim + m_dim, feats_dim * 2),
            drop,
            SiLU(),
            nn.Linear(feats_dim * 2, feats_dim),
        )
        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4), drop, SiLU(), nn.Linear(m_dim * 4, 1)
        )
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            # upstream: keeps deep stacks from diverging to NaN
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        batch: Tensor,
        n_ligand: int,
    ) -> Tensor:
        coors, feats = x[:, : self.pos_dim], x[:, self.pos_dim :]
        src, dst = edge_index[0], edge_index[1]

        rel_coors = coors[src] - coors[dst]
        rel_dist = (rel_coors**2).sum(dim=-1, keepdim=True)
        edge_feats = (
            rel_dist if edge_attr is None else torch.cat([edge_attr, rel_dist], dim=-1)
        )

        m_ij = self.edge_mlp(torch.cat([feats[dst], feats[src], edge_feats], dim=-1))
        if self.soft_edge:
            m_ij = m_ij * self.edge_weight(m_ij)

        n = feats.size(0)
        mhat_i = scatter_add(
            self.coors_mlp(m_ij) * self.coors_norm(rel_coors), dst, dim=0, dim_size=n
        )
        coors_out = torch.cat(
            [coors[:n_ligand] + mhat_i[:n_ligand], coors[n_ligand:]], dim=0
        )

        m_i = scatter_add(m_ij, dst, dim=0, dim_size=n)
        hidden = self.node_norm(feats, batch) if self.node_norm is not None else feats
        hidden_out = feats + self.node_mlp(torch.cat([hidden, m_i], dim=-1))

        return torch.cat([coors_out, hidden_out], dim=-1)


class EGNNSparseNetwork(nn.Module):
    """Stack of :class:`EGNNSparseLayer` over the joined ligand+pocket cloud.

    Returns the ligand slice only: ``(node_feats, coord_update)``, where the
    coordinate output is the *displacement* from the input positions (the
    equivariant score), as upstream does.
    """

    def __init__(
        self,
        n_layers: int,
        feats_dim: int,
        pos_dim: int = 3,
        edge_attr_dim: int = 0,
        m_dim: int = 16,
        soft_edge: int = 0,
        norm_feats: bool = True,
        norm_coors: bool = False,
        norm_coors_scale_init: float = 1e-2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pos_dim = pos_dim
        self.feats_dim = feats_dim
        self.mpnn_layers = nn.ModuleList(
            EGNNSparseLayer(
                feats_dim=feats_dim,
                pos_dim=pos_dim,
                edge_attr_dim=edge_attr_dim,
                m_dim=m_dim,
                soft_edge=soft_edge,
                norm_feats=norm_feats,
                norm_coors=norm_coors,
                norm_coors_scale_init=norm_coors_scale_init,
                dropout=dropout,
            )
            for _ in range(n_layers)
        )

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
        n_ligand: int,
    ) -> Tuple[Tensor, Tensor]:
        x = torch.cat([pos, z], dim=1)
        for layer in self.mpnn_layers:
            x = layer(x, edge_index, edge_attr, batch=batch, n_ligand=n_ligand)
        coors, feats = x[:, : self.pos_dim], x[:, self.pos_dim :]
        coors = coors - pos
        return feats[:n_ligand], coors[:n_ligand]

    def __repr__(self) -> str:
        return f"EGNNSparseNetwork of: {len(self.mpnn_layers)} layers"
