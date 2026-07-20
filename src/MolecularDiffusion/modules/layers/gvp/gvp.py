"""GVP building blocks ported from FlowMol (flowmol/models/gvp.py).

The GVP, GVPDropout and GVPLayerNorm classes originate from lucidrains'
geometric-vector-perceptron repo, adapted by FlowMol. GVPConv is FlowMol's
homogeneous-graph GVP convolution. NodePositionUpdate and EdgeUpdate were
lifted out of FlowMol's vector_field.py because they are generic reusable
blocks (position/edge-feature refinement), not FlowMol-specific.

Only cosmetic changes vs. the upstream source: the ``_rbf`` radial-basis
helper (originally in flowmol.utils.embedding) is defined here so this module
is self-contained.
"""

import math
from typing import Union

import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from torch import einsum
from dgl.nn.functional import edge_softmax


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16):
    """Radial-basis expansion of a distance tensor along a new last axis.

    From https://github.com/jingraham/neurips19-graph-protein-design
    """
    device = D.device
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    return torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))


def get_time_embedding(timesteps, embedding_dim=256, max_positions=1000):
    """Sinusoidal time embedding (ported from FlowMol utils.embedding).

    Only used when ``time_embedding_dim > 1``; the default FlowMol config here
    uses a scalar time feature, but the vector field imports this so the name
    must resolve.
    """
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        * -emb
    )
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def exists(val):
    return val is not None


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    """L2 norm of a tensor clamped above a minimum value ``eps``.

    :param sqrt: if ``False``, returns the square of the L2 norm.
    """
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


class GVP(nn.Module):
    def __init__(
        self,
        dim_vectors_in,
        dim_vectors_out,
        dim_feats_in,
        dim_feats_out,
        n_cp_feats=0,
        hidden_vectors=None,
        feats_activation=nn.SiLU(),
        vectors_activation=nn.Sigmoid(),
        vector_gating=True,
        xavier_init=False,
    ):
        super().__init__()
        self.dim_vectors_in = dim_vectors_in
        self.dim_feats_in = dim_feats_in
        self.n_cp_feats = n_cp_feats

        self.dim_vectors_out = dim_vectors_out
        dim_h = (
            max(dim_vectors_in, dim_vectors_out)
            if hidden_vectors is None
            else hidden_vectors
        )

        wh_k = 1 / math.sqrt(dim_vectors_in)
        self.Wh = torch.zeros(
            dim_vectors_in, dim_h, dtype=torch.float32
        ).uniform_(-wh_k, wh_k)
        self.Wh = nn.Parameter(self.Wh)

        if n_cp_feats > 0:
            wcp_k = 1 / math.sqrt(dim_vectors_in)
            self.Wcp = torch.zeros(
                dim_vectors_in, n_cp_feats * 2, dtype=torch.float32
            ).uniform_(-wcp_k, wcp_k)
            self.Wcp = nn.Parameter(self.Wcp)

        if n_cp_feats > 0:
            wu_in_dim = dim_h + n_cp_feats
        else:
            wu_in_dim = dim_h
        wu_k = 1 / math.sqrt(wu_in_dim)
        self.Wu = torch.zeros(
            wu_in_dim, dim_vectors_out, dtype=torch.float32
        ).uniform_(-wu_k, wu_k)
        self.Wu = nn.Parameter(self.Wu)

        self.vectors_activation = vectors_activation

        self.to_feats_out = nn.Sequential(
            nn.Linear(dim_h + n_cp_feats + dim_feats_in, dim_feats_out),
            feats_activation,
        )

        if vector_gating:
            self.scalar_to_vector_gates = nn.Linear(
                dim_feats_out, dim_vectors_out
            )
            if xavier_init:
                nn.init.xavier_uniform_(
                    self.scalar_to_vector_gates.weight, gain=1
                )
                nn.init.constant_(self.scalar_to_vector_gates.bias, 0)
        else:
            self.scalar_to_vector_gates = None

    def forward(self, data):
        feats, vectors = data
        b, n, _, v, c = *feats.shape, *vectors.shape

        assert (
            c == 3 and v == self.dim_vectors_in
        ), "vectors have wrong dimensions"
        assert n == self.dim_feats_in, "scalar features have wrong dimensions"

        Vh = einsum("b v c, v h -> b h c", vectors, self.Wh)

        if self.n_cp_feats > 0:
            Vcp = einsum("b v c, v p -> b p c", vectors, self.Wcp)
            cp_src, cp_dst = torch.split(Vcp, self.n_cp_feats, dim=1)
            cp = torch.linalg.cross(cp_src, cp_dst, dim=-1)
            Vh = torch.cat((Vh, cp), dim=1)

        Vu = einsum("b h c, h u -> b u c", Vh, self.Wu)

        sh = _norm_no_nan(Vh)
        s = torch.cat((feats, sh), dim=1)
        feats_out = self.to_feats_out(s)

        if exists(self.scalar_to_vector_gates):
            gating = self.scalar_to_vector_gates(feats_out)
            gating = gating.unsqueeze(dim=-1)
        else:
            gating = _norm_no_nan(Vu)

        vectors_out = self.vectors_activation(gating) * Vu
        return (feats_out, vectors_out)


class _VDropout(nn.Module):
    """Vector channel dropout (whole vector channels dropped together)."""

    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        return mask * x / (1 - self.drop_rate)


class GVPDropout(nn.Module):
    """Separate dropout for scalars and vectors."""

    def __init__(self, rate):
        super().__init__()
        self.vector_dropout = _VDropout(rate)
        self.feat_dropout = nn.Dropout(rate)

    def forward(self, feats, vectors):
        return self.feat_dropout(feats), self.vector_dropout(vectors)


class GVPLayerNorm(nn.Module):
    """Layer norm for scalars, non-trainable norm for vectors."""

    def __init__(self, feats_h_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.feat_norm = nn.LayerNorm(feats_h_size)

    def forward(self, data):
        feats, vectors = data
        normed_feats = self.feat_norm(feats)
        vn = _norm_no_nan(vectors, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True) + self.eps)
        vn = vn + self.eps
        normed_vectors = vectors / vn
        return normed_feats, normed_vectors


class GVPConv(nn.Module):
    """GVP graph convolution on a homogeneous graph."""

    def __init__(
        self,
        scalar_size: int = 128,
        vector_size: int = 16,
        n_cp_feats: int = 0,
        scalar_activation=nn.SiLU,
        vector_activation=nn.Sigmoid,
        n_message_gvps: int = 1,
        n_update_gvps: int = 1,
        attention: bool = False,
        s_message_dim: int = None,
        v_message_dim: int = None,
        n_heads: int = 1,
        n_expansion_gvps: int = 1,
        use_dst_feats: bool = False,
        dst_feat_msg_reduction_factor: float = 4,
        rbf_dmax: float = 20,
        rbf_dim: int = 16,
        edge_feat_size: int = 0,
        message_norm: Union[float, str] = 10,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.scalar_size = scalar_size
        self.vector_size = vector_size
        self.n_cp_feats = n_cp_feats
        self.scalar_activation = scalar_activation
        self.vector_activation = vector_activation
        self.n_message_gvps = n_message_gvps
        self.n_update_gvps = n_update_gvps
        self.edge_feat_size = edge_feat_size
        self.use_dst_feats = use_dst_feats
        self.rbf_dmax = rbf_dmax
        self.rbf_dim = rbf_dim
        self.dropout_rate = dropout
        self.message_norm = message_norm

        self.s_message_dim = s_message_dim
        self.v_message_dim = v_message_dim
        self.attention = attention
        self.n_heads = n_heads

        if s_message_dim is None:
            self.s_message_dim = scalar_size
        if v_message_dim is None:
            self.v_message_dim = vector_size

        if self.s_message_dim != scalar_size or self.v_message_dim != vector_size:
            self.compressed_messaging = True
        else:
            self.compressed_messaging = False

        if self.compressed_messaging:
            compression_gvps = []
            for i in range(n_expansion_gvps):
                if i == 0:
                    dim_feats_in = scalar_size
                    dim_vectors_in = vector_size
                else:
                    dim_feats_in = max(self.s_message_dim, scalar_size)
                    dim_vectors_in = max(self.v_message_dim, vector_size)

                if i == n_expansion_gvps - 1:
                    dim_feats_out = self.s_message_dim
                    dim_vectors_out = self.v_message_dim
                else:
                    dim_feats_out = max(self.s_message_dim, scalar_size)
                    dim_vectors_out = max(self.v_message_dim, vector_size)

                compression_gvps.append(
                    GVP(
                        dim_vectors_in=dim_vectors_in,
                        dim_vectors_out=dim_vectors_out,
                        n_cp_feats=n_cp_feats,
                        dim_feats_in=dim_feats_in,
                        dim_feats_out=dim_feats_out,
                        feats_activation=scalar_activation(),
                        vectors_activation=vector_activation(),
                        vector_gating=True,
                    )
                )
            self.node_compression = nn.Sequential(*compression_gvps)
        else:
            self.node_compression = nn.Identity()

        if attention:
            if (
                self.s_message_dim % n_heads != 0
                or self.v_message_dim % n_heads != 0
            ):
                raise ValueError(
                    "Number of attention heads must divide the message size."
                )
            self.s_feats_per_head = self.s_message_dim // n_heads
            self.v_feats_per_head = self.v_message_dim // n_heads
            extra_scalar_feats = n_heads * 2
            self.att_weight_projection = nn.Sequential(
                nn.Linear(extra_scalar_feats, extra_scalar_feats, bias=False),
                nn.LayerNorm(extra_scalar_feats),
            )
        else:
            extra_scalar_feats = 0

        if use_dst_feats:
            if dst_feat_msg_reduction_factor != 1:
                s_dst_feats_for_messages = int(
                    self.s_message_dim / dst_feat_msg_reduction_factor
                )
                v_dst_feats_for_messages = int(
                    self.v_message_dim / dst_feat_msg_reduction_factor
                )
                self.dst_feat_msg_projection = GVP(
                    dim_vectors_in=self.v_message_dim,
                    dim_vectors_out=v_dst_feats_for_messages,
                    dim_feats_in=self.s_message_dim,
                    dim_feats_out=s_dst_feats_for_messages,
                    n_cp_feats=0,
                    feats_activation=scalar_activation(),
                    vectors_activation=vector_activation(),
                )
            else:
                s_dst_feats_for_messages = self.s_message_dim
                v_dst_feats_for_messages = self.v_message_dim
                self.dst_feat_msg_projection = nn.Identity()
        else:
            s_dst_feats_for_messages = 0
            v_dst_feats_for_messages = 0

        message_gvps = []
        s_slope = (
            self.s_message_dim + extra_scalar_feats - scalar_size
        ) / n_message_gvps
        v_slope = (self.v_message_dim - vector_size) / n_message_gvps
        for i in range(n_message_gvps):
            dim_vectors_in = self.v_message_dim
            dim_feats_in = self.s_message_dim

            if i == 0:
                dim_vectors_in += 1
                dim_feats_in += rbf_dim + edge_feat_size
            else:
                dim_feats_in = dim_feats_out
                dim_vectors_in = dim_vectors_out

            if use_dst_feats and i == 0:
                dim_vectors_in += v_dst_feats_for_messages
                dim_feats_in += s_dst_feats_for_messages

            if self.s_message_dim < scalar_size:
                dim_feats_out = int(s_slope * i + scalar_size)
                if i == n_message_gvps - 1:
                    dim_feats_out = self.s_message_dim + extra_scalar_feats
            else:
                dim_feats_out = self.s_message_dim + extra_scalar_feats

            if self.v_message_dim < vector_size:
                dim_vectors_out = int(v_slope * i + vector_size)
                if i == n_message_gvps - 1:
                    dim_vectors_out = self.v_message_dim
            else:
                dim_vectors_out = self.v_message_dim

            message_gvps.append(
                GVP(
                    dim_vectors_in=dim_vectors_in,
                    dim_vectors_out=dim_vectors_out,
                    n_cp_feats=n_cp_feats,
                    dim_feats_in=dim_feats_in,
                    dim_feats_out=dim_feats_out,
                    feats_activation=scalar_activation(),
                    vectors_activation=vector_activation(),
                    vector_gating=True,
                )
            )
        self.edge_message = nn.Sequential(*message_gvps)

        update_gvps = []
        for i in range(n_update_gvps):
            update_gvps.append(
                GVP(
                    dim_vectors_in=vector_size,
                    dim_vectors_out=vector_size,
                    n_cp_feats=n_cp_feats,
                    dim_feats_in=scalar_size,
                    dim_feats_out=scalar_size,
                    feats_activation=scalar_activation(),
                    vectors_activation=vector_activation(),
                    vector_gating=True,
                )
            )
        self.node_update = nn.Sequential(*update_gvps)

        self.dropout = GVPDropout(self.dropout_rate)
        self.message_layer_norm = GVPLayerNorm(self.scalar_size)
        self.update_layer_norm = GVPLayerNorm(self.scalar_size)

        if (
            isinstance(self.message_norm, str)
            and self.message_norm not in ["mean", "sum"]
        ):
            raise ValueError(
                "message_norm must be either 'mean', 'sum', or a number, "
                f"got {self.message_norm}"
            )
        else:
            if self.message_norm not in ["mean", "sum"]:
                assert isinstance(
                    self.message_norm, (float, int)
                ), "message_norm must be either 'mean', 'sum', or a number"

        if self.message_norm == "mean":
            self.agg_func = fn.mean
        else:
            self.agg_func = fn.sum

        if self.compressed_messaging:
            projection_gvps = []
            for i in range(n_expansion_gvps):
                if i == 0:
                    dim_feats_in = self.s_message_dim
                    dim_vectors_in = self.v_message_dim
                else:
                    dim_feats_in = scalar_size
                    dim_vectors_in = vector_size

                dim_feats_out = scalar_size
                dim_vectors_out = vector_size

                projection_gvps.append(
                    GVP(
                        dim_vectors_in=dim_vectors_in,
                        dim_vectors_out=dim_vectors_out,
                        n_cp_feats=n_cp_feats,
                        dim_feats_in=dim_feats_in,
                        dim_feats_out=dim_feats_out,
                        feats_activation=scalar_activation(),
                        vectors_activation=vector_activation(),
                        vector_gating=True,
                    )
                )
            self.message_expansion = nn.Sequential(*projection_gvps)
        else:
            self.message_expansion = nn.Identity()

    def forward(
        self,
        g: dgl.DGLGraph,
        scalar_feats: torch.Tensor,
        coord_feats: torch.Tensor,
        vec_feats: torch.Tensor,
        edge_feats: torch.Tensor = None,
        x_diff: torch.Tensor = None,
        d: torch.Tensor = None,
    ):
        with g.local_scope():
            g.ndata["s"] = scalar_feats
            g.ndata["x"] = coord_feats
            g.ndata["v"] = vec_feats

            if x_diff is not None and d is not None:
                g.edata["x_diff"] = x_diff
                g.edata["d"] = d

            if self.edge_feat_size > 0:
                assert edge_feats is not None, "Edge features must be provided."
                g.edata["ef"] = edge_feats

            if "x_diff" not in g.edata:
                g.apply_edges(fn.u_sub_v("x", "x", "x_diff"))
                dij = _norm_no_nan(g.edata["x_diff"], keepdims=True) + 1e-8
                g.edata["x_diff"] = g.edata["x_diff"] / dij
                g.edata["d"] = _rbf(
                    dij.squeeze(1), D_max=self.rbf_dmax, D_count=self.rbf_dim
                )

            g.ndata["s"], g.ndata["v"] = self.node_compression(
                (g.ndata["s"], g.ndata["v"])
            )

            if self.use_dst_feats:
                g.ndata["s_dst_msg"], g.ndata["v_dst_msg"] = (
                    self.dst_feat_msg_projection((g.ndata["s"], g.ndata["v"]))
                )

            g.apply_edges(self.message)

            if self.attention:
                scalar_msg = g.edata["scalar_msg"][:, : self.s_message_dim]
                att_logits = g.edata["scalar_msg"][:, self.s_message_dim :]
                att_logits = self.att_weight_projection(att_logits)
                att_weights = edge_softmax(g, att_logits)
                s_att_weights = att_weights[:, : self.n_heads]
                v_att_weights = att_weights[:, self.n_heads :]
                s_att_weights = s_att_weights.repeat_interleave(
                    self.s_feats_per_head, dim=1
                )
                v_att_weights = v_att_weights.repeat_interleave(
                    self.v_feats_per_head, dim=1
                )
                g.edata["scalar_msg"] = scalar_msg * s_att_weights
                g.edata["vec_msg"] = (
                    g.edata["vec_msg"] * v_att_weights.unsqueeze(-1)
                )

            g.update_all(
                fn.copy_e("scalar_msg", "m"), self.agg_func("m", "scalar_msg")
            )
            g.update_all(
                fn.copy_e("vec_msg", "m"), self.agg_func("m", "vec_msg")
            )

            if isinstance(self.message_norm, str):
                z = 1
            else:
                z = self.message_norm

            scalar_msg = g.ndata["scalar_msg"] / z
            vec_msg = g.ndata["vec_msg"] / z

            scalar_msg, vec_msg = self.message_expansion((scalar_msg, vec_msg))
            scalar_msg, vec_msg = self.dropout(scalar_msg, vec_msg)

            scalar_feat = scalar_feats + scalar_msg
            vec_feat = vec_feats + vec_msg
            scalar_feat, vec_feat = self.message_layer_norm(
                (scalar_feat, vec_feat)
            )

            scalar_residual, vec_residual = self.node_update(
                (scalar_feat, vec_feat)
            )
            scalar_residual, vec_residual = self.dropout(
                scalar_residual, vec_residual
            )
            scalar_feat = scalar_feat + scalar_residual
            vec_feat = vec_feat + vec_residual
            scalar_feat, vec_feat = self.update_layer_norm(
                (scalar_feat, vec_feat)
            )

        return scalar_feat, vec_feat

    def message(self, edges):
        vec_feats = [edges.data["x_diff"].unsqueeze(1), edges.src["v"]]
        if self.use_dst_feats:
            vec_feats.append(edges.dst["v_dst_msg"])
        vec_feats = torch.cat(vec_feats, dim=1)

        scalar_feats = [edges.src["s"], edges.data["d"]]
        if self.edge_feat_size > 0:
            scalar_feats.append(edges.data["ef"])
        if self.use_dst_feats:
            scalar_feats.append(edges.dst["s_dst_msg"])
        scalar_feats = torch.cat(scalar_feats, dim=1)

        scalar_message, vector_message = self.edge_message(
            (scalar_feats, vec_feats)
        )
        return {"scalar_msg": scalar_message, "vec_msg": vector_message}


class NodePositionUpdate(nn.Module):
    """Refine node positions from scalar/vector features (equivariant)."""

    def __init__(
        self, n_scalars, n_vec_channels, n_gvps: int = 3, n_cp_feats: int = 0
    ):
        super().__init__()
        self.gvps = []
        for i in range(n_gvps):
            if i == n_gvps - 1:
                vectors_out = 1
                vectors_activation = nn.Identity()
            else:
                vectors_out = n_vec_channels
                vectors_activation = nn.Sigmoid()

            self.gvps.append(
                GVP(
                    dim_feats_in=n_scalars,
                    dim_feats_out=n_scalars,
                    dim_vectors_in=n_vec_channels,
                    dim_vectors_out=vectors_out,
                    n_cp_feats=n_cp_feats,
                    vectors_activation=vectors_activation,
                )
            )
        self.gvps = nn.Sequential(*self.gvps)

    def forward(self, scalars, positions, vectors):
        _, vector_updates = self.gvps((scalars, vectors))
        return positions + vector_updates.squeeze(1)


class EdgeUpdate(nn.Module):
    """Refine edge features from adjacent node scalars (+ optional distance)."""

    def __init__(
        self,
        n_node_scalars,
        n_edge_feats,
        update_edge_w_distance=False,
        rbf_dim=16,
    ):
        super().__init__()
        self.update_edge_w_distance = update_edge_w_distance

        input_dim = n_node_scalars * 2 + n_edge_feats
        if update_edge_w_distance:
            input_dim += rbf_dim

        self.edge_update_fn = nn.Sequential(
            nn.Linear(input_dim, n_edge_feats),
            nn.SiLU(),
            nn.Linear(n_edge_feats, n_edge_feats),
            nn.SiLU(),
        )
        self.edge_norm = nn.LayerNorm(n_edge_feats)

    def forward(self, g: dgl.DGLGraph, node_scalars, edge_feats, d):
        src_idxs, dst_idxs = g.edges()
        mlp_inputs = [
            node_scalars[src_idxs],
            node_scalars[dst_idxs],
            edge_feats,
        ]
        if self.update_edge_w_distance:
            mlp_inputs.append(d)
        edge_feats = self.edge_norm(
            edge_feats + self.edge_update_fn(torch.cat(mlp_inputs, dim=-1))
        )
        return edge_feats
