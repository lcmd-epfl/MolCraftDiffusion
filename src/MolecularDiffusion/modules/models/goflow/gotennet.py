"""GotenNet: the CGR-conditioned equivariant backbone GoFlow's velocity
field is built from.

Ported from ``gotennet/models/representation/gotennet.py`` (commit
``3ec00a09``). Aykent & Xia, arXiv:2410.14670 -- cited from the GotenNet
repository's own README, not independently verified against it (see
``INTEGRATION_PLAN.md``, Repo Inspection).

Two deviations from upstream, both mechanical, neither changing any tensor
shape or learned behaviour:

* The two ``einops.rearrange``/``reduce`` calls in ``GATA.rej``
  (``gotennet.py:296-297``) are replaced by their exact plain-torch
  equivalents -- verified by their own docstrings below.
* ``RankedLogger`` (a Hydra-rank-zero-only wrapper the rest of
  ``gotennet/utils`` is not ported for) is replaced by a plain
  ``logging.getLogger``; the two ``log.info`` calls it backs are cosmetic
  (weight-init choice logging), not model behaviour.

``self.distance`` (upstream's ``Distance`` module) is not constructed here:
see ``ops.py``'s module docstring for why it is dead code in ``forward``.
"""

from __future__ import annotations

import logging
import math
from functools import partial
from typing import Callable, Mapping, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter, softmax

from .cgr_graph_utils import _extend_condensed_graph_edge
from .ops import (
    AtomCGREmbedding,
    CosineCutoff,
    Dense,
    EdgeCGREmbedding,
    EdgeInit,
    MLP,
    NodeInit,
    TensorInit,
    VecLayerNorm,
    get_weight_init_by_string,
    parse_update_info,
    str2act,
    str2basis,
)

log = logging.getLogger(__name__)


def lmax_tensor_size(lmax: int) -> int:
    return ((lmax + 1) ** 2) - 1


def split_degree(tensor: Tensor, lmax: int, dim: int = -1):
    """Split a stacked-irreps tensor into one chunk per degree ``l``.

    Verbatim from ``gotennet.py:41-51``.
    """
    cumsum = 0
    tensors = []
    for i in range(1, lmax + 1):
        l_vec_size = lmax_tensor_size(i) - lmax_tensor_size(i - 1)
        slc = [slice(None)] * tensor.ndim
        slc[dim] = slice(cumsum, cumsum + l_vec_size)
        tensors.append(tensor[tuple(slc)])
        cumsum += l_vec_size
    return tensors


class GATA(MessagePassing):
    """Attention-weighted equivariant message passing.

    Verbatim from ``gotennet.py:54-370``, except :meth:`rej`'s two
    ``einops`` calls.
    """

    def __init__(
        self, n_atom_basis: int, activation: Callable, weight_init=nn.init.xavier_uniform_,
        bias_init=nn.init.zeros_, aggr="add", node_dim=0, epsilon: float = 1e-7,
        layer_norm="", vector_norm="", cutoff=5.0, scaling=1.0, num_heads=8, dropout=0.0,
        edge_updates=True, last_layer=False, scale_edge=True, edge_ln="", evec_dim=None,
        emlp_dim=None, sep_vecj=True, lmax=1,
    ) -> None:
        super().__init__(aggr=aggr, node_dim=node_dim)
        self.lmax = lmax
        self.sep_vecj = sep_vecj
        self.epsilon = epsilon
        self.last_layer = last_layer
        self.edge_updates = edge_updates
        self.scale_edge = scale_edge
        self.activation = activation

        self.update_info = parse_update_info(edge_updates)
        self.dropout = dropout
        self.n_atom_basis = n_atom_basis

        init_dense = partial(Dense, weight_init=weight_init, bias_init=bias_init)
        self.gamma_s = nn.Sequential(
            init_dense(n_atom_basis, n_atom_basis, activation=activation),
            init_dense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )

        self.num_heads = num_heads
        self.q_w = init_dense(n_atom_basis, n_atom_basis, activation=None)
        self.k_w = init_dense(n_atom_basis, n_atom_basis, activation=None)

        self.gamma_v = nn.Sequential(
            init_dense(n_atom_basis, n_atom_basis, activation=activation),
            init_dense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )

        self.phik_w_ra = init_dense(n_atom_basis, n_atom_basis, activation=activation)

        init_mlp = partial(MLP, weight_init=weight_init, bias_init=bias_init)
        self.edge_vec_dim = n_atom_basis if evec_dim is None else evec_dim
        self.edge_mlp_dim = n_atom_basis if emlp_dim is None else emlp_dim
        if not self.last_layer and self.edge_updates:
            if self.update_info["mlp"] or self.update_info["mlpa"]:
                dims = [n_atom_basis, self.edge_mlp_dim, n_atom_basis]
            else:
                dims = [n_atom_basis, n_atom_basis]
            self.edge_attr_up = init_mlp(
                dims, activation=activation,
                last_activation=None if self.update_info["mlp"] else self.activation,
                norm=edge_ln,
            )
            self.vecq_w = init_dense(n_atom_basis, self.edge_vec_dim, activation=None, bias=False)

            if self.sep_vecj:
                self.veck_w = nn.ModuleList(
                    [
                        init_dense(n_atom_basis, self.edge_vec_dim, activation=None, bias=False)
                        for _ in range(self.lmax)
                    ]
                )
            else:
                self.veck_w = init_dense(n_atom_basis, self.edge_vec_dim, activation=None, bias=False)

            if self.update_info["lin_w"] > 0:
                modules = []
                if self.update_info["lin_w"] % 10 == 2:
                    modules.append(self.activation)
                self.lin_w_linear = init_dense(
                    self.edge_vec_dim, n_atom_basis, activation=None,
                    norm="layer" if self.update_info["lin_ln"] == 2 else "",
                )
                modules.append(self.lin_w_linear)
                self.lin_w = nn.Sequential(*modules)

        self.down_proj = nn.Identity()
        self.cutoff = CosineCutoff(cutoff, scaling)
        self._alpha = None

        self.w_re = init_dense(n_atom_basis, n_atom_basis * 3, None)

        self.layernorm_ = layer_norm
        self.vector_norm_ = vector_norm
        self.layernorm = nn.LayerNorm(n_atom_basis) if layer_norm != "" else nn.Identity()
        self.tln = (
            VecLayerNorm(n_atom_basis, trainable=False, norm_type=vector_norm)
            if vector_norm != ""
            else nn.Identity()
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.layernorm_:
            self.layernorm.reset_parameters()
        if self.vector_norm_:
            self.tln.reset_parameters()
        for m in self.gamma_s:
            m.reset_parameters()
        self.q_w.reset_parameters()
        self.k_w.reset_parameters()
        for m in self.gamma_v:
            m.reset_parameters()
        self.w_re.reset_parameters()
        if not self.last_layer and self.edge_updates:
            self.edge_attr_up.reset_parameters()
            self.vecq_w.reset_parameters()
            if self.sep_vecj:
                for w in self.veck_w:
                    w.reset_parameters()
            else:
                self.veck_w.reset_parameters()
            if self.update_info["lin_w"] > 0:
                self.lin_w_linear.reset_parameters()

    def forward(self, edge_index, s, t, dir_ij, r_ij, d_ij, num_edges_expanded):
        s = self.layernorm(s)
        t = self.tln(t)

        q = self.q_w(s).reshape(-1, self.num_heads, self.n_atom_basis // self.num_heads)
        k = self.k_w(s).reshape(-1, self.num_heads, self.n_atom_basis // self.num_heads)

        x = self.gamma_s(s)
        val = self.gamma_v(s)
        f_ij = r_ij
        r_ij_attn = self.phik_w_ra(r_ij)
        r_ij = self.w_re(r_ij)

        su, tu = self.propagate(
            edge_index=edge_index, x=x, q=q, k=k, val=val, ten=t, r_ij=r_ij,
            r_ij_attn=r_ij_attn, d_ij=d_ij, dir_ij=dir_ij,
            num_edges_expanded=num_edges_expanded,
        )

        s = s + su
        t = t + tu

        if not self.last_layer and self.edge_updates:
            vec = t
            w1 = self.vecq_w(vec)
            if self.sep_vecj:
                vec_split = split_degree(vec, self.lmax, dim=1)
                w_out = torch.concat(
                    [w(vec_split[i]) for i, w in enumerate(self.veck_w)], dim=1
                )
            else:
                w_out = self.veck_w(vec)

            df_ij = self.edge_updater(edge_index, w1=w1, w2=w_out, d_ij=dir_ij, f_ij=f_ij)
            df_ij = f_ij + df_ij
            self._alpha = None
            return s, t, df_ij
        self._alpha = None
        return s, t, f_ij

    def message(
        self, edge_index, x_i, x_j, q_i, k_j, val_j, ten_j, r_ij, r_ij_attn, d_ij,
        dir_ij, num_edges_expanded, index, ptr: OptTensor, dim_size: Optional[int],
    ) -> Tuple[Tensor, Tensor]:
        r_ij_attn = r_ij_attn.reshape(-1, self.num_heads, self.n_atom_basis // self.num_heads)
        attn = (q_i * k_j * r_ij_attn).sum(dim=-1, keepdim=True)
        attn = softmax(attn, index, ptr, dim_size)

        if self.scale_edge:
            norm = torch.sqrt(num_edges_expanded.reshape(-1, 1, 1)) / math.sqrt(self.n_atom_basis)
        else:
            norm = 1.0 / math.sqrt(self.n_atom_basis)
        attn = attn * norm
        self._alpha = attn
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        self_attn = attn * val_j.reshape(-1, self.num_heads, (self.n_atom_basis * 3) // self.num_heads)
        sea = self_attn.reshape(-1, 1, self.n_atom_basis * 3)

        x = sea + (r_ij.unsqueeze(1) * x_j * self.cutoff(d_ij.unsqueeze(-1).unsqueeze(-1)))

        o_s, o_d, o_t = torch.split(x, self.n_atom_basis, dim=-1)
        dmu = o_d * dir_ij[..., None] + o_t * ten_j
        return o_s, dmu

    @staticmethod
    def rej(vec: Tensor, d_ij: Tensor) -> Tensor:
        """Reject ``vec`` off the ``d_ij`` direction, per irrep degree.

        Replaces ``gotennet.py:294-298``'s two ``einops`` calls with their
        exact plain-torch equivalents: ``rearrange(d_ij, 'b l -> b l 1')``
        is ``d_ij.unsqueeze(-1)``, and ``reduce(vec * d_ij_1, 'b l c -> b 1
        c', 'sum')`` is ``(vec * d_ij_1).sum(dim=1, keepdim=True)`` -- both
        named explicitly in ``INTEGRATION_PLAN.md``'s Naming section.
        """
        d_ij_1 = d_ij.unsqueeze(-1)
        vec_proj = (vec * d_ij_1).sum(dim=1, keepdim=True)
        return vec - vec_proj * d_ij_1

    def edge_update(self, w1_i, w2_j, d_ij, f_ij):
        if self.sep_vecj:
            vi_split = split_degree(w1_i, self.lmax, dim=1)
            vj_split = split_degree(w2_j, self.lmax, dim=1)
            d_ij_split = split_degree(d_ij, self.lmax, dim=1)

            pairs = []
            for i in range(len(vi_split)):
                if self.update_info["rej"]:
                    w1 = self.rej(vi_split[i], d_ij_split[i])
                    w2 = self.rej(vj_split[i], -d_ij_split[i])
                else:
                    w1 = vi_split[i]
                    w2 = vj_split[i]
                pairs.append((w1, w2))
        elif not self.update_info["rej"]:
            pairs = [(w1_i, w2_j)]
        else:
            pairs = [(self.rej(w1_i, d_ij), self.rej(w2_j, -d_ij))]

        w_dot_sum = None
        for w1, w2 in pairs:
            w_dot = (w1 * w2).sum(dim=1)
            w_dot_sum = w_dot if w_dot_sum is None else w_dot_sum + w_dot
        w_dot = w_dot_sum
        if self.update_info["lin_w"] > 0:
            w_dot = self.lin_w(w_dot)

        if self.update_info["gated"] == "gatedt":
            w_dot = torch.tanh(w_dot)
        elif self.update_info["gated"] == "gated":
            w_dot = torch.sigmoid(w_dot)
        elif self.update_info["gated"] == "act":
            w_dot = self.activation(w_dot)

        return self.edge_attr_up(f_ij) * w_dot

    def aggregate(self, features, index, ptr, dim_size):
        x, vec = features
        x_ = scatter(x, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        vec_ = scatter(vec, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return x_, vec_

    def update(self, inputs):
        return inputs


class EQFF(nn.Module):
    """Equivariant feed-forward mixing. Verbatim from ``gotennet.py:372-414``."""

    def __init__(
        self, n_atom_basis: int, activation: Callable, epsilon: float = 1e-8,
        weight_init=nn.init.xavier_uniform_, bias_init=nn.init.zeros_, vec_dim=None,
    ) -> None:
        super().__init__()
        self.n_atom_basis = n_atom_basis
        init_dense = partial(Dense, weight_init=weight_init, bias_init=bias_init)
        vec_dim = n_atom_basis if vec_dim is None else vec_dim

        self.gamma_m = nn.Sequential(
            init_dense(2 * n_atom_basis, n_atom_basis, activation=activation),
            init_dense(n_atom_basis, 2 * n_atom_basis, activation=None),
        )
        self.w_vu = init_dense(n_atom_basis, vec_dim, activation=None, bias=False)
        self.epsilon = epsilon

    def reset_parameters(self) -> None:
        self.w_vu.reset_parameters()
        for m in self.gamma_m:
            m.reset_parameters()

    def forward(self, s, v):
        t_prime = self.w_vu(v)
        t_prime_mag = torch.sqrt(torch.sum(t_prime**2, dim=-2, keepdim=True) + self.epsilon)
        combined = torch.cat([s, t_prime_mag], dim=-1)
        m12 = self.gamma_m(combined)
        m_1, m_2 = torch.split(m12, self.n_atom_basis, dim=-1)
        return s + m_1, v + m_2 * t_prime


def get_timestep_embedding(timesteps: Tensor, embedding_dim: int, max_positions: int = 10000) -> Tensor:
    """Verbatim from ``gotennet.py:417-428``."""
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class TimestepEmbedding(nn.Module):
    """Verbatim from ``gotennet.py:431-447``."""

    def __init__(self, embedding_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            Dense(embedding_dim, hidden_dim, norm="layer", activation=nn.SiLU()),
            Dense(hidden_dim, output_dim, norm="layer", activation=nn.SiLU()),
        )

    def forward(self, timesteps: Tensor) -> Tensor:
        t_emb = get_timestep_embedding(timesteps.squeeze(), self.embedding_dim)
        return self.mlp(t_emb)


class GotenNet(nn.Module):
    """The CGR-conditioned equivariant backbone.

    Ported from ``gotennet.py:450-618``. ``forward``'s signature and body
    are unchanged from upstream: ``inputs`` reads only ``.edge_index,
    .edge_type, .batch, .r_feat, .p_feat, .atom_type`` off the PyG batch
    (never ``.pos`` -- the noisy TS coordinate is always the separate
    ``x_t_N_3`` argument), exactly as ``INTEGRATION_PLAN.md``'s Repo
    Inspection records.
    """

    def __init__(
        self,
        n_atom_basis: int = 128,
        n_atom_feat_basis: int = 128,  # noqa: ARG002 - dead upstream too; kept for config fidelity
        n_atom_rdkit_feats: int = 28,
        n_interactions: int = 8,
        radial_basis: Union[Callable, str] = "expnorm",
        n_rbf: int = 20,
        cutoff_fn: Optional[Callable] = None,
        edge_order: int = 4,
        activation: Optional[Union[Callable, str]] = F.silu,
        max_z: int = 100,
        epsilon: float = 1e-8,
        weight_init=nn.init.xavier_uniform_,
        bias_init=nn.init.zeros_,
        max_num_neighbors: int = 32,  # noqa: ARG002 - only Distance read this; not ported (ops.py)
        int_layer_norm="",
        int_vector_norm="",
        num_heads=8,
        attn_dropout=0.0,
        edge_updates=True,
        scale_edge=True,
        lmax=2,
        aggr="add",
        edge_ln="",
        evec_dim=None,
        emlp_dim=None,
        sep_int_vec=True,
    ) -> None:
        super().__init__()
        self.scale_edge = scale_edge
        if isinstance(weight_init, str):
            log.info("Using %s weight initialization", weight_init)
            weight_init = get_weight_init_by_string(weight_init)
        if isinstance(bias_init, str):
            bias_init = get_weight_init_by_string(bias_init)
        if isinstance(activation, str):
            activation = str2act(activation)

        self.n_atom_basis = self.hidden_dim = n_atom_basis
        self.n_interactions = n_interactions
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.scaling = cutoff_fn.scaling
        self.edge_order = edge_order

        self.neighbor_embedding = NodeInit(
            [self.hidden_dim // 2, self.hidden_dim], n_atom_rdkit_feats, n_rbf,
            self.cutoff, self.scaling, max_z=max_z, weight_init=weight_init,
            bias_init=bias_init, proj_ln="layer", activation=activation,
        )
        self.edge_embedding = EdgeInit(
            n_rbf, [self.hidden_dim // 2, self.hidden_dim], weight_init=weight_init,
            bias_init=bias_init, proj_ln="",
        )
        self.time_embedding = TimestepEmbedding(128, 128, self.hidden_dim)

        radial_basis_cls = str2basis(radial_basis)
        self.radial_basis = radial_basis_cls(cutoff=self.cutoff, scaling=self.scaling, n_rbf=n_rbf)

        self.atom_cgr_embedding = AtomCGREmbedding(n_atom_rdkit_feats, n_atom_basis)
        self.edge_cgr_embedding = EdgeCGREmbedding(self.hidden_dim)

        self.tensor_init = TensorInit(l=lmax)

        self.gata = nn.ModuleList(
            [
                GATA(
                    n_atom_basis=self.n_atom_basis, activation=activation, aggr=aggr,
                    weight_init=weight_init, bias_init=bias_init, layer_norm=int_layer_norm,
                    vector_norm=int_vector_norm, cutoff=self.cutoff, scaling=self.scaling,
                    epsilon=epsilon, num_heads=num_heads, dropout=attn_dropout,
                    edge_updates=edge_updates, last_layer=(i == self.n_interactions - 1),
                    scale_edge=scale_edge, edge_ln=edge_ln, evec_dim=evec_dim, emlp_dim=emlp_dim,
                    sep_vecj=sep_int_vec, lmax=lmax,
                )
                for i in range(self.n_interactions)
            ]
        )
        self.eqff = nn.ModuleList(
            [
                EQFF(
                    n_atom_basis=self.n_atom_basis, activation=activation, epsilon=epsilon,
                    weight_init=weight_init, bias_init=bias_init,
                )
                for _ in range(self.n_interactions)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.edge_embedding.reset_parameters()
        self.neighbor_embedding.reset_parameters()
        for m in self.gata:
            m.reset_parameters()
        for m in self.eqff:
            m.reset_parameters()

    def forward(
        self, x_t_N_3: Tensor, t_G: Tensor, inputs: Mapping[str, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """Compute atomic representations.

        Args:
            x_t_N_3: ``(N, 3)`` current (noisy) node coordinates.
            t_G: ``(B, 1)`` flow time, one value per graph.
            inputs: the PyG batch built by ``goflow_data.goflow_collate``;
                only ``.edge_index, .edge_type, .batch, .r_feat, .p_feat,
                .atom_type`` are read.

        Returns:
            ``(q, mu)``: ``q`` is ``(N, hidden_dim)`` scalar features, ``mu``
            is ``(N, (lmax+1)**2 - 1, hidden_dim)`` higher-order features.
        """
        edge_index, edge_type, batch = inputs.edge_index, inputs.edge_type, inputs.batch
        r_feat, p_feat, atom_type = inputs.r_feat, inputs.p_feat, inputs.atom_type

        edge_index, _, edge_type_r, edge_type_p = _extend_condensed_graph_edge(
            x_t_N_3, edge_index, edge_type, batch, cutoff=self.cutoff, edge_order=self.edge_order
        )

        edge_vec = x_t_N_3[edge_index[0]] - x_t_N_3[edge_index[1]]
        edge_weight = torch.norm(edge_vec, dim=-1)
        edge_attr = self.radial_basis(edge_weight)

        q = self.atom_cgr_embedding(atom_type, r_feat, p_feat)
        q = self.neighbor_embedding(
            atom_type, r_feat, p_feat, q, edge_index, edge_weight, edge_attr, edge_type_r, edge_type_p
        )
        t_emb_g = self.time_embedding(t_G)
        if batch is None:
            batch = torch.zeros(len(atom_type), dtype=torch.long, device=q.device)
        t_emb_n = t_emb_g[batch]
        q = q + t_emb_n

        edge_emb = self.edge_embedding(edge_index, edge_attr, edge_type_r, edge_type_p)
        edge_emb_t = edge_emb + t_emb_n[edge_index[0]]
        assert torch.allclose(t_emb_n[edge_index[0]], t_emb_n[edge_index[1]])

        mask = edge_index[0] != edge_index[1]
        dist = torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
        edge_vec[mask] = edge_vec[mask] / dist

        edge_vec = self.tensor_init(edge_vec)
        equi_dim = ((self.tensor_init.l + 1) ** 2) - 1
        num_edges = scatter(torch.ones_like(edge_weight), edge_index[0], dim=0, reduce="sum")
        num_edges_expanded = num_edges[edge_index[0]]

        qs = q.shape
        mu = torch.zeros((qs[0], equi_dim, qs[1]), device=q.device)
        q = q.unsqueeze(1)
        for interaction, mixing in zip(self.gata, self.eqff):
            q, mu, edge_attr = interaction(
                edge_index, q, mu, dir_ij=edge_vec, r_ij=edge_emb_t, d_ij=edge_weight,
                num_edges_expanded=num_edges_expanded,
            )
            q, mu = mixing(q, mu)

        q = q.squeeze(1)
        return q, mu
