"""DMT -- the Diffusion Molecular Transformer, NExT-Mol's 3D half.

Ported from ``others/NExT-Mol/model/diffusion_model_dgt.py`` (``DGTDiffusion``).
A relational graph transformer over the **fully connected** node-pair graph with
paired node and edge tracks, Gaussian-basis distance edge features, adaLN-style
time modulation, and an MLP head that predicts the position noise.

**Only coordinates are diffused.** ``x`` (atom features) and ``edge_attr`` (bond
one-hots) are fixed conditioning on every forward and every sampling step.

Deliberately not ported (all "explicitly out of scope" in INTEGRATION_PLAN.md,
and none of them appear in the released checkpoints' 275 tensors):

* ``enable_equiv`` / ``use_original_dgt`` -- the equivariant-coordinate-update
  variants (``CondEquiUpdate``, ``CoorsNorm``, ``dist_gbf2``). Dead for the
  released weights, which set both False.
* ``use_llm`` / ``llm_cond`` / ``delta_train`` -- MoLlama hidden states as extra
  node features (``ExtendedProjector``, ``projector``). A different use of the
  LM than the de-novo pipeline, and the path with no published weights.
* ``context`` property conditioning (``cond_mlp`` / ``cond_lin`` and the EGNN
  MAE classifier). Note these two submodules are built unconditionally upstream
  but are **absent from the released checkpoints**, so keeping them here would
  make every conversion report bogus missing keys.
* ``torch.compile`` decorators and the AMD-GPU probe
  (``torch.cuda.get_device_name(0)`` at import time, which raises on a CPU-only
  box). Pure speed, no numerics.

The module tree below is otherwise **name-for-name** with upstream, which is
what makes checkpoint conversion a plain ``diffusion_model.`` -> ``net.`` prefix
swap with zero dropped tensors.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
import torch_geometric.utils as pyg_utils
from torch import Tensor, nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax
from torch_scatter import scatter

__all__ = [
    "DGTDiffusion",
    "coord2dist",
    "get_align_noise",
    "kabsch_batch",
    "remove_mean",
    "sample_com_rand_pos",
]


def coord2dist(x: Tensor, edge_index: Tensor) -> Tensor:
    """Squared interatomic distance per edge -- upstream's edge distance feat."""
    row, col = edge_index
    coord_diff = x[row] - x[col]
    return torch.sum(coord_diff**2, 1).unsqueeze(1)


def remove_mean(pos: Tensor, batch: Tensor) -> Tensor:
    """Subtract the per-molecule centre of mass from a flat ``(sum N_i, 3)``."""
    mean_pos = scatter(pos, batch, dim=0, reduce="mean")
    return pos - mean_pos[batch]


def remove_mean_with_mask(x: Tensor, node_mask: Tensor, return_mean: bool = False):
    """Dense ``(B, N, 3)`` version, masked. Used by the Kabsch alignment."""
    n = node_mask.sum(1, keepdims=True)
    mean = torch.sum(x, dim=1, keepdim=True) / n
    x = x - mean * node_mask
    if return_mean:
        return x, mean
    return x


def sample_com_rand_pos(pos_shape, batch: Tensor) -> Tensor:
    """COM-free Gaussian noise (``diffusion_data_module.py:23``)."""
    noise = torch.randn(pos_shape, device=batch.device)
    return noise - scatter(noise, batch, dim=0, reduce="mean")[batch]


@torch.no_grad()
def kabsch_batch(coords_pred: Tensor, coords_tar: Tensor) -> Tensor:
    """Batched Kabsch rotation (``model/diffusion_pl.py:107``)."""
    a = torch.einsum("...ki, ...kj -> ...ij", coords_pred, coords_tar).to(torch.float32)
    u, _s, vt = torch.linalg.svd(a)
    sign_det = torch.sign(torch.det(a))
    corr_diag = torch.ones((a.size(0), u.size(-1)), device=a.device)
    corr_diag[:, -1] = sign_det
    corr = torch.diag_embed(corr_diag)
    return torch.einsum("...ij, ...jk, ...kl -> ...il", u, corr, vt)


@torch.no_grad()
def get_align_noise(  # noqa: PLR0913
    pos_t: Tensor,
    pos_0: Tensor,
    alpha_t: Tensor,
    sigma_t: Tensor,
    batch_mask: Tensor | None = None,
    translation_correction: bool = False,
) -> Tensor:
    """Rotation-aligned epsilon target (``model/diffusion_pl.py:136``).

    ``align_prediction`` is not ported: it is False for every released
    checkpoint (``get_noise_loss`` is only ever called with the default).
    """
    if translation_correction:
        mask = batch_mask.unsqueeze(-1)
        pos_0_c, _ = remove_mean_with_mask(pos_0, mask, return_mean=True)
        pos_t_c, pos_t_mean = remove_mean_with_mask(pos_t, mask, return_mean=True)
        rotations = kabsch_batch(pos_t_c, pos_0_c)
        align_pos_0 = (
            torch.einsum("...ki, ...ji -> ...jk", rotations, pos_0_c) + pos_t_mean
        )
    else:
        rotations = kabsch_batch(pos_t, pos_0)
        align_pos_0 = torch.einsum("...ki, ...ji -> ...jk", rotations, pos_0)
    return (pos_t - alpha_t * align_pos_0) / sigma_t


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class LearnedSinusodialposEmb(nn.Module):
    """Learned Fourier features of the scalar time/noise level."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0  # noqa: S101
        self.weights = nn.Parameter(torch.randn(dim // 2))

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(-1)
        freqs = x * self.weights.unsqueeze(0) * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return torch.cat((x, fouriered), dim=-1)


@torch.jit.script
def _gaussian(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    a = (2 * 3.14159) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    """Gaussian radial basis over the (squared) distance, plus the raw value."""

    def __init__(self, k: int) -> None:
        super().__init__()
        self.K = k - 1
        self.means = nn.Embedding(1, self.K)
        self.stds = nn.Embedding(1, self.K)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)

    def forward(self, x: Tensor) -> Tensor:
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return torch.cat([x, _gaussian(x, mean, std).type_as(self.means.weight)], dim=-1)


class TransLayer(MessagePassing):
    """Edge-conditioned multiplicative attention. No FFN, no norm (those live
    in :class:`EquivariantBlock`).
    """

    _alpha: OptTensor

    def __init__(  # noqa: PLR0913
        self,
        x_channels: int,
        out_channels: int,
        heads: int = 1,
        dropout: float = 0.0,
        edge_dim: int | None = None,
        bias: bool = True,
        **kwargs,
    ) -> None:
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)
        self.x_channels = x_channels
        self.in_channels = x_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lin_key = nn.Linear(x_channels, heads * out_channels, bias=bias)
        self.lin_query = nn.Linear(x_channels, heads * out_channels, bias=bias)
        self.lin_value = nn.Linear(x_channels, heads * out_channels, bias=bias)
        self.lin_edge0 = nn.Linear(edge_dim, heads * out_channels, bias=False)
        self.lin_edge1 = nn.Linear(edge_dim, heads * out_channels, bias=False)
        self.proj = nn.Linear(heads * out_channels, heads * out_channels, bias=bias)

    def forward(
        self, x: OptTensor, edge_index: Adj, edge_attr: OptTensor = None
    ) -> Tensor:
        h, c = self.heads, self.out_channels
        query = self.lin_query(x).view(-1, h, c)
        key = self.lin_key(x).view(-1, h, c)
        value = self.lin_value(x).view(-1, h, c)
        out = self.propagate(
            edge_index, query=query, key=key, value=value, edge_attr=edge_attr, size=None
        )
        return self.proj(out.view(-1, h * c))

    def message(  # noqa: PLR0913
        self,
        query_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: int | None,
    ) -> Tensor:
        edge_attn = torch.tanh(
            self.lin_edge0(edge_attr).view(-1, self.heads, self.out_channels)
        )
        alpha = (query_i * key_j * edge_attn).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        msg = value_j * torch.tanh(
            self.lin_edge1(edge_attr).view(-1, self.heads, self.out_channels)
        )
        return msg * alpha.view(-1, self.heads, 1)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads})"
        )


class TransLayerOptim(MessagePassing):
    """``--fuse_qkv`` variant: one fused QKV projection and one fused edge
    projection. Different parameter *shapes*, so a checkpoint trained with
    ``fuse_qkv`` needs this class and one trained without needs
    :class:`TransLayer`. ``drugs_dmt_b_e2999.ckpt`` is the fused one.
    """

    def __init__(  # noqa: PLR0913
        self,
        x_channels: int,
        out_channels: int,
        heads: int = 1,
        dropout: float = 0.0,
        edge_dim: int | None = None,
        bias: bool = True,
        **kwargs,
    ) -> None:
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)
        self.x_channels = x_channels
        self.in_channels = x_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.lin_qkv = nn.Linear(x_channels, heads * out_channels * 3, bias=bias)
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels * 2, bias=False)
        self.proj = nn.Linear(heads * out_channels, heads * out_channels, bias=bias)

    def forward(
        self, x: OptTensor, edge_index: Adj, edge_attr: OptTensor = None
    ) -> Tensor:
        h, c = self.heads, self.out_channels
        query, key, value = self.lin_qkv(x).view(-1, h, 3, c).unbind(dim=2)
        out = self.propagate(
            edge_index, query=query, key=key, value=value, edge_attr=edge_attr, size=None
        )
        return self.proj(out.view(-1, h * c))

    def message(  # noqa: PLR0913
        self,
        query_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: int | None,
    ) -> Tensor:
        edge_key, edge_value = (
            torch.tanh(self.lin_edge(edge_attr))
            .view(-1, self.heads, 2, self.out_channels)
            .unbind(dim=2)
        )
        alpha = (query_i * key_j * edge_key).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return value_j * edge_value * alpha.view(-1, self.heads, 1)


class EquivariantBlock(nn.Module):
    """One DMT block: edge-conditioned attention + adaLN time modulation +
    node and edge FFNs.

    Despite the upstream name this block is **not** equivariant in the
    configuration the released weights use (``equi_pos=False``); that is why the
    random-rotation augmentation at training time is load-bearing.
    """

    def __init__(  # noqa: PLR0913
        self,
        node_dim: int,
        edge_dim: int,
        time_dim: int,
        num_heads: int,
        cond_time: bool = True,
        mlp_ratio: int = 4,
        act=nn.GELU,
        dropout: float = 0.1,
        pair_update: bool = True,
        fuse_qkv: bool = False,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.act1 = act()
        self.act2 = act()
        self.cond_time = cond_time
        self.pair_update = pair_update

        if self.pair_update:
            self.edge_emb = nn.Linear(edge_dim, edge_dim)
        else:
            self.edge_emb = nn.Sequential(
                nn.Linear(edge_dim, edge_dim * 2),
                nn.GELU(),
                nn.Linear(edge_dim * 2, edge_dim),
                nn.LayerNorm(edge_dim),
            )

        layer_cls = TransLayerOptim if fuse_qkv else TransLayer
        self.attn_mpnn = layer_cls(
            node_dim, node_dim // num_heads, num_heads, edge_dim=edge_dim, dropout=dropout
        )

        self.ff_linear1 = nn.Linear(node_dim, node_dim * mlp_ratio)
        self.ff_linear2 = nn.Linear(node_dim * mlp_ratio, node_dim)
        if pair_update:
            self.node2edge_lin = nn.Linear(node_dim, edge_dim)
        self.ff_linear3 = nn.Linear(edge_dim, edge_dim * mlp_ratio)
        self.ff_linear4 = nn.Linear(edge_dim * mlp_ratio, edge_dim)

        if self.cond_time:
            self.node_time_mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(time_dim, node_dim * 6)
            )
            self.norm1_node = nn.LayerNorm(node_dim, elementwise_affine=False, eps=1e-6)
            self.norm2_node = nn.LayerNorm(node_dim, elementwise_affine=False, eps=1e-6)
            if self.pair_update:
                self.edge_time_mlp = nn.Sequential(
                    nn.SiLU(), nn.Linear(time_dim, edge_dim * 6)
                )
                self.norm1_edge = nn.LayerNorm(
                    edge_dim, elementwise_affine=False, eps=1e-6
                )
            self.norm2_edge = nn.LayerNorm(edge_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1_node = nn.LayerNorm(node_dim, elementwise_affine=True, eps=1e-6)
            self.norm2_node = nn.LayerNorm(node_dim, elementwise_affine=True, eps=1e-6)
            if self.pair_update:
                self.norm1_edge = nn.LayerNorm(
                    edge_dim, elementwise_affine=True, eps=1e-6
                )
            self.norm2_edge = nn.LayerNorm(edge_dim, elementwise_affine=True, eps=1e-6)

    def _ff_block_node(self, x: Tensor) -> Tensor:
        x = F.dropout(self.act1(self.ff_linear1(x)), p=self.dropout, training=self.training)
        return F.dropout(self.ff_linear2(x), p=self.dropout, training=self.training)

    def _ff_block_edge(self, x: Tensor) -> Tensor:
        x = F.dropout(self.act2(self.ff_linear3(x)), p=self.dropout, training=self.training)
        return F.dropout(self.ff_linear4(x), p=self.dropout, training=self.training)

    def forward(  # noqa: PLR0913
        self,
        pos: Tensor,
        h: Tensor,
        edge_attr: Tensor,
        edge_index: Tensor,
        node_mask: Tensor,
        node_time_emb: Tensor | None = None,
        edge_time_emb: Tensor | None = None,
    ):
        h_in_node, h_in_edge = h, edge_attr
        edge_attr = self.edge_emb(edge_attr)

        edge_gate_msa = edge_shift_mlp = edge_scale_mlp = edge_gate_mlp = None
        node_gate_msa = node_shift_mlp = node_scale_mlp = node_gate_mlp = None
        if self.cond_time:
            (
                node_shift_msa,
                node_scale_msa,
                node_gate_msa,
                node_shift_mlp,
                node_scale_mlp,
                node_gate_mlp,
            ) = self.node_time_mlp(node_time_emb).chunk(6, dim=1)
            h = modulate(self.norm1_node(h), node_shift_msa, node_scale_msa)
            if self.pair_update:
                (
                    edge_shift_msa,
                    edge_scale_msa,
                    edge_gate_msa,
                    edge_shift_mlp,
                    edge_scale_mlp,
                    edge_gate_mlp,
                ) = self.edge_time_mlp(edge_time_emb).chunk(6, dim=1)
                edge_attr = modulate(
                    self.norm1_edge(edge_attr), edge_shift_msa, edge_scale_msa
                )
        else:
            h = self.norm1_node(h)
            if self.pair_update:
                edge_attr = self.norm1_edge(edge_attr)

        h_node = self.attn_mpnn(h, edge_index, edge_attr)
        h_out = self._node_update(
            h_in_node,
            h_node,
            node_gate_msa,
            node_shift_mlp,
            node_scale_mlp,
            node_gate_mlp,
            node_mask,
        )
        if self.pair_update:
            h_edge = h_node[edge_index[0]] + h_node[edge_index[1]]
            h_edge_out = self._edge_update(
                h_in_edge,
                h_edge,
                edge_gate_msa,
                edge_shift_mlp,
                edge_scale_mlp,
                edge_gate_mlp,
            )
        else:
            h_edge_out = h_in_edge
        return h_out, h_edge_out, pos

    def _node_update(  # noqa: PLR0913
        self,
        h_in_node,
        h_node,
        node_gate_msa,
        node_shift_mlp,
        node_scale_mlp,
        node_gate_mlp,
        node_mask,
    ):
        if self.cond_time:
            h_node = h_in_node + node_gate_msa * h_node
            _h = modulate(self.norm2_node(h_node), node_shift_mlp, node_scale_mlp)
            _h = _h * node_mask
            return (h_node + node_gate_mlp * self._ff_block_node(_h)) * node_mask
        h_node = h_in_node + h_node
        _h = self.norm2_node(h_node) * node_mask
        return (h_node + self._ff_block_node(_h)) * node_mask

    def _edge_update(  # noqa: PLR0913
        self,
        h_in_edge,
        h_edge,
        edge_gate_msa,
        edge_shift_mlp,
        edge_scale_mlp,
        edge_gate_mlp,
    ):
        h_edge = self.node2edge_lin(h_edge)
        if self.cond_time:
            h_edge = h_in_edge + edge_gate_msa * h_edge
            _h = modulate(self.norm2_edge(h_edge), edge_shift_mlp, edge_scale_mlp)
            return h_edge + edge_gate_mlp * self._ff_block_edge(_h)
        h_edge = h_in_edge + h_edge
        return h_edge + self._ff_block_edge(self.norm2_edge(h_edge))


class DGTDiffusion(nn.Module):
    """Predicts the position noise (and the implied clean positions).

    ``forward(data) -> (pred_pos, pred_noise)``, where ``data`` is a PyG
    ``Batch`` carrying ``x, pos, edge_index, edge_attr, batch, smiles,
    max_seqlen, t_cond, alpha_t, sigma_t``.

    Sizes: DMT-B = ``hidden_size 512, n_blocks 10``; DMT-L = ``768 / 12``.
    """

    def __init__(  # noqa: PLR0913
        self,
        in_node_features: int = 44,
        in_edge_features: int = 4,
        hidden_size: int = 512,
        n_blocks: int = 10,
        n_heads: int = 8,
        dropout: float = 0.1,
        mlp_ratio: int = 4,
        disable_com: bool = True,
        not_pair_update: bool = False,
        fuse_qkv: bool = False,
    ) -> None:
        super().__init__()
        self.disable_com = disable_com
        self.pair_update = not not_pair_update
        self.n_blocks = n_blocks

        time_dim = hidden_dim = hidden_size
        edge_dim = hidden_dim // 4

        learned_dim = 16
        self.time_mlp = nn.Sequential(
            LearnedSinusodialposEmb(learned_dim),
            nn.Linear(learned_dim + 1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.dist_gbf = GaussianLayer(edge_dim)
        self.node_emb = nn.Sequential(
            nn.Linear(in_node_features + 3, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.edge_emb = nn.Linear(in_edge_features + edge_dim, edge_dim)
        for i in range(n_blocks):
            self.add_module(
                f"block_{i}",
                EquivariantBlock(
                    hidden_dim,
                    edge_dim,
                    time_dim,
                    n_heads,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                    act=nn.GELU,
                    pair_update=self.pair_update,
                    fuse_qkv=fuse_qkv,
                ),
            )
        self.final_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3, bias=False),
        )

    def forward(self, data):
        bs = len(data["smiles"])
        max_n = data.max_seqlen

        x = torch.cat((data.x, data.pos, data.t_cond.reshape(-1, 1)), dim=-1)
        dense_x, node_mask = pyg_utils.to_dense_batch(
            x, data.batch, batch_size=bs, max_num_nodes=max_n
        )
        # node_h keeps the coordinates: node_emb's input width is
        # in_node_features + 3 (47 for QM9), NOT in_node_features. Slicing this
        # `:-1` off by three columns silently changes the model.
        node_h, pos, t_cond = dense_x[:, :, :-1], dense_x[:, :, -4:-1], dense_x[:, :, -1]
        edge_h = pyg_utils.to_dense_adj(
            data.edge_index,
            data.batch,
            data.edge_attr,
            batch_size=bs,
            max_num_nodes=max_n,
        )

        # Fully connected node-pair graph INCLUDING self-loops. Non-bonded pairs
        # arrive as an all-zero 4-vector -- that is the model's implicit
        # "no bond", and why canonical class 0 is never materialized.
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        bs, n_nodes = node_mask.size()
        edge_h = edge_h[edge_mask.nonzero(as_tuple=True)]
        edge_index, _ = pyg_utils.dense_to_sparse(edge_mask)

        time_emb = self.time_mlp(t_cond[:, 0])
        node_time_emb = (
            time_emb.unsqueeze(1).expand(bs, n_nodes, -1).reshape(bs * n_nodes, -1)
        )
        edge_batch_id = torch.div(edge_index[0], n_nodes, rounding_mode="floor")
        edge_time_emb = time_emb[edge_batch_id]

        pos = pos.reshape(bs * n_nodes, -1)
        edge_h = torch.cat([edge_h, self.dist_gbf(coord2dist(pos, edge_index))], dim=-1)

        node_h = self.node_emb(node_h).reshape(bs * n_nodes, -1)
        edge_h = self.edge_emb(edge_h)

        for i in range(self.n_blocks):
            node_h, edge_h, pos = self._modules[f"block_{i}"](
                pos,
                node_h,
                edge_h,
                edge_index,
                node_mask.reshape(-1, 1),
                node_time_emb,
                edge_time_emb,
            )

        pred_noise = self.final_linear(node_h)
        pred_noise = pred_noise.reshape(bs * n_nodes, -1)[node_mask.reshape(-1)]
        if not self.disable_com:
            pred_noise = remove_mean(pred_noise, data.batch)
        pred_pos = (data.pos - pred_noise.detach() * data.sigma_t) / data.alpha_t
        return pred_pos, pred_noise
