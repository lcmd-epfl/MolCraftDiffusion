"""LoQI / Megalodon ``MegaFNV3Conf`` backbone.

Ported from ``others/LoQI/src/megalodon/dynamics/fn_model.py`` (NVIDIA,
Apache-2.0). A 10-layer interleaved stack of ``DiTeBlock`` (DiT-with-edges,
invariant) and ``XEGNNK`` (coordinate-only EGNN, equivariant). Only the
coordinate head exists: this model denoises ``x`` against a *fixed* conditioning
graph (atom types, formal charges, bond orders, stereo edges), so upstream's
``MegaFNV3`` atom/edge prediction heads and ``BondRefine`` are not ported --
``MegaFNV3Conf`` never instantiates them, and porting dead modules would only
add state-dict keys the released weights do not have.

Two deliberate deviations from upstream, both behaviour-preserving:

1. ``einops`` is replaced by plain ``unflatten``/``transpose``/``flatten``
   (upstream lines 512 and 541). ``einops`` is not installed in this
   environment, and the precedent for this substitution is
   ``modules/models/pmdm/encoders.py:15``.
2. ``BondRefine``/``MegaFNV3``/``PredictionHead`` are not ported (dead here).

Module names and construction order are load-bearing: they are the state-dict
keys the converted pretrained checkpoint is remapped onto.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.norm import LayerNorm as BatchLayerNorm
from torch_scatter import scatter, scatter_mean

NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "silu": nn.SiLU(),
    "gelu": nn.GELU(),
    "gelu_tanh": nn.GELU(approximate="tanh"),
    "sigmoid": nn.Sigmoid(),
}


class E3Norm(nn.Module):
    """Per-channel norm over the 3-vector axis, mean-normalized per molecule."""

    def __init__(self, n_vector_features: int = 1, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        if n_vector_features > 1:
            self.weight = nn.Parameter(torch.ones((1, 1, n_vector_features)))
        else:
            self.weight = nn.Parameter(torch.ones((1, 1)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def forward(self, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(pos, dim=1, keepdim=True)
        batch_size = int(batch.max()) + 1
        mean_norm = scatter_mean(norm, batch, dim=0, dim_size=batch_size)
        return self.weight * pos / (mean_norm[batch] + self.eps)


class MLP(nn.Module):
    """Upstream's ``fn_model.MLP`` (note: the ``bias`` flag, unlike the copy in
    ``dynamics/utils.py``)."""

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        hidden_size: int,
        output_dim: int,
        num_hidden_layers: int = 0,
        activation: str = "silu",
        dropout: float = 0.0,
        last_act: str | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if activation not in NONLINEARITIES:
            msg = f"Activation must be one of {list(NONLINEARITIES)}"
            raise ValueError(msg)
        self.act_layer = NONLINEARITIES[activation]

        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_size, bias=bias),
            NONLINEARITIES[activation],
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size, bias=bias))
            layers.append(NONLINEARITIES[activation])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size, output_dim, bias=bias))
        if last_act:
            layers.append(NONLINEARITIES[last_act])
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding + 2-layer MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], -1)
        return embedding

    def forward(self, t: torch.Tensor, batch=None) -> torch.Tensor:  # noqa: ARG002
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    """Nearest multiple of 256 after the expansion ratio (from ESM3)."""
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def swiglu_ffn(d_model: int, expansion_ratio: float, bias: bool) -> nn.Sequential:
    hidden = swiglu_correction_fn(expansion_ratio, d_model)
    return nn.Sequential(
        nn.Linear(d_model, hidden * 2, bias=bias),
        SwiGLU(),
        nn.Linear(hidden, d_model, bias=bias),
    )


def swiglu_ffn_edge(d_model: int, bias: bool) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, d_model * 2, bias=bias),
        SwiGLU(),
        nn.Linear(d_model, d_model, bias=bias),
    )


class XEGNNK(nn.Module):
    """Coordinate-only EGNN update (``X`` in, ``X`` out), with a cross-product
    term. Node features are read but never written."""

    def __init__(  # noqa: PLR0913
        self,
        invariant_node_feat_dim: int = 64,
        invariant_edge_feat_dim: int = 64,
        n_vector_features: int = 128,
        dist_size: int = 4,
        prune_edges: bool = False,
    ) -> None:
        super().__init__()
        self.h_projection = nn.Sequential(
            nn.Linear(invariant_node_feat_dim, invariant_edge_feat_dim), nn.SiLU()
        )
        self.coord_projection = nn.Linear(n_vector_features, dist_size)
        self.message_input_size = 4 * invariant_edge_feat_dim + dist_size
        self.phi_message = MLP(
            self.message_input_size, invariant_edge_feat_dim, invariant_edge_feat_dim
        )
        self.phi_x = MLP(
            invariant_edge_feat_dim, invariant_edge_feat_dim, n_vector_features
        )
        self.coor_update_clamp_value = 10.0
        self.h_norm = BatchLayerNorm(invariant_edge_feat_dim)
        self.use_cross_product = True
        self.phi_x_cross = MLP(
            invariant_edge_feat_dim, invariant_edge_feat_dim, n_vector_features
        )
        self.x_norm = E3Norm(n_vector_features)
        self.prune_edges = prune_edges

    def forward(  # noqa: PLR0913
        self, batch, X, H, edge_index, edge_attr=None, te=None
    ):
        X = X - scatter_mean(X, index=batch, dim=0, dim_size=X.shape[0])[batch]
        X = self.x_norm(X, batch)
        H = self.h_projection(H)
        H = self.h_norm(H, batch)
        source, target = edge_index
        rel_coors = X[source] - X[target]
        rel_dist = (rel_coors.transpose(1, 2) ** 2).sum(dim=-1, keepdim=False)
        if self.prune_edges:
            test = scatter_mean(rel_dist.sum(-1), batch[source])
            edge_cut_mask = rel_dist.sum(-1) < test[batch[source]] / 2
            edge_index = edge_index[:, edge_cut_mask]
            source, target = edge_index
            rel_coors = X[source] - X[target]
            rel_dist = (rel_coors.transpose(1, 2) ** 2).sum(dim=-1, keepdim=False)
            edge_attr = edge_attr[edge_cut_mask]

        dist_coord = self.coord_projection(X)
        dist_rel_coords = dist_coord[source] - dist_coord[target]
        rel_dist_feat = (dist_rel_coords.transpose(1, 2) ** 2).sum(-1, keepdim=False)
        if edge_attr is not None:
            edge_attr_feat = torch.cat([edge_attr, rel_dist_feat], dim=-1)
        else:
            edge_attr_feat = rel_dist_feat

        m_ij = self.phi_message(
            torch.cat([H[target], H[source], edge_attr_feat, te[batch[source]]], -1)
        )
        coor_wij = self.phi_x(m_ij)
        if self.coor_update_clamp_value:
            coor_wij.clamp_(
                min=-self.coor_update_clamp_value, max=self.coor_update_clamp_value
            )
        X_rel_norm = rel_coors / (1 + torch.sqrt(rel_dist.unsqueeze(1) + 1e-8))
        x_update = scatter(
            X_rel_norm * coor_wij.unsqueeze(1),
            index=target,
            dim=0,
            reduce="sum",
            dim_size=X.shape[0],
        )
        X_out = X + x_update

        if self.use_cross_product:
            mean = scatter(X, index=batch, dim=0, reduce="mean", dim_size=X.shape[0])
            x_src = X[source] - mean[source]
            x_tgt = X[target] - mean[target]
            cross = torch.cross(x_src, x_tgt, dim=1)
            cross = cross / (1 + torch.linalg.norm(cross, dim=1, keepdim=True))
            coor_wij_cross = self.phi_x_cross(m_ij)
            if self.coor_update_clamp_value:
                coor_wij_cross.clamp_(
                    min=-self.coor_update_clamp_value,
                    max=self.coor_update_clamp_value,
                )
            X_out = X_out + scatter(
                cross * coor_wij_cross.unsqueeze(1),
                index=target,
                dim=0,
                reduce="sum",
                dim_size=X.shape[0],
            )
        return X_out


def coord2distfn(x, edge_index, scale_dist_features=1, batch=None):  # noqa: ARG001
    """Pairwise squared distances per vector channel, plus optional extras."""
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff**2, 1)
    if scale_dist_features >= 2:
        dotproduct = (x[row] * x[col]).sum(dim=-2, keepdim=False)
        radial = torch.cat([radial, dotproduct], dim=-1)
    if scale_dist_features == 4:
        p_i, p_j = x[edge_index[0]], x[edge_index[1]]
        d_i = torch.pow(p_i, 2).sum(-2, keepdim=False).clamp(min=1e-6).sqrt()
        d_j = torch.pow(p_j, 2).sum(-2, keepdim=False).clamp(min=1e-6).sqrt()
        radial = torch.cat([radial, d_i, d_j], dim=-1)
    return radial


class DiTeBlock(nn.Module):
    """DiT block with an edge channel, PyG-batched (batch dim of 1 + attn mask)."""

    def __init__(  # noqa: PLR0913
        self,
        hidden_size: int,
        edge_hidden_size: int,
        num_heads: int,
        mlp_expansion_ratio: float = 4.0,
        use_z: bool = True,
        mask_z: bool = True,
        use_rotary: bool = False,
        n_vector_features: int = 128,  # noqa: ARG002 - kept for signature parity
        dist_size: int = 128,
        **block_kwargs,  # noqa: ARG002
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.edge_hidden_size = edge_hidden_size
        self.num_heads = num_heads
        self.norm1 = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.norm2 = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.feature_embedder = MLP(
            hidden_size + hidden_size + edge_hidden_size + dist_size,
            hidden_size,
            hidden_size,
        )
        self.norm1_edge = BatchLayerNorm(edge_hidden_size, affine=False, eps=1e-6)
        self.norm2_edge = BatchLayerNorm(edge_hidden_size, affine=False, eps=1e-6)

        self.ffn_norm = BatchLayerNorm(hidden_size)
        self.ffn = swiglu_ffn(hidden_size, mlp_expansion_ratio, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.adaLN_edge_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(edge_hidden_size, 6 * edge_hidden_size, bias=True)
        )

        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.norm_q = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.norm_k = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.out_projection = nn.Linear(hidden_size, hidden_size, bias=False)

        self.use_rotary = use_rotary
        self.d_head = hidden_size // num_heads

        if use_z:
            self.use_z = use_z
            self.pair_bias = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 1, bias=False)
            )
            self.mask_z = mask_z

        self.lin_edge0 = nn.Linear(hidden_size, edge_hidden_size, bias=False)
        # Upstream assigns lin_edge1 twice; the second assignment is the one
        # that survives, and is what the released weights contain.
        self.lin_edge1 = nn.Linear(
            edge_hidden_size + dist_size, edge_hidden_size, bias=False
        )
        self.ffn_norm_edge = BatchLayerNorm(edge_hidden_size)
        self.ffn_edge = swiglu_ffn_edge(edge_hidden_size, bias=False)

    def _heads(self, t: torch.Tensor) -> torch.Tensor:
        """``(b, s, h*d) -> (b, h, s, d)``. Plain-tensor form of upstream's
        ``einops.rearrange(t, "b s (h d) -> b h s d", h=num_heads)``."""
        return t.unflatten(-1, (self.num_heads, self.d_head)).transpose(1, 2)

    def forward(  # noqa: PLR0913
        self,
        batch: torch.Tensor,
        x: torch.Tensor,
        t_emb_h: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        t_emb_e: torch.Tensor | None = None,
        dist: torch.Tensor | None = None,
        edge_batch: torch.Tensor | None = None,
        Z: torch.Tensor | None = None,
    ):
        src, tgt = edge_index
        if Z is not None:
            assert self.use_z  # noqa: S101

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(t_emb_h)[batch].chunk(6, dim=1)
        (
            edge_shift_msa,
            edge_scale_msa,
            edge_gate_msa,
            edge_shift_mlp,
            edge_scale_mlp,
            edge_gate_mlp,
        ) = self.adaLN_edge_modulation(t_emb_e)[batch[src]].chunk(6, dim=1)

        x_norm = modulate(self.norm1(x, batch), shift_msa, scale_msa)
        edge_attr_norm = modulate(
            self.norm1_edge(edge_attr, edge_batch), edge_shift_msa, edge_scale_msa
        )
        messages = self.feature_embedder(
            torch.cat([x_norm[src], x_norm[tgt], edge_attr_norm, dist], dim=-1)
        )
        x_norm = scatter_mean(messages, src, dim=0)

        qkv = self.qkv_proj(x_norm)
        Q, K, V = qkv.chunk(3, dim=-1)
        Q, K = self.norm_q(Q, batch), self.norm_k(K, batch)
        if x.dim() == 2:
            Q, K, V = Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0)
            self.use_rotary = False

        Q, K, V = self._heads(Q), self._heads(K), self._heads(V)

        if x.dim() == 2:
            attn_mask = batch.unsqueeze(0) == batch.unsqueeze(1)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        else:
            attn_mask = batch

        if Z is not None:
            if x.dim() != 2:
                msg = "Batch-wise pair embedding update is not implemented"
                raise ValueError(msg)
            mask = torch.ones((x.size(0), x.size(0)))
            if self.mask_z:
                mask.fill_diagonal_(0)
            attn_mask = attn_mask.float()
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf"))
            attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
            bias = (self.pair_bias(Z).squeeze(-1) * mask).unsqueeze(0).unsqueeze(0)
            attn_mask = attn_mask + bias

        attn_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
        # einops.rearrange(attn_output, "b h s d -> b s (h d)")
        attn_output = attn_output.transpose(1, 2).flatten(-2).squeeze(0)
        y = self.out_projection(attn_output)

        x = x + gate_msa * y
        edge = edge_attr + edge_gate_msa * self.lin_edge0((y[src] + y[tgt]))
        x = x + gate_mlp * self.ffn(
            self.ffn_norm(modulate(self.norm2(x, batch), shift_mlp, scale_mlp), batch)
        )
        e_in = self.lin_edge1(torch.cat([edge, dist], dim=-1))
        edge_attr = edge + edge_gate_mlp * self.ffn_edge(
            self.ffn_norm_edge(
                modulate(
                    self.norm2_edge(e_in, edge_batch), edge_shift_mlp, edge_scale_mlp
                ),
                edge_batch,
            )
        )
        return x, edge_attr


class MegaFNV3Conf(nn.Module):
    """The conformer backbone: coordinates out, conditioning graph in.

    ``forward(batch, X, H, E_idx, E, t) -> {"x_hat": (N,3), "H": (N,D)}``,
    with ``x_hat`` centre-of-mass free.
    """

    def __init__(  # noqa: PLR0913
        self,
        num_layers: int = 8,
        equivariant_node_feature_dim: int = 3,  # noqa: ARG002 - config parity
        invariant_node_feat_dim: int = 256,
        invariant_edge_feat_dim: int = 256,
        atom_classes: int = 16,
        edge_classes: int = 5,
        num_heads: int = 16,
        n_vector_features: int = 128,
        scale_dist_features: int = 4,
        dist_size: int = 4,
        prune_edges: bool = False,
    ) -> None:
        super().__init__()
        self.scale_dist_features = scale_dist_features
        self.atom_embedder = MLP(
            atom_classes, invariant_node_feat_dim, invariant_node_feat_dim
        )
        self.edge_embedder = MLP(
            edge_classes, invariant_edge_feat_dim, invariant_edge_feat_dim
        )
        self.num_atom_classes = atom_classes
        self.num_edge_classes = edge_classes
        self.n_vector_features = n_vector_features
        self.coord_emb = nn.Linear(1, n_vector_features, bias=False)
        self.coord_pred = nn.Linear(n_vector_features, 1, bias=False)
        self.node_time_embedding = TimestepEmbedder(invariant_node_feat_dim)
        self.edge_time_embedding = TimestepEmbedder(invariant_edge_feat_dim)
        self.dit_layers = nn.ModuleList()
        self.egnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.dit_layers.append(
                DiTeBlock(
                    invariant_node_feat_dim,
                    invariant_edge_feat_dim,
                    num_heads,
                    use_z=False,
                    dist_size=scale_dist_features * dist_size,
                    n_vector_features=n_vector_features,
                )
            )
            self.egnn_layers.append(
                XEGNNK(
                    invariant_node_feat_dim,
                    invariant_edge_feat_dim,
                    n_vector_features=n_vector_features,
                    dist_size=dist_size,
                    prune_edges=prune_edges,
                )
            )
        self.dist_projection = nn.Linear(n_vector_features, dist_size, bias=False)

    def forward(self, batch, X, H, E_idx, E, t):  # noqa: PLR0913
        pos = self.coord_emb(X.unsqueeze(-1))  # N x 3 x K

        H = self.atom_embedder(H)
        E = self.edge_embedder(E)
        edge_batch = batch[E_idx[0]]
        te_h = self.node_time_embedding(t)
        te_e = self.edge_time_embedding(t)
        edge_attr = E

        for layer_index in range(len(self.dit_layers)):
            proj_pos = self.dist_projection(pos)
            distances = coord2distfn(proj_pos, E_idx, self.scale_dist_features, batch)
            H, edge_attr = self.dit_layers[layer_index](
                batch, H, te_h, edge_attr, E_idx, te_e, distances, edge_batch
            )
            pos = self.egnn_layers[layer_index](batch, pos, H, E_idx, edge_attr, te_e)

        X = self.coord_pred(pos).squeeze(-1)
        x = X - scatter_mean(X, index=batch, dim=0)[batch]
        return {"x_hat": x, "H": H}
