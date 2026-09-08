"""Dense self-attention adapter binding a plain (non-equivariant)
Transformer to the ``dynamics._forward(t, xh, node_mask, edge_mask,
context)`` contract that ``EnVariationalDiffusion.phi``
(``modules/models/en_diffusion.py:234``) calls.

Novel-model ablation (``docs/model_novel/diffusion_transformer/
INTEGRATION_PLAN.md``): swaps EGCL's equivariant pairwise-difference
message passing (``modules/models/egcl.py::EGNN_dynamics``) for plain
multi-head self-attention over absolute coordinates, with no distance/edge
features and no built-in rotation-equivariance -- the hypothesis is that
the platform's existing rotation-augmentation flag
(``GeomMolecularGenerative.data_augmentation``) substitutes for the
architectural guarantee. Reuses the platform's TABASCO-family attention
layers by import rather than copying them: unlike ``PaiNNDynamics``
(``modules/models/painn_dynamics.py``), no dense<->flat packing is needed
at all, since ``Transformer`` already speaks dense ``(B, N, dim)``.
"""

from __future__ import annotations

import torch
from torch import nn

from MolecularDiffusion.modules.layers.tabasco.positional_encoder import (
    SinusoidEncoding,
    TimeFourierEncoding,
)
from MolecularDiffusion.modules.layers.tabasco.transformer import (
    Transformer,
)
from MolecularDiffusion.utils.geom_utils import remove_mean_with_mask_v2


class TransformerDynamics(nn.Module):
    """Denoising network: plain multi-head self-attention, EDM interface.

    Args:
        in_node_nf: Node feature channels the diffusion model expects
            back (atom-type one-hot + atomic number [+ extra values]),
            excluding time and context, which are added internally.
        context_node_nf: Conditioning channels, concatenated to the node
            features before embedding.
        n_dims: Spatial dimensions (3).
        hidden_dim: Transformer token width.
        num_layers: Transformer block depth (``Transformer``'s ``depth``).
        num_heads: Multi-head self-attention heads.
        mlp_dim: Feed-forward hidden width; ``None`` defaults to
            ``4 * hidden_dim`` inside ``Transformer``.
        dropout: Dropout probability.
        activation_type: ``Transformer``'s feed-forward activation
            string knob (e.g. ``"gelu"``).
        add_sinusoid_posenc: Ablation-only knob, off by default. Atoms
            are an unordered set here, so this has no principled reason
            to help; it exists only for later ablation curiosity and is
            never load-bearing.
    """

    def __init__(
        self,
        in_node_nf: int,
        context_node_nf: int = 0,
        n_dims: int = 3,
        hidden_dim: int = 192,
        num_layers: int = 9,
        num_heads: int = 8,
        mlp_dim: int | None = None,
        dropout: float = 0.0,
        activation_type: str = "gelu",
        add_sinusoid_posenc: bool = False,
    ) -> None:
        super().__init__()
        self.in_node_nf = in_node_nf
        self.context_node_nf = context_node_nf
        self.n_dims = n_dims
        self.add_sinusoid_posenc = add_sinusoid_posenc

        self.pos_embed = nn.Linear(n_dims, hidden_dim, bias=False)
        self.feat_embed = nn.Linear(
            in_node_nf + context_node_nf, hidden_dim, bias=False
        )
        self.time_encoding = TimeFourierEncoding(hidden_dim)
        self.transformer = Transformer(
            dim=hidden_dim,
            depth=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            activation_type=activation_type,
        )
        self.out_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_dims + in_node_nf),
        )
        if add_sinusoid_posenc:
            self.sinusoid_posenc = SinusoidEncoding(hidden_dim)

    # -- EnVariationalDiffusion dynamics interface --------------------- #

    def _forward(
        self,
        t: torch.Tensor,
        xh: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,  # noqa: ARG002 -- see class docstring
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict eps for a dense padded batch.

        Args:
            t: scalar or ``(B,)``/``(B, 1)`` diffusion time in ``[0, 1]``.
            xh: ``(B, N, 3 + in_node_nf)`` noisy positions ++ features.
            node_mask: ``(B, N, 1)``, 1 = valid node, 0 = padding.
            edge_mask: unused. Every dynamics wrapper under
                ``EnVariationalDiffusion`` shares this call signature
                (``en_diffusion.py:234``); the dense ``edge_mask`` here
                is always ``node_mask`` outer-producted with itself, no
                extra information the attention ``key_padding_mask``
                built from ``node_mask`` alone doesn't already carry.
            context: ``(B, N, context_node_nf)`` or ``None``.

        Returns:
            ``(B, N, 3 + in_node_nf)``, zero on padded rows, with the
            position channels projected to the zero-CoM subspace that
            ``EnVariationalDiffusion`` asserts everywhere.
        """
        b, n, _ = xh.shape
        x = xh[..., : self.n_dims]
        h = xh[..., self.n_dims :]
        if context is not None and self.context_node_nf > 0:
            h = torch.cat([h, context], dim=-1)

        # TimeFourierEncoding asserts its own output shape (B, dim) and
        # requires a strict 1-D (B,) input -- unlike PaiNN's own
        # FourierTimeFeatures, which wants (N, 1).
        if torch.numel(t) == 1:
            t_flat = t.reshape(1).expand(b)
        else:
            t_flat = t.reshape(b)
        time_emb = self.time_encoding(t_flat).unsqueeze(1)

        tokens = self.pos_embed(x) + self.feat_embed(h) + time_emb
        if self.add_sinusoid_posenc:
            tokens = tokens + self.sinusoid_posenc(b, n)

        # This platform's node_mask is 1 = valid, but Transformer's
        # padding_mask (a thin wrapper over nn.MultiheadAttention's
        # key_padding_mask) is True = ignore -- inverted here.
        key_padding_mask = ~node_mask.squeeze(-1).bool()
        h_out = self.transformer(tokens, padding_mask=key_padding_mask)
        raw_out = self.out_head(h_out)

        # Zero padded rows BEFORE the CoM projection -- load-bearing, not
        # stylistic. remove_mean_with_mask_v2's mean is
        # sum(pos, dim=1) / num_valid_nodes; unzeroed padding would bias
        # every molecule's centroid without raising, and the corruption
        # would only surface later as assert_mean_zero_with_mask failing
        # deep inside EnVariationalDiffusion.
        out = raw_out * node_mask
        vel = (
            remove_mean_with_mask_v2(out[..., : self.n_dims], node_mask)
            * node_mask
        )

        if torch.any(torch.isnan(vel)):
            vel = torch.zeros_like(vel)
            out = torch.zeros_like(out)

        return torch.cat([vel, out[..., self.n_dims :]], dim=2)

    # Deliberately no `_forward_pyG`: its only caller
    # (`en_diffusion.py::phi_pyg`) is reached only for batches carrying a
    # "graph" key, which this model's `data_type: pointcloud` batches
    # never set (see INTEGRATION_PLAN.md's scope decision).
