"""DiT blocks for ChefNMR (MIT, (c) 2025 Ziyu Xiong).

Upstream: ``src/model/modules/layers.py``, modified from DiT
(facebookresearch/DiT) and timm's ``VisionTransformer``.

Kept in ``modules/models/chefnmr/`` rather than promoted to
``modules/layers/``: these blocks carry an explicit **atom mask** through
attention and use adaLN-zero conditioning on a fused
``time + spectrum`` vector. Nothing else in the tree consumes that shape
today, and a shared block would have to grow the mask argument anyway.
"""

from __future__ import annotations

import collections.abc
import math
from functools import partial
from itertools import repeat
from typing import Type

import torch
import torch.nn.functional as F
from torch import nn


class TimestepEmbedder(nn.Module):
    """Sinusoidal noise-level embedding + 2-layer MLP."""

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
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class Attention(nn.Module):
    """Masked multi-head self-attention over the atom axis.

    ``fused_attn`` is hard-off, exactly as upstream: the released weights
    were trained through the explicit softmax path, and
    ``scaled_dot_product_attention`` differs in how it handles a fully
    masked row.
    """

    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            msg = "dim should be divisible by num_heads"
            raise ValueError(msg)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = False

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        attn_mask = mask.unsqueeze(1).unsqueeze(2).to(torch.bool)  # (B, 1, 1, N)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                attn_mask=attn_mask,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


def _ntuple(n: int):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    """timm's ViT MLP."""

    def __init__(  # noqa: PLR0913
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias: bool = True,
        drop: float = 0.0,
        use_conv: bool = False,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        return self.drop2(self.fc2(self.norm(x)))


def modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        mlp_ratio: float = 4,
        **block_kwargs,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            n_heads,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
            **block_kwargs,
        )
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, atom_mask: torch.Tensor
    ) -> torch.Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), atom_mask
        )
        return x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )


class FinalLayer(nn.Module):
    """adaLN-zero output head: hidden -> coordinate update."""

    def __init__(self, hidden_size: int, out_atom_coords_size: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_atom_coords_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))
