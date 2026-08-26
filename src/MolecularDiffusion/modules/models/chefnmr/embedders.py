"""NMR spectrum tokenizers + encoder for ChefNMR (MIT, (c) 2025 Ziyu Xiong).

Upstream: ``src/model/modules/embedders.py`` (conv tokenizer modified from
MarklandGroup/NMR2Struct).

Ported verbatim except that upstream's three ``einops.rearrange`` calls are
inlined as plain ``reshape``/``permute`` -- ``einops`` is not installed in
this environment and these three sites are the only thing that wanted it::

    rearrange(t, "b n (h d) -> b h n d")  ->  t.reshape(b, n, h, d).permute(0, 2, 1, 3)
    rearrange(o, "b h n d -> b n (h d)")  ->  o.permute(0, 2, 1, 3).reshape(b, n, h * d)

**The ``embed`` tokenizer only works on a binary vector.** ``_embed_spectrum``
does ``x.long() * arange(1, L+1)`` into an ``nn.Embedding(L+1, D,
padding_idx=0)``: a bin holding 0 maps to the pad row, a bin holding 1 maps
to that bin's own row, and any other value indexes a *different bin's* row.
The 80-bin 13C grid is stored as strictly {0.0, 1.0}, and
``scripts/convert_dataset.py`` asserts that rather than assuming it.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torch import nn

from MolecularDiffusion.modules.models.chefnmr.utils import (
    get_1d_sincos_pos_embed_from_grid,
)


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------
class SpectraTokenizerPatch1D(nn.Module):
    """Split a 1-D spectrum into fixed patches and project them."""

    def __init__(self, patch_size: int, stride: int, hidden_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.proj = nn.Linear(patch_size, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=1, size=self.patch_size, step=self.stride)
        return self.proj(patches)

    @staticmethod
    def num_tokens(input_size: int, patch_size: int, stride: int) -> int:
        return (input_size - patch_size) // stride + 1


class SpectraTokenizerConv1D(nn.Module):
    """Conv1d/ReLU/MaxPool stack -> ``(B, T, D)`` tokens."""

    def __init__(  # noqa: PLR0913
        self,
        input_size: int,
        hidden_dim: int,
        pool_sizes: list,
        kernel_sizes: list,
        out_channels: list,
    ) -> None:
        super().__init__()
        num_layers = len(pool_sizes)
        if num_layers <= 0:
            msg = "Must specify at least one convolutional layer."
            raise ValueError(msg)
        if len(kernel_sizes) != num_layers or len(out_channels) != num_layers:
            msg = "kernel_sizes/out_channels must match pool_sizes in length."
            raise ValueError(msg)

        self.conv_blocks = nn.ModuleList()
        in_channel = 1
        for i in range(num_layers):
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channel,
                        out_channels=out_channels[i],
                        kernel_size=kernel_sizes[i],
                        stride=1,
                        padding="valid",
                    ),
                    nn.ReLU(),
                    nn.MaxPool1d(pool_sizes[i]),
                )
            )
            in_channel = out_channels[i]

        self.linear_after_conv = nn.Linear(out_channels[-1], hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            x = x.unsqueeze(dim=1)
        for block in self.conv_blocks:
            x = block(x)
        x = torch.transpose(x, 1, 2)
        return self.linear_after_conv(x)

    @staticmethod
    def _calculate_dim_after_conv(
        l_in: int, kernel: int, padding: int, dilation: int, stride: int
    ) -> int:
        numerator = l_in + (2 * padding) - (dilation * (kernel - 1)) - 1
        return math.floor((numerator / stride) + 1)

    @staticmethod
    def _calculate_dim_after_pool(  # noqa: PLR0913
        pool_variation: str,
        l_in: int,
        kernel: int,
        padding: int,
        dilation: int,
        stride: int,
    ) -> int:
        if pool_variation == "max":
            numerator = l_in + (2 * padding) - (dilation * (kernel - 1)) - 1
        else:
            numerator = l_in + (2 * padding) - kernel
        return math.floor((numerator / stride) + 1)

    @staticmethod
    def num_tokens(input_size: int, kernel_sizes: list, pool_sizes: list) -> int:
        l_current = input_size
        for conv_kernel, pool_kernel in zip(kernel_sizes, pool_sizes):
            l_current = SpectraTokenizerConv1D._calculate_dim_after_conv(
                l_in=l_current, kernel=conv_kernel, padding=0, dilation=1, stride=1
            )
            l_current = SpectraTokenizerConv1D._calculate_dim_after_pool(
                pool_variation="max",
                l_in=l_current,
                kernel=pool_kernel,
                padding=0,
                dilation=1,
                stride=pool_kernel,
            )
        return l_current


# ---------------------------------------------------------------------------
# Transformer primitives
# ---------------------------------------------------------------------------
class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class Attention(nn.Module):
    """Unmasked multi-head self-attention over spectrum tokens."""

    def __init__(self, dim: int, heads: int, dim_head: Optional[int] = None) -> None:
        super().__init__()
        if dim_head is None:
            if dim % heads != 0:
                msg = "Dimension must be divisible by number of heads"
                raise ValueError(msg)
            dim_head = dim // heads
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.attn = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        h, d = self.heads, self.dim_head
        # einops-free: "b n (h d) -> b h n d"
        q, k, v = (
            t.reshape(b, n, h, d).permute(0, 2, 1, 3)
            for t in self.to_qkv(x).chunk(3, dim=-1)
        )
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        out = torch.matmul(self.attn(dots), v)
        # einops-free: "b h n d -> b n (h d)"
        out = out.permute(0, 2, 1, 3).reshape(b, n, h * d)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoder(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        mlp_dim = dim * mlp_ratio
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
                    ]
                )
            )
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)
        return self.final_norm(x)


class AttnPoolToken(nn.Module):
    """CLS-token attention pooling to a fixed-size vector."""

    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        out_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.cls, mean=0.0, std=0.02)
        self.attn = Attention(dim, heads, dim_head)
        self.proj = nn.Sequential(
            nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls = self.cls.expand(x.size(0), -1, -1)
        y = torch.cat([cls, x], dim=1)
        return self.proj(self.attn(y)[:, 0])


# ---------------------------------------------------------------------------
# Main embedding module
# ---------------------------------------------------------------------------
class NMRSpectraEmbedder(nn.Module):
    """Concatenated 1H/13C spectra -> one ``(B, output_dim)`` vector."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        use_hnmr: bool = True,
        use_cnmr: bool = False,
        hnmr_dim: int = 10000,
        cnmr_dim: int = 10000,
        hidden_dim: int = 256,
        output_dim: int = 768,
        dropout: float = 0.1,
        pooling: str = "flatten",
        tokenizer_args: dict = None,
        transformer_args: dict = None,
    ) -> None:
        super().__init__()
        self.use_hnmr = use_hnmr
        self.use_cnmr = use_cnmr
        self.hnmr_dim = hnmr_dim
        self.cnmr_dim = cnmr_dim
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling
        self.hidden_dim = hidden_dim

        default_tokenizer_args = {
            "h_tokenizer": "conv",
            "c_tokenizer": "conv",
            "h_pool_sizes": [8, 12],
            "h_kernel_sizes": [5, 9],
            "h_out_channels": [64, 128],
            "c_pool_sizes": [8, 12],
            "c_kernel_sizes": [5, 9],
            "c_out_channels": [64, 128],
            "h_patch_size": 256,
            "h_patch_stride": 128,
            "c_patch_size": 256,
            "c_patch_stride": 128,
            "h_mask_token": False,
            "c_mask_token": False,
        }
        tokenizer_args = {**default_tokenizer_args, **(tokenizer_args or {})}

        default_transformer_args = {
            "pos_enc": "learnable",
            "type_enc": True,
            "depth": 4,
            "heads": 8,
            "dim_head": None,
            "mlp_ratio": 4,
        }
        transformer_args = {**default_transformer_args, **(transformer_args or {})}

        self.h_tokenizer = tokenizer_args["h_tokenizer"]
        self.c_tokenizer = tokenizer_args["c_tokenizer"]
        self.use_h_mask_token = tokenizer_args["h_mask_token"]
        self.use_c_mask_token = tokenizer_args["c_mask_token"]

        pos_enc = transformer_args["pos_enc"]
        type_enc = transformer_args["type_enc"]
        depth = transformer_args["depth"]
        heads = transformer_args["heads"]
        dim_head = transformer_args["dim_head"]
        mlp_ratio = transformer_args["mlp_ratio"]

        self.h_embed, self.h_token_num = None, 0
        self.c_embed, self.c_token_num = None, 0

        if use_hnmr:
            self.h_embed, self.h_token_num = self._initialize_tokenizer(
                tokenizer_type=self.h_tokenizer,
                input_dim=hnmr_dim,
                hidden_dim=hidden_dim,
                pool_sizes=tokenizer_args["h_pool_sizes"],
                kernel_sizes=tokenizer_args["h_kernel_sizes"],
                out_channels=tokenizer_args["h_out_channels"],
                patch_size=tokenizer_args["h_patch_size"],
                stride=tokenizer_args["h_patch_stride"],
            )
            if self.use_h_mask_token:
                self.h_mask_token = nn.Embedding(2, hidden_dim)
                nn.init.normal_(self.h_mask_token.weight, std=0.02)
                self.h_token_num += 1

        if use_cnmr:
            self.c_embed, self.c_token_num = self._initialize_tokenizer(
                tokenizer_type=self.c_tokenizer,
                input_dim=cnmr_dim,
                hidden_dim=hidden_dim,
                pool_sizes=tokenizer_args["c_pool_sizes"],
                kernel_sizes=tokenizer_args["c_kernel_sizes"],
                out_channels=tokenizer_args["c_out_channels"],
                patch_size=tokenizer_args["c_patch_size"],
                stride=tokenizer_args["c_patch_stride"],
            )
            if self.use_c_mask_token:
                self.c_mask_token = nn.Embedding(2, hidden_dim)
                nn.init.normal_(self.c_mask_token.weight, std=0.02)
                self.c_token_num += 1

        total_tokens = self.h_token_num + self.c_token_num
        # Upstream stores a closure in `self.pos_encode`. A local lambda on a
        # module attribute is UNPICKLABLE, and `core/engine.py:877` saves the
        # task object itself with `full_state=True` -- so the mode is stored
        # and branched on in `pos_encode` instead. Behaviour is identical and
        # the parameter names (and therefore the state_dict) are unchanged.
        self.pos_enc_mode = pos_enc
        if pos_enc == "sincos":
            self.pos_embed = nn.Parameter(
                torch.zeros(1, total_tokens, hidden_dim), requires_grad=False
            )
            pos_embed = get_1d_sincos_pos_embed_from_grid(
                self.pos_embed.shape[-1], np.arange(total_tokens)
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        elif pos_enc == "learnable":
            self.learnable_pos_embed = nn.Parameter(
                torch.zeros(1, total_tokens, hidden_dim)
            )
            nn.init.normal_(self.learnable_pos_embed, std=0.02)
        elif pos_enc is not None:
            msg = f"Unknown positional encoding: {pos_enc}"
            raise ValueError(msg)

        if use_hnmr and use_cnmr and type_enc:
            self.type_embedding = nn.Embedding(2, hidden_dim)
            nn.init.normal_(self.type_embedding.weight, std=0.02)
        else:
            self.type_embedding = None

        if depth > 0:
            self.transformer = TransformerEncoder(
                dim=hidden_dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
        else:
            self.transformer = nn.Identity()

        if pooling == "flatten":
            flatten_dim = total_tokens * hidden_dim
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.LayerNorm(flatten_dim),
                nn.Dropout(dropout),
                nn.Linear(flatten_dim, output_dim),
            )
        elif pooling == "attn":
            self.head = AttnPoolToken(
                hidden_dim, output_dim, heads=heads, dim_head=dim_head, dropout=dropout
            )
        else:
            msg = f"Unknown pooling method: {pooling}"
            raise ValueError(msg)

    def _initialize_tokenizer(
        self, tokenizer_type: str, input_dim: int, hidden_dim: int, **kwargs
    ):
        if tokenizer_type == "conv":
            embed_layer = SpectraTokenizerConv1D(
                input_size=input_dim,
                hidden_dim=hidden_dim,
                pool_sizes=kwargs["pool_sizes"],
                kernel_sizes=kwargs["kernel_sizes"],
                out_channels=kwargs["out_channels"],
            )
            num_tokens = SpectraTokenizerConv1D.num_tokens(
                input_size=input_dim,
                kernel_sizes=kwargs["kernel_sizes"],
                pool_sizes=kwargs["pool_sizes"],
            )
        elif tokenizer_type == "patch":
            embed_layer = SpectraTokenizerPatch1D(
                patch_size=kwargs["patch_size"],
                stride=kwargs["stride"],
                hidden_dim=hidden_dim,
            )
            num_tokens = SpectraTokenizerPatch1D.num_tokens(
                input_size=input_dim,
                patch_size=kwargs["patch_size"],
                stride=kwargs["stride"],
            )
        elif tokenizer_type == "embed":
            # Binary occupancy grid (e.g. the 80-bin 13C spectrum).
            embed_layer = nn.Embedding(input_dim + 1, hidden_dim, padding_idx=0)
            num_tokens = input_dim
            nn.init.normal_(embed_layer.weight, mean=0.0, std=0.02)
        else:
            msg = f"Unknown tokenizer type: {tokenizer_type}"
            raise ValueError(msg)
        return embed_layer, num_tokens

    def _embed_spectrum(
        self,
        x: torch.Tensor,
        tokenizer_type: str,
        embed_layer: nn.Module,
        input_dim: int,
    ) -> torch.Tensor:
        if tokenizer_type in ("conv", "patch"):
            return embed_layer(x)
        if tokenizer_type == "embed":
            # x must be binary: 0 -> padding row, 1 -> that bin's own row.
            indices = torch.arange(
                1, input_dim + 1, device=x.device, dtype=torch.long
            )
            return embed_layer(x.long() * indices)
        msg = f"Unknown tokenizer type '{tokenizer_type}' during embedding."
        raise ValueError(msg)

    def _embed_hnmr(self, x: torch.Tensor) -> torch.Tensor:
        if self.h_embed is None:
            msg = "HNMR embedding layer is not initialized. Check use_hnmr flag."
            raise RuntimeError(msg)
        h_embed = self._embed_spectrum(x, self.h_tokenizer, self.h_embed, self.hnmr_dim)
        if not self.use_h_mask_token:
            return h_embed
        h_missing_mask = (x == 0).all(dim=1)
        h_missing_tokens = self.h_mask_token(h_missing_mask.long()).unsqueeze(1)
        return torch.cat([h_missing_tokens, h_embed], dim=1)

    def _embed_cnmr(self, x: torch.Tensor) -> torch.Tensor:
        if self.c_embed is None:
            msg = "CNMR embedding layer is not initialized. Check use_cnmr flag."
            raise RuntimeError(msg)
        c_embed = self._embed_spectrum(x, self.c_tokenizer, self.c_embed, self.cnmr_dim)
        if not self.use_c_mask_token:
            return c_embed
        c_missing_mask = (x == 0).all(dim=1)
        c_missing_tokens = self.c_mask_token(c_missing_mask.long()).unsqueeze(1)
        return torch.cat([c_missing_tokens, c_embed], dim=1)

    def pos_encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_enc_mode == "sincos":
            return x + self.pos_embed
        if self.pos_enc_mode == "learnable":
            return x + self.learnable_pos_embed
        return x

    def _separate_spectra_components(self, x: torch.Tensor):
        hnmr_x = x[:, : self.hnmr_dim]
        cnmr_x = x[:, self.hnmr_dim : self.hnmr_dim + self.cnmr_dim]
        return hnmr_x, cnmr_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hnmr_spectra, cnmr_spectra = self._separate_spectra_components(x)

        tokens = []
        type_ids = []
        if self.use_hnmr:
            t = self._embed_hnmr(hnmr_spectra)
            tokens.append(t)
            type_ids.append(torch.zeros(t.size(1), device=t.device, dtype=torch.long))
        if self.use_cnmr:
            t = self._embed_cnmr(cnmr_spectra)
            tokens.append(t)
            type_ids.append(torch.ones(t.size(1), device=t.device, dtype=torch.long))

        x = torch.cat(tokens, dim=1)
        x = self.pos_encode(x)

        if self.type_embedding is not None:
            type_emb = self.type_embedding(torch.cat(type_ids))
            x = x + type_emb.unsqueeze(0).expand(x.size(0), -1, -1)

        x = self.dropout(x)
        x = self.transformer(x)
        return self.head(x)
