"""SpecFormer: DiffSpectra's patch-based transformer spectral encoder.

Ported from ``others/DiffSpectra/models/specformer.py`` (itself adapted from
PatchTST, https://github.com/yuqinie98/PatchTST). Takes 1-3 raw spectra
(UV/IR/Raman, depending on ``spectra_version``) and returns one
``output_dim``-wide vector per molecule, which
:class:`~MolecularDiffusion.modules.models.diffspectra.dmt.DMT` adds directly
into its timestep embedding (no cross-attention, no CFG branch -- see the
integration plan's Repo Inspection section).

No behaviour changes from upstream beyond explicit imports (``from
.specformer_layers import *`` -> named imports) and dropping the
``__main__`` smoke block; ``reset_parameters()`` is kept even though loading
a checkpoint overwrites its effect, because it is cheap and keeps this
diffable against upstream.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from .specformer_layers import (
    Transpose,
    get_activation_fn,
    positional_encoding,
)

# Fixed by QM9S: UV, IR, Raman spectra are always this long (raw wavenumber/
# wavelength grid, before patching). ``configs/diffspectra_qm9s.py``'s
# patch_len/stride ([20,50,50] / [10,25,25]) are derived from these.
_SPECTRUM_LEN = {"uv": 701, "ir": 3501, "raman": 3501}
_SPECTRA_ORDER = ("uv", "ir", "raman")  # index 0/1/2, fixed by upstream


class SpecFormer(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        patch_len: list = None,
        stride: list = None,
        output_dim: int = 256,
        spectra_version: str = "ir",
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 16,
        d_k: int | None = None,
        d_v: int | None = None,
        d_ff: int = 256,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        act: str = "gelu",
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        fc_dropout: float = 0.0,
        head_dropout: float = 0.0,
        individual: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.patch_len = list(patch_len or [20, 50, 50])
        self.stride = list(stride or [10, 25, 25])
        list_len_spectrum = [_SPECTRUM_LEN[k] for k in _SPECTRA_ORDER]

        self.spectra_version = spectra_version
        if spectra_version == "uv":
            self.used_spectra_type = [0]
        elif spectra_version == "ir":
            self.used_spectra_type = [1]
        elif spectra_version == "raman":
            self.used_spectra_type = [2]
        elif spectra_version == "allspectra":
            self.used_spectra_type = [0, 1, 2]
        else:
            msg = "spectra_version should be uv, ir, raman or allspectra"
            raise ValueError(msg)

        patch_nums = [
            int(
                (list_len_spectrum[i] - self.patch_len[i]) / self.stride[i] + 1
            )
            for i in self.used_spectra_type
        ]
        self.patch_nums = patch_nums
        all_patch_num = sum(patch_nums)

        self.backbone = TSTiEncoder(
            patch_nums=patch_nums,
            patch_len=self.patch_len,
            spectra_version=spectra_version,
            used_spectra_type=self.used_spectra_type,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=act,
            res_attention=res_attention,
            pre_norm=pre_norm,
            store_attn=store_attn,
            pe=pe,
            learn_pe=learn_pe,
            **kwargs,
        )

        self.head_nf = d_model * all_patch_num
        self.head = Flatten_Head(
            individual, self.head_nf, output_dim, head_dropout=head_dropout
        )
        self.out_norm = nn.LayerNorm(output_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.backbone.reset_parameters()
        self.head.reset_parameters()
        self.out_norm.reset_parameters()

    def forward(self, spectra_tensor):
        """``spectra_tensor``: one ``(B, L)``/``(B, 1, L)`` tensor for a
        single spectrum version, or a 3-list ``[uv, ir, raman]`` for
        ``allspectra`` -- matching :class:`~.dmt.DMT`'s own ``context``
        contract, which is in turn what the task's ``_context_from_batch``
        builds.
        """
        if self.spectra_version in ("uv", "ir", "raman"):
            spectra = [spectra_tensor.squeeze()]
        elif self.spectra_version == "allspectra":
            spectra = [
                spectra_tensor[0].squeeze(),
                spectra_tensor[1].squeeze(),
                spectra_tensor[2].squeeze(),
            ]
        else:
            msg = "spectra_version should be uv, ir, raman or allspectra"
            raise ValueError(msg)

        patched_spectra = []
        if self.spectra_version in ("uv", "ir", "raman"):
            spec = spectra[0]
            if spec.dim() == 1:
                spec = spec.unsqueeze(0)
            if spec.dim() == 3 and spec.size(1) == 1:  # noqa: PLR2004
                spec = spec.squeeze(1)
            spec = spec.unfold(
                dimension=-1,
                size=self.patch_len[self.used_spectra_type[0]],
                step=self.stride[self.used_spectra_type[0]],
            )
            patched_spectra.append(spec.permute(0, 2, 1))
        else:  # allspectra
            for i, spec in enumerate(spectra):
                if spec.dim() == 1:
                    spec = spec.unsqueeze(0)
                if spec.dim() == 3 and spec.size(1) == 1:  # noqa: PLR2004
                    spec = spec.squeeze(1)
                spec = spec.unfold(
                    dimension=-1, size=self.patch_len[i], step=self.stride[i]
                )
                patched_spectra.append(spec.permute(0, 2, 1))

        z = self.backbone(patched_spectra)
        z = self.head(z)
        return self.out_norm(z)


class TSTiEncoder(nn.Module):
    """Channel-independent patch encoder (one branch per spectrum type)."""

    def __init__(  # noqa: PLR0913
        self,
        patch_nums,
        patch_len,
        spectra_version,
        used_spectra_type,
        n_layers=3,
        d_model=128,
        n_heads=16,
        d_k=None,
        d_v=None,
        d_ff=256,
        norm="BatchNorm",
        attn_dropout=0.0,
        dropout=0.0,
        act="gelu",
        res_attention=True,
        pre_norm=False,
        store_attn=False,
        pe="zeros",
        learn_pe=True,
        **kwargs: object,  # noqa: ARG002
    ) -> None:
        super().__init__()
        self.patch_nums = patch_nums
        self.patch_len = patch_len
        self.spectra_version = spectra_version
        self.used_spectra_type = used_spectra_type

        self.W_P = nn.ModuleList(
            [nn.Linear(patch_len[i], d_model) for i in used_spectra_type]
        )
        if spectra_version in ("uv", "ir", "raman"):
            self.W_pos = positional_encoding(
                pe, learn_pe, patch_nums[0], d_model
            )
        elif spectra_version == "allspectra":
            self.W_pos_uv = positional_encoding(
                pe, learn_pe, patch_nums[0], d_model
            )
            self.W_pos_ir = positional_encoding(
                pe, learn_pe, patch_nums[1], d_model
            )
            self.W_pos_raman = positional_encoding(
                pe, learn_pe, patch_nums[2], d_model
            )

        self.dropout = nn.Dropout(dropout)

        all_patch_nums = sum(patch_nums)
        self.encoder = TSTEncoder(
            all_patch_nums,
            d_model,
            n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            pre_norm=pre_norm,
            activation=act,
            res_attention=res_attention,
            n_layers=n_layers,
            store_attn=store_attn,
        )

    def reset_parameters(self) -> None:
        for w in self.W_P:
            nn.init.xavier_uniform_(w.weight)
            w.bias.data.fill_(0)
        self.encoder.reset_parameters()

    def forward(self, patched_spectra) -> Tensor:
        encoded_spectra = []
        if self.spectra_version in ("uv", "ir", "raman"):
            patched_spec = patched_spectra[0].permute(0, 2, 1)
            patched_spec = self.W_P[0](patched_spec)
            patched_spec = self.dropout(patched_spec + self.W_pos)
            encoded_spectra.append(patched_spec)
        else:  # allspectra
            for i, patched_spec in enumerate(patched_spectra):
                patched_spec = patched_spec.permute(0, 2, 1)
                patched_spec = self.W_P[i](patched_spec)
                if i == 0:
                    patched_spec = self.dropout(patched_spec + self.W_pos_uv)
                elif i == 1:
                    patched_spec = self.dropout(patched_spec + self.W_pos_ir)
                else:
                    patched_spec = self.dropout(
                        patched_spec + self.W_pos_raman
                    )
                encoded_spectra.append(patched_spec)

        z = torch.cat(encoded_spectra, dim=1)
        return self.encoder(z)


class TSTEncoder(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        q_len,
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        d_ff=None,
        norm="BatchNorm",
        attn_dropout=0.0,
        dropout=0.0,
        activation="gelu",
        res_attention=False,
        n_layers=1,
        pre_norm=False,
        store_attn=False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TSTEncoderLayer(
                    q_len,
                    d_model,
                    n_heads=n_heads,
                    d_k=d_k,
                    d_v=d_v,
                    d_ff=d_ff,
                    norm=norm,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    activation=activation,
                    res_attention=res_attention,
                    pre_norm=pre_norm,
                    store_attn=store_attn,
                )
                for _ in range(n_layers)
            ]
        )
        self.res_attention = res_attention

    def reset_parameters(self) -> None:
        for layer in self.layers:
            layer.reset_parameters()

    def forward(
        self,
        src: Tensor,
        key_padding_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(
                    output,
                    prev=scores,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                )
            return output
        for mod in self.layers:
            output = mod(
                output, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        return output


class TSTEncoderLayer(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        q_len,  # noqa: ARG002 - part of upstream's signature, unused in the body
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        d_ff=256,
        store_attn=False,
        norm="BatchNorm",
        attn_dropout=0,
        dropout=0.0,
        bias=True,
        activation="gelu",
        res_attention=False,
        pre_norm=False,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            msg = (
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
            raise ValueError(msg)
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(
            d_model,
            n_heads,
            d_k,
            d_v,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            res_attention=res_attention,
        )

        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias),
        )

        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def reset_parameters(self) -> None:
        self.self_attn.reset_parameters()
        if isinstance(self.norm_attn, nn.LayerNorm):
            self.norm_attn.reset_parameters()
        for name, param in self.ff.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, val=0)
        if isinstance(self.norm_ffn, nn.LayerNorm):
            self.norm_ffn.reset_parameters()

    def forward(
        self,
        src: Tensor,
        prev: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        if self.pre_norm:
            src = self.norm_attn(src)
        if self.res_attention:
            src2, attn, scores = self.self_attn(
                src,
                src,
                src,
                prev,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            src2, attn = self.self_attn(
                src,
                src,
                src,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        if self.store_attn:
            self.attn = attn
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)

        if self.pre_norm:
            src = self.norm_ffn(src)
        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        return src


class _MultiheadAttention(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        res_attention=False,
        attn_dropout=0.0,
        proj_dropout=0.0,
        qkv_bias=True,
        lsa=False,
    ) -> None:
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(
            d_model,
            n_heads,
            attn_dropout=attn_dropout,
            res_attention=res_attention,
            lsa=lsa,
        )
        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout)
        )

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_Q.weight)
        self.W_Q.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.W_K.weight)
        self.W_K.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.W_V.weight)
        self.W_V.bias.data.fill_(0)

    def forward(  # noqa: PLR0913
        self,
        q: Tensor,
        k: Tensor | None = None,
        v: Tensor | None = None,
        prev: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ):
        bs = q.size(0)
        if k is None:
            k = q
        if v is None:
            v = q

        q_s = self.W_Q(q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = (
            self.W_K(k)
            .view(bs, -1, self.n_heads, self.d_k)
            .permute(0, 2, 3, 1)
        )
        v_s = self.W_V(v).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)

        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(
                q_s,
                k_s,
                v_s,
                prev=prev,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            output, attn_weights = self.sdp_attn(
                q_s,
                k_s,
                v_s,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )

        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(bs, -1, self.n_heads * self.d_v)
        )
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        attn_dropout=0.0,
        res_attention=False,
        lsa=False,
    ) -> None:
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(
            torch.tensor(head_dim**-0.5), requires_grad=lsa
        )
        self.lsa = lsa

    def forward(  # noqa: PLR0913
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        prev: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ):
        attn_scores = torch.matmul(q, k) * self.scale
        if prev is not None:
            attn_scores = attn_scores + prev
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask
        if key_padding_mask is not None:
            attn_scores.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)

        if self.res_attention:
            return output, attn_weights, attn_scores
        return output, attn_weights


class Flatten_Head(nn.Module):  # noqa: N801 - upstream name, kept for diffability
    def __init__(
        self, individual, nf, target_window, head_dropout=0, n_vars=1
    ) -> None:
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for _ in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def reset_parameters(self) -> None:
        if self.individual:
            for lin in self.linears:
                nn.init.xavier_uniform_(lin.weight)
                lin.bias.data.fill_(0)
        else:
            nn.init.xavier_uniform_(self.linear.weight)
            self.linear.bias.data.fill_(0)

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            return torch.stack(x_out, dim=1)
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)
