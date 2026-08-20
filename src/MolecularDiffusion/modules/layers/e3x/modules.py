"""e3x ``Dense``, ``Tensor`` and ``SelfAttention``, in PyTorch.

Flax modules infer their input shapes lazily; PyTorch cannot, so every class
here takes the input ``(num_parity, max_degree, in_features)`` explicitly. The
*parameter* layout is unchanged, which is what makes the published Orbax
checkpoints convertible:

======================================================  ==============================
Flax path                                               here
======================================================  ==============================
``<mod>/query/<l><parity>/kernel``  ``(in, out)``        ``query.dense['<l><parity>'].weight`` (transposed)
``<mod>/out/0+/bias``                                   ``out.dense['0+'].bias`` (only l=0 has one)
``<mod>/relative_positional_encoding/kernel``           ``rel_pos.weight`` (transposed, no bias)
``<mod>/tensor/kernel``  ``(P1,L1+1,P2,L2+1,P3,L3+1,F)`` ``tensor.kernel`` (**no** transpose)
======================================================  ==============================

Port of the ``e3x/nn/modules.py`` surface DiTMC reaches. ``FusedTensor``,
``MessagePass`` and ``TensorDense`` are deliberately not ported -- no DiTMC
config path calls them (``use_fused_tensor=False`` at both call sites).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from . import initializers, so3
from .features import change_max_degree_or_type, extract_max_degree
from .indexed import gather_dst, gather_src, indexed_softmax, indexed_sum


def duplication_indices_for_max_degree(max_degree: int) -> torch.Tensor:
    """``[0, 1,1,1, 2,2,2,2,2, ...]`` -- degree ``l`` repeated ``2l+1`` times."""
    return torch.repeat_interleave(
        torch.arange(max_degree + 1), torch.arange(max_degree + 1) * 2 + 1
    )


def _degree_names(max_degree: int, suffix: str | None) -> list[str]:
    if suffix is None:
        parity = ["+", "-"]
        return [f"{l}{parity[l % 2]}" for l in range(max_degree + 1)]  # noqa: E741
    return [f"{l}{suffix}" for l in range(max_degree + 1)]  # noqa: E741


class Dense(nn.Module):
    """Per-degree (and per-parity) linear map over the feature axis.

    e3x's ``Dense`` is **not one kernel**: it instantiates one ``nn.Linear`` per
    degree (and per parity when ``P=2``), applied to the slice
    ``inputs[..., l**2:(l+1)**2, :]``. A bias exists **only on the ``l=0``
    sublayer**, and only when ``use_bias=True``; odd-parity sublayers never get
    one.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_degree: int,
        num_parity: int,
        use_bias: bool = True,
        zero_init: bool = False,
    ) -> None:
        super().__init__()
        if num_parity not in (1, 2):
            msg = f"num_parity must be 1 or 2, received {num_parity}"
            raise ValueError(msg)
        self.in_features = in_features
        self.out_features = out_features
        self.max_degree = max_degree
        self.num_parity = num_parity
        self.use_bias = use_bias

        self.dense = nn.ModuleDict()
        if num_parity == 2:
            for l, name in enumerate(_degree_names(max_degree, "+")):  # noqa: E741
                self.dense[name] = nn.Linear(
                    in_features, out_features, bias=use_bias and l == 0
                )
            for name in _degree_names(max_degree, "-"):
                self.dense[name] = nn.Linear(in_features, out_features, bias=False)
        else:
            for l, name in enumerate(_degree_names(max_degree, None)):  # noqa: E741
                self.dense[name] = nn.Linear(
                    in_features, out_features, bias=use_bias and l == 0
                )
        self._even_names = (
            _degree_names(max_degree, "+")
            if num_parity == 2
            else _degree_names(max_degree, None)
        )
        self._odd_names = _degree_names(max_degree, "-") if num_parity == 2 else []
        self.reset_parameters(zero_init=zero_init)

    def reset_parameters(self, zero_init: bool = False) -> None:
        for layer in self.dense.values():
            if zero_init:
                nn.init.zeros_(layer.weight)
            else:
                initializers.lecun_normal_(layer.weight, self.in_features)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        max_degree = extract_max_degree(inputs.shape)
        if max_degree != self.max_degree or inputs.shape[-3] != self.num_parity:
            msg = (
                f"Dense was built for (P={self.num_parity}, L={self.max_degree}) "
                f"but received shape {tuple(inputs.shape)}"
            )
            raise ValueError(msg)

        def _branch(x, names):
            return torch.cat(
                [
                    self.dense[names[l]](x[..., l**2 : (l + 1) ** 2, :])  # noqa: E741
                    for l in range(max_degree + 1)  # noqa: E741
                ],
                dim=-2,
            )

        if self.num_parity == 2:
            even = _branch(inputs[..., 0, :, :], self._even_names)
            odd = _branch(inputs[..., 1, :, :], self._odd_names)
            return torch.stack((even, odd), dim=-3)
        return _branch(inputs.squeeze(-3), self._even_names).unsqueeze(-3)


def resolve_tensor_output(
    max_degree1: int,
    num_parity1: int,
    max_degree2: int,
    num_parity2: int,
    max_degree: int | None,
    include_pseudotensors: bool,
) -> tuple[int, int]:
    """``(num_parity3, max_degree3)`` a :class:`Tensor` will emit."""
    max_degree3 = (
        max(max_degree1, max_degree2) if max_degree is None else max_degree
    )
    if max_degree3 > max_degree1 + max_degree2:
        msg = (
            f"max_degree for the tensor product of inputs with max_degree "
            f"{max_degree1} and {max_degree2} can be at most "
            f"{max_degree1 + max_degree2}, received max_degree={max_degree3}"
        )
        raise ValueError(msg)
    if (num_parity1 == num_parity2 == 1) and (
        max_degree1 == 0 or max_degree2 == 0 or max_degree3 == 0
    ):
        include_pseudotensors = False
    return (2 if include_pseudotensors else 1), max_degree3


class Tensor(nn.Module):
    r"""Learnable tensor product of two equivariant feature representations.

    ``einsum('...plf,...qmf,plqmrnf,lmn->...rnf', inputs1, inputs2, kernel, cg)``
    with the kernel expanded per-degree and masked on parity-forbidden paths.

    At ``max_degree=0`` with ``P=1`` everywhere (the non-equivariant DiT path)
    this does **not** collapse to the identity: the kernel is ``(1,1,1,1,1,1,F)``
    and ``cg[0,0,0] == 1``, so it is a learnable **per-feature multiplicative
    gate**, initialized by ``tensor_lecun_normal`` with ``fan_in = 1`` -- a
    truncated normal with stddev ~1.1368, not 1.0. Dropping it would neither
    match a checkpoint nor train the same way.
    """

    def __init__(
        self,
        features: int,
        max_degree1: int,
        num_parity1: int,
        max_degree2: int,
        num_parity2: int,
        max_degree: int | None = None,
        include_pseudotensors: bool = True,
    ) -> None:
        super().__init__()
        num_parity3, max_degree3 = resolve_tensor_output(
            max_degree1,
            num_parity1,
            max_degree2,
            num_parity2,
            max_degree,
            include_pseudotensors,
        )
        self.features = features
        self.max_degree1, self.num_parity1 = max_degree1, num_parity1
        self.max_degree2, self.num_parity2 = max_degree2, num_parity2
        self.max_degree3, self.num_parity3 = max_degree3, num_parity3

        kernel_shape = (
            num_parity1,
            max_degree1 + 1,
            num_parity2,
            max_degree2 + 1,
            num_parity3,
            max_degree3 + 1,
            features,
        )
        self.kernel = nn.Parameter(torch.empty(kernel_shape))
        initializers.tensor_lecun_normal_(self.kernel)

        self.mixed_coupling_paths = not (
            num_parity1 == num_parity2 == num_parity3 == 2
        )
        self.register_buffer(
            "cg",
            so3.clebsch_gordan(max_degree1, max_degree2, max_degree3),
            persistent=False,
        )
        if self.mixed_coupling_paths:
            mask = initializers.tensor_product_mask(kernel_shape[:-1])
            self.register_buffer(
                "mask", torch.as_tensor(mask, dtype=torch.float32), persistent=False
            )
        else:
            self.mask = None
        self.register_buffer(
            "idx1", duplication_indices_for_max_degree(max_degree1), persistent=False
        )
        self.register_buffer(
            "idx2", duplication_indices_for_max_degree(max_degree2), persistent=False
        )
        self.register_buffer(
            "idx3", duplication_indices_for_max_degree(max_degree3), persistent=False
        )

    def forward(self, inputs1: torch.Tensor, inputs2: torch.Tensor) -> torch.Tensor:
        if inputs1.shape[-1] != inputs2.shape[-1]:
            msg = (
                "axis -1 of inputs1 and inputs2 must have the same size, received "
                f"{tuple(inputs1.shape)} and {tuple(inputs2.shape)}"
            )
            raise ValueError(msg)

        kernel = self.kernel
        if self.mixed_coupling_paths:
            kernel = kernel * self.mask
        kernel = kernel.index_select(1, self.idx1)
        kernel = kernel.index_select(3, self.idx2)
        kernel = kernel.index_select(5, self.idx3)

        cg = self.cg.to(inputs1.dtype)
        if self.mixed_coupling_paths:
            return torch.einsum(
                "...plf,...qmf,plqmrnf,lmn->...rnf", inputs1, inputs2, kernel, cg
            )

        def _couple(i: int, j: int, k: int) -> torch.Tensor:
            return torch.einsum(
                "...lf,...mf,lmnf,lmn->...nf",
                inputs1[..., i, :, :],
                inputs2[..., j, :, :],
                kernel[i, :, j, :, k, :, :],
                cg,
            )

        eee = _couple(0, 0, 0)
        ooe = _couple(1, 1, 0)
        eoo = _couple(0, 1, 1)
        oeo = _couple(1, 0, 1)
        return torch.stack((eee + ooe, eoo + oeo), dim=-3)


class SelfAttention(nn.Module):
    r"""Equivariant multi-head self-attention over a sparse edge list.

    Reproduces ``e3x.nn.SelfAttention`` -> ``MultiHeadAttention`` -> ``_Conv``,
    including three behaviours that are arbitrary but load-bearing:

    1. **The q/k head fold and the value-weight fold disagree.** ``query``/
       ``key`` are reshaped to ``(..., head_dim, num_heads)``, so feature ``f``
       belongs to head ``f % num_heads``; the attention weight is then
       ``repeat_interleave``-d, so **value** feature ``f`` is weighted by head
       ``f // head_dim``. Both conventions are upstream's; neither is a bug to
       fix here.
    2. **The ``1/sqrt(depth)`` scale includes parity and ``(L+1)**2``**:
       ``depth = P · (L_qk+1)**2 · head_dim``. Only at ``max_degree=0, P=1``
       does it reduce to ``1/sqrt(head_dim)``.
    3. **The relative positional encoding is multiplicative, not additive**: it
       enters as the third factor of an elementwise triple-product einsum, a
       learned per-degree per-feature distance-dependent gate on each ``q·k``
       term. It is fed **only the scalar slice** ``basis[..., 0, 0, :]``.
    """

    def __init__(
        self,
        in_features: int,
        in_max_degree: int,
        in_num_parity: int,
        num_heads: int,
        max_degree: int | None = None,
        include_pseudotensors: bool = True,
        num_basis: int | None = None,
        basis_max_degree: int = 0,
        basis_num_parity: int = 1,
        qkv_features: int | None = None,
        out_features: int | None = None,
        use_relative_positional_encoding_qk: bool = True,
        use_relative_positional_encoding_v: bool = True,
        use_basis_bias: bool = False,
    ) -> None:
        super().__init__()
        qkv_features = in_features if qkv_features is None else qkv_features
        out_features = in_features if out_features is None else out_features
        if qkv_features % num_heads != 0:
            msg = (
                f"qkv_features ({qkv_features}) must be divisible by "
                f"num_heads ({num_heads})"
            )
            raise ValueError(msg)
        if (
            use_relative_positional_encoding_qk or use_relative_positional_encoding_v
        ) and num_basis is None:
            msg = (
                "when using relative positional encodings, 'num_basis' is a "
                "required argument, received None"
            )
            raise TypeError(msg)

        self.num_heads = num_heads
        self.qkv_features = qkv_features
        self.head_dim = qkv_features // num_heads
        self.rpe_qk = use_relative_positional_encoding_qk
        self.rpe_v = use_relative_positional_encoding_v

        # Self-attention: q and k come from the same tensor, so harmonizing is a
        # no-op, but the resolved values still drive every shape below.
        self.max_degree_qk = in_max_degree
        self.num_parity_qk = in_num_parity

        self.query = Dense(
            in_features, qkv_features, self.max_degree_qk, self.num_parity_qk, use_bias=False
        )
        self.key = Dense(
            in_features, qkv_features, self.max_degree_qk, self.num_parity_qk, use_bias=False
        )
        self.value = Dense(
            in_features, qkv_features, in_max_degree, in_num_parity, use_bias=False
        )

        if self.rpe_qk:
            self.rel_pos = nn.Linear(
                num_basis,
                self.num_parity_qk * (self.max_degree_qk + 1) * qkv_features,
                bias=use_basis_bias,
            )
            initializers.lecun_normal_(self.rel_pos.weight, num_basis)
            if self.rel_pos.bias is not None:
                nn.init.zeros_(self.rel_pos.bias)
            self.register_buffer(
                "_rpe_dup_idx",
                duplication_indices_for_max_degree(self.max_degree_qk),
                persistent=False,
            )
        else:
            self.rel_pos = None

        if self.rpe_v:
            self.filter = Dense(
                num_basis,
                qkv_features,
                basis_max_degree,
                basis_num_parity,
                use_bias=use_basis_bias,
            )
            self.tensor = Tensor(
                features=qkv_features,
                max_degree1=basis_max_degree,
                num_parity1=basis_num_parity,
                max_degree2=in_max_degree,
                num_parity2=in_num_parity,
                max_degree=max_degree,
                include_pseudotensors=include_pseudotensors,
            )
            out_parity, out_degree = self.tensor.num_parity3, self.tensor.max_degree3
        else:
            self.filter = None
            self.tensor = None
            out_parity, out_degree = in_num_parity, in_max_degree

        self.out_num_parity, self.out_max_degree = out_parity, out_degree
        self.out = Dense(
            qkv_features, out_features, out_degree, out_parity, use_bias=True
        )

    def forward(
        self,
        inputs: torch.Tensor,
        basis: torch.Tensor | None = None,
        *,
        dst_idx: torch.Tensor,
        src_idx: torch.Tensor,
        num_segments: int,
        cutoff_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (self.rpe_qk or self.rpe_v) and basis is None:
            msg = (
                "when using relative positional encodings, 'basis' is a required "
                "argument, received basis=None"
            )
            raise TypeError(msg)

        query_inputs = change_max_degree_or_type(
            inputs,
            max_degree=self.max_degree_qk,
            include_pseudotensors=self.num_parity_qk == 2,
        )
        query = self.query(query_inputs)
        key = self.key(query_inputs)
        value = self.value(inputs)

        # Split heads: (..., P, D, head_dim, num_heads). Heads are the INNERMOST
        # axis, so feature f belongs to head f % num_heads.
        query = query.reshape(*query.shape[:-1], -1, self.num_heads)
        key = key.reshape(*key.shape[:-1], -1, self.num_heads)

        depth = math.prod(query.shape[-4:-1])  # parity * degrees * head_dim
        query = query / math.sqrt(depth)

        query = gather_dst(query, dst_idx)
        key = gather_src(key, src_idx)

        if self.rpe_qk:
            rpe = self.rel_pos(basis[..., 0, 0, :])
            rpe = rpe.reshape(
                *rpe.shape[:-1],
                self.num_parity_qk,
                self.max_degree_qk + 1,
                self.qkv_features,
            )
            rpe = rpe.index_select(-2, self._rpe_dup_idx)
            rpe = rpe.reshape(*rpe.shape[:-1], -1, self.num_heads)
            dot = torch.einsum("...plfh,...plfh,...plfh->...h", query, key, rpe)
        else:
            dot = torch.einsum("...plfh,...plfh->...h", query, key)

        weight = indexed_softmax(
            dot, dst_idx, num_segments, multiplicative_mask=cutoff_value
        )
        # Value feature f is weighted by head f // head_dim -- the OPPOSITE
        # grouping from the q/k reshape above. Upstream behaviour, reproduced.
        weight = torch.repeat_interleave(weight, self.head_dim, dim=-1)
        weight = weight[..., None, None, :]

        value = gather_src(value, src_idx)
        products = weight * value

        if self.rpe_v:
            filters = self.filter(basis)
            products = self.tensor(filters, products)

        attention = indexed_sum(products, dst_idx, num_segments)
        return self.out(attention)
