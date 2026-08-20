"""Shape/parity bookkeeping for e3x rank-4 feature tensors.

Every equivariant tensor is ``(..., P, (L+1)**2, F)``:

* axis ``-3`` -- **parity**, size 1 or 2. Index ``0`` is even (``p = +1``),
  index ``1`` is odd (``p = -1``).
* axis ``-2`` -- **degree/order**, size ``(L+1)**2``, degree ``l`` occupying the
  contiguous slice ``[l**2, (l+1)**2)``, degrees ascending.
* ``P = 1`` is the "proper tensors only" form: parity is implicit,
  ``p = (-1)**l``.
* axis ``-1`` -- features.

Port of ``e3x/nn/features.py``.
"""

from __future__ import annotations

import math

import torch


def extract_max_degree(shape) -> int:
    """``max_degree`` from a feature shape, with e3x's validity checks."""
    if len(shape) < 3:
        msg = f"shape of features must have at least length 3, received {tuple(shape)}"
        raise ValueError(msg)
    if shape[-3] not in (1, 2):
        msg = f"expected 1 or 2 for axis -3 of feature shape, received {tuple(shape)}"
        raise ValueError(msg)
    max_degree = round(math.sqrt(shape[-2]) - 1)
    if shape[-2] != (max_degree + 1) ** 2:
        msg = (
            f"received invalid size {shape[-2]} for axis -2 of feature shape, "
            f"closest valid size is {(max_degree + 1) ** 2}"
        )
        raise ValueError(msg)
    return max_degree


def change_max_degree_or_type(
    x: torch.Tensor,
    max_degree: int | None = None,
    include_pseudotensors: bool | None = None,
) -> torch.Tensor:
    """Pad with zeros / slice away degree and parity channels.

    Growing ``max_degree`` zero-pads axis -2; shrinking it slices. ``P=1 -> 2``
    routes even degrees into the even block and odd degrees into the odd block,
    zeros elsewhere; ``P=2 -> 1`` keeps the proper-tensor slot ``l % 2`` of each
    degree.
    """
    in_max_degree = extract_max_degree(x.shape)
    input_has_pseudo = x.shape[-3] != 1

    max_degree = in_max_degree if max_degree is None else max_degree
    include_pseudotensors = (
        input_has_pseudo if include_pseudotensors is None else include_pseudotensors
    )

    if max_degree < in_max_degree:
        x = x[..., : (max_degree + 1) ** 2, :]
    elif max_degree > in_max_degree:
        pad = [0, 0, 0, (max_degree + 1) ** 2 - (in_max_degree + 1) ** 2]
        x = torch.nn.functional.pad(x, pad)

    if input_has_pseudo and not include_pseudotensors:
        x = torch.cat(
            [
                x[..., l % 2 : l % 2 + 1, l**2 : (l + 1) ** 2, :]  # noqa: E741
                for l in range(max_degree + 1)  # noqa: E741
            ],
            dim=-2,
        )
    elif include_pseudotensors and not input_has_pseudo:
        even = torch.cat(
            [
                x[..., 0:1, l**2 : (l + 1) ** 2, :]
                if l % 2 == 0
                else torch.zeros_like(x[..., 0:1, l**2 : (l + 1) ** 2, :])
                for l in range(max_degree + 1)  # noqa: E741
            ],
            dim=-2,
        )
        odd = torch.cat(
            [
                x[..., 0:1, l**2 : (l + 1) ** 2, :]
                if l % 2 != 0
                else torch.zeros_like(x[..., 0:1, l**2 : (l + 1) ** 2, :])
                for l in range(max_degree + 1)  # noqa: E741
            ],
            dim=-2,
        )
        x = torch.cat((even, odd), dim=-3)

    return x


def add(*inputs: torch.Tensor) -> torch.Tensor:
    """Union-broadcast addition of equivariant features.

    ``L = max(L_i)``, ``P = 2`` if any operand has ``P = 2``; smaller operands
    are **zero-padded up**, never truncated. Raises on batch/feature/dtype
    mismatch rather than silently broadcasting a size-1 axis, which is the
    whole point of the function existing.
    """
    if inputs[0].dim() < 3:
        msg = (
            "all inputs must be at least three-dimensional, received input with "
            f"shape {tuple(inputs[0].shape)} at position 0"
        )
        raise ValueError(msg)

    max_degree = 0
    has_pseudo = False
    features = inputs[0].shape[-1]
    batch_shape = tuple(inputs[0].shape[:-3])
    dtype = inputs[0].dtype
    for i, x in enumerate(inputs):
        if tuple(x.shape[:-3]) != batch_shape:
            msg = (
                "all inputs must have the same leading dimensions, received "
                f"{tuple(x.shape[:-3])} at position {i}, expected {batch_shape}"
            )
            raise ValueError(msg)
        if x.shape[-1] != features:
            msg = (
                f"all inputs must have the same number of features, received "
                f"{x.shape[-1]} at position {i}, expected {features}"
            )
            raise ValueError(msg)
        if x.dtype != dtype:
            msg = (
                f"all inputs must have the same dtype, received {x.dtype} at "
                f"position {i}, expected {dtype}"
            )
            raise ValueError(msg)
        max_degree = max(max_degree, extract_max_degree(x.shape))
        has_pseudo = has_pseudo or x.shape[-3] == 2

    y = inputs[0].new_zeros(
        (*batch_shape, 2 if has_pseudo else 1, (max_degree + 1) ** 2, features)
    )
    for x in inputs:
        y = y + change_max_degree_or_type(
            x, max_degree=max_degree, include_pseudotensors=has_pseudo
        )
    return y


def reflect(x: torch.Tensor) -> torch.Tensor:
    """Apply a parity inversion to equivariant features.

    ``P=2``: negate the odd-parity block. ``P=1``: negate the odd-``l`` blocks.
    Used only by the fidelity suite.
    """
    max_degree = extract_max_degree(x.shape)
    if x.shape[-3] == 2:
        even, odd = x[..., 0:1, :, :], x[..., 1:2, :, :]
        return torch.cat((even, -odd), dim=-3)
    sign = torch.cat(
        [
            x.new_full((2 * l + 1,), (-1.0) ** l)  # noqa: E741
            for l in range(max_degree + 1)  # noqa: E741
        ]
    )
    return x * sign.view(*([1] * (x.dim() - 2)), -1, 1)


def promote_to_e3x(x: torch.Tensor) -> torch.Tensor:
    """``(n, F) -> (n, 1, 1, F)``. Port of ``backbones/utils.promote_to_e3x``."""
    if x.dim() != 2:
        msg = f"expected a 2D tensor, received shape {tuple(x.shape)}"
        raise ValueError(msg)
    return x[:, None, None, :]


def broadcast_equivariant_multiplication(
    factor: torch.Tensor, tensor: torch.Tensor
) -> torch.Tensor:
    """Per-``(parity, degree)`` scaling of an equivariant tensor.

    ``factor`` is ``(n, P, L+1, F)`` (one value per degree), ``tensor`` is
    ``(n, P, (L+1)**2, F)``. Port of
    ``backbones/utils.broadcast_equivariant_multiplication``.
    """
    max_degree_tensor = extract_max_degree(tensor.shape)
    max_degree_factor = factor.shape[-2] - 1
    if factor.shape[-1] != tensor.shape[-1]:
        msg = f"feature dims must align: {tuple(factor.shape)} vs {tuple(tensor.shape)}"
        raise ValueError(msg)
    if len(factor) != len(tensor):
        msg = f"leading axis must align: {len(factor)} vs {len(tensor)}"
        raise ValueError(msg)
    if max_degree_factor != max_degree_tensor:
        msg = f"max_degree must align: {max_degree_factor} vs {max_degree_tensor}"
        raise ValueError(msg)
    repeats = torch.tensor(
        [2 * l + 1 for l in range(max_degree_factor + 1)],  # noqa: E741
        device=factor.device,
    )
    return torch.repeat_interleave(factor, repeats, dim=-2) * tensor
