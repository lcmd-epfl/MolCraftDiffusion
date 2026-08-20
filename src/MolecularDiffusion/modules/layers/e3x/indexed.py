"""Sparse indexed (segment) ops. Port of the ``e3x/ops/indexed.py`` surface.

Only the sparse ``dst_idx``/``src_idx`` path is ported; DiTMC never uses the
dense adjacency form.
"""

from __future__ import annotations

import torch


def gather_src(inputs: torch.Tensor, src_idx: torch.Tensor) -> torch.Tensor:
    """``inputs[src_idx]`` along the leading axis."""
    return inputs.index_select(0, src_idx)


def gather_dst(inputs: torch.Tensor, dst_idx: torch.Tensor) -> torch.Tensor:
    """``inputs[dst_idx]`` along the leading axis."""
    return inputs.index_select(0, dst_idx)


def indexed_sum(
    inputs: torch.Tensor, dst_idx: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Segment sum over the leading axis."""
    out = inputs.new_zeros((num_segments, *inputs.shape[1:]))
    return out.index_add(0, dst_idx, inputs)


def indexed_max(
    inputs: torch.Tensor, dst_idx: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Segment max over the leading axis; empty segments give ``-inf``."""
    out = inputs.new_full((num_segments, *inputs.shape[1:]), float("-inf"))
    idx = dst_idx.view(-1, *([1] * (inputs.dim() - 1))).expand_as(inputs)
    return out.scatter_reduce(0, idx, inputs, reduce="amax", include_self=True)


def indexed_softmax(
    inputs: torch.Tensor,
    dst_idx: torch.Tensor,
    num_segments: int,
    multiplicative_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Softmax over each ``dst_idx`` segment.

    The per-segment maximum is subtracted for numerical stability **under
    ``stop_gradient``** (``detach()``), and an optional multiplicative mask is
    applied to the raw exponentials *before* normalization -- both exactly as
    e3x does.
    """
    maximum = indexed_max(inputs, dst_idx, num_segments)
    numerator = torch.exp(inputs - gather_dst(maximum, dst_idx).detach())
    if multiplicative_mask is not None:
        numerator = numerator * multiplicative_mask
    denominator = indexed_sum(numerator, dst_idx, num_segments)
    return numerator / gather_dst(denominator, dst_idx)


def segment_mean(
    inputs: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Segment mean; empty segments give 0 (``jraph.segment_mean`` semantics)."""
    total = indexed_sum(inputs, segment_ids, num_segments)
    count = torch.zeros(
        num_segments, dtype=inputs.dtype, device=inputs.device
    ).index_add(0, segment_ids, torch.ones_like(segment_ids, dtype=inputs.dtype))
    count = count.view(-1, *([1] * (total.dim() - 1)))
    return total / torch.where(count > 0, count, torch.ones_like(count))
