from typing import Optional

import torch

from ..scatter import broadcast, scatter_max, scatter_sum


def scatter_softmax(
    src: torch.Tensor, index: torch.Tensor, dim: int = -1, dim_size: Optional[int] = None
) -> torch.Tensor:
    if not torch.is_floating_point(src):
        raise ValueError("`scatter_softmax` can only be computed over tensors with floating point data types.")
    index = broadcast(index, src, dim)
    max_value_per_index = scatter_max(src, index, dim=dim, dim_size=dim_size)[0]
    max_per_src_element = max_value_per_index.gather(dim, index)
    recentered_scores_exp = (src - max_per_src_element).exp_()
    sum_per_index = scatter_sum(recentered_scores_exp, index, dim, dim_size=dim_size)
    normalizing_constants = sum_per_index.gather(dim, index)
    return recentered_scores_exp.div(normalizing_constants)


def scatter_log_softmax(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-12,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    if not torch.is_floating_point(src):
        raise ValueError("`scatter_log_softmax` can only be computed over tensors with floating point data types.")
    index = broadcast(index, src, dim)
    max_value_per_index = scatter_max(src, index, dim=dim, dim_size=dim_size)[0]
    max_per_src_element = max_value_per_index.gather(dim, index)
    recentered_scores = src - max_per_src_element
    sum_per_index = scatter_sum(recentered_scores.exp(), index, dim, dim_size=dim_size)
    normalizing_constants = sum_per_index.add_(eps).log_().gather(dim, index)
    return recentered_scores.sub_(normalizing_constants)


def scatter_logsumexp(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    if not torch.is_floating_point(src):
        raise ValueError("`scatter_logsumexp` can only be computed over tensors with floating point data types.")
    index = broadcast(index, src, dim)
    max_value_per_index = scatter_max(src, index, dim=dim, dim_size=dim_size)[0]
    max_per_src_element = max_value_per_index.gather(dim, index)
    recentered = src - max_per_src_element
    recentered.masked_fill_(torch.isnan(recentered), float("-inf"))
    sum_per_index = scatter_sum(recentered.exp_(), index, dim, dim_size=dim_size)
    result = sum_per_index.add_(eps).log_().add_(max_value_per_index)
    if out is not None:
        return out.copy_(result)
    return result
