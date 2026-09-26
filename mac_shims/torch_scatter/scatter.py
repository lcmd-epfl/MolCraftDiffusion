from typing import Optional, Tuple

import torch


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int) -> torch.Tensor:
    """torch_scatter.utils.broadcast: expand a (usually 1-D) index to ``other``."""
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    return src.expand(other.size())


def _out_size(src, index, dim, dim_size):
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    return size


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        out = torch.zeros(_out_size(src, index, dim, dim_size), dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index, src)


def scatter_add(src, index, dim=-1, out=None, dim_size=None):
    return scatter_sum(src, index, dim, out, dim_size)


def scatter_mul(src, index, dim=-1, out=None, dim_size=None):
    index = broadcast(index, src, dim)
    if out is None:
        out = torch.ones(_out_size(src, index, dim, dim_size), dtype=src.dtype, device=src.device)
    return out.scatter_reduce_(dim, index, src, "prod", include_self=True)


def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode="floor")
    return out


def _scatter_arg(src, index, dim, out, dim_size, reduce) -> Tuple[torch.Tensor, torch.Tensor]:
    if dim < 0:
        dim = src.dim() + dim
    index = broadcast(index, src, dim)
    size = _out_size(src, index, dim, dim_size)
    if out is None:
        # torch_scatter leaves empty segments at 0 (include_self=False keeps the init).
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
    out.scatter_reduce_(dim, index, src, reduce, include_self=False)

    # argmax/argmin: position along `dim` of the winning element; empty -> src.size(dim).
    n = src.size(dim)
    shape = [1] * src.dim()
    shape[dim] = n
    # int32 because MPS has no int64 scatter_reduce; cast back to torch_scatter's long.
    pos = torch.arange(n, dtype=torch.int32, device=src.device).view(shape).expand_as(src)
    winner = src == out.gather(dim, index)
    cand = torch.where(winner, pos, torch.full_like(pos, n))
    arg = torch.full(out.size(), n, dtype=torch.int32, device=src.device)
    arg.scatter_reduce_(dim, index, cand, "amin", include_self=True)
    return out, arg.long()


def scatter_max(src, index, dim=-1, out=None, dim_size=None):
    return _scatter_arg(src, index, dim, out, dim_size, "amax")


def scatter_min(src, index, dim=-1, out=None, dim_size=None):
    return _scatter_arg(src, index, dim, out, dim_size, "amin")


def scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
    if reduce in ("sum", "add"):
        return scatter_sum(src, index, dim, out, dim_size)
    if reduce == "mul":
        return scatter_mul(src, index, dim, out, dim_size)
    if reduce == "mean":
        return scatter_mean(src, index, dim, out, dim_size)
    if reduce == "min":
        return scatter_min(src, index, dim, out, dim_size)[0]
    if reduce == "max":
        return scatter_max(src, index, dim, out, dim_size)[0]
    raise ValueError(f"Unsupported reduce: {reduce!r}")
