from typing import Optional

import torch

from .scatter import scatter


def segment_coo(
    src: torch.Tensor,
    index: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    """Reduce over sorted ``index`` along ``index.dim() - 1``."""
    return scatter(src, index, index.dim() - 1, out, dim_size, reduce)


def segment_csr(
    src: torch.Tensor,
    indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    """Reduce ``src`` segments delimited by a 1-D ``indptr`` along dim 0."""
    if indptr.dim() != 1:
        raise NotImplementedError("mac shim segment_csr supports 1-D indptr only")
    counts = indptr[1:] - indptr[:-1]
    num_segments = counts.numel()
    index = torch.repeat_interleave(
        torch.arange(num_segments, device=src.device), counts.to(src.device)
    )
    # Elements before indptr[0] / after indptr[-1] belong to no segment.
    start = int(indptr[0])
    body = src.narrow(0, start, index.numel())
    return scatter(body, index, 0, out, num_segments, reduce)
