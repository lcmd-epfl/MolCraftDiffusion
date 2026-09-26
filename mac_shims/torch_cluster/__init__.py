"""Pure-torch stand-in for ``torch_cluster`` (macOS / Apple MPS only).

``radius``, ``radius_graph``, ``knn`` and ``knn_graph`` with torch_cluster 1.6
signatures and edge conventions, computed with blocked dense distances so they
run on MPS as well as CPU. torch_geometric < 2.8 routes
``torch_geometric.nn.radius_graph`` / ``knn_graph`` through this module.
Dense is fine at molecule scale; queries are processed in row blocks to bound
memory on large batches.
"""

from typing import Optional

import torch

__version__ = "1.6.3+molcraft.shim"

_BLOCK = 4096


def _batch_or_zeros(batch, n, device):
    if batch is None:
        return torch.zeros(n, dtype=torch.long, device=device)
    return batch.to(device)


def radius(
    x: torch.Tensor,
    y: torch.Tensor,
    r: float,
    batch_x: Optional[torch.Tensor] = None,
    batch_y: Optional[torch.Tensor] = None,
    max_num_neighbors: int = 32,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """For each ``y[i]`` find up to ``max_num_neighbors`` points ``x[j]`` with
    ``|x[j] - y[i]| <= r`` in the same batch. Returns ``[row (y), col (x)]``.
    When more than ``max_num_neighbors`` qualify, the nearest are kept (as the
    torch_cluster CPU kernel does), so a point's self-match is always kept.
    """
    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    device = x.device
    batch_x = _batch_or_zeros(batch_x, x.size(0), device)
    batch_y = _batch_or_zeros(batch_y, y.size(0), device)
    if x.numel() == 0 or y.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=device)

    kk = min(max_num_neighbors, x.size(0))
    rows, cols = [], []
    for start in range(0, y.size(0), _BLOCK):
        yb = y[start : start + _BLOCK]
        dist = torch.cdist(yb, x)
        hit = (dist <= r) & (batch_y[start : start + _BLOCK, None] == batch_x[None, :])
        # Keep the nearest `max_num_neighbors` hits per query, like the CPU kernel.
        val, idx = dist.masked_fill(~hit, float("inf")).topk(kk, dim=1, largest=False)
        keep = torch.isfinite(val)
        r_idx = torch.arange(yb.size(0), device=device)[:, None].expand_as(idx)
        rows.append(r_idx[keep] + start)
        cols.append(idx[keep])
    return torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)


def radius_graph(
    x: torch.Tensor,
    r: float,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = "source_to_target",
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    assert flow in ["source_to_target", "target_to_source"]
    edge_index = radius(
        x, x, r, batch, batch,
        max_num_neighbors if loop else max_num_neighbors + 1,
        num_workers, batch_size,
    )
    if flow == "source_to_target":
        row, col = edge_index[1], edge_index[0]
    else:
        row, col = edge_index[0], edge_index[1]
    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0)


def knn(
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
    batch_x: Optional[torch.Tensor] = None,
    batch_y: Optional[torch.Tensor] = None,
    cosine: bool = False,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """For each ``y[i]`` the ``k`` nearest ``x[j]`` in the same batch
    (``batch_size`` accepted for API parity). Returns ``[row (y), col (x)]``."""
    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    device = x.device
    batch_x = _batch_or_zeros(batch_x, x.size(0), device)
    batch_y = _batch_or_zeros(batch_y, y.size(0), device)
    if x.numel() == 0 or y.numel() == 0 or k <= 0:
        return torch.empty(2, 0, dtype=torch.long, device=device)

    kk = min(k, x.size(0))
    rows, cols = [], []
    for start in range(0, y.size(0), _BLOCK):
        yb = y[start : start + _BLOCK]
        if cosine:
            dist = 1 - torch.nn.functional.normalize(yb, dim=-1) @ torch.nn.functional.normalize(x, dim=-1).T
        else:
            dist = torch.cdist(yb, x)
        same = batch_y[start : start + _BLOCK, None] == batch_x[None, :]
        dist = dist.masked_fill(~same, float("inf"))
        val, idx = dist.topk(kk, dim=1, largest=False)
        keep = torch.isfinite(val)
        r_idx = torch.arange(yb.size(0), device=device)[:, None].expand_as(idx)
        rows.append(r_idx[keep] + start)
        cols.append(idx[keep])
    return torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)


def knn_graph(
    x: torch.Tensor,
    k: int,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
    flow: str = "source_to_target",
    cosine: bool = False,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    assert flow in ["source_to_target", "target_to_source"]
    edge_index = knn(x, x, k if loop else k + 1, batch, batch, cosine, num_workers, batch_size)
    if flow == "source_to_target":
        row, col = edge_index[1], edge_index[0]
    else:
        row, col = edge_index[0], edge_index[1]
    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0)


def _unsupported(name):
    def fn(*args, **kwargs):
        raise NotImplementedError(
            f"torch_cluster.{name} is not provided by the MolCraftDiffusion mac shim "
            "(no MolCraftDiffusion model uses it)."
        )

    fn.__name__ = name
    return fn


# torch_geometric imports these at module load; nothing in MolCraftDiffusion calls them.
fps = _unsupported("fps")
nearest = _unsupported("nearest")
graclus_cluster = _unsupported("graclus_cluster")
grid_cluster = _unsupported("grid_cluster")
random_walk = _unsupported("random_walk")

__all__ = ["radius", "radius_graph", "knn", "knn_graph"]
