"""Dense-tensor building blocks for MiDi's relational graph transformer.

These live here rather than in ``modules/layers/`` on purpose: every one of
them is hardcoded to MiDi's dense ``(B,N,·)`` / ``(B,N,N,·)`` layout and is
reused by nothing else in the platform.

Upstream's ``SetNorm``/``GraphNorm`` are omitted -- they are commented out at
every call site in ``midi/models/transformer_model.py`` (lines 38-39, 50-51,
89, 97, 106, 111) in favour of plain ``LayerNorm``, so they carry no weights
in any released checkpoint.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import init


class PositionsMLP(nn.Module):
    """Rescale coordinates by a learned function of their norm (SE(3)-safe)."""

    def __init__(self, hidden_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self, pos: torch.Tensor, node_mask: torch.Tensor
    ) -> torch.Tensor:
        """``pos (B,N,3)``, ``node_mask (B,N)`` -> rescaled, re-centred pos."""
        norm = torch.norm(pos, dim=-1, keepdim=True)  # bs, n, 1
        new_norm = self.mlp(norm)  # bs, n, 1
        new_pos = pos * new_norm / (norm + self.eps)
        new_pos = new_pos * node_mask.unsqueeze(-1)
        return new_pos - torch.mean(new_pos, dim=1, keepdim=True)


class SE3Norm(nn.Module):
    """Normalize positions by their mean norm over real nodes."""

    def __init__(
        self,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.normalized_shape = (1,)
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(self.normalized_shape, **factory_kwargs)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the single scale parameter to one."""
        init.ones_(self.weight)

    def forward(
        self, pos: torch.Tensor, node_mask: torch.Tensor
    ) -> torch.Tensor:
        """``pos (B,N,3)``, ``node_mask (B,N,1)`` -> normalized positions."""
        norm = torch.norm(pos, dim=-1, keepdim=True)  # bs, n, 1
        mean_norm = torch.sum(norm, dim=1, keepdim=True) / torch.sum(
            node_mask, dim=1, keepdim=True
        )  # bs, 1, 1
        return self.weight * pos / (mean_norm + self.eps)

    def extra_repr(self) -> str:
        """Describe the layer for ``repr``."""
        return f"{self.normalized_shape}, eps={self.eps}"


class Xtoy(nn.Module):
    """Pool node features (mean/min/max/std) into the global feature."""

    def __init__(self, dx: int, dy: int) -> None:
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:  # noqa: N803
        """``X (B,N,dx)``, ``x_mask (B,N,1)`` -> ``(B,dy)``."""
        x_mask = x_mask.expand(-1, -1, X.shape[-1])
        float_imask = 1 - x_mask.float()
        m = X.sum(dim=1) / torch.sum(x_mask, dim=1)
        mi = (X + 1e5 * float_imask).min(dim=1)[0]
        ma = (X - 1e5 * float_imask).max(dim=1)[0]
        std = torch.sum(((X - m[:, None, :]) ** 2) * x_mask, dim=1) / torch.sum(
            x_mask, dim=1
        )
        return self.lin(torch.hstack((m, mi, ma, std)))


class Etoy(nn.Module):
    """Pool edge features (mean/min/max/std) into the global feature."""

    def __init__(self, d: int, dy: int) -> None:
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(
        self,
        E: torch.Tensor,  # noqa: N803
        e_mask1: torch.Tensor,
        e_mask2: torch.Tensor,
    ) -> torch.Tensor:
        """``E (B,N,N,de)`` -> ``(B,dy)``."""
        mask = (e_mask1 * e_mask2).expand(-1, -1, -1, E.shape[-1])
        float_imask = 1 - mask.float()
        divide = torch.sum(mask, dim=(1, 2))
        m = E.sum(dim=(1, 2)) / divide
        mi = (E + 1e5 * float_imask).min(dim=2)[0].min(dim=1)[0]
        ma = (E - 1e5 * float_imask).max(dim=2)[0].max(dim=1)[0]
        std = (
            torch.sum(((E - m[:, None, None, :]) ** 2) * mask, dim=(1, 2))
            / divide
        )
        return self.lin(torch.hstack((m, mi, ma, std)))


class EtoX(nn.Module):
    """Pool edge features over the second axis into node features."""

    def __init__(self, de: int, dx: int) -> None:
        super().__init__()
        self.lin = nn.Linear(4 * de, dx)

    def forward(self, E: torch.Tensor, e_mask2: torch.Tensor) -> torch.Tensor:  # noqa: N803
        """``E (B,N,N,de)`` -> ``(B,N,dx)``."""
        _bs, n, _n2, de = E.shape
        e_mask2 = e_mask2.expand(-1, n, -1, de)
        float_imask = 1 - e_mask2.float()
        m = E.sum(dim=2) / torch.sum(e_mask2, dim=2)
        mi = (E + 1e5 * float_imask).min(dim=2)[0]
        ma = (E - 1e5 * float_imask).max(dim=2)[0]
        std = torch.sum(
            ((E - m[:, :, None, :]) ** 2) * e_mask2, dim=2
        ) / torch.sum(e_mask2, dim=2)
        return self.lin(torch.cat((m, mi, ma, std), dim=2))


def masked_softmax(
    x: torch.Tensor, mask: torch.Tensor, **kwargs: object
) -> torch.Tensor:
    """Softmax over ``x`` with ``mask == 0`` entries driven to zero."""
    if torch.sum(mask) == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)
