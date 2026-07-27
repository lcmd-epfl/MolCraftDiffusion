"""Vector Neuron (VN) equivariant building blocks.

Ported from DiffSMol (``source/models/shape_vn_layers.py``), which in turn
derives from Deng et al., "Vector Neurons: A General Framework for SO(3)-
Equivariant Networks" (ICCV 2021).

A VN feature is a tensor of shape ``[B, C, 3, N, ...]`` -- ``C`` channels of
3-vectors. Every op here commutes with a global rotation applied to the
``3`` axis, so a stack of them is exactly SO(3)-equivariant. That is what
makes the DiffSMol shape latent a ``(128, 3)`` *equivariant* embedding
rather than an invariant descriptor.

Note these blocks are NOT translation-equivariant: ``get_graph_feature_cross``
concatenates the raw coordinate ``x`` as a channel, so the caller is
responsible for centering its input first.
"""

from __future__ import annotations

import torch
import torch.nn as nn

EPS = 1e-6

__all__ = [
    "VNLinear",
    "VNLeakyReLU",
    "VNBatchNorm",
    "VNMaxPool",
    "VNLinearLeakyReLU",
    "VNResnetBlockFC",
    "VNStdFeature",
    "ResnetBlockFC",
    "mean_pool",
    "knn",
    "get_graph_feature_cross",
]


class VNLinear(nn.Module):
    """Channel-mixing linear map; acts on the channel axis only."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: ``[B, C_in, 3, N, ...]`` -> ``[B, C_out, 3, N, ...]``."""
        return self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)


class VNLeakyReLU(nn.Module):
    """Equivariant leaky ReLU: reflect the component of ``x`` that lies on
    the negative side of a learned direction ``d``."""

    def __init__(
        self,
        in_channels: int,
        share_nonlinearity: bool = False,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        out = 1 if share_nonlinearity else in_channels
        self.map_to_dir = nn.Linear(in_channels, out, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        return self.negative_slope * x + (1 - self.negative_slope) * (
            mask * x
            + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d)
        )


class VNBatchNorm(nn.Module):
    """BatchNorm on the per-channel vector *norms*, leaving directions
    untouched (and therefore equivariance intact)."""

    def __init__(self, num_features: int, dim: int) -> None:
        super().__init__()
        self.dim = dim
        if dim in (3, 4):
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)
        else:
            raise ValueError(f"VNBatchNorm: unsupported dim {dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        return x / norm.unsqueeze(2) * norm_bn.unsqueeze(2)


class VNMaxPool(nn.Module):
    """Max-pool over the last axis, selecting by projection onto a learned
    direction (an argmax over a scalar, hence equivariant)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = tuple(
            torch.meshgrid(
                *[torch.arange(j, device=x.device) for j in x.size()[:-1]],
                indexing="ij",
            )
        ) + (idx,)
        return x[index_tuple]


class VNLinearLeakyReLU(nn.Module):
    """``VNLinear`` -> optional ``VNBatchNorm`` -> equivariant leaky ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int = 5,
        share_nonlinearity: bool = False,
        negative_slope: float = 0.2,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.negative_slope = negative_slope
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.batchnorm = VNBatchNorm(out_channels, dim=dim)
        out = 1 if share_nonlinearity else out_channels
        self.map_to_dir = nn.Linear(in_channels, out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        if self.use_batchnorm:
            p = self.batchnorm(p)
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (p * d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdims=True)
        return self.negative_slope * p + (1 - self.negative_slope) * (
            mask * p
            + (1 - mask) * (p - (dotprod / (d_norm_sq + EPS)) * d)
        )


class VNResnetBlockFC(nn.Module):
    """Fully connected VN ResNet block."""

    def __init__(
        self,
        size_in: int,
        size_out: int | None = None,
        size_h: int | None = None,
    ) -> None:
        super().__init__()
        size_out = size_in if size_out is None else size_out
        size_h = min(size_in, size_out) if size_h is None else size_h
        self.size_in, self.size_h, self.size_out = size_in, size_h, size_out

        self.fc_0 = VNLinear(size_in, size_h)
        self.fc_1 = VNLinear(size_h, size_out)
        self.actvn_0 = VNLeakyReLU(size_in, negative_slope=0.0)
        self.actvn_1 = VNLeakyReLU(size_h, negative_slope=0.0)
        self.shortcut = (
            None if size_in == size_out else VNLinear(size_in, size_out)
        )
        nn.init.zeros_(self.fc_1.map_to_feat.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dx = self.fc_1(self.actvn_1(self.fc_0(self.actvn_0(x))))
        x_s = x if self.shortcut is None else self.shortcut(x)
        return x_s + dx


class VNStdFeature(nn.Module):
    """Project VN features onto a learned equivariant frame, producing an
    *invariant* descriptor plus the frame itself."""

    def __init__(
        self,
        in_channels: int,
        dim: int = 4,
        normalize_frame: bool = False,
        share_nonlinearity: bool = False,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame
        self.vn1 = VNLinearLeakyReLU(
            in_channels,
            in_channels // 2,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
        )
        self.vn2 = VNLinearLeakyReLU(
            in_channels // 2,
            in_channels // 4,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
        )
        self.vn_lin = nn.Linear(
            in_channels // 4, 2 if normalize_frame else 3, bias=False
        )

    def forward(self, x: torch.Tensor):
        z0 = self.vn2(self.vn1(x))
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)

        if self.normalize_frame:
            v1 = z0[:, 0, :]
            u1 = v1 / (torch.sqrt((v1 * v1).sum(1, keepdims=True)) + EPS)
            v2 = z0[:, 1, :]
            v2 = v2 - (v2 * u1).sum(1, keepdims=True) * u1
            u2 = v2 / (torch.sqrt((v2 * v2).sum(1, keepdims=True)) + EPS)
            u3 = torch.cross(u1, u2, dim=-1)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)

        if self.dim == 4:
            x_std = torch.einsum("bijm,bjkm->bikm", x, z0)
        elif self.dim == 3:
            x_std = torch.einsum("bij,bjk->bik", x, z0)
        elif self.dim == 5:
            x_std = torch.einsum("bijmn,bjkmn->bikmn", x, z0)
        else:
            raise ValueError(f"VNStdFeature: unsupported dim {self.dim}")
        return x_std, z0


class ResnetBlockFC(nn.Module):
    """Plain (non-VN) fully connected ResNet block, used by the decoder that
    consumes the invariant features produced from VN latents."""

    def __init__(
        self,
        size_in: int,
        size_out: int | None = None,
        size_h: int | None = None,
    ) -> None:
        super().__init__()
        size_out = size_in if size_out is None else size_out
        size_h = min(size_in, size_out) if size_h is None else size_h
        self.size_in, self.size_h, self.size_out = size_in, size_h, size_out
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()
        self.shortcut = (
            None
            if size_in == size_out
            else nn.Linear(size_in, size_out, bias=False)
        )
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dx = self.fc_1(self.actvn(self.fc_0(self.actvn(x))))
        x_s = x if self.shortcut is None else self.shortcut(x)
        return x_s + dx


def mean_pool(
    x: torch.Tensor, dim: int = -1, keepdim: bool = False
) -> torch.Tensor:
    return x.mean(dim=dim, keepdim=keepdim)


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """k-nearest-neighbour indices for ``x`` of shape ``[B, C, N]``."""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]


def get_graph_feature_cross(
    x: torch.Tensor,
    k: int = 20,
    idx: torch.Tensor | None = None,
    if_cross: bool = False,
) -> torch.Tensor:
    """Build a kNN edge feature tensor from VN features.

    ``x``: ``[B, C, 3, N]`` -> ``[B, 2C (or 3C), 3, N, k]``.

    The raw ``x`` is concatenated as a channel, so this is rotation- but NOT
    translation-equivariant: center the point cloud before calling.
    """
    batch_size = x.size(0)
    num_points = x.size(3)

    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)

    idx_base = (
        torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)
        * num_points
    )
    idx = (idx + idx_base).view(-1).long()

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    if if_cross:
        cross = torch.cross(feature, x, dim=-1)
        feature = torch.cat((feature - x, x, cross), dim=3)
    else:
        feature = torch.cat((feature - x, x), dim=3)
    return feature.permute(0, 3, 4, 1, 2).contiguous()
