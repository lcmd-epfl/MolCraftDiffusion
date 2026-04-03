"""
Minimal OCP compatibility layer for ShEPhERD's EquiformerV2 encoder.

Provides lightweight stubs for OCP utilities that ShEPhERD depends on,
without requiring the full OCP (ocpmodels) package.

Ported from:
- shepherd/src/shepherd/model/equiformer_v2/ocpmodels/common/registry.py
- shepherd/src/shepherd/model/equiformer_v2/ocpmodels/common/utils.py
- shepherd/src/shepherd/model/equiformer_v2/ocpmodels/models/base.py
- shepherd/src/shepherd/model/equiformer_v2/ocpmodels/models/scn/sampling.py
- shepherd/src/shepherd/model/equiformer_v2/ocpmodels/models/scn/smearing.py
"""

import math
import logging

import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph


# ============================================================
# Registry stub — only used as a decorator, no real registry needed
# ============================================================

class _Registry:
    def register_model(self, name):
        """No-op decorator that just returns the class unchanged."""
        def decorator(cls):
            return cls
        return decorator

registry = _Registry()


# ============================================================
# conditional_grad — not actually called in ShEPhERD's encoder
# ============================================================

def conditional_grad(dec):
    """Identity decorator (conditional_grad is unused in ShEPhERD)."""
    def decorator(func):
        return func
    return decorator


# ============================================================
# BaseModel — minimal base class
# ============================================================

class BaseModel(nn.Module):
    def __init__(self, num_atoms=None, bond_feat_dim=None, num_targets=None):
        super(BaseModel, self).__init__()
        self.num_atoms = num_atoms
        self.bond_feat_dim = bond_feat_dim
        self.num_targets = num_targets

    def forward(self, data):
        raise NotImplementedError

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# CalcSpherePoints — Fibonacci sphere sampling
# ============================================================

def CalcSpherePoints(num_points, device):
    goldenRatio = (1 + 5**0.5) / 2
    i = torch.arange(num_points, device=device).view(-1, 1)
    theta = 2 * math.pi * i / goldenRatio
    phi = torch.arccos(1 - 2 * (i + 0.5) / num_points)
    points = torch.cat(
        [
            torch.cos(theta) * torch.sin(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(phi),
        ],
        dim=1,
    )

    # weight the points by their density
    pt_cross = points.view(1, -1, 3) - points.view(-1, 1, 3)
    pt_cross = torch.sum(pt_cross**2, dim=2)
    pt_cross = torch.exp(-pt_cross / (0.5 * 0.3))
    scalar = 1.0 / torch.sum(pt_cross, dim=1)
    scalar = num_points * scalar / torch.sum(scalar)
    return points * (scalar.view(-1, 1))


# ============================================================
# Smearing functions — distance encodings
# ============================================================

class GaussianSmearing(torch.nn.Module):
    def __init__(
        self, start=-5.0, stop=5.0, num_gaussians=50, basis_width_scalar=1.0
    ):
        super(GaussianSmearing, self).__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = (
            -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        )
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class SigmoidSmearing(torch.nn.Module):
    def __init__(
        self, start=-5.0, stop=5.0, num_sigmoid=50, basis_width_scalar=1.0
    ):
        super(SigmoidSmearing, self).__init__()
        self.num_output = num_sigmoid
        offset = torch.linspace(start, stop, num_sigmoid)
        self.coeff = (basis_width_scalar / (offset[1] - offset[0])).item()
        self.register_buffer("offset", offset)

    def forward(self, dist):
        exp_dist = self.coeff * (dist.view(-1, 1) - self.offset.view(1, -1))
        return torch.sigmoid(exp_dist)


class LinearSigmoidSmearing(torch.nn.Module):
    def __init__(
        self, start=-5.0, stop=5.0, num_sigmoid=50, basis_width_scalar=1.0
    ):
        super(LinearSigmoidSmearing, self).__init__()
        self.num_output = num_sigmoid
        offset = torch.linspace(start, stop, num_sigmoid)
        self.coeff = (basis_width_scalar / (offset[1] - offset[0])).item()
        self.register_buffer("offset", offset)

    def forward(self, dist):
        exp_dist = self.coeff * (dist.view(-1, 1) - self.offset.view(1, -1))
        x_dist = torch.sigmoid(exp_dist) + 0.001 * exp_dist
        return x_dist


class SiLUSmearing(torch.nn.Module):
    def __init__(
        self, start=-5.0, stop=5.0, num_output=50, basis_width_scalar=1.0
    ):
        super(SiLUSmearing, self).__init__()
        self.num_output = num_output
        self.fc1 = nn.Linear(2, num_output)
        self.act = nn.SiLU()

    def forward(self, dist):
        x_dist = dist.view(-1, 1)
        x_dist = torch.cat([x_dist, torch.ones_like(x_dist)], dim=1)
        x_dist = self.act(self.fc1(x_dist))
        return x_dist
