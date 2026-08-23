# Ported from SynCoGen (https://github.com/andreirekesh/SynCoGen,
# commit 6b38eec26ccc687b32808c37aefdf75a4a30f1da, MIT) --
# upstream path: syncogen/diffusion/noise/__init__.py
# gin decorators stripped (Hydra replaces gin); imports repointed.
# Any behavioural change from upstream carries an inline UPSTREAM comment.

"""Noise schedules for diffusion models."""

from MolecularDiffusion.modules.models.syncogen.diffusion.noise.base import NoiseBase
from MolecularDiffusion.modules.models.syncogen.diffusion.noise.cosine import CosineNoise, CosineSqrNoise
from MolecularDiffusion.modules.models.syncogen.diffusion.noise.linear import LinearNoise
from MolecularDiffusion.modules.models.syncogen.diffusion.noise.geometric import GeometricNoise
from MolecularDiffusion.modules.models.syncogen.diffusion.noise.loglinear import LogLinearNoise
from MolecularDiffusion.modules.models.syncogen.diffusion.noise.brownian import BrownianBridgeNoise


__all__ = [
    "Noise",
    "CosineNoise",
    "CosineSqrNoise",
    "LinearNoise",
    "GeometricNoise",
    "LogLinearNoise",
    "BrownianBridgeNoise",
]
