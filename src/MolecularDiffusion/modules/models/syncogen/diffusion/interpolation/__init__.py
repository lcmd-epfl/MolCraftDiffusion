# Ported from SynCoGen (https://github.com/andreirekesh/SynCoGen,
# commit 6b38eec26ccc687b32808c37aefdf75a4a30f1da, MIT) --
# upstream path: syncogen/diffusion/interpolation/__init__.py
# gin decorators stripped (Hydra replaces gin); imports repointed.
# Any behavioural change from upstream carries an inline UPSTREAM comment.

"""Interpolators for flow matching."""

from MolecularDiffusion.modules.models.syncogen.diffusion.interpolation.base import InterpolatorBase
from MolecularDiffusion.modules.models.syncogen.diffusion.interpolation.linear import LinearInterpolator
from MolecularDiffusion.modules.models.syncogen.diffusion.interpolation.geometric import GeometricInterpolator


__all__ = [
    "InterpolatorBase",
    "LinearInterpolator",
    "GeometricInterpolator",
]
