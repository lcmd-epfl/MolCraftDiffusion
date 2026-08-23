# Ported from SynCoGen (https://github.com/andreirekesh/SynCoGen,
# commit 6b38eec26ccc687b32808c37aefdf75a4a30f1da, MIT) --
# upstream path: syncogen/diffusion/noise/geometric.py
# gin decorators stripped (Hydra replaces gin); imports repointed.
# Any behavioural change from upstream carries an inline UPSTREAM comment.

"""Geometric noise schedule."""

import torch
from MolecularDiffusion.modules.models.syncogen.diffusion.noise.base import NoiseBase


class GeometricNoise(NoiseBase):
    """Geometric noise schedule."""

    def __init__(self, sigma_min: float = 1e-3, sigma_max: float = 1):
        super().__init__()
        self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])

    def rate_noise(self, t):
        """Rate of change of noise."""
        return (
            self.sigmas[0] ** (1 - t)
            * self.sigmas[1] ** t
            * (self.sigmas[1].log() - self.sigmas[0].log())
        )

    def total_noise(self, t):
        """Total noise."""
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t
