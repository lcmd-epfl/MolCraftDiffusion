# Ported from SynCoGen (https://github.com/andreirekesh/SynCoGen,
# commit 6b38eec26ccc687b32808c37aefdf75a4a30f1da, MIT) --
# upstream path: syncogen/diffusion/noise/brownian.py
# gin decorators stripped (Hydra replaces gin); imports repointed.
# Any behavioural change from upstream carries an inline UPSTREAM comment.

"""Brownian bridge noise schedule."""

import torch
from MolecularDiffusion.modules.models.syncogen.diffusion.noise.base import NoiseBase


class BrownianBridgeNoise(NoiseBase):
    """Brownian bridge noise schedule (from ETFlow)."""

    def __init__(self, eps: float = 1e-3, weight: float = 0.1):
        super().__init__()
        self.eps = eps
        self.weight = weight

    def rate_noise(self, t):
        """Rate of change of noise."""
        numerator = 1 - 2 * t
        denominator = 2 * torch.sqrt(t * (1 - t) + self.eps)
        return numerator / denominator * self.weight

    def total_noise(self, t):
        """Total noise."""
        return torch.sqrt(t * (1 - t)) * self.weight
