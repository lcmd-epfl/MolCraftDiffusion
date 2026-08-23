# Ported from SynCoGen (https://github.com/andreirekesh/SynCoGen,
# commit 6b38eec26ccc687b32808c37aefdf75a4a30f1da, MIT) --
# upstream path: syncogen/diffusion/noise/base.py
# gin decorators stripped (Hydra replaces gin); imports repointed.
# Any behavioural change from upstream carries an inline UPSTREAM comment.

"""Base noise schedule class."""

import abc
import torch
import torch.nn as nn


class NoiseBase(abc.ABC, nn.Module):
    """Base noise schedule class."""

    def forward(self, t):
        """Get total and rate of noise at timestep t."""
        return self.total_noise(t), self.rate_noise(t)

    @abc.abstractmethod
    def rate_noise(self, t):
        """Rate of change of noise (g(t))."""
        pass

    @abc.abstractmethod
    def total_noise(self, t):
        """Total noise (integral of g(t) from 0 to t + g(0))."""
        pass
