# Ported from SynCoGen (https://github.com/andreirekesh/SynCoGen,
# commit 6b38eec26ccc687b32808c37aefdf75a4a30f1da, MIT) --
# upstream path: syncogen/diffusion/interpolation/base.py
# gin decorators stripped (Hydra replaces gin); imports repointed.
# Any behavioural change from upstream carries an inline UPSTREAM comment.

"""Base interpolator class."""

import abc
import torch
import torch.nn as nn


class InterpolatorBase(abc.ABC, nn.Module):
    """Base interpolator class for flow matching.

    Interpolators define how to interpolate between two coordinate sets C0 and C1
    at time t in [0, 1], where t=0 corresponds to C0 and t=1 corresponds to C1.
    """

    def forward(
        self, C0: torch.Tensor, C1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Interpolate between C0 and C1 at time t.

        Args:
            C0: Initial coordinates tensor, shape [B, N, 3] or [B, N, M, 3]
            C1: Final coordinates tensor, shape [B, N, 3] or [B, N, M, 3]
            t: Time tensor, shape [B] or [B, 1, ...]

        Returns:
            Interpolated coordinates at time t, same shape as C0/C1
        """
        return self.interpolate(C0, C1, t)

    @abc.abstractmethod
    def interpolate(
        self, C0: torch.Tensor, C1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute interpolated coordinates.

        Args:
            C0: Initial coordinates tensor
            C1: Final coordinates tensor
            t: Time tensor

        Returns:
            Interpolated coordinates
        """
        pass
