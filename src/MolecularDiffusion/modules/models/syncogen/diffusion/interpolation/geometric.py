# Ported from SynCoGen (https://github.com/andreirekesh/SynCoGen,
# commit 6b38eec26ccc687b32808c37aefdf75a4a30f1da, MIT) --
# upstream path: syncogen/diffusion/interpolation/geometric.py
# gin decorators stripped (Hydra replaces gin); imports repointed.
# Any behavioural change from upstream carries an inline UPSTREAM comment.

"""Geometric interpolator."""

import torch
from MolecularDiffusion.modules.models.syncogen.diffusion.interpolation.base import InterpolatorBase


class GeometricInterpolator(InterpolatorBase):
    """Geometric interpolator for flow matching.

    Implements geometric interpolation: φ_t(C0, C1) = C0^(1-t) * C1^t
    where exponentiation is applied element-wise.

    Note: This interpolator requires C0 and C1 to have the same sign.
    For coordinates, you may want to use a linear interpolator instead.
    """

    def interpolate(
        self, C0: torch.Tensor, C1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Geometrically interpolate between C0 and C1.

        Args:
            C0: Initial coordinates tensor
            C1: Final coordinates tensor
            t: Time tensor, shape [B] or [B, 1, ...]

        Returns:
            Geometrically interpolated coordinates
        """
        # Ensure t has compatible shape for broadcasting
        while t.dim() < C0.dim():
            t = t.unsqueeze(-1)

        # Use log-space for numerical stability: exp((1-t)*log(C0) + t*log(C1))
        # Note: For coordinates, this may not be meaningful since they can be negative
        # This is primarily useful for positive-valued features
        C0_log = torch.sign(C0) * torch.log(torch.abs(C0) + 1e-8)
        C1_log = torch.sign(C1) * torch.log(torch.abs(C1) + 1e-8)

        interpolated_log = (1 - t) * C0_log + t * C1_log
        return torch.sign(interpolated_log) * torch.exp(torch.abs(interpolated_log))
