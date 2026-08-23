# Ported from SynCoGen (https://github.com/andreirekesh/SynCoGen,
# commit 6b38eec26ccc687b32808c37aefdf75a4a30f1da, MIT) --
# upstream path: syncogen/diffusion/interpolation/linear.py
# gin decorators stripped (Hydra replaces gin); imports repointed.
# Any behavioural change from upstream carries an inline UPSTREAM comment.

"""Linear interpolator."""

import torch
from MolecularDiffusion.modules.models.syncogen.diffusion.interpolation.base import InterpolatorBase


class LinearInterpolator(InterpolatorBase):
    """Linear interpolator for flow matching.

    Implements simple linear interpolation: φ_t(C0, C1) = (1-t) * C0 + t * C1
    """

    def interpolate(
        self, C0: torch.Tensor, C1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Linearly interpolate between C0 and C1.

        Args:
            C0: Initial coordinates tensor
            C1: Final coordinates tensor
            t: Time tensor, shape [B] or [B, 1, ...]

        Returns:
            Linearly interpolated coordinates
        """
        # Ensure t has compatible shape for broadcasting
        while t.dim() < C0.dim():
            t = t.unsqueeze(-1)

        return (1 - t) * C0 + t * C1
