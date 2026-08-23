# Ported from SynCoGen (https://github.com/andreirekesh/SynCoGen,
# commit 6b38eec26ccc687b32808c37aefdf75a4a30f1da, MIT) --
# upstream path: syncogen/diffusion/sampling/integrators/base.py
# gin decorators stripped (Hydra replaces gin); imports repointed.
# Any behavioural change from upstream carries an inline UPSTREAM comment.

"""Base class for continuous (coordinate) integrators."""

import torch
from abc import ABC, abstractmethod

from MolecularDiffusion.modules.models.syncogen.api.atomics.coordinates import Coordinates


class IntegratorBase(ABC):
    """Base class for continuous integrators (coordinates).

    Integrators accept Coordinates objects and extract what they need internally.
    This keeps the call site clean and provides access to masks, pharmacophores, etc.
    """

    @abstractmethod
    def step(
        self,
        coords: Coordinates,
        C0_pred: torch.Tensor,
        t: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Perform one integration step.

        Args:
            coords: Coordinates object with current noisy coordinates
            C0_pred: Predicted clean coordinates from backbone
            t: Current timestep (B, 1) or scalar
            dt: Time step size

        Returns:
            C_next: Updated coordinates tensor
        """
        raise NotImplementedError
