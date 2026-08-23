# Ported from SynCoGen (https://github.com/andreirekesh/SynCoGen,
# commit 6b38eec26ccc687b32808c37aefdf75a4a30f1da, MIT) --
# upstream path: syncogen/diffusion/loss/__init__.py
# gin decorators stripped (Hydra replaces gin); imports repointed.
# Any behavioural change from upstream carries an inline UPSTREAM comment.

"""Loss functions for diffusion models."""

from MolecularDiffusion.modules.models.syncogen.diffusion.loss.base import LossBase, LossMode, LossList
from MolecularDiffusion.modules.models.syncogen.diffusion.loss.nll import NLLLoss
from MolecularDiffusion.modules.models.syncogen.diffusion.loss.mse import MSELoss
from MolecularDiffusion.modules.models.syncogen.diffusion.loss.bond_length import BondLengthLoss
from MolecularDiffusion.modules.models.syncogen.diffusion.loss.pairwise_distance import PairwiseDistanceLoss
from MolecularDiffusion.modules.models.syncogen.diffusion.loss.smooth_lddt import SmoothLDDTLoss


__all__ = [
    "LossBase",
    "LossMode",
    "LossList",
    "NLLLoss",
    "MSELoss",
    "BondLengthLoss",
    "PairwiseDistanceLoss",
    "SmoothLDDTLoss",
]
