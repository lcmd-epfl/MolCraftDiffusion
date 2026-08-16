"""MiDi backbone: mixed graph + 3D denoising diffusion.

Vignac, Osman, Toni, Frossard, "MiDi: Mixed Graph and 3D Denoising Diffusion
for Molecule Generation", ECML 2023, arXiv:2302.09048.

Ported from the reference implementation with the training loop, PyG data
loading, wandb/torchmetrics accumulators, molecular metrics, ``ExtraFeatures``
and the variational-NLL machinery left behind -- the platform already covers
those. What remains is the network, the noise process and the sampling math,
which is all ``modules/tasks/diffusion_midi.py`` needs.
"""

from .noise_model import (
    DiscreteUniformTransition,
    MarginalUniformTransition,
    NoiseModel,
)
from .placeholder import Dims, PlaceHolder, remove_mean_with_mask
from .transformer_model import GraphTransformer

__all__ = [
    "Dims",
    "DiscreteUniformTransition",
    "GraphTransformer",
    "MarginalUniformTransition",
    "NoiseModel",
    "PlaceHolder",
    "remove_mean_with_mask",
]
