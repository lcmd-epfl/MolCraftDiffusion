"""Reusable Vector Neuron (VN) SO(3)-equivariant building blocks.

Ported from DiffSMol (``source/models/shape_vn_layers.py``); originally from
Deng et al., "Vector Neurons" (ICCV 2021). Architecture-agnostic, so they
live under ``modules/layers`` rather than inside a single model package.
"""

from MolecularDiffusion.modules.layers.vn.vn_layers import (
    ResnetBlockFC,
    VNBatchNorm,
    VNLeakyReLU,
    VNLinear,
    VNLinearLeakyReLU,
    VNMaxPool,
    VNResnetBlockFC,
    VNStdFeature,
    get_graph_feature_cross,
    knn,
    mean_pool,
)

__all__ = [
    "ResnetBlockFC",
    "VNBatchNorm",
    "VNLeakyReLU",
    "VNLinear",
    "VNLinearLeakyReLU",
    "VNMaxPool",
    "VNResnetBlockFC",
    "VNStdFeature",
    "get_graph_feature_cross",
    "knn",
    "mean_pool",
]
