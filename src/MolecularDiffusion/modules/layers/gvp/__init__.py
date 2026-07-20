"""Reusable Geometric Vector Perceptron (GVP) equivariant building blocks.

Ported from FlowMol (``flowmol/models/gvp.py``) — the generic GVP / GVPConv
message-passing primitives plus the node-position and edge-feature update
blocks used by the FlowMol vector field. These are architecture-agnostic
SE(3)-equivariant blocks, so they live under ``modules/layers`` rather than
inside a single model package.
"""

from MolecularDiffusion.modules.layers.gvp.gvp import (
    GVP,
    GVPConv,
    GVPDropout,
    GVPLayerNorm,
    EdgeUpdate,
    NodePositionUpdate,
    _norm_no_nan,
    _rbf,
    get_time_embedding,
)

__all__ = [
    "GVP",
    "GVPConv",
    "GVPDropout",
    "GVPLayerNorm",
    "EdgeUpdate",
    "NodePositionUpdate",
    "_norm_no_nan",
    "_rbf",
    "get_time_embedding",
]
