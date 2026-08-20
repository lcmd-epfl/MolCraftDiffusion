"""LoQI: low-energy molecular conformer generation on a fixed 2D graph.

Ported from https://github.com/.../LoQI (NVIDIA / Megalodon lineage,
Apache-2.0). Unlike every other generative model in this package, LoQI does not
invent molecules: it takes a molecule you already have -- atom types, formal
charges, bond orders and stereochemistry -- and denoises only its 3D
coordinates. Two variants share the same backbone:

* ``diffusion_loqi``      -- VDM diffusion, 25 discrete steps, self-conditioned
* ``diffusion_loqi_flow`` -- continuous flow matching, velocity prediction,
  rigid (Kabsch) optimal transport, no self-conditioning

References
----------
Nikitin, Anstine, Zubatyuk, Paliwal, Isayev, "Scalable Low-Energy Molecular
Conformer Generation with Quantum Mechanical Accuracy" (2025),
DOI 10.26434/chemrxiv-2025-k4h7v.

Reidenbach, Nikitin, Isayev, Paliwal, "Applications of Modular Co-Design for
De Novo 3D Molecule Generation" (Megalodon), arXiv:2505.18392 (2025).
"""

from .fn_model import MegaFNV3Conf
from .graph_utils import (
    CHI_BONDS,
    N_EDGE_CLASSES,
    derive_stereo_edges,
    make_graph_fully_connected,
)
from .interpolant import (
    ContinuousDiffusionInterpolant,
    ContinuousFlowMatchingInterpolant,
    build_interpolant,
)
from .self_conditioning import BaseSelfConditioningModule

__all__ = [
    "CHI_BONDS",
    "N_EDGE_CLASSES",
    "BaseSelfConditioningModule",
    "ContinuousDiffusionInterpolant",
    "ContinuousFlowMatchingInterpolant",
    "MegaFNV3Conf",
    "build_interpolant",
    "derive_stereo_edges",
    "make_graph_fully_connected",
]
