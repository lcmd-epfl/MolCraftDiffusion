"""DiTMC -- graph-conditioned Diffusion Transformer conformer generator.

Frank, Ripken, Lied, Mueller, Unke, Chmiela. *Sampling 3D Molecular Conformers
with Diffusion Transformers.* NeurIPS 2025. arXiv:2506.15378.
Data + checkpoints: doi:10.5281/zenodo.15489212.

Unlike every other generator in this repo, DiTMC does **not** invent molecules:
you hand it a molecule you already have (atoms and bonds) and it returns 3D
conformers of exactly that molecule. No atom types, bonds or charges come out.

A faithful PyTorch port of all three published variants (``dit_ape``,
``dit_rpe``, ``dit_so3``), including classifier-free guidance, self-conditioning
and sampler trajectories. The ``e3x`` surface ``dit_so3`` depends on is
reimplemented directly in ``modules/layers/e3x`` -- not substituted with e3nn,
whose irrep ordering and normalization would make the published checkpoints
unconvertible.
"""

from .build import (
    SHIPPED_GLOBALS,
    SHIPPED_VARIANTS,
    VARIANTS,
    build_variant,
    make_molecular_dit,
    make_molecular_dit_so3,
)
from .dit import DiTLayer, GenerativeModel, SO3DiTLayer
from .flow_matching import (
    FlowMatching,
    aggregate_node_error,
    center_data,
    kabsch_align,
    rotation_augmentation,
)
from .graph_features import (
    ATOMIC_TYPES,
    UNREACHABLE_HOPS,
    MoleculeFeatureCache,
    all_pairs_edges,
    node_attr_dim,
)
from .graphs import CondGraph, LatentGraph, PriorGraph
from .priors import GaussianPrior, HarmonicPrior, build_prior
from .readout import EquivariantReadout, SimpleReadout

__all__ = [
    "ATOMIC_TYPES",
    "SHIPPED_GLOBALS",
    "SHIPPED_VARIANTS",
    "UNREACHABLE_HOPS",
    "VARIANTS",
    "CondGraph",
    "DiTLayer",
    "EquivariantReadout",
    "FlowMatching",
    "GaussianPrior",
    "GenerativeModel",
    "HarmonicPrior",
    "LatentGraph",
    "MoleculeFeatureCache",
    "PriorGraph",
    "SO3DiTLayer",
    "SimpleReadout",
    "aggregate_node_error",
    "all_pairs_edges",
    "build_prior",
    "build_variant",
    "center_data",
    "kabsch_align",
    "make_molecular_dit",
    "make_molecular_dit_so3",
    "node_attr_dim",
    "rotation_augmentation",
]
