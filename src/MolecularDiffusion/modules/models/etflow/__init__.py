"""ET-Flow: equivariant flow matching for molecular conformer generation.

Hassan, Shenoy, Lee, Stark, Thaler, Beaini, NeurIPS 2024, arXiv:2410.22388.
Upstream: https://github.com/shenoynikhil/ETFlow (MIT).

Three pieces land here and nothing else:

``torchmd``
    The time-conditioned TorchMD-ET transformer that is the vector field.
    Module names match the released checkpoints exactly.
``flow``
    The harmonic prior over the bond-graph Laplacian, the Kabsch alignment
    that makes the regression target rotation-free, the edge-set rebuild and
    the batchwise L2 objective. Parameter-free.
``features``
    ET-Flow's 10-column atom featurization and its chiral-centre tensors.

Upstream's Lightning/optimizer plumbing, dataset code, CLI and evaluation
scripts are deliberately NOT ported -- the platform engine owns all of that.
"""

from MolecularDiffusion.modules.models.etflow.features import (
    NODE_ATTR_DIM,
    ETFlowFeatureCache,
    atom_to_feature_vector,
    get_chiral_tensors,
    graph_key,
)
from MolecularDiffusion.modules.models.etflow.flow import (
    HarmonicSampler,
    batchwise_l2_loss,
    center_of_mass,
    extend_bond_index,
    find_rigid_alignment,
    rmsd_align,
    signed_volume,
    switch_parity_of_pos,
    unsqueeze_like,
)
from MolecularDiffusion.modules.models.etflow.torchmd import TorchMDDynamics

__all__ = [
    "NODE_ATTR_DIM",
    "ETFlowFeatureCache",
    "HarmonicSampler",
    "TorchMDDynamics",
    "atom_to_feature_vector",
    "batchwise_l2_loss",
    "center_of_mass",
    "extend_bond_index",
    "find_rigid_alignment",
    "get_chiral_tensors",
    "graph_key",
    "rmsd_align",
    "signed_volume",
    "switch_parity_of_pos",
    "unsqueeze_like",
]
