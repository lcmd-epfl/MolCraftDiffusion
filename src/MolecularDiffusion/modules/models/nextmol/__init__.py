"""NExT-Mol: 3D diffusion (DMT) meets 1D language modeling (MoLlama).

Liu*, Luo*, Huang et al., *NExT-Mol: 3D Diffusion Meets 1D Language Modeling
for 3D Molecule Generation*, ICLR 2025 (arXiv:2502.12638).

The two halves are fully decoupled -- their only interface is a flat list of
SMILES. :mod:`mollama` writes the molecule down; :mod:`dgt` places it in 3D.
"""

from MolecularDiffusion.modules.models.nextmol.dgt import (
    DGTDiffusion,
    coord2dist,
    get_align_noise,
    kabsch_batch,
    remove_mean,
    sample_com_rand_pos,
)
from MolecularDiffusion.modules.models.nextmol.featurize import (
    BOND_CLASSES,
    atom_types_for,
    drugs_types,
    featurize_mol,
    qm9_types,
)
from MolecularDiffusion.modules.models.nextmol.scheduler import (
    NoiseScheduleVPV2,
)

__all__ = [
    "BOND_CLASSES",
    "DGTDiffusion",
    "NoiseScheduleVPV2",
    "atom_types_for",
    "coord2dist",
    "drugs_types",
    "featurize_mol",
    "get_align_noise",
    "kabsch_batch",
    "qm9_types",
    "remove_mean",
    "sample_com_rand_pos",
]

# `mollama` is NOT re-exported: it imports `transformers` lazily, but keeping it
# out of this namespace means `from ...nextmol import DGTDiffusion` never even
# touches it. Import it explicitly where the LM half is actually used.
