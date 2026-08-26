"""ChefNMR: 3D structure elucidation from NMR spectra.

Xiong, Zhang, Alauddin, Cheng, An, Seyedsayamdost, Zhong,
*"Atomic Diffusion Models for Small Molecule Structure Elucidation from NMR
Spectra"*, NeurIPS 2025. arXiv:2512.03127.
Upstream: https://github.com/ml-struct-bio/chefnmr (MIT).

EDM/AlphaFold3 diffusion over Cartesian atom coordinates with a plain
(non-equivariant) DiT denoiser, conditioned on a binned 1H/13C NMR pair and
on the **known chemical formula** -- the atom types are an input, never a
prediction. Bonds are perceived from the generated geometry afterwards by
RDKit; the model has no bond channel at all.
"""

from MolecularDiffusion.modules.models.chefnmr.diffusion import AtomDiffusion
from MolecularDiffusion.modules.models.chefnmr.score_models import (
    DiffusionModuleTransformer,
)
from MolecularDiffusion.modules.models.chefnmr.sidecar import (
    ChefNMRSidecar,
    load_sidecar,
    parse_row_index,
)

__all__ = [
    "AtomDiffusion",
    "ChefNMRSidecar",
    "DiffusionModuleTransformer",
    "load_sidecar",
    "parse_row_index",
]
