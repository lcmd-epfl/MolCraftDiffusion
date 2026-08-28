"""DiffSpectra: DMT (SE(3)-equivariant graph transformer) + SpecFormer
(spectral encoder). Liang Wang et al., "DiffSpectra: Molecular Structure
Elucidation from Spectra using Diffusion Models", arXiv:2507.06853.
https://github.com/AzureLeon1/DiffSpectra
"""

from MolecularDiffusion.modules.models.diffspectra.dmt import DMT
from MolecularDiffusion.modules.models.diffspectra.specformer import SpecFormer

__all__ = ["DMT", "SpecFormer"]
