"""LigandDiff architecture port (JCTC 2024, https://github.com/lgd6/LigandDiff).

Regenerates one ligand of a 3D transition-metal complex while keeping the
metal and the remaining ligands fixed. A continuous 3D DDPM (EDM) over
coordinates + an 8-wide heavy-atom one-hot, denoised by a GVP network wrapped
in ``egnn.Dynamics``.

Ported verbatim from the target repo's ``src/`` (commit b89b423), minus the
``egnn_dynamics`` backbone branch, the training loop, the CLI and the
RDKit/molSimplify metric stack -- see
``docs/model_integrations/ligandiff/INTEGRATION_PLAN.md``.
"""

from MolecularDiffusion.modules.models.ligandiff.edm import EDM
from MolecularDiffusion.modules.models.ligandiff.egnn import Dynamics

__all__ = ["EDM", "Dynamics"]
