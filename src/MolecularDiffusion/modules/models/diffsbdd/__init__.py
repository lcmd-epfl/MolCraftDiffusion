"""DiffSBDD: E(3)-equivariant pocket-conditioned ligand diffusion.

Ported from https://github.com/arneschneuing/DiffSBDD (commit 5d0d38d).
See ``docs/model_integrations/diffsbdd/INTEGRATION_PLAN.md``.
"""

from MolecularDiffusion.modules.models.diffsbdd.dynamics import EGNNDynamics
from MolecularDiffusion.modules.models.diffsbdd.egnn import EGNN
from MolecularDiffusion.modules.models.diffsbdd.en_diffusion import (
    ConditionalDDPM,
    DistributionNodes,
    EnVariationalDiffusion,
    PredefinedNoiseSchedule,
    cosine_beta_schedule,
    polynomial_schedule,
)

#: ``mode`` -> DDPM class (``lightning_modules.py:59-61``). The third upstream
#: entry, ``pocket_conditioning_simple`` (``SimpleConditionalDDPM``), is an
#: ablation with no released weights and is out of scope.
DDPM_MODES = {
    "pocket_conditioning": ConditionalDDPM,
    "joint": EnVariationalDiffusion,
}

__all__ = [
    "DDPM_MODES",
    "ConditionalDDPM",
    "DistributionNodes",
    "EGNN",
    "EGNNDynamics",
    "EnVariationalDiffusion",
    "PredefinedNoiseSchedule",
    "cosine_beta_schedule",
    "polynomial_schedule",
]
