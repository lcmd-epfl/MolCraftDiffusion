"""DiffSMol: shape-conditioned hybrid 3D molecular diffusion.

Ported from https://github.com/NingQuanwei/DiffSMol (commit 2f72f80).
Bond-free scope; shape conditioning and D3PM atom-type diffusion kept.
See ``docs/model_integrations/diffsmol/INTEGRATION_PLAN.md``.

``shape_utils`` is deliberately NOT re-exported here -- it pulls optional
``scikit-image``/``trimesh`` deps that training and generation must not
require.
"""

from MolecularDiffusion.modules.models.diffsmol.score_model import (
    ScorePosNet3D,
)
from MolecularDiffusion.modules.models.diffsmol.shape_ae import (
    DEFAULT_SHAPE_AE_CHECKPOINT,
    PointCloudAE,
    load_shape_ae,
)
from MolecularDiffusion.modules.models.diffsmol.uni_transformer import (
    UniTransformerO2TwoUpdateGeneral,
)

__all__ = [
    "DEFAULT_SHAPE_AE_CHECKPOINT",
    "PointCloudAE",
    "ScorePosNet3D",
    "UniTransformerO2TwoUpdateGeneral",
    "load_shape_ae",
]
