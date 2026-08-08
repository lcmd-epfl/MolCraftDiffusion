"""IPDiff: TargetDiff's denoiser conditioned on a pretrained binding prior.

Only two genuinely new pieces live here -- :class:`BAPNet` (the frozen
"IPNet" interaction prior) and :class:`IPDiffScorePosNet3D` (a thin subclass
of KGDiff's ``ScorePosNet3D`` adding prior conditioning and prior shifting).
The backbone (``uni_transformer.py``) and its building blocks (``common.py``)
are reused by import from ``modules/models/kgdiff/``, having been verified
byte-identical to IPDiff's copies modulo import paths and docstrings.

See ``docs/model_integrations/ipdiff/INTEGRATION_PLAN.md``.
"""

from MolecularDiffusion.modules.models.ipdiff.bapnet import BAPNet
from MolecularDiffusion.modules.models.ipdiff.score_model import (
    IPDiffScorePosNet3D,
)

__all__ = ["BAPNet", "IPDiffScorePosNet3D"]
