"""PMDM: pocket-conditioned 3D molecular diffusion.

Ported from https://github.com/Layne-Huang/PMDM ("A dual diffusion model
enables 3D molecule generation and lead optimization based on target pockets").
See ``docs/model_integrations/pmdm/INTEGRATION_PLAN.md`` for what is and is
not in scope.
"""

from .epsnet import PMDMEpsNet, get_beta_schedule

__all__ = ["PMDMEpsNet", "get_beta_schedule"]
