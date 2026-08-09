"""EquiFM: equivariant flow matching with hybrid probability transport.

Paper: Song, Gong et al., NeurIPS 2023 (arXiv:2312.07168).
Target repo: github.com/AlgoMole/MolFM (sampling-only release).

The EGNN backbone is deliberately NOT duplicated here -- reuse
``MolecularDiffusion.modules.models.geoldm.networks.EGNN_dynamics_QM9``, which
is token-identical to MolFM's and already matches the released checkpoint keys.
"""

from .cnflows import DISCRETE_PATHS, Cnflows
from .eot import solve_eot

__all__ = ["Cnflows", "DISCRETE_PATHS", "solve_eot"]
