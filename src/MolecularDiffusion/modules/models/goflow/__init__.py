"""GoFlow: transition-state geometry from a reaction's condensed graph,
via conditional flow matching and an E(3)-equivariant transformer.

Galustian, Mark, Karwounopoulos, Kovar & Heid, *GoFlow: Efficient
Transition State Geometry Prediction with Flow Matching and E(3)-Equivariant
Neural Networks*, ChemRxiv (2025), doi:10.26434/chemrxiv-2025-bk2rh. Ported
from the repo checked out at ``others/nice/goflow`` (commit
``3ec00a09d9b283e3258ae01fe5d3e35bb3812bff``).

The package holds the network (:mod:`.gotennet`, :mod:`.ops`,
:mod:`.outputs`, :mod:`.cgr_graph_utils`) and the flow-matching algorithm
(:mod:`.flow`) only. Its data layer lives in
``data/component/goflow_data.py`` (beside the shared ``reaction_data.py``
container) and its task in ``modules/tasks/diffusion_goflow.py``.

See ``docs/model_integrations/goflow/INTEGRATION_PLAN.md``.
"""

from MolecularDiffusion.modules.models.goflow.flow import (
    euler_integrate,
    get_perturbed_flow_point_and_time,
    get_shortest_path_fast_batched_x_1,
    rmsd_loss,
)
from MolecularDiffusion.modules.models.goflow.gotennet import GotenNet
from MolecularDiffusion.modules.models.goflow.outputs import Atomwise3DOut

__all__ = [
    "Atomwise3DOut",
    "GotenNet",
    "euler_integrate",
    "get_perturbed_flow_point_and_time",
    "get_shortest_path_fast_batched_x_1",
    "rmsd_loss",
]
