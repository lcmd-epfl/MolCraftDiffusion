"""FlowMol3 backbone: bond-carrying, CTMC discrete-flow-matching GVP field.

Sibling of the coordinate-only ``modules/models/flowmol`` package, which stays
byte-identical. The two share ``build_edge_idxs``, ``InterpolantScheduler`` and
the whole of ``modules/layers/gvp``; everything bond- or CTMC-specific lives
here.

See ``docs/model_integrations/flowmol_graph3d/`` for the integration plan and
the provenance of the released weights this reproduces.
"""

from MolecularDiffusion.modules.models.flowmol.interpolant_scheduler import (
    InterpolantScheduler,
)
from MolecularDiffusion.modules.models.flowmol_graph3d.ctmc_vector_field import (
    CTMCVectorField,
)
from MolecularDiffusion.modules.models.flowmol_graph3d.graph_utils import (
    build_edge_idxs,
    get_batch_idxs,
    get_edge_batch_idxs,
    get_node_batch_idxs,
    get_upper_edge_mask,
)
from MolecularDiffusion.modules.models.flowmol_graph3d.priors import (
    centered_normal_prior_batched_graph,
    ctmc_masked_edge_prior,
    ctmc_masked_prior,
)
from MolecularDiffusion.modules.models.flowmol_graph3d.self_conditioning import (
    SelfConditioningResidualLayer,
)
from MolecularDiffusion.modules.models.flowmol_graph3d.vector_field import (
    EndpointVectorField,
)

__all__ = [
    "CTMCVectorField",
    "EndpointVectorField",
    "InterpolantScheduler",
    "SelfConditioningResidualLayer",
    "build_edge_idxs",
    "centered_normal_prior_batched_graph",
    "ctmc_masked_edge_prior",
    "ctmc_masked_prior",
    "get_batch_idxs",
    "get_edge_batch_idxs",
    "get_node_batch_idxs",
    "get_upper_edge_mask",
]
