"""A faithful PyTorch reimplementation of the ``e3x`` surface DiTMC calls.

Scope is deliberately "the call sites, not the library". Not ported, because no
DiTMC config path reaches them: ``FusedTensor`` (both call sites pass
``use_fused_tensor=False``), the cutoff functions (``dit_so3`` sets
``cutoff_fn: null`` and the factory raises otherwise), ``e3x.nn.Embed``,
``MessagePass``, ``TensorDense``, ``ExponentialBasis``, ``damping_fn``, and the
``sinc``/``gaussian``/``chebyshev``/``exponential_*`` radial families.

Global e3x config pins, hard-coded here rather than left implicit:
``cartesian_order = True``, ``normalization = 'racah'``,
``use_fused_tensor = False``.
"""

from .activations import (
    gelu,
    get_activation_fn,
    get_e3x_activation_fn,
    silu,
    swish,
)
from .features import (
    add,
    broadcast_equivariant_multiplication,
    change_max_degree_or_type,
    extract_max_degree,
    promote_to_e3x,
    reflect,
)
from .indexed import (
    gather_dst,
    gather_src,
    indexed_max,
    indexed_softmax,
    indexed_sum,
    segment_mean,
)
from .modules import (
    Dense,
    SelfAttention,
    Tensor,
    duplication_indices_for_max_degree,
    resolve_tensor_output,
)
from .radial import (
    basic_fourier,
    basis,
    get_radial_fn,
    reciprocal_bernstein,
    reciprocal_mapping,
)
from .so3 import (
    cartesian_permutation,
    cartesian_permutation_for_degree,
    clebsch_gordan,
    random_rotation,
    spherical_harmonics,
)

__all__ = [
    "Dense",
    "SelfAttention",
    "Tensor",
    "add",
    "basic_fourier",
    "basis",
    "broadcast_equivariant_multiplication",
    "cartesian_permutation",
    "cartesian_permutation_for_degree",
    "change_max_degree_or_type",
    "clebsch_gordan",
    "duplication_indices_for_max_degree",
    "extract_max_degree",
    "gather_dst",
    "gather_src",
    "gelu",
    "get_activation_fn",
    "get_e3x_activation_fn",
    "get_radial_fn",
    "indexed_max",
    "indexed_softmax",
    "indexed_sum",
    "promote_to_e3x",
    "reciprocal_bernstein",
    "reciprocal_mapping",
    "reflect",
    "resolve_tensor_output",
    "random_rotation",
    "segment_mean",
    "silu",
    "spherical_harmonics",
    "swish",
]
