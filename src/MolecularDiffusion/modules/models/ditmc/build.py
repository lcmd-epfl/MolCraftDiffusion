"""Factories for the three published DiTMC variants.

Port of ``dit_mc/model_zoo/GeomDiT.py``. All three shipped configs are covered:

===========  ===========================  =========================================
variant      absolute / relative pos.     edge embedding
===========  ===========================  =========================================
``dit_ape``  absolute=True, relative=False ``DiTEdgeEmbed`` (hops only)
``dit_rpe``  absolute=False, relative=True ``DiTEdgeEmbed`` (hops + displacements)
``dit_so3``  neither                       ``RadialSphericalEdgeEmbedding``
===========  ===========================  =========================================

Because ``embed_shortest_hops_bool`` is ``True`` in ``globals``,
``relative_embedding_bool = relative_positional_embedding_bool or
embed_shortest_hops_bool`` is **True for ``dit_ape`` too** -- so both
non-equivariant variants instantiate ``DiTEdgeEmbed`` and run ``SelfAttention``
with both relative-positional-encoding flags on. They differ only in
``embed_distances_bool``.
"""

from __future__ import annotations

import inspect

from torch import nn

from .dit import DiTLayer, GenerativeModel, SO3DiTLayer
from .embedding import (
    DiTEdgeEmbed,
    DiTNodeEmbed,
    RadialSphericalEdgeEmbedding,
    TimeEmbedding,
)
from .meshgraphnet import MeshGraphNetEncoder
from .readout import EquivariantReadout, SimpleReadout

VARIANTS = ("ape", "rpe", "so3")


def _check_mgn_width(num_features: int, mgn_num_features: int) -> None:
    """The conditioner's width must equal the DiT's.

    ``DiTLayer``/``SO3DiTLayer`` and both readouts add the conditioner features
    to the time embedding, so a mismatch is a hard error. Upstream's shipped
    configs happen to satisfy it (256 = 8 heads x 32), and the failure without
    this guard is an opaque shape error from deep inside ``e3x.add``.
    """
    if mgn_num_features != num_features:
        msg = (
            f"mgn_num_features ({mgn_num_features}) must equal num_heads * "
            f"num_features_head ({num_features}) -- the conditioner's node "
            f"features are ADDED to the time embedding."
        )
        raise ValueError(msg)


def make_molecular_dit(  # noqa: PLR0913
    node_attr_dim: int,
    num_layers: int = 6,
    num_heads: int = 8,
    num_features_head: int = 32,
    mgn_num_features: int = 256,
    mgn_num_layers: int = 2,
    mgn_activation_fn: str = "silu",
    num_features_mlp: int | None = None,
    activation_fn_mlp: str = "gelu",
    activation_fn: str = "silu",
    absolute_positional_embedding_bool: bool = True,
    relative_positional_embedding_bool: bool = False,
    rpe_radial_basis_bool: bool = False,
    rpe_num_radial_basis: int = 8,
    rpe_max_frequency: float = 2 * 3.141592653589793,
    self_conditioning_bool: bool = False,
    positional_encoding_bool: bool = False,
    embed_shortest_hops_bool: bool = False,
    act_dense_correct_bool: bool = False,
    output: str = "drift_and_noise",
) -> GenerativeModel:
    """``dit_ape`` / ``dit_rpe``."""
    num_features = num_heads * num_features_head
    _check_mgn_width(num_features, mgn_num_features)
    if num_features_mlp is None:
        num_features_mlp = 4 * num_features

    conditioner = MeshGraphNetEncoder(
        node_attr_dim=node_attr_dim,
        num_layers=mgn_num_layers,
        num_features=mgn_num_features,
        activation_fn=mgn_activation_fn,
    )
    time_embedding = TimeEmbedding(num_features, activation_fn=activation_fn)
    node_embedding = DiTNodeEmbed(
        num_features=num_features,
        activation_fn=activation_fn,
        self_conditioning_bool=self_conditioning_bool,
        positional_encoding_bool=positional_encoding_bool,
        positional_embedding_bool=absolute_positional_embedding_bool,
    )

    relative_embedding_bool = (
        relative_positional_embedding_bool or embed_shortest_hops_bool
    )
    edge_embedding = (
        DiTEdgeEmbed(
            num_features=num_features,
            activation_fn=activation_fn,
            radial_basis_bool=rpe_radial_basis_bool,
            num_radial_basis=rpe_num_radial_basis,
            max_frequency=rpe_max_frequency,
            embed_distances_bool=relative_positional_embedding_bool,
            embed_shortest_hops_bool=embed_shortest_hops_bool,
        )
        if relative_embedding_bool
        else None
    )

    layers = nn.ModuleList(
        [
            DiTLayer(
                num_features=num_features,
                num_heads=num_heads,
                num_features_mlp=num_features_mlp,
                activation_fn_mlp=activation_fn_mlp,
                activation_fn=activation_fn,
                relative_embedding_qk_bool=relative_embedding_bool,
                relative_embedding_v_bool=relative_embedding_bool,
                act_dense_correct_bool=act_dense_correct_bool,
            )
            for _ in range(num_layers)
        ]
    )

    return GenerativeModel(
        node_embedding=node_embedding,
        time_embedding=time_embedding,
        layers=layers,
        readout=SimpleReadout(num_features, activation_fn, output=output),
        edge_embedding=edge_embedding,
        conditioner=conditioner,
        conditioning_bool=True,
        variant="rpe" if relative_positional_embedding_bool else "ape",
    )


def make_molecular_dit_so3(  # noqa: PLR0913
    node_attr_dim: int,
    num_layers: int = 6,
    num_heads: int = 8,
    num_features_head: int = 32,
    cutoff: float | None = None,
    mgn_num_features: int = 256,
    mgn_num_layers: int = 2,
    mgn_activation_fn: str = "silu",
    max_degree: int = 1,
    num_features_mlp: int | None = None,
    activation_fn_mlp: str = "gelu",
    activation_fn: str = "silu",
    include_pseudotensors: bool = True,
    radial_basis: str = "reciprocal_bernstein",
    num_radial_basis: int = 64,
    radial_basis_kwargs: dict | None = None,
    cutoff_fn: str | None = None,
    self_conditioning_bool: bool = False,
    positional_encoding_bool: bool = False,
    embed_shortest_hops_bool: bool = False,
    scale_spherical_basis_with_shortest_hops_bool: bool = False,
    act_dense_correct_bool: bool = False,
    output: str = "drift_and_noise",
) -> GenerativeModel:
    """``dit_so3``. Upstream raises unless ``cutoff`` is ``None``."""
    if cutoff is not None:
        msg = f"Cutoff not None not supported yet. Received cutoff={cutoff}"
        raise NotImplementedError(msg)
    if cutoff_fn is not None:
        msg = (
            f"When no cutoff is used, cutoff_fn must also be None. Received "
            f"cutoff={cutoff} and cutoff_fn={cutoff_fn}."
        )
        raise ValueError(msg)

    num_features = num_heads * num_features_head
    _check_mgn_width(num_features, mgn_num_features)
    if num_features_mlp is None:
        num_features_mlp = 4 * num_features

    conditioner = MeshGraphNetEncoder(
        node_attr_dim=node_attr_dim,
        num_layers=mgn_num_layers,
        num_features=mgn_num_features,
        activation_fn=mgn_activation_fn,
    )
    time_embedding = TimeEmbedding(num_features, activation_fn=activation_fn)
    node_embedding = DiTNodeEmbed(
        num_features=num_features,
        activation_fn=activation_fn,
        self_conditioning_bool=self_conditioning_bool,
        positional_encoding_bool=positional_encoding_bool,
        positional_embedding_bool=False,
    )
    edge_embedding = RadialSphericalEdgeEmbedding(
        cutoff=cutoff,
        max_degree=max_degree,
        radial_basis=radial_basis,
        num_radial_basis=num_radial_basis,
        radial_basis_kwargs=radial_basis_kwargs,
        cutoff_fn=cutoff_fn,
        activation_fn=activation_fn,
        embed_shortest_hops_bool=embed_shortest_hops_bool,
        scale_spherical_basis_with_shortest_hops_bool=scale_spherical_basis_with_shortest_hops_bool,
    )

    # The node features widen after the first block: the node embedding is
    # (P=1, L=0), and every skip connection unions it with the attention output
    # (P=2 when include_pseudotensors, L=max_degree). Flax infers this lazily;
    # here it is computed once and passed down.
    layers = []
    in_max_degree, in_num_parity = 0, 1
    for _ in range(num_layers):
        layer = SO3DiTLayer(
            num_features=num_features,
            num_heads=num_heads,
            num_features_mlp=num_features_mlp,
            max_degree=max_degree,
            include_pseudotensors=include_pseudotensors,
            in_max_degree=in_max_degree,
            in_num_parity=in_num_parity,
            num_radial_basis=num_radial_basis,
            basis_max_degree=max_degree,
            activation_fn_mlp=activation_fn_mlp,
            activation_fn=activation_fn,
            act_dense_correct_bool=act_dense_correct_bool,
        )
        layers.append(layer)
        in_max_degree, in_num_parity = layer.out_max_degree, layer.out_num_parity

    return GenerativeModel(
        node_embedding=node_embedding,
        time_embedding=time_embedding,
        layers=nn.ModuleList(layers),
        readout=EquivariantReadout(
            num_features,
            activation_fn,
            output=output,
            in_max_degree=in_max_degree,
            in_num_parity=in_num_parity,
        ),
        edge_embedding=edge_embedding,
        conditioner=conditioner,
        conditioning_bool=True,
        variant="so3",
    )


#: The three shipped ``configs/model/*.yaml`` verbatim, plus the ``globals``
#: block from ``configs/config.yaml`` that applies to all of them.
SHIPPED_GLOBALS = {
    "embed_shortest_hops_bool": True,
    "act_dense_correct_bool": True,
    "positional_encoding_bool": False,
    "self_conditioning_bool": False,
}

SHIPPED_VARIANTS = {
    "ape": {
        "absolute_positional_embedding_bool": True,
        "relative_positional_embedding_bool": False,
        "rpe_radial_basis_bool": False,
    },
    "rpe": {
        "absolute_positional_embedding_bool": False,
        "relative_positional_embedding_bool": True,
        "rpe_radial_basis_bool": False,
    },
    "so3": {
        "max_degree": 1,
        "include_pseudotensors": True,
        "radial_basis": "reciprocal_bernstein",
        "num_radial_basis": 64,
        "scale_spherical_basis_with_shortest_hops_bool": True,
    },
}


def build_variant(variant: str, node_attr_dim: int, **overrides) -> GenerativeModel:
    """Build one of ``ape`` / ``rpe`` / ``so3`` with the shipped defaults.

    One config serves all three variants, so ``overrides`` is filtered to the
    chosen factory's signature: ``num_radial_basis`` means nothing to
    ``dit_ape``, and ``absolute_positional_embedding_bool`` means nothing to
    ``dit_so3``. A key that matches **no** variant's signature is still an
    error -- silently swallowing a typo'd hyperparameter is how a checkpoint
    quietly stops matching.
    """
    if variant not in VARIANTS:
        msg = f"variant must be one of {VARIANTS}, received {variant!r}"
        raise ValueError(msg)
    factory = make_molecular_dit_so3 if variant == "so3" else make_molecular_dit
    accepted = set(inspect.signature(factory).parameters)
    known = accepted | set(inspect.signature(
        make_molecular_dit if variant == "so3" else make_molecular_dit_so3
    ).parameters)
    unknown = set(overrides) - known
    if unknown:
        msg = f"unknown DiTMC hyperparameters: {sorted(unknown)}"
        raise TypeError(msg)

    kwargs = dict(SHIPPED_GLOBALS)
    kwargs.update(SHIPPED_VARIANTS[variant])
    kwargs.update(overrides)
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    return factory(node_attr_dim=node_attr_dim, **kwargs)
