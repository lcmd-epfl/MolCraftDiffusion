"""Node, edge and time embeddings. Port of ``dit_mc/backbones/embedding.py``."""

from __future__ import annotations

import functools
import math

import torch
from torch import nn

from MolecularDiffusion.modules.layers.e3x import (
    add as e3x_add,
)
from MolecularDiffusion.modules.layers.e3x import (
    basic_fourier,
    basis,
    broadcast_equivariant_multiplication,
    get_radial_fn,
    promote_to_e3x,
)

from .graphs import CondGraph, LatentGraph
from .layers import (
    MLP,
    GaussianRandomFourierFeatures,
    flax_layer_norm,
)

#: ``nn.Embed`` over atomic numbers; upstream hard-codes 119 (all elements).
NUM_ATOMIC_NUMBERS = 119

#: ``algos.pyx`` writes 510 for unreachable pairs, hence 512 embedding rows.
NUM_SHORTEST_HOPS = 512
UNREACHABLE_HOPS = 510


def flax_embedding(num_embeddings: int, features: int) -> nn.Embedding:
    """``flax.linen.Embed``: ``variance_scaling(1.0, 'fan_in', 'normal')``.

    For a ``(num_embeddings, features)`` table with ``out_axis=0`` this resolves
    to ``fan_in = features`` and a **plain** (not truncated) normal.
    """
    emb = nn.Embedding(num_embeddings, features)
    nn.init.normal_(emb.weight, mean=0.0, std=math.sqrt(1.0 / features))
    return emb


def get_index_embedding(
    indices: torch.Tensor, emb_dim: int, max_len: int = 256
) -> torch.Tensor:
    """Sinusoidal positional embedding from per-molecule atom offsets."""
    k = torch.arange(emb_dim // 2, dtype=torch.float32, device=indices.device)
    arg = indices[..., None].float() * math.pi / (max_len ** (2 * k / emb_dim))
    return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)


def get_pos_indices(batch_segments: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Index of each atom within its own molecule."""
    counts = torch.zeros(
        num_graphs, dtype=torch.long, device=batch_segments.device
    ).index_add(0, batch_segments, torch.ones_like(batch_segments))
    offsets = torch.cat(
        [counts.new_zeros(1), torch.cumsum(counts, 0)[:-1]]
    )
    return (
        torch.arange(batch_segments.shape[0], device=batch_segments.device)
        - offsets[batch_segments]
    )


class TimeEmbedding(nn.Module):
    """Fourier features of the latent time, then a 2-layer MLP."""

    def __init__(
        self,
        num_features: int,
        num_features_fourier: int | None = None,
        activation_fn: str = "silu",
    ) -> None:
        super().__init__()
        nf = num_features // 2 if num_features_fourier is None else num_features_fourier
        self.ff = GaussianRandomFourierFeatures(1, nf)
        self.mlp = MLP(
            nf,
            num_features,
            num_layers=2,
            activation_fn=activation_fn,
            use_bias=True,
        )

    def forward(self, time_latent: torch.Tensor) -> torch.Tensor:
        if time_latent.dim() != 1:
            msg = (
                f"latent times must be an array of single dimension, received "
                f"shape {tuple(time_latent.shape)}"
            )
            raise ValueError(msg)
        return promote_to_e3x(self.mlp(self.ff(time_latent.unsqueeze(-1))))


class NodeAttributeEmbedding(nn.Module):
    """MeshGraphNet node encoder: 2-layer MLP + LayerNorm."""

    def __init__(
        self, in_features: int, num_features: int, activation_fn: str
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            in_features, num_features, num_layers=2, activation_fn=activation_fn
        )
        self.norm = flax_layer_norm(num_features)

    def forward(self, node_attr: torch.Tensor) -> torch.Tensor:
        return promote_to_e3x(self.norm(self.mlp(node_attr)))


class EdgeAttributeEmbedding(nn.Module):
    """MeshGraphNet edge encoder over the 4-class bond one-hot."""

    def __init__(
        self, in_features: int, num_features: int, activation_fn: str
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            in_features, num_features, num_layers=2, activation_fn=activation_fn
        )
        self.norm = flax_layer_norm(num_features)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return promote_to_e3x(self.norm(self.mlp(edge_attr)))


class DiTNodeEmbed(nn.Module):
    """Atomic-number embedding, optionally plus self-conditioning / positions."""

    def __init__(
        self,
        num_features: int,
        activation_fn: str,
        self_conditioning_bool: bool,
        positional_encoding_bool: bool,
        positional_embedding_bool: bool,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.self_conditioning_bool = self_conditioning_bool
        self.positional_encoding_bool = positional_encoding_bool
        self.positional_embedding_bool = positional_embedding_bool

        self.embed = flax_embedding(NUM_ATOMIC_NUMBERS, num_features)
        self.sc_mlp = (
            MLP(3, num_features, num_layers=2, use_bias=False, activation_fn=activation_fn)
            if self_conditioning_bool
            else None
        )
        self.pos_mlp = (
            MLP(3, num_features, num_layers=2, use_bias=False, activation_fn=activation_fn)
            if positional_embedding_bool
            else None
        )

    def forward(self, graph: LatentGraph) -> torch.Tensor:
        h = self.embed(graph.atomic_numbers)
        if self.sc_mlp is not None:
            self_cond = graph.self_cond
            if self_cond is None:
                self_cond = torch.zeros_like(graph.positions)
            h = h + self.sc_mlp(self_cond)
        if self.positional_encoding_bool:
            indices = get_pos_indices(graph.batch_segments, graph.num_graphs)
            h = h + get_index_embedding(indices, self.num_features).to(h.dtype)
        if self.pos_mlp is not None:
            h = h + self.pos_mlp(graph.positions)
        return promote_to_e3x(h)


class DiTEdgeEmbed(nn.Module):
    """Non-equivariant relative edge embedding (``dit_ape`` / ``dit_rpe``).

    ``embed_shortest_hops_bool`` is ``True`` in ``globals``, which is why
    ``relative_embedding_bool`` is True for **both** ``dit_ape`` and
    ``dit_rpe``; they differ only in ``embed_distances_bool``.

    The shortest-hops MLP deliberately keeps ``use_bias=True`` -- upstream's own
    comment: "we need a bias here s.t. the output is non-zero in case of CFG".
    """

    def __init__(
        self,
        num_features: int,
        activation_fn: str,
        embed_distances_bool: bool = True,
        embed_shortest_hops_bool: bool = True,
        radial_basis_bool: bool | None = None,
        num_radial_basis: int | None = None,
        max_frequency: float | None = None,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.embed_distances_bool = embed_distances_bool
        self.embed_shortest_hops_bool = embed_shortest_hops_bool
        self.radial_basis_bool = bool(radial_basis_bool)

        if self.radial_basis_bool:
            if num_radial_basis is None or max_frequency is None:
                msg = (
                    "`num_radial_basis` and `max_frequency` must be provided if "
                    "`radial_basis_bool` is True."
                )
                raise ValueError(msg)
            if num_radial_basis <= 1:
                msg = (
                    f"`num_radial_basis` must be greater than one, received "
                    f"{num_radial_basis}."
                )
                raise ValueError(msg)
        self.num_radial_basis = num_radial_basis
        self.max_frequency = max_frequency

        if embed_distances_bool:
            in_dim = 3 * num_radial_basis if self.radial_basis_bool else 3
            self.re_mlp = MLP(
                in_dim,
                num_features,
                num_layers=2,
                use_bias=False,
                activation_fn=activation_fn,
            )
        else:
            self.re_mlp = None

        if embed_shortest_hops_bool:
            self.hops_embed = flax_embedding(NUM_SHORTEST_HOPS, num_features)
            self.hops_mlp = MLP(
                num_features,
                num_features,
                num_layers=2,
                use_bias=True,
                activation_fn=activation_fn,
            )
        else:
            self.hops_embed = None
            self.hops_mlp = None

    def forward(self, graph: LatentGraph) -> torch.Tensor:
        positions = graph.positions
        num_edges = graph.senders.shape[0]

        if self.re_mlp is not None:
            # r_j - r_i: senders is j, receivers is i. See graphs.py.
            displacements = positions[graph.senders] - positions[graph.receivers]
            if self.radial_basis_bool:
                limit = (self.num_radial_basis - 1) * math.pi / self.max_frequency
                displacements = basic_fourier(
                    displacements, num=self.num_radial_basis, limit=limit
                ).reshape(num_edges, -1)
            re = self.re_mlp(displacements)
        else:
            re = positions.new_zeros((num_edges, self.num_features))

        if self.hops_mlp is not None:
            re_hops = self.hops_mlp(self.hops_embed(graph.shortest_hops))
            re = re + re_hops * graph.cond_scaling_edges[:, None]

        return promote_to_e3x(re)


class RadialSphericalEdgeEmbedding(nn.Module):
    """SO(3)-equivariant edge basis (``dit_so3``).

    ``e3x.nn.basis`` with ``reciprocal_bernstein`` radial functions and no
    cutoff (``dit_so3`` sets ``cutoff: null``; the factory raises otherwise).
    The shortest-hops term is zero-initialized so it starts as a pure identity
    scaling, and is gated by ``cond_scaling`` for classifier-free guidance.
    """

    def __init__(
        self,
        max_degree: int,
        activation_fn: str,
        cutoff: float | None = None,
        embed_shortest_hops_bool: bool = False,
        scale_spherical_basis_with_shortest_hops_bool: bool = False,
        radial_basis: str = "reciprocal_bernstein",
        num_radial_basis: int = 32,
        radial_basis_kwargs: dict | None = None,
        cutoff_fn: str | None = None,
    ) -> None:
        super().__init__()
        if cutoff is not None or cutoff_fn is not None:
            msg = (
                f"Only the no-cutoff configuration is ported (dit_so3 sets both "
                f"to null). Received cutoff={cutoff}, cutoff_fn={cutoff_fn}."
            )
            raise NotImplementedError(msg)
        self.max_degree = max_degree
        self.num_radial_basis = num_radial_basis
        self.embed_shortest_hops_bool = embed_shortest_hops_bool
        self.scale_with_hops = scale_spherical_basis_with_shortest_hops_bool
        self.radial_fn = functools.partial(
            get_radial_fn(radial_basis), **(radial_basis_kwargs or {})
        )

        if embed_shortest_hops_bool:
            nf = (
                num_radial_basis * (max_degree + 1)
                if self.scale_with_hops
                else num_radial_basis
            )
            self.hops_features = nf
            self.hops_embed = flax_embedding(NUM_SHORTEST_HOPS, nf)
            self.hops_mlp = MLP(
                nf,
                nf,
                num_layers=2,
                use_bias=False,
                activation_fn=activation_fn,
                output_is_zero_at_init=True,
            )
        else:
            self.hops_embed = None
            self.hops_mlp = None

    def forward(self, graph: LatentGraph) -> torch.Tensor:
        positions = graph.positions
        num_edges = graph.senders.shape[0]
        displacements = positions[graph.senders] - positions[graph.receivers]

        b = basis(
            displacements,
            max_degree=self.max_degree,
            num=self.num_radial_basis,
            radial_fn=self.radial_fn,
            cutoff_fn=None,
        )

        if self.hops_mlp is None:
            return b

        re_hops = self.hops_mlp(self.hops_embed(graph.shortest_hops))
        re_hops = graph.cond_scaling_edges[:, None] * re_hops
        re_hops = promote_to_e3x(re_hops)

        if self.scale_with_hops:
            re_hops = re_hops.reshape(
                num_edges, 1, self.max_degree + 1, self.num_radial_basis
            )
            return broadcast_equivariant_multiplication(
                factor=1 + re_hops, tensor=b
            )
        # e3x.add zero-pads the scalar-only term up to the basis' degree axis.
        re_hops = e3x_add(re_hops, b.new_zeros((*b.shape[:3], re_hops.shape[-1])))
        return e3x_add(b, re_hops)


def build_cond_embeddings(
    node_attr_dim: int, num_features: int, activation_fn: str
) -> tuple[NodeAttributeEmbedding, EdgeAttributeEmbedding]:
    """Convenience pair used by the MeshGraphNet encoder."""
    return (
        NodeAttributeEmbedding(node_attr_dim, num_features, activation_fn),
        EdgeAttributeEmbedding(4, num_features, activation_fn),
    )


__all__ = [
    "NUM_ATOMIC_NUMBERS",
    "NUM_SHORTEST_HOPS",
    "UNREACHABLE_HOPS",
    "CondGraph",
    "DiTEdgeEmbed",
    "DiTNodeEmbed",
    "EdgeAttributeEmbedding",
    "NodeAttributeEmbedding",
    "RadialSphericalEdgeEmbedding",
    "TimeEmbedding",
    "build_cond_embeddings",
    "flax_embedding",
    "get_index_embedding",
    "get_pos_indices",
]
