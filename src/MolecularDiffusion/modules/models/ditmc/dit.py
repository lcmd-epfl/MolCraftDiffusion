"""DiT blocks and the top-level generative model.

Port of ``dit_mc/backbones/neural_network_layer.{DiTLayer,SO3DiTLayer}`` and
``backbones/generative_model.GenerativeModel``. ``IdentityMerger`` is a literal
no-op upstream (``merger.py:13-23``) and is dropped; the ``GenerativeLayer``
namedtuple collapses to a plain module list.
"""

from __future__ import annotations

import torch
from torch import nn

from MolecularDiffusion.modules.layers.e3x import (
    SelfAttention,
)
from MolecularDiffusion.modules.layers.e3x import (
    add as e3x_add,
)
from MolecularDiffusion.modules.layers.e3x import (
    broadcast_equivariant_multiplication,
    get_activation_fn,
    promote_to_e3x,
)

from .graphs import CondGraph, LatentGraph
from .layers import (
    MLP,
    E3MLP,
    EquivariantLayerNorm,
    flax_dense,
    flax_layer_norm,
    modulate_adaLN,
    modulate_E3adaLN,
)


class DiTLayer(nn.Module):
    """Non-equivariant DiT block with adaLN-Zero conditioning.

    ``act_dense_correct_bool`` is ``True`` in ``globals``, so the shipped models
    compute ``Dense(act_fn(c))``; the zero-init makes the modulation vector zero
    at init either way. Both branches are ported.
    """

    def __init__(
        self,
        num_features: int,
        num_heads: int,
        num_features_mlp: int,
        activation_fn_mlp: str = "gelu",
        activation_fn: str = "silu",
        relative_embedding_qk_bool: bool = True,
        relative_embedding_v_bool: bool = True,
        act_dense_correct_bool: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.act_dense_correct_bool = act_dense_correct_bool
        self.act_fn = get_activation_fn(activation_fn)

        self.cond_norm = flax_layer_norm(num_features)
        self.ada_dense = flax_dense(
            num_features, 6 * num_features, bias=True, zero_init=True
        )
        self.norm1 = flax_layer_norm(num_features, use_scale=False, use_bias=False)
        self.norm2 = flax_layer_norm(num_features, use_scale=False, use_bias=False)

        self.attention = SelfAttention(
            in_features=num_features,
            in_max_degree=0,
            in_num_parity=1,
            num_heads=num_heads,
            max_degree=0,
            include_pseudotensors=False,
            num_basis=num_features,
            basis_max_degree=0,
            basis_num_parity=1,
            use_relative_positional_encoding_qk=relative_embedding_qk_bool,
            use_relative_positional_encoding_v=relative_embedding_v_bool,
        )
        self.mlp = MLP(
            num_features,
            [num_features_mlp, num_features],
            num_layers=2,
            activation_fn=activation_fn_mlp,
        )

    def forward(
        self,
        graph: LatentGraph,
        features_nodes: torch.Tensor,
        features_edges: torch.Tensor | None,
        features_cond: torch.Tensor | None,
        features_time: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = graph.num_nodes
        if features_nodes.dim() != 4:
            msg = "Features are assumed to be in the e3x convention."
            raise ValueError(msg)
        if features_nodes.shape[1] != 1 or features_nodes.shape[2] != 1:
            msg = "Parity must be 1 and maximal degree must be 0."
            raise ValueError(msg)

        h = features_nodes.squeeze(1).squeeze(1)
        t = features_time.squeeze(1).squeeze(1)
        edges = (
            features_edges.squeeze(1).squeeze(1)
            if features_edges is not None
            else None
        )

        if features_cond is not None:
            cond = features_cond.squeeze(1).squeeze(1)
            cond = cond * graph.cond_scaling_nodes[:, None]
        else:
            cond = t.new_zeros((num_nodes, self.num_features))

        c = self.cond_norm(cond + t)
        if self.act_dense_correct_bool:
            mod = self.ada_dense(self.act_fn(c))
        else:
            mod = self.act_fn(self.ada_dense(c))
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)

        pre_att = modulate_adaLN(self.norm1(h), shift=beta1, scale=gamma1)
        post_att = self.attention(
            promote_to_e3x(pre_att),
            promote_to_e3x(edges) if edges is not None else None,
            dst_idx=graph.receivers,
            src_idx=graph.senders,
            num_segments=num_nodes,
        ).squeeze(1).squeeze(1)

        h = h + post_att * alpha1
        pre_mlp = modulate_adaLN(self.norm2(h), shift=beta2, scale=gamma2)
        h = h + self.mlp(pre_mlp) * alpha2
        return promote_to_e3x(h)


class SO3DiTLayer(nn.Module):
    """SO(3)-equivariant DiT block.

    Unlike :class:`DiTLayer`, the input shape *changes* between layer 0 and the
    rest: the node embedding is ``(N, 1, 1, F)``, and after one block the skip
    connection has widened it to ``(N, P_out, (L+1)**2, F)``. ``in_max_degree``
    / ``in_num_parity`` therefore differ per layer and are supplied by
    ``build.py`` rather than inferred lazily as Flax does.
    """

    def __init__(
        self,
        num_features: int,
        num_heads: int,
        num_features_mlp: int,
        max_degree: int,
        include_pseudotensors: bool,
        in_max_degree: int,
        in_num_parity: int,
        num_radial_basis: int,
        basis_max_degree: int,
        activation_fn_mlp: str = "gelu",
        activation_fn: str = "silu",
        act_dense_correct_bool: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.max_degree = max_degree
        self.in_max_degree = in_max_degree
        self.in_num_parity = in_num_parity
        self.parity_output = 2 if include_pseudotensors else 1
        self.act_dense_correct_bool = act_dense_correct_bool
        self.act_fn = get_activation_fn(activation_fn)

        self.cond_norm = flax_layer_norm(num_features)

        # Six modulation vectors, each with its own degree/parity width.
        self._sizes = (
            num_features * (in_max_degree + 1) * in_num_parity,  # gamma1
            num_features,  # beta1
            num_features * (max_degree + 1) * self.parity_output,  # alpha1
            num_features * (max_degree + 1) * self.parity_output,  # gamma2
            num_features,  # beta2
            num_features * (max_degree + 1) * self.parity_output,  # alpha2
        )
        self.ada_dense = flax_dense(
            num_features, sum(self._sizes), bias=True, zero_init=True
        )

        self.norm1 = EquivariantLayerNorm(
            num_features,
            in_max_degree,
            in_num_parity,
            use_scale=False,
            use_bias=False,
        )
        self.attention = SelfAttention(
            in_features=num_features,
            in_max_degree=in_max_degree,
            in_num_parity=in_num_parity,
            num_heads=num_heads,
            max_degree=max_degree,
            include_pseudotensors=include_pseudotensors,
            num_basis=num_radial_basis,
            basis_max_degree=basis_max_degree,
            basis_num_parity=1,
        )
        att_parity = self.attention.out_num_parity
        att_degree = self.attention.out_max_degree
        if att_degree != max_degree:
            msg = (
                f"attention emits L={att_degree} but the layer was configured "
                f"for L={max_degree}"
            )
            raise ValueError(msg)
        # NOTE: att_parity can be 1 even when include_pseudotensors=True. e3x's
        # Tensor force-disables pseudotensors when both factors have P=1 and one
        # of the degrees is 0 -- which is exactly layer 0, whose node features
        # are still the invariant (P=1, L=0) embedding. Upstream then relies on
        # JAX broadcasting the (N, 2, ., F) alpha against the (N, 1, ., F)
        # attention output to widen it back to P=2. Reproduced, not "fixed".
        self.att_num_parity = att_parity

        # After the first skip connection the running features are the union of
        # the input shape, the alpha shape and the attention output shape.
        self.mid_max_degree = max(in_max_degree, max_degree)
        self.mid_num_parity = max(in_num_parity, self.parity_output, att_parity)
        self.norm2 = EquivariantLayerNorm(
            num_features,
            self.mid_max_degree,
            self.mid_num_parity,
            use_scale=False,
            use_bias=False,
        )
        self.mlp = E3MLP(
            num_features,
            [num_features_mlp, num_features],
            max_degree=self.mid_max_degree,
            num_parity=self.mid_num_parity,
            num_layers=2,
            activation_fn=activation_fn_mlp,
        )
        self.out_max_degree = max(self.mid_max_degree, max_degree)
        self.out_num_parity = max(self.mid_num_parity, self.parity_output)

    def forward(
        self,
        graph: LatentGraph,
        features_nodes: torch.Tensor,
        features_edges: torch.Tensor,
        features_cond: torch.Tensor | None,
        features_time: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = graph.num_nodes
        if features_nodes.dim() != 4:
            msg = "Features are assumed to be in the e3x convention."
            raise ValueError(msg)

        if features_cond is not None:
            if features_cond.shape[1:3] != (1, 1):
                msg = (
                    "Node features for conditioning must be invariant, i.e. "
                    f"max_degree = 0 and parity = 1. Received "
                    f"{tuple(features_cond.shape)}."
                )
                raise ValueError(msg)
            scaling = promote_to_e3x(graph.cond_scaling_nodes[:, None])
            cond = features_cond * scaling
        else:
            cond = torch.zeros_like(features_time)

        c = self.cond_norm(e3x_add(cond, features_time))
        if self.act_dense_correct_bool:
            mod = self.ada_dense(self.act_fn(c))
        else:
            mod = self.act_fn(self.ada_dense(c))
        mod = mod.squeeze(1).squeeze(1)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = torch.split(
            mod, self._sizes, dim=-1
        )

        f = self.num_features
        gamma1 = gamma1.reshape(num_nodes, self.in_num_parity, self.in_max_degree + 1, f)
        beta1 = beta1.reshape(num_nodes, 1, 1, f)
        alpha1 = alpha1.reshape(num_nodes, self.parity_output, self.max_degree + 1, f)
        gamma2 = gamma2.reshape(num_nodes, self.parity_output, self.max_degree + 1, f)
        beta2 = beta2.reshape(num_nodes, 1, 1, f)
        alpha2 = alpha2.reshape(num_nodes, self.parity_output, self.max_degree + 1, f)

        pre_att = modulate_E3adaLN(
            self.norm1(features_nodes), shift=beta1, scale=gamma1
        )
        post_att = self.attention(
            pre_att,
            features_edges,
            dst_idx=graph.receivers,
            src_idx=graph.senders,
            num_segments=num_nodes,
        )
        features_nodes = e3x_add(
            features_nodes,
            broadcast_equivariant_multiplication(factor=alpha1, tensor=post_att),
        )

        pre_mlp = modulate_E3adaLN(
            self.norm2(features_nodes), shift=beta2, scale=gamma2
        )
        post_mlp = self.mlp(pre_mlp)
        return e3x_add(
            features_nodes,
            broadcast_equivariant_multiplication(factor=alpha2, tensor=post_mlp),
        )


class GenerativeModel(nn.Module):
    """``conditioner -> embeddings -> DiT blocks -> readout``.

    Returns ``(drift, noise)`` when ``output='drift_and_noise'``, otherwise a
    single ``(N, 3)`` tensor.
    """

    def __init__(
        self,
        node_embedding: nn.Module,
        time_embedding: nn.Module,
        layers: nn.ModuleList,
        readout: nn.Module,
        edge_embedding: nn.Module | None = None,
        conditioner: nn.Module | None = None,
        conditioning_bool: bool = False,
        variant: str = "dit",
    ) -> None:
        super().__init__()
        self.node_embedding = node_embedding
        self.time_embedding = time_embedding
        self.layers = layers
        self.readout = readout
        self.edge_embedding = edge_embedding
        self.conditioner = conditioner
        self.conditioning_bool = conditioning_bool
        self.variant = variant

    @property
    def output(self) -> str:
        return self.readout.output

    def forward(
        self,
        time_latent: torch.Tensor,
        graph_latent: LatentGraph,
        graph_cond: CondGraph | None = None,
    ):
        features_cond = (
            self.conditioner(graph_cond)
            if (self.conditioning_bool and self.conditioner is not None)
            else None
        )
        features_nodes = self.node_embedding(graph_latent)
        features_time = self.time_embedding(time_latent)
        features_edges = (
            self.edge_embedding(graph_latent)
            if self.edge_embedding is not None
            else None
        )

        for layer in self.layers:
            features_nodes = layer(
                graph_latent,
                features_nodes,
                features_edges,
                features_cond,
                features_time,
            )

        return self.readout(
            features_nodes, features_time=features_time, features_cond=features_cond
        )
