"""MeshGraphNet bond-graph conditioner.

Port of ``neural_network_layer.MeshGraphNetLayer`` + ``encoder.EncoderModel``.
This is the only place the covalent bonds enter as message passing; the DiT
attention itself runs over the *fully connected* latent graph.
"""

from __future__ import annotations

import torch
from torch import nn

from MolecularDiffusion.modules.layers.e3x import promote_to_e3x

from .embedding import EdgeAttributeEmbedding, NodeAttributeEmbedding
from .graphs import CondGraph
from .layers import MLP, flax_layer_norm


class MeshGraphNetLayer(nn.Module):
    """One edge update followed by one node update, both LayerNorm-then-MLP."""

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        activation_fn: str = "silu",
    ) -> None:
        super().__init__()
        edge_in = 2 * num_node_features + num_edge_features
        self.edge_norm = flax_layer_norm(edge_in)
        self.edge_mlp = MLP(
            edge_in,
            num_edge_features,
            num_layers=2,
            activation_fn=activation_fn,
            use_bias=True,
        )
        node_in = num_node_features + num_edge_features
        self.node_norm = flax_layer_norm(node_in)
        self.node_mlp = MLP(
            node_in,
            num_node_features,
            num_layers=2,
            activation_fn=activation_fn,
            use_bias=True,
        )

    def forward(
        self,
        graph: CondGraph,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cat = torch.cat(
            [
                node_features[graph.senders],
                node_features[graph.receivers],
                edge_features,
            ],
            dim=-1,
        )
        edge_features = self.edge_mlp(self.edge_norm(cat))

        num_nodes = node_features.shape[0]
        aggregated = node_features.new_zeros(
            (num_nodes, edge_features.shape[-1])
        ).index_add(0, graph.receivers, edge_features)
        node_features = self.node_mlp(
            self.node_norm(torch.cat([node_features, aggregated], dim=-1))
        )
        return node_features, edge_features


class MeshGraphNetEncoder(nn.Module):
    """``make_graph_mesh_net_encoder``: embeddings + a stack of layers."""

    def __init__(
        self,
        node_attr_dim: int,
        num_layers: int,
        num_features: int,
        activation_fn: str = "silu",
    ) -> None:
        super().__init__()
        self.node_embedding = NodeAttributeEmbedding(
            node_attr_dim, num_features, activation_fn
        )
        self.edge_embedding = EdgeAttributeEmbedding(4, num_features, activation_fn)
        self.layers = nn.ModuleList(
            [
                MeshGraphNetLayer(num_features, num_features, activation_fn)
                for _ in range(num_layers)
            ]
        )
        self.num_features = num_features

    def forward(self, graph: CondGraph) -> torch.Tensor:
        """Returns node features in e3x form, ``(N, 1, 1, num_features)``."""
        # The embeddings promote to e3x; the layers work on the plain (N, F)
        # form, exactly as upstream does (its MLPs broadcast over the two
        # singleton axes).
        node_features = self.node_embedding(graph.node_attr).squeeze(1).squeeze(1)
        edge_features = self.edge_embedding(graph.edge_attr).squeeze(1).squeeze(1)
        for layer in self.layers:
            node_features, edge_features = layer(graph, node_features, edge_features)
        return promote_to_e3x(node_features)
