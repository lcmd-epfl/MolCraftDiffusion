"""Transformer decoder for VAE.

Copyright (c) Meta Platforms, Inc. and affiliates.
Adapted for MolecularDiffusion.
"""

import math
from typing import Dict

import torch
from torch import nn
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter


def get_index_embedding(indices, emb_dim, max_len=2048):
    """Creates sine / cosine positional embeddings from a prespecified indices."""
    K = torch.arange(emb_dim // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


class TransformerDecoder(nn.Module):
    """Transformer decoder as part of VAE.

    Takes encoded latent tokens and decodes to atom types and positions.
    For molecules, lattice and frac_coords outputs are ignored.

    Args:
        max_num_elements: Maximum number of elements (atomic numbers) supported
        d_model: Dimension of the model
        nhead: Number of attention heads
        dim_feedforward: Dimension of the feedforward network
        activation: Activation function to use
        dropout: Dropout rate
        norm_first: Whether to use pre-normalization in Transformer blocks
        bias: Whether to use bias
        num_layers: Number of layers
    """

    def __init__(
        self,
        max_num_elements: int = 100,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_first: bool = True,
        bias: bool = True,
        num_layers: int = 4,
    ):
        super().__init__()

        self.max_num_elements = max_num_elements
        self.d_model = d_model
        self.num_layers = num_layers

        activation_fn = {
            "gelu": nn.GELU(approximate="tanh"),
            "relu": nn.ReLU(),
        }[activation]
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=activation_fn,
                dropout=dropout,
                batch_first=True,
                norm_first=norm_first,
                bias=bias,
            ),
            norm=nn.LayerNorm(d_model),
            num_layers=num_layers,
        )

        # Output heads
        self.atom_types_head = nn.Linear(d_model, max_num_elements, bias=True)
        self.pos_head = nn.Linear(d_model, 3, bias=False)
        
        # Crystal heads (silenced for molecules - output zeros)
        self.frac_coords_head = nn.Linear(d_model, 3, bias=False)
        self.lattice_head = nn.Linear(d_model, 6, bias=False)

    def forward(self, encoded_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            encoded_batch: Dict with keys:
                x (torch.Tensor): Encoded latent tokens (n, d)
                num_atoms (torch.Tensor): Number of atoms per sample
                batch (torch.Tensor): Batch index for each atom
                token_idx (torch.Tensor): Token index for each atom

        Returns:
            Dict with keys: atom_types (n, max_elements), pos (n, 3), 
                            frac_coords (n, 3), lattices (bsz, 6)
        """
        x = encoded_batch["x"]

        # Positional embedding
        x = x + get_index_embedding(encoded_batch["token_idx"], self.d_model)

        # Convert from PyG batch to dense batch with padding
        x, token_mask = to_dense_batch(x, encoded_batch["batch"])

        # Transformer forward pass
        x = self.transformer.forward(x, src_key_padding_mask=(~token_mask))
        x = x[token_mask]

        # Global pooling for lattice prediction: (n, d) -> (bsz, d)
        x_global = scatter(x, encoded_batch["batch"], dim=0, reduce="mean")

        # Prediction heads
        atom_types_out = self.atom_types_head(x)  # (n, max_elements)
        pos_out = self.pos_head(x)  # (n, 3)
        
        # Crystal outputs (silenced for molecules)
        frac_coords_out = self.frac_coords_head(x)  # (n, 3)
        lattices_out = self.lattice_head(x_global)  # (bsz, 6)

        return {
            "atom_types": atom_types_out,
            "pos": pos_out,
            "frac_coords": frac_coords_out,
            "lattices": lattices_out,
            "lengths": lattices_out[:, :3],
            "angles": lattices_out[:, 3:],
        }
