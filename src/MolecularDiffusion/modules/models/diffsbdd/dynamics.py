"""``EGNNDynamics``, ported from DiffSBDD ``equivariant_diffusion/dynamics.py``.

Two node sets -- a diffused ligand and a protein pocket -- are projected into
a shared ``joint_nf`` space by separate encoder/decoder MLPs, concatenated
into one flat node set, message-passed over a distance-cutoff graph, then
split and decoded again.

The graph is rebuilt every forward pass from batch membership plus optional
distance cutoffs (:meth:`EGNNDynamics.get_edges`). It carries no chemistry:
``edge_embedding`` is a 3-way *node provenance* label (lig-lig / lig-pocket /
pocket-pocket), not a bond type, and is ``None`` unless
``edge_embedding_dim`` is set -- which no shipped DiffSBDD config does.

``mode='gnn_dynamics'`` is not ported (unused by every shipped config), so
``self.mode`` is gone and the EGNN branch is unconditional.

Submodule names are state-dict keys of the released checkpoints; do not
rename ``atom_encoder`` / ``atom_decoder`` / ``residue_encoder`` /
``residue_decoder`` / ``egnn``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch_scatter import scatter_mean

from MolecularDiffusion.modules.models.diffsbdd.egnn import EGNN


class EGNNDynamics(nn.Module):
    """``(xh_ligand, xh_pocket, t, mask_lig, mask_pocket) -> (out_lig, out_pocket)``."""

    def __init__(
        self,
        atom_nf: int,
        residue_nf: int,
        n_dims: int = 3,
        joint_nf: int = 16,
        hidden_nf: int = 64,
        act_fn: Optional[nn.Module] = None,
        n_layers: int = 4,
        attention: bool = False,
        condition_time: bool = True,
        tanh: bool = False,
        norm_constant: float = 0,
        inv_sublayers: int = 2,
        sin_embedding: bool = False,
        normalization_factor: float = 100,
        aggregation_method: str = "sum",
        update_pocket_coords: bool = True,
        edge_cutoff_ligand: Optional[float] = None,
        edge_cutoff_pocket: Optional[float] = None,
        edge_cutoff_interaction: Optional[float] = None,
        reflection_equivariant: bool = True,
        edge_embedding_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        act_fn = nn.SiLU() if act_fn is None else act_fn
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_cutoff_p = edge_cutoff_pocket
        self.edge_cutoff_i = edge_cutoff_interaction
        self.edge_nf = edge_embedding_dim

        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_nf, 2 * atom_nf),
            act_fn,
            nn.Linear(2 * atom_nf, joint_nf),
        )
        self.atom_decoder = nn.Sequential(
            nn.Linear(joint_nf, 2 * atom_nf),
            act_fn,
            nn.Linear(2 * atom_nf, atom_nf),
        )
        self.residue_encoder = nn.Sequential(
            nn.Linear(residue_nf, 2 * residue_nf),
            act_fn,
            nn.Linear(2 * residue_nf, joint_nf),
        )
        self.residue_decoder = nn.Sequential(
            nn.Linear(joint_nf, 2 * residue_nf),
            act_fn,
            nn.Linear(2 * residue_nf, residue_nf),
        )

        self.edge_embedding = (
            nn.Embedding(3, self.edge_nf) if self.edge_nf is not None else None
        )
        self.edge_nf = 0 if self.edge_nf is None else self.edge_nf

        dynamics_node_nf = joint_nf + 1 if condition_time else joint_nf
        self.egnn = EGNN(
            in_node_nf=dynamics_node_nf,
            in_edge_nf=self.edge_nf,
            hidden_nf=hidden_nf,
            act_fn=act_fn,
            n_layers=n_layers,
            attention=attention,
            tanh=tanh,
            norm_constant=norm_constant,
            inv_sublayers=inv_sublayers,
            sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            reflection_equiv=reflection_equivariant,
        )
        self.node_nf = dynamics_node_nf
        self.update_pocket_coords = update_pocket_coords
        self.n_dims = n_dims
        self.condition_time = condition_time

    def forward(
        self,
        xh_atoms: torch.Tensor,
        xh_residues: torch.Tensor,
        t: torch.Tensor,
        mask_atoms: torch.Tensor,
        mask_residues: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_lig = len(mask_atoms)
        x_atoms = xh_atoms[:, : self.n_dims].clone()
        h_atoms = xh_atoms[:, self.n_dims :].clone()
        x_residues = xh_residues[:, : self.n_dims].clone()
        h_residues = xh_residues[:, self.n_dims :].clone()

        h_atoms = self.atom_encoder(h_atoms)
        h_residues = self.residue_encoder(h_residues)

        x = torch.cat((x_atoms, x_residues), dim=0)
        h = torch.cat((h_atoms, h_residues), dim=0)
        mask = torch.cat([mask_atoms, mask_residues])

        if self.condition_time:
            if t.numel() == 1:
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                h_time = t[mask]
            h = torch.cat([h, h_time], dim=1)

        edges = self.get_edges(mask_atoms, mask_residues, x_atoms, x_residues)

        if self.edge_nf > 0:
            # 0: ligand-pocket, 1: ligand-ligand, 2: pocket-pocket.
            # Node provenance, NOT a chemical bond type.
            edge_types = torch.zeros(
                edges.size(1), dtype=torch.long, device=edges.device
            )
            edge_types[(edges[0] < n_lig) & (edges[1] < n_lig)] = 1
            edge_types[(edges[0] >= n_lig) & (edges[1] >= n_lig)] = 2
            edge_attr = self.edge_embedding(edge_types)
        else:
            edge_attr = None

        update_coords_mask = (
            None
            if self.update_pocket_coords
            else torch.cat(
                (torch.ones_like(mask_atoms), torch.zeros_like(mask_residues))
            ).unsqueeze(1)
        )
        h_final, x_final = self.egnn(
            h,
            x,
            edges,
            update_coords_mask=update_coords_mask,
            batch_mask=mask,
            edge_attr=edge_attr,
        )
        vel = x_final - x

        if self.condition_time:
            h_final = h_final[:, :-1]

        h_final_atoms = self.atom_decoder(h_final[:n_lig])
        h_final_residues = self.residue_decoder(h_final[n_lig:])

        if torch.any(torch.isnan(vel)):
            if self.training:
                vel[torch.isnan(vel)] = 0.0
            else:
                raise ValueError("NaN detected in EGNN output")

        if self.update_pocket_coords:
            # joint mode: the whole system lives in the zero-CoM subspace
            vel = vel - scatter_mean(vel, mask, dim=0)[mask]

        return (
            torch.cat([vel[:n_lig], h_final_atoms], dim=-1),
            torch.cat([vel[n_lig:], h_final_residues], dim=-1),
        )

    def get_edges(
        self,
        batch_mask_ligand: torch.Tensor,
        batch_mask_pocket: torch.Tensor,
        x_ligand: torch.Tensor,
        x_pocket: torch.Tensor,
    ) -> torch.Tensor:
        """Complete graph within each complex, optionally distance-masked."""
        adj_ligand = batch_mask_ligand[:, None] == batch_mask_ligand[None, :]
        adj_pocket = batch_mask_pocket[:, None] == batch_mask_pocket[None, :]
        adj_cross = batch_mask_ligand[:, None] == batch_mask_pocket[None, :]

        if self.edge_cutoff_l is not None:
            adj_ligand = adj_ligand & (
                torch.cdist(x_ligand, x_ligand) <= self.edge_cutoff_l
            )
        if self.edge_cutoff_p is not None:
            adj_pocket = adj_pocket & (
                torch.cdist(x_pocket, x_pocket) <= self.edge_cutoff_p
            )
        if self.edge_cutoff_i is not None:
            adj_cross = adj_cross & (
                torch.cdist(x_ligand, x_pocket) <= self.edge_cutoff_i
            )

        adj = torch.cat(
            (
                torch.cat((adj_ligand, adj_cross), dim=1),
                torch.cat((adj_cross.T, adj_pocket), dim=1),
            ),
            dim=0,
        )
        return torch.stack(torch.where(adj), dim=0)
