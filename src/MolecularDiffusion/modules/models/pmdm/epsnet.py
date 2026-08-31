"""PMDM's pocket-conditioned epsilon network.

Port of ``models/epsnet/MDM_pocket_coor_shared.py::MDM_full_pocket_coor_shared``
(the only network PMDM's ``get_model`` can return), re-exported as
:class:`PMDMEpsNet`.

Shape of the model, unchanged from upstream:

* the **pocket** is embedded once by a SchNet tower and is never noised;
* the **ligand** is the diffused variable -- coordinates score-matching style,
  atom features VP style;
* a cross-attention block mixes ligand and pocket tokens, then two EGNN
  stacks (global @ ``g_cutoff``, local @ ``cutoff``) run over the
  *concatenated* ligand+pocket point set, with only ligand coordinates
  updated;
* two MLP heads produce ``grad_{global,local}_node``; the position score is
  the EGNN coordinate output itself.

Differences from upstream, all of them removals of code that this
integration's scope (see ``docs/model_integrations/pmdm/INTEGRATION_PLAN.md``)
puts out of reach:

* the ``vae_context`` VAE latent branch and the ``context`` property list
  (both default-off upstream);
* ``atom_num_emb`` (default-off), ``is_sidechain`` (always ``None``),
  and the ``gaussian`` edge encoder (config is ``mlp``);
* ``self.model_global``/``model_local``, which were ``ModuleList`` aliases of
  already-registered submodules and only duplicated state_dict keys.

The config object is replaced by explicit keyword arguments so the whole
network is Hydra-instantiable from ``configs/tasks/diffusion_pmdm.yaml``.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_scatter import scatter_mean
from tqdm.auto import tqdm

from .common import (
    center_pos_lp,
    center_pos_pl,
    clip_norm,
    eq_transform,
    extend_graph_order_radius,
    get_distance,
    get_edges,
    get_num_embedding,
    MultiLayerPerceptron,
)
from .encoders import CrossAttentionBlock, EGNNSparseNetwork, MLPEdgeEncoder, SchNetProteinEncoder


def get_beta_schedule(
    beta_schedule: str, beta_start: float, beta_end: float, num_diffusion_timesteps: int
) -> np.ndarray:
    """Upstream's noise schedules (``sigmoid`` is what the released config uses)."""
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5, beta_end**0.5, num_diffusion_timesteps, dtype=np.float64
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        x = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = (1 / (np.exp(-x) + 1)) * (beta_end - beta_start) + beta_start
    elif beta_schedule == "cosine":
        betas = []
        alpha_bar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2  # noqa: E731
        for i in range(num_diffusion_timesteps):
            t1, t2 = i / num_diffusion_timesteps, (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        betas = np.array(betas)
    else:
        raise NotImplementedError(f"unknown beta_schedule: {beta_schedule}")
    if betas.shape != (num_diffusion_timesteps,):
        raise ValueError(f"bad beta schedule shape {betas.shape}")
    return betas


def _compute_alpha(betas: Tensor, t: Tensor) -> Tensor:
    """Cumulative ``alpha`` at timestep ``t``. Shared by every sampler below."""
    betas = torch.cat([torch.zeros(1).to(betas.device), betas], dim=0)
    return (1 - betas).cumprod(dim=0).index_select(0, t + 1)


class PMDMEpsNet(nn.Module):
    """Dual-encoder, pocket-conditioned score network."""

    def __init__(
        self,
        num_atom: int = 10,
        protein_feature_dim: int = 31,
        hidden_dim: int = 128,
        protein_hidden_dim: int = 128,
        num_convs: int = 3,
        num_convs_local: int = 3,
        protein_num_convs: int = 2,
        cutoff: float = 3.0,
        g_cutoff: float = 6.0,
        encoder_cutoff: float = 6.0,
        edge_order: int = 3,
        mlp_act: str = "relu",
        edge_encoder: str = "mlp",
        soft_edge: bool = True,
        norm_coors: bool = True,
        beta_schedule: str = "sigmoid",
        beta_start: float = 1.0e-7,
        beta_end: float = 2.0e-3,
        num_diffusion_timesteps: int = 1000,
    ) -> None:
        super().__init__()
        if edge_encoder != "mlp":
            raise NotImplementedError(
                f"edge_encoder={edge_encoder!r}: only 'mlp' is ported "
                "(the released PMDM config uses it)."
            )
        if hidden_dim != protein_hidden_dim:
            raise ValueError(
                "hidden_dim and protein_hidden_dim must match: the cross-attention "
                "block mixes ligand and pocket tokens in one space."
            )

        self.num_atom = num_atom
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        self.g_cutoff = g_cutoff
        self.edge_order = edge_order

        self.edge_encoder_global = MLPEdgeEncoder(hidden_dim, mlp_act)
        self.edge_encoder_local = MLPEdgeEncoder(hidden_dim, mlp_act)
        self.atten_layer = CrossAttentionBlock(
            hidden_dim, 4, hidden_dim // 4, 0.1, hidden_dim
        )

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim * 4), nn.Linear(hidden_dim * 4, hidden_dim * 4)]
        )
        self.temb_proj = nn.Linear(hidden_dim * 4, hidden_dim)

        self.protein_encoder = SchNetProteinEncoder(
            hidden_channels=protein_hidden_dim,
            num_filters=protein_hidden_dim,
            num_interactions=protein_num_convs,
            edge_channels=self.edge_encoder_global.out_channels,
            cutoff=encoder_cutoff,
            input_dim=protein_feature_dim,
        )
        self.ligand_encoder = SchNetProteinEncoder(
            hidden_channels=protein_hidden_dim,
            num_filters=protein_hidden_dim,
            num_interactions=protein_num_convs,
            edge_channels=self.edge_encoder_global.out_channels,
            cutoff=encoder_cutoff,
            input_dim=num_atom,
        )

        self.encoder_global = EGNNSparseNetwork(
            n_layers=num_convs,
            feats_dim=hidden_dim,
            edge_attr_dim=hidden_dim,
            m_dim=hidden_dim,
            soft_edge=int(soft_edge),
            norm_coors=norm_coors,
        )
        self.encoder_local = EGNNSparseNetwork(
            n_layers=num_convs_local,
            feats_dim=hidden_dim,
            edge_attr_dim=hidden_dim,
            m_dim=hidden_dim,
            soft_edge=int(soft_edge),
            norm_coors=norm_coors,
        )

        # NB: upstream also builds grad_{global,local}_dist_mlp heads, but
        # never calls them -- the position score comes from the EGNN
        # coordinate output, not a distance head. They are omitted so DDP
        # does not trip over permanently-unused parameters.
        self.grad_global_node_mlp = MultiLayerPerceptron(
            hidden_dim, [hidden_dim, hidden_dim // 2, num_atom], activation=mlp_act
        )
        self.grad_local_node_mlp = MultiLayerPerceptron(
            hidden_dim, [hidden_dim, hidden_dim // 2, num_atom], activation=mlp_act
        )

        betas = torch.from_numpy(
            get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps)
        ).float()
        self.betas = nn.Parameter(betas, requires_grad=False)
        self.alphas = nn.Parameter((1.0 - betas).cumprod(dim=0), requires_grad=False)
        self.num_timesteps = int(self.betas.size(0))
        #: Default number of reverse steps. Named ``T`` because that is the
        #: attribute ``cli/generate.py`` reaches for when the generation config
        #: sets ``diffusion_steps``; the task passes it to ``n_steps``.
        self.T = self.num_timesteps

    # ------------------------------------------------------------------ #
    # score network
    # ------------------------------------------------------------------ #
    def net(
        self,
        ligand_atom_type: Tensor,
        ligand_pos: Tensor,
        ligand_bond_index: Tensor,
        ligand_bond_type: Optional[Tensor],
        ligand_batch: Tensor,
        protein_embeddings: Tensor,
        protein_pos: Tensor,
        protein_batch: Tensor,
        time_step: Tensor,
        linker_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """One denoiser pass. Returns the global/local position and node scores
        plus the ligand edge bookkeeping the loss needs.

        ``linker_mask`` (bool, ``(n_ligand,)``, ``True`` = regenerate): used
        only by :meth:`linker_sample`, to bias the EGNN coordinate update
        toward the region being regenerated (see ``encoders.py``'s module
        docstring). ``None`` (default) reproduces training/default-sampling
        behaviour exactly.
        """
        n_ligand = ligand_atom_type.size(0)

        # pooled pocket context, broadcast onto every ligand token
        context = scatter_mean(protein_embeddings, protein_batch, dim=0).index_select(
            0, ligand_batch
        )

        nonlinearity = F.relu
        temb = get_num_embedding(time_step, self.hidden_dim)
        temb = self.temb.dense[0](temb)
        temb = self.temb.dense[1](nonlinearity(temb))
        temb = self.temb_proj(nonlinearity(temb))
        context = temb.index_select(0, ligand_batch) + context

        ligand_feats = (
            self.ligand_encoder(node_attr=ligand_atom_type, pos=ligand_pos, batch=ligand_batch)
            + context
        )
        ligand_feats, protein_feats = self.atten_layer(ligand_feats, protein_embeddings)

        pocket_atom = torch.cat([ligand_feats, protein_feats], dim=0)
        pocket_pos = torch.cat([ligand_pos, protein_pos], dim=0)
        pocket_batch = torch.cat([ligand_batch, protein_batch])

        # ligand-only edge bookkeeping for the distance-denoising target
        full_bond_type = torch.ones(
            ligand_bond_index.size(1), dtype=torch.long, device=ligand_bond_index.device
        )
        edge_index, edge_type = extend_graph_order_radius(
            num_nodes=n_ligand,
            pos=ligand_pos,
            edge_index=ligand_bond_index,
            edge_type=full_bond_type,
            batch=ligand_batch,
            order=self.edge_order,
            cutoff=self.cutoff,
        )
        edge_length = get_distance(ligand_pos, edge_index).unsqueeze(-1)

        if ligand_bond_type is not None:
            if ligand_bond_type.numel() != edge_index.size(1):
                raise RuntimeError(
                    "ligand_bond_type has "
                    f"{ligand_bond_type.numel()} entries but the extended ligand "
                    f"graph has {edge_index.size(1)} edges. PMDM's transform makes "
                    "the ligand graph fully connected, which makes those equal; "
                    "a mismatch means the collate emitted a different graph."
                )
            local_edge_mask = ligand_bond_type > 0
        else:
            local_edge_mask = edge_type == 0

        # joined ligand+pocket geometric graphs
        local_pocket_edge = get_edges(pocket_pos, pocket_batch, self.cutoff)
        global_pocket_edge = get_edges(pocket_pos, pocket_batch, self.g_cutoff)

        edge_attr_global = self.edge_encoder_global(
            get_distance(pocket_pos, global_pocket_edge).unsqueeze(-1)
        )
        node_attr_global, pos_attr_global = self.encoder_global(
            z=pocket_atom,
            pos=pocket_pos,
            edge_index=global_pocket_edge,
            edge_attr=edge_attr_global,
            batch=pocket_batch,
            n_ligand=n_ligand,
            linker_mask=linker_mask,
        )

        edge_attr_local = self.edge_encoder_local(
            get_distance(pocket_pos, local_pocket_edge).unsqueeze(-1)
        )
        node_attr_local, pos_attr_local = self.encoder_local(
            z=pocket_atom,
            pos=pocket_pos,
            edge_index=local_pocket_edge,
            edge_attr=edge_attr_local,
            batch=pocket_batch,
            n_ligand=n_ligand,
            linker_mask=linker_mask,
        )

        node_score_global = self.grad_global_node_mlp(node_attr_global)
        node_score_local = self.grad_local_node_mlp(node_attr_local)

        return (
            pos_attr_global,
            pos_attr_local,
            node_score_global,
            node_score_local,
            edge_index,
            edge_type,
            edge_length,
            local_edge_mask,
        )

    # ------------------------------------------------------------------ #
    # training objective
    # ------------------------------------------------------------------ #
    def forward(self, batch, return_unreduced_loss: bool = False):
        """Score-matching loss for one batch of pocket-ligand complexes.

        ``batch`` is any attribute-access object carrying the keys
        ``pmdm_collate`` emits (a ``SimpleNamespace`` in practice).
        """
        ligand_atom_type = batch.ligand_atom_feature.float()
        ligand_pos = batch.ligand_pos
        ligand_bond_index = batch.ligand_bond_index
        ligand_bond_type = batch.ligand_bond_type
        ligand_batch = batch.ligand_element_batch
        protein_atom_feature_full = batch.protein_atom_feature_full.float()
        protein_pos = batch.protein_pos
        protein_batch = batch.protein_element_batch

        node2graph = ligand_batch
        num_graphs = int(node2graph.max().item()) + 1

        # antithetic timestep sampling (upstream)
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=ligand_pos.device
        )
        time_step = torch.cat([time_step, self.num_timesteps - time_step - 1], dim=0)[
            :num_graphs
        ]

        a = self.alphas.index_select(0, time_step)
        a_pos = a.index_select(0, node2graph).unsqueeze(-1)

        pos_noise = torch.randn_like(ligand_pos)
        atom_noise = torch.randn_like(ligand_atom_type)

        # only the ligand is perturbed; the pocket just rides the recentring
        ligand_pos, protein_pos = center_pos_pl(
            ligand_pos, protein_pos, ligand_batch, protein_batch
        )
        ligand_pos_perturbed = ligand_pos + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()
        ligand_pos_perturbed, protein_pos = center_pos_pl(
            ligand_pos_perturbed, protein_pos, ligand_batch, protein_batch
        )
        ligand_atom_perturbed = (
            a_pos.sqrt() * ligand_atom_type + (1.0 - a_pos).sqrt() * atom_noise
        )

        protein_ctx = self.protein_encoder(
            node_attr=protein_atom_feature_full, pos=protein_pos, batch=protein_batch
        )

        (
            pos_eq_global,
            pos_eq_local,
            node_score_global,
            node_score_local,
            edge_index,
            _edge_type,
            edge_length,
            local_edge_mask,
        ) = self.net(
            ligand_atom_type=ligand_atom_perturbed,
            ligand_pos=ligand_pos_perturbed,
            ligand_bond_index=ligand_bond_index,
            ligand_bond_type=ligand_bond_type,
            ligand_batch=ligand_batch,
            protein_embeddings=protein_ctx,
            protein_pos=protein_pos,
            protein_batch=protein_batch,
            time_step=time_step,
        )

        edge2graph = node2graph.index_select(0, edge_index[0])
        a_edge = a.index_select(0, edge2graph).unsqueeze(-1)

        d_gt = get_distance(ligand_pos, edge_index).unsqueeze(-1)
        d_perturbed = edge_length
        d_target = (d_gt - d_perturbed) / (1.0 - a_edge).sqrt() * a_edge.sqrt()

        global_mask = torch.logical_and(
            torch.logical_or(
                torch.logical_and(
                    d_perturbed > self.cutoff, d_perturbed <= self.g_cutoff
                ),
                local_edge_mask.unsqueeze(-1),
            ),
            ~local_edge_mask.unsqueeze(-1),
        )
        target_d_global = torch.where(global_mask, d_target, torch.zeros_like(d_target))
        target_pos_global = eq_transform(
            target_d_global, ligand_pos_perturbed, edge_index, edge_length
        )
        target_pos_local = eq_transform(
            d_target[local_edge_mask],
            ligand_pos_perturbed,
            edge_index[:, local_edge_mask],
            edge_length[local_edge_mask],
        )

        loss_pos = F.mse_loss(
            pos_eq_global + pos_eq_local, target_pos_global + target_pos_local, reduction="none"
        ).sum(dim=-1, keepdim=True)
        loss_node = F.mse_loss(
            node_score_global + node_score_local, atom_noise, reduction="none"
        ).sum(dim=-1, keepdim=True)
        loss = loss_pos + loss_node

        if return_unreduced_loss:
            return loss, loss_pos, loss_node
        return loss

    # ------------------------------------------------------------------ #
    # sampling
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def langevin_dynamics_sample(
        self,
        ligand_atom_type: Tensor,
        ligand_pos_init: Tensor,
        ligand_bond_index: Tensor,
        ligand_bond_type: Optional[Tensor],
        ligand_batch: Tensor,
        protein_atom_feature_full: Tensor,
        protein_pos: Tensor,
        protein_batch: Tensor,
        num_graphs: int,
        n_steps: int = 100,
        step_lr: float = 1.0e-6,
        clip: float = 1000.0,
        clip_local: Optional[float] = None,
        clip_pos: Optional[float] = None,
        global_start_sigma: float = float("inf"),
        local_start_sigma: float = float("inf"),
        w_global_pos: float = 1.0,
        w_global_node: float = 1.0,
        w_local_pos: float = 1.0,
        w_local_node: float = 1.0,
        sampling_type: str = "generalized",
        eta: float = 1.0,
        keep_traj: bool = False,
    ) -> Tuple[Tensor, List[Tensor], Tensor, List[Tensor]]:
        """Reverse process inside a fixed pocket.

        Returns ``(ligand_pos, pos_traj, ligand_atom_type, atom_traj)``, flat
        and concatenated (not padded). Final coordinates are translated back
        into the *input* pocket's frame.
        """

        def compute_alpha(beta: Tensor, t: Tensor) -> Tensor:
            beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
            return (1 - beta).cumprod(dim=0).index_select(0, t + 1)

        sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
        pos_traj: List[Tensor] = []
        atom_traj: List[Tensor] = []

        n_steps = max(1, min(int(n_steps), self.num_timesteps))
        skip = max(1, self.num_timesteps // n_steps)
        seq = range(0, self.num_timesteps, skip)
        seq_next = [-1] + list(seq[:-1])

        protein_com = scatter_mean(protein_pos, protein_batch, dim=0)
        ligand_pos, protein_pos = center_pos_pl(
            ligand_pos_init + protein_com[ligand_batch],
            protein_pos,
            ligand_batch,
            protein_batch,
        )

        protein_ctx = self.protein_encoder(
            node_attr=protein_atom_feature_full, pos=protein_pos, batch=protein_batch
        )

        for i, j in tqdm(
            list(zip(reversed(seq), reversed(seq_next))), desc="pmdm sample", leave=False
        ):
            t = torch.full(
                size=(num_graphs,), fill_value=i, dtype=torch.long, device=ligand_pos.device
            )
            (
                pos_eq_global,
                pos_eq_local,
                node_score_global,
                node_score_local,
                _edge_index,
                _edge_type,
                _edge_length,
                _local_edge_mask,
            ) = self.net(
                ligand_atom_type=ligand_atom_type,
                ligand_pos=ligand_pos,
                ligand_bond_index=ligand_bond_index,
                ligand_bond_type=ligand_bond_type,
                ligand_batch=ligand_batch,
                protein_embeddings=protein_ctx,
                protein_pos=protein_pos,
                protein_batch=protein_batch,
                time_step=t,
            )

            if sigmas[i] < local_start_sigma:
                node_eq_local = pos_eq_local
                if clip_local is not None:
                    node_eq_local = clip_norm(node_eq_local, limit=clip_local)
            else:
                node_eq_local, node_score_local = 0, 0

            if sigmas[i] < global_start_sigma:
                node_eq_global = clip_norm(pos_eq_global, limit=clip)
            else:
                node_eq_global, node_score_global = 0, 0

            eps_pos = w_local_pos * node_eq_local + w_global_pos * node_eq_global
            eps_node = w_local_node * node_score_local + w_global_node * node_score_global

            noise = torch.randn_like(ligand_pos)
            noise_node = torch.randn_like(ligand_atom_type)
            t0 = t[0]
            next_t = (torch.ones(1) * j).to(ligand_pos.device)
            at = compute_alpha(self.betas, t0.long())
            at_next = compute_alpha(self.betas, next_t.long())

            if sampling_type == "generalized":
                et = -eps_pos
                c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                c2 = ((1 - at_next) - c1**2).sqrt()

                step_size_pos_ld = step_lr * (sigmas[i] / 0.01) ** 2 / sigmas[i]
                step_size_pos_gen = 3 * ((1 - at).sqrt() / at.sqrt() - c2 / at_next.sqrt())
                step_size_pos = min(step_size_pos_ld, step_size_pos_gen)

                step_size_noise_ld = torch.sqrt((step_lr * (sigmas[i] / 0.01) ** 2) * 2)
                step_size_noise_gen = 5 * (c1 / at_next.sqrt())
                step_size_noise = min(step_size_noise_ld, step_size_noise_gen)

                eps_node = eps_node / (1 - at).sqrt()
                pos_next = ligand_pos - et * step_size_pos + noise * step_size_noise
                atom_next = (
                    ligand_atom_type - eps_node * step_size_pos + noise_node * step_size_noise
                )
            elif sampling_type == "ddpm_noisy":
                atm1 = at_next
                beta_t = 1 - at / atm1
                e = -eps_pos
                mean = (ligand_pos - beta_t * e) / (1 - beta_t).sqrt()
                mask = 1 - (t0 == 0).float()
                logvar = beta_t.log()
                pos_next = mean + mask * torch.exp(0.5 * logvar) * noise

                e = eps_node
                node0_from_e = (1.0 / at).sqrt() * ligand_atom_type - (
                    1.0 / at - 1
                ).sqrt() * e
                mean = (
                    (atm1.sqrt() * beta_t) * node0_from_e
                    + ((1 - beta_t).sqrt() * (1 - atm1)) * ligand_atom_type
                ) / (1.0 - at)
                atom_next = mean + mask * torch.exp(0.5 * logvar) * noise_node
            elif sampling_type == "ld":
                step_size = step_lr * (sigmas[i] / 0.01) ** 2
                pos_next = (
                    ligand_pos
                    + step_size * eps_pos / sigmas[i]
                    + noise * torch.sqrt(step_size * 2)
                )
                eps_node = eps_node / (1 - at).sqrt()
                atom_next = (
                    ligand_atom_type
                    - step_size * eps_node / sigmas[i]
                    + noise_node * torch.sqrt(step_size * 2)
                )
            else:
                raise ValueError(
                    "sampling_type must be one of [generalized, ddpm_noisy, ld], "
                    f"got {sampling_type!r}"
                )

            ligand_pos, ligand_atom_type = pos_next, atom_next
            if torch.isnan(ligand_pos).any():
                raise FloatingPointError(
                    "NaN in sampled ligand coordinates -- lower step_lr / w_*_pos, "
                    "or set interference.clip_pos."
                )
            ligand_pos, protein_pos = center_pos_pl(
                ligand_pos, protein_pos, ligand_batch, protein_batch
            )
            if clip_pos is not None:
                ligand_pos = torch.clamp(ligand_pos, min=-clip_pos, max=clip_pos)
            if keep_traj:
                pos_traj.append(ligand_pos.clone().cpu())
                atom_traj.append(ligand_atom_type.clone().cpu())

        # back into the input pocket's frame
        protein_final = scatter_mean(protein_pos, protein_batch, dim=0)
        shift = protein_com - protein_final
        ligand_pos = ligand_pos + shift[ligand_batch]
        return ligand_pos, pos_traj, ligand_atom_type, atom_traj

    # ------------------------------------------------------------------ #
    # constrained sampling (lead optimisation / linker design)
    # ------------------------------------------------------------------ #
    def _reverse_step(
        self,
        ligand_pos: Tensor,
        ligand_atom_type: Tensor,
        pos_eq_global: Tensor,
        pos_eq_local: Tensor,
        node_score_global,
        node_score_local,
        sigma_i: Tensor,
        at: Tensor,
        at_next: Tensor,
        t0: Tensor,
        clip: float,
        clip_local: Optional[float],
        global_start_sigma: float,
        local_start_sigma: float,
        w_global_pos: float,
        w_global_node: float,
        w_local_pos: float,
        w_local_node: float,
        sampling_type: str,
        step_lr: float,
        eta: float,
    ) -> Tuple[Tensor, Tensor]:
        """One reverse-diffusion update, given the network's scores for the
        current step. Identical math to the per-step body of
        :meth:`langevin_dynamics_sample`, factored out here since both
        :meth:`inpainting_sample` and :meth:`linker_sample` need it verbatim.
        """
        if sigma_i < local_start_sigma:
            node_eq_local = pos_eq_local
            if clip_local is not None:
                node_eq_local = clip_norm(node_eq_local, limit=clip_local)
        else:
            node_eq_local, node_score_local = 0, 0

        if sigma_i < global_start_sigma:
            node_eq_global = clip_norm(pos_eq_global, limit=clip)
        else:
            node_eq_global, node_score_global = 0, 0

        eps_pos = w_local_pos * node_eq_local + w_global_pos * node_eq_global
        eps_node = w_local_node * node_score_local + w_global_node * node_score_global

        noise = torch.randn_like(ligand_pos)
        noise_node = torch.randn_like(ligand_atom_type)

        if sampling_type == "generalized":
            et = -eps_pos
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1**2).sqrt()

            step_size_pos_ld = step_lr * (sigma_i / 0.01) ** 2 / sigma_i
            step_size_pos_gen = 3 * ((1 - at).sqrt() / at.sqrt() - c2 / at_next.sqrt())
            step_size_pos = min(step_size_pos_ld, step_size_pos_gen)

            step_size_noise_ld = torch.sqrt((step_lr * (sigma_i / 0.01) ** 2) * 2)
            step_size_noise_gen = 5 * (c1 / at_next.sqrt())
            step_size_noise = min(step_size_noise_ld, step_size_noise_gen)

            eps_node = eps_node / (1 - at).sqrt()
            pos_next = ligand_pos - et * step_size_pos + noise * step_size_noise
            atom_next = (
                ligand_atom_type - eps_node * step_size_pos + noise_node * step_size_noise
            )
        elif sampling_type == "ddpm_noisy":
            atm1 = at_next
            beta_t = 1 - at / atm1
            e = -eps_pos
            mean = (ligand_pos - beta_t * e) / (1 - beta_t).sqrt()
            mask = 1 - (t0 == 0).float()
            logvar = beta_t.log()
            pos_next = mean + mask * torch.exp(0.5 * logvar) * noise

            e = eps_node
            node0_from_e = (1.0 / at).sqrt() * ligand_atom_type - (
                1.0 / at - 1
            ).sqrt() * e
            mean = (
                (atm1.sqrt() * beta_t) * node0_from_e
                + ((1 - beta_t).sqrt() * (1 - atm1)) * ligand_atom_type
            ) / (1.0 - at)
            atom_next = mean + mask * torch.exp(0.5 * logvar) * noise_node
        elif sampling_type == "ld":
            step_size = step_lr * (sigma_i / 0.01) ** 2
            pos_next = (
                ligand_pos
                + step_size * eps_pos / sigma_i
                + noise * torch.sqrt(step_size * 2)
            )
            eps_node = eps_node / (1 - at).sqrt()
            atom_next = (
                ligand_atom_type
                - step_size * eps_node / sigma_i
                + noise_node * torch.sqrt(step_size * 2)
            )
        else:
            raise ValueError(
                "sampling_type must be one of [generalized, ddpm_noisy, ld], "
                f"got {sampling_type!r}"
            )
        return pos_next, atom_next

    @torch.no_grad()
    def inpainting_sample(
        self,
        ligand_atom_type: Tensor,
        ligand_pos_init: Tensor,
        ligand_bond_index: Tensor,
        ligand_bond_type: Optional[Tensor],
        ligand_batch: Tensor,
        frag_mask: Tensor,
        protein_atom_feature_full: Tensor,
        protein_pos: Tensor,
        protein_batch: Tensor,
        num_graphs: int,
        n_steps: int = 100,
        step_lr: float = 1.0e-6,
        clip: float = 1000.0,
        clip_local: Optional[float] = None,
        clip_pos: Optional[float] = None,
        global_start_sigma: float = float("inf"),
        local_start_sigma: float = float("inf"),
        w_global_pos: float = 1.0,
        w_global_node: float = 1.0,
        w_local_pos: float = 1.0,
        w_local_node: float = 1.0,
        sampling_type: str = "generalized",
        eta: float = 1.0,
        keep_traj: bool = False,
    ) -> Tuple[Tensor, List[Tensor], Tensor, List[Tensor]]:
        """RePaint-style constrained sampling ("lead optimisation" upstream):
        keep ``frag_mask`` atoms of a starting ligand fixed, regenerate the
        rest, inside the same fixed pocket.

        ``frag_mask`` (bool, ``(n_ligand,)``): ``True`` = keep exactly as
        given in ``ligand_pos_init``/``ligand_atom_type``; ``False`` =
        regenerate. Every reverse step re-noises the kept fragment's atom
        *type* to the current timestep before scoring, then force-restores
        both its position and type afterward -- classic RePaint. Position is
        never re-noised (upstream's matching line is dead/commented-out
        code, not ported). Port of upstream ``inpainting_sample``,
        ``MDM_pocket_coor_shared.py:929-1183``.
        """
        sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
        pos_traj: List[Tensor] = []
        atom_traj: List[Tensor] = []

        n_steps = max(1, min(int(n_steps), self.num_timesteps))
        skip = max(1, self.num_timesteps // n_steps)
        seq = range(0, self.num_timesteps, skip)
        seq_next = [-1] + list(seq[:-1])

        protein_com = scatter_mean(protein_pos, protein_batch, dim=0)
        # The fragment's positions are real, already-placed coordinates (the
        # user's SDF); the newly-appended atoms are raw N(0,1) noise centred
        # on world-origin, not on the pocket. center_pos_lp always subtracts
        # the pocket's own (real-world) centroid, so without this shift the
        # noise atoms land ~|protein_com| away from the pocket/fragment after
        # centering -- often tens of angstroms, outside every radius-graph
        # cutoff, leaving them permanently edge-less and unable to receive
        # any score signal pulling them toward the fragment (upstream has a
        # dead, commented-out line at this exact spot -- applying it to the
        # WHOLE tensor would instead un-centre the fragment; shifting only
        # the new atoms is the part that is actually needed).
        new_atom_mask = ~frag_mask
        ligand_pos_init = ligand_pos_init.clone()
        ligand_pos_init[new_atom_mask] = (
            ligand_pos_init[new_atom_mask] + protein_com[ligand_batch][new_atom_mask]
        )
        ligand_pos, protein_pos = center_pos_lp(
            ligand_pos_init, protein_pos, ligand_batch, protein_batch
        )
        ligand_atom_type = ligand_atom_type.clone()

        protein_ctx = self.protein_encoder(
            node_attr=protein_atom_feature_full, pos=protein_pos, batch=protein_batch
        )

        for i, j in tqdm(
            list(zip(reversed(seq), reversed(seq_next))), desc="pmdm inpaint", leave=False
        ):
            t = torch.full(
                size=(num_graphs,), fill_value=i, dtype=torch.long, device=ligand_pos.device
            )
            at = _compute_alpha(self.betas, t[0].long())
            step_mask = 1.0 - (t[0] == 0).float()

            frag_pos = ligand_pos[frag_mask]
            frag_atom_type = ligand_atom_type[frag_mask]

            atom_noise = torch.randn_like(frag_atom_type)
            ligand_atom_type = ligand_atom_type.clone()
            ligand_atom_type[frag_mask] = (
                at.sqrt() * frag_atom_type + (1.0 - at).sqrt() * atom_noise * step_mask
            )

            (
                pos_eq_global,
                pos_eq_local,
                node_score_global,
                node_score_local,
                _edge_index,
                _edge_type,
                _edge_length,
                _local_edge_mask,
            ) = self.net(
                ligand_atom_type=ligand_atom_type,
                ligand_pos=ligand_pos,
                ligand_bond_index=ligand_bond_index,
                ligand_bond_type=ligand_bond_type,
                ligand_batch=ligand_batch,
                protein_embeddings=protein_ctx,
                protein_pos=protein_pos,
                protein_batch=protein_batch,
                time_step=t,
            )

            t0 = t[0]
            next_t = (torch.ones(1) * j).to(ligand_pos.device)
            at_next = _compute_alpha(self.betas, next_t.long())
            pos_next, atom_next = self._reverse_step(
                ligand_pos,
                ligand_atom_type,
                pos_eq_global,
                pos_eq_local,
                node_score_global,
                node_score_local,
                sigmas[i],
                at,
                at_next,
                t0,
                clip,
                clip_local,
                global_start_sigma,
                local_start_sigma,
                w_global_pos,
                w_global_node,
                w_local_pos,
                w_local_node,
                sampling_type,
                step_lr,
                eta,
            )
            ligand_pos, ligand_atom_type = pos_next, atom_next
            if torch.isnan(ligand_pos).any():
                raise FloatingPointError(
                    "NaN in sampled ligand coordinates -- lower step_lr / w_*_pos, "
                    "or set interference.clip_pos."
                )

            # RePaint restore: force the kept fragment back to its true value
            ligand_pos = ligand_pos.clone()
            ligand_pos[frag_mask] = frag_pos
            ligand_atom_type = ligand_atom_type.clone()
            ligand_atom_type[frag_mask] = frag_atom_type

            ligand_pos, protein_pos = center_pos_pl(
                ligand_pos, protein_pos, ligand_batch, protein_batch
            )
            if clip_pos is not None:
                ligand_pos = torch.clamp(ligand_pos, min=-clip_pos, max=clip_pos)
            if keep_traj:
                pos_traj.append(ligand_pos.clone().cpu())
                atom_traj.append(ligand_atom_type.clone().cpu())

        protein_final = scatter_mean(protein_pos, protein_batch, dim=0)
        shift = protein_com - protein_final
        ligand_pos = ligand_pos + shift[ligand_batch]
        return ligand_pos, pos_traj, ligand_atom_type, atom_traj

    @torch.no_grad()
    def linker_sample(
        self,
        ligand_atom_type: Tensor,
        ligand_pos_init: Tensor,
        ligand_bond_index: Tensor,
        ligand_bond_type: Optional[Tensor],
        ligand_batch: Tensor,
        frag_mask: Tensor,
        protein_atom_feature_full: Tensor,
        protein_pos: Tensor,
        protein_batch: Tensor,
        num_graphs: int,
        n_steps: int = 100,
        step_lr: float = 1.0e-6,
        clip: float = 1000.0,
        clip_local: Optional[float] = None,
        clip_pos: Optional[float] = None,
        global_start_sigma: float = float("inf"),
        local_start_sigma: float = float("inf"),
        w_global_pos: float = 1.0,
        w_global_node: float = 1.0,
        w_local_pos: float = 1.0,
        w_local_node: float = 1.0,
        sampling_type: str = "generalized",
        eta: float = 1.0,
        keep_traj: bool = False,
    ) -> Tuple[Tensor, List[Tensor], Tensor, List[Tensor]]:
        """RePaint-style constrained sampling for linker design: keep two (or
        more) disjoint fragments of a starting ligand fixed (``frag_mask``),
        regenerate the region between them, inside the same fixed pocket.

        Structurally identical to :meth:`inpainting_sample` (same
        restore-after-every-step loop), with two differences: the kept
        fragment is fed to the network exactly as given, with no per-step
        forward-noise on its atom type (upstream computes a matching noise
        tensor here but never uses it -- dead code, not ported); and the
        network additionally receives ``linker_mask`` (the region being
        regenerated), which biases its coordinate update toward that region
        (see ``encoders.py``). Port of upstream ``linker_sample``,
        ``MDM_pocket_coor_shared.py:1186-1425``.
        """
        sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
        pos_traj: List[Tensor] = []
        atom_traj: List[Tensor] = []

        n_steps = max(1, min(int(n_steps), self.num_timesteps))
        skip = max(1, self.num_timesteps // n_steps)
        seq = range(0, self.num_timesteps, skip)
        seq_next = [-1] + list(seq[:-1])

        linker_mask = ~frag_mask

        protein_com = scatter_mean(protein_pos, protein_batch, dim=0)
        # Same fix as inpainting_sample: shift only the newly-appended
        # (linker) atoms' raw N(0,1) noise into the pocket's real-world
        # frame before center_pos_lp subtracts it back out -- otherwise
        # they land tens of angstroms from the kept fragments, outside
        # every radius-graph cutoff, and never receive a score signal
        # pulling them toward the gap they are meant to bridge.
        ligand_pos_init = ligand_pos_init.clone()
        ligand_pos_init[linker_mask] = (
            ligand_pos_init[linker_mask] + protein_com[ligand_batch][linker_mask]
        )
        ligand_pos, protein_pos = center_pos_lp(
            ligand_pos_init, protein_pos, ligand_batch, protein_batch
        )
        ligand_atom_type = ligand_atom_type.clone()

        protein_ctx = self.protein_encoder(
            node_attr=protein_atom_feature_full, pos=protein_pos, batch=protein_batch
        )

        for i, j in tqdm(
            list(zip(reversed(seq), reversed(seq_next))), desc="pmdm linker", leave=False
        ):
            t = torch.full(
                size=(num_graphs,), fill_value=i, dtype=torch.long, device=ligand_pos.device
            )
            at = _compute_alpha(self.betas, t[0].long())

            frag_pos = ligand_pos[frag_mask]
            frag_atom_type = ligand_atom_type[frag_mask]

            (
                pos_eq_global,
                pos_eq_local,
                node_score_global,
                node_score_local,
                _edge_index,
                _edge_type,
                _edge_length,
                _local_edge_mask,
            ) = self.net(
                ligand_atom_type=ligand_atom_type,
                ligand_pos=ligand_pos,
                ligand_bond_index=ligand_bond_index,
                ligand_bond_type=ligand_bond_type,
                ligand_batch=ligand_batch,
                protein_embeddings=protein_ctx,
                protein_pos=protein_pos,
                protein_batch=protein_batch,
                time_step=t,
                linker_mask=linker_mask,
            )

            t0 = t[0]
            next_t = (torch.ones(1) * j).to(ligand_pos.device)
            at_next = _compute_alpha(self.betas, next_t.long())
            pos_next, atom_next = self._reverse_step(
                ligand_pos,
                ligand_atom_type,
                pos_eq_global,
                pos_eq_local,
                node_score_global,
                node_score_local,
                sigmas[i],
                at,
                at_next,
                t0,
                clip,
                clip_local,
                global_start_sigma,
                local_start_sigma,
                w_global_pos,
                w_global_node,
                w_local_pos,
                w_local_node,
                sampling_type,
                step_lr,
                eta,
            )
            ligand_pos, ligand_atom_type = pos_next, atom_next
            if torch.isnan(ligand_pos).any():
                raise FloatingPointError(
                    "NaN in sampled ligand coordinates -- lower step_lr / w_*_pos, "
                    "or set interference.clip_pos."
                )

            ligand_pos = ligand_pos.clone()
            ligand_pos[frag_mask] = frag_pos
            ligand_atom_type = ligand_atom_type.clone()
            ligand_atom_type[frag_mask] = frag_atom_type

            ligand_pos, protein_pos = center_pos_pl(
                ligand_pos, protein_pos, ligand_batch, protein_batch
            )
            if clip_pos is not None:
                ligand_pos = torch.clamp(ligand_pos, min=-clip_pos, max=clip_pos)
            if keep_traj:
                pos_traj.append(ligand_pos.clone().cpu())
                atom_traj.append(ligand_atom_type.clone().cpu())

        protein_final = scatter_mean(protein_pos, protein_batch, dim=0)
        shift = protein_com - protein_final
        ligand_pos = ligand_pos + shift[ligand_batch]
        return ligand_pos, pos_traj, ligand_atom_type, atom_traj
