"""Apo2Mol's ``ScorePosNet3D``: three simultaneous generative processes.

Ported from ``others/Apo2Mol/models/molopt_score_model.py``. Upstream takes an
OmegaConf ``config`` object; this takes plain keyword arguments and rebuilds a
namespace internally, so the Hydra task config stays flat and the ported body
stays a near-verbatim copy.

What is actually being generated, and how:

1. **Ligand coordinates** -- continuous Gaussian DDPM, sigmoid beta schedule,
   ``T = 1000``, the network predicts ``x0`` (``model_mean_type: C0``). Same
   as KGDiff / IPDiff / TargetDiff.
2. **Ligand atom types** -- D3PM categorical diffusion in log space over the
   13-class ``(element, is_aromatic)`` vocabulary, cosine schedule.
3. **Pocket conformation** -- *not* a DDPM. A ``lambda_schedule`` interpolant
   from the holo pose towards the apo pose: translations and chi angles
   interpolate linearly, rotations by SLERP from the identity, each with
   Gaussian jitter scaled by ``beta.sqrt()``. The network predicts the
   **inverse** transform, i.e. how to get back to holo.

Loss (``:772``)::

    loss_ligand_pos + 100 * loss_v + loss_prot_tr + loss_prot_rot
                    + 5 * loss_prot_chi

Two cost-shaping details worth knowing before reading :meth:`sample_diffusion`:

* the pocket is updated on only **5 of the 1000** reverse steps
  (:attr:`protein_update_steps`), because each update runs a Python loop over
  residues in :func:`~.residue_ops.apply_transforms_tensor_batch`;
* the frozen PMINet prior is re-evaluated on **every** step from the current
  prediction, and it builds a dense complex graph, so memory scales with
  ``(pocket atoms x batch)^2``.

Out of scope this pass (see the integration plan): the retrieval-prompt branch
(``topk_prompt: 0`` in the release, so it is dead), ``pos_only`` sampling, and
the ``egnn`` backbone alternative.
"""

from __future__ import annotations

import logging

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_mean
from tqdm.auto import tqdm

from .attn import RetAugmentationLinearAttention
from .common import ShiftedSoftplus
from .residue_ops import (
    apply_transforms_tensor_batch,
    axis_angle_to_quaternion,
    quaternion_product,
    quaternion_to_rotation_matrix,
    slerp_identity_to_q,
)
from .uni_transformer import UniTransformerO2TwoUpdateGeneral

logger = logging.getLogger(__name__)

#: ``molopt_score_model.py:389`` -- the padded ligand block the retrieval
#: attention runs over. Sampling more atoms than this raises.
MAX_LIG_LEN = 150


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5, beta_end**0.5,
                num_diffusion_timesteps, dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule; returns ``sqrt(alphas)`` (the paper's alpha)."""
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    alphas = np.clip(alphas, a_min=0.001, a_max=1.0)
    return np.sqrt(alphas)


def calculate_tm_score(predicted_pos, reference_pos):
    """TM-score after optimal superposition. Diagnostic only.

    ``Bio.SVDSuperimposer`` is imported lazily -- upstream imports it at
    module scope, which would make ``biopython`` (an optional ``[bio]`` extra
    here) a hard import dependency of the model itself.
    """
    from Bio.SVDSuperimposer import SVDSuperimposer  # noqa: PLC0415

    device = predicted_pos.device
    predicted_pos = predicted_pos.cpu().numpy()
    reference_pos = reference_pos.cpu().numpy()

    sup = SVDSuperimposer()
    sup.set(reference_pos, predicted_pos)
    sup.run()
    rot, tran = sup.get_rotran()

    aligned = np.dot(predicted_pos, rot) + tran
    length = aligned.shape[0]
    d_0 = 1.24 * (length ** (1 / 3)) - 1.8
    distances = np.linalg.norm(aligned - reference_pos, axis=1)
    tm_score = np.sum(1 / (1 + (distances / d_0) ** 2)) / length
    return torch.tensor(tm_score).to(device)


def to_torch_const(x):
    """Schedule constant as a frozen Parameter (so it lands in state_dict)."""
    return nn.Parameter(torch.from_numpy(np.asarray(x)).float(), requires_grad=False)


def center_pos(
    protein_pos, protein_pos_holo, ligand_pos, batch_protein, batch_ligand,
    mode="protein",
):
    """Subtract the APO pocket centroid from everything; return the offset."""
    if mode == "none":
        return protein_pos, protein_pos_holo, ligand_pos, 0.0
    if mode == "protein":
        offset = scatter_mean(protein_pos, batch_protein, dim=0)
        return (
            protein_pos - offset[batch_protein],
            protein_pos_holo - offset[batch_protein],
            ligand_pos - offset[batch_ligand],
            offset,
        )
    raise NotImplementedError(mode)


def index_to_log_onehot(x, num_classes):
    if x.max().item() >= num_classes:
        raise ValueError(f"class index {x.max().item()} >= {num_classes}")
    return torch.log(F.one_hot(x, num_classes).float().clamp(min=1e-30))


def categorical_kl(log_prob1, log_prob2):
    return (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def normal_kl(mean1, logvar1, mean2, logvar2):
    kl = 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + (mean1 - mean2) ** 2 * torch.exp(-logvar2)
    )
    return kl.sum(-1)


def log_normal(values, means, log_scales):
    var = torch.exp(log_scales * 2)
    log_prob = (
        -((values - means) ** 2) / (2 * var)
        - log_scales
        - np.log(np.sqrt(2 * np.pi))
    )
    return log_prob.sum(-1)


def log_sample_categorical(logits):
    """Gumbel-max sample; returns CLASS INDICES, not a one-hot."""
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    return (gumbel_noise + logits).argmax(dim=-1)


def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(coef, t, batch):
    return coef[t][batch].unsqueeze(-1)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ScorePosNet3D(nn.Module):
    """Joint ligand + pocket diffusion model (``molopt_score_model.py:231``)."""

    def __init__(
        self,
        protein_atom_feature_dim: int = 27,
        ligand_atom_feature_dim: int = 13,
        cond_dim: int = 128,
        topk_prompt: int = 0,
        model_mean_type: str = "C0",
        beta_schedule: str = "sigmoid",
        beta_start: float = 1.0e-7,
        beta_end: float = 2.0e-3,
        pos_beta_s: float = 0.01,
        v_beta_schedule: str = "cosine",
        v_beta_s: float = 0.01,
        lambda_schedule: str = "sigmoid",
        num_diffusion_timesteps: int = 1000,
        loss_v_weight: float = 100.0,
        loss_chi_weight: float = 5.0,
        sample_time_method: str = "symmetric",
        time_emb_dim: int = 0,
        time_emb_mode: str = "simple",
        center_pos_mode: str = "protein",
        node_indicator: bool = True,
        model_type: str = "uni_o2",
        num_blocks: int = 1,
        num_layers: int = 9,
        hidden_dim: int = 128,
        n_heads: int = 16,
        edge_feat_dim: int = 5,
        num_r_gaussian: int = 20,
        knn: int = 32,
        num_node_types: int = 8,
        act_fn: str = "relu",
        norm: bool = True,
        cutoff_mode: str = "knn",
        ew_net_type: str = "global",
        num_x2h: int = 1,
        num_h2x: int = 1,
        r_max: float = 10.0,
        x2h_out_fc: bool = False,
        sync_twoup: bool = False,
        num_protein_update_steps: int = 5,
    ) -> None:
        super().__init__()
        self.model_mean_type = model_mean_type
        self.loss_v_weight = loss_v_weight
        self.loss_chi_weight = loss_chi_weight
        self.sample_time_method = sample_time_method

        if beta_schedule == "cosine":
            alphas = cosine_beta_schedule(num_diffusion_timesteps, pos_beta_s) ** 2
            betas = 1.0 - alphas
        else:
            betas = get_beta_schedule(
                beta_schedule=beta_schedule,
                beta_start=beta_start,
                beta_end=beta_end,
                num_diffusion_timesteps=num_diffusion_timesteps,
            )
            alphas = 1.0 - betas

        if lambda_schedule != "sigmoid":
            raise NotImplementedError(
                f"lambda_schedule={lambda_schedule!r}: only 'sigmoid' is "
                "implemented upstream (molopt_score_model.py:254)."
            )
        lambdas = np.linspace(-6, 6, num_diffusion_timesteps)
        lambdas = 1 - 1 / (1 + np.exp(-lambdas))

        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.betas = to_torch_const(betas)
        self.num_timesteps = self.betas.size(0)
        self.alphas_cumprod = to_torch_const(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)
        self.lambdas = to_torch_const(lambdas)

        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(
            np.sqrt(1.0 - alphas_cumprod)
        )
        self.sqrt_recip_alphas_cumprod = to_torch_const(np.sqrt(1.0 / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_torch_const(
            np.sqrt(1.0 / alphas_cumprod - 1)
        )

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_c0_coef = to_torch_const(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_ct_coef = to_torch_const(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        )
        self.posterior_var = to_torch_const(posterior_variance)
        self.posterior_logvar = to_torch_const(
            np.log(np.append(posterior_variance[1], posterior_variance[1:]))
        )

        if v_beta_schedule != "cosine":
            raise NotImplementedError(
                f"v_beta_schedule={v_beta_schedule!r}: only 'cosine' is "
                "implemented upstream."
            )
        alphas_v = cosine_beta_schedule(self.num_timesteps, v_beta_s)
        log_alphas_v = np.log(alphas_v)
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)
        self.log_alphas_v = to_torch_const(log_alphas_v)
        self.log_one_minus_alphas_v = to_torch_const(log_1_min_a(log_alphas_v))
        self.log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)
        self.log_one_minus_alphas_cumprod_v = to_torch_const(
            log_1_min_a(log_alphas_cumprod_v)
        )

        # Present in the released checkpoint; upstream never reads it back.
        # Kept so the key set round-trips (molopt_score_model.py:289-293).
        self.custom_noise = to_torch_const(
            np.array([
                1.19691158e-04, 3.37258288e-01, 3.08534372e-01, 6.15134064e-02,
                5.82621236e-02, 1.98585290e-01, 1.93860432e-03, 9.45952575e-03,
                8.06443701e-03, 0.00000000e00, 7.97025380e-03, 2.89809573e-03,
                5.39591284e-03,
            ])
        )

        self.register_buffer("Lt_history", torch.zeros(self.num_timesteps))
        self.register_buffer("Lt_count", torch.zeros(self.num_timesteps))

        self.hidden_dim = hidden_dim
        self.num_classes = ligand_atom_feature_dim
        self.node_indicator = node_indicator
        emb_dim = hidden_dim - 1 if node_indicator else hidden_dim

        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)
        self.center_pos_mode = center_pos_mode

        self.time_emb_dim = time_emb_dim
        self.time_emb_mode = time_emb_mode
        if self.time_emb_dim > 0:
            if time_emb_mode == "simple":
                self.ligand_atom_emb = nn.Linear(
                    ligand_atom_feature_dim + 1, emb_dim
                )
            elif time_emb_mode == "sin":
                self.time_emb = nn.Sequential(
                    SinusoidalPosEmb(self.time_emb_dim),
                    nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.time_emb_dim * 4, self.time_emb_dim),
                )
                self.ligand_atom_emb = nn.Linear(
                    ligand_atom_feature_dim + self.time_emb_dim, emb_dim
                )
            else:
                raise NotImplementedError(time_emb_mode)
        else:
            self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)

        self.refine_net_type = model_type
        if model_type != "uni_o2":
            raise NotImplementedError(
                f"model_type={model_type!r}: only 'uni_o2' is ported "
                "(the 'egnn' branch is dead in the released configuration)."
            )
        self.refine_net = UniTransformerO2TwoUpdateGeneral(
            num_blocks=num_blocks,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            k=knn,
            edge_feat_dim=edge_feat_dim,
            num_r_gaussian=num_r_gaussian,
            num_node_types=num_node_types,
            act_fn=act_fn,
            norm=norm,
            cutoff_mode=cutoff_mode,
            ew_net_type=ew_net_type,
            num_x2h=num_x2h,
            num_h2x=num_h2x,
            r_max=r_max,
            x2h_out_fc=x2h_out_fc,
            sync_twoup=sync_twoup,
        )

        self.v_inference = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(hidden_dim, ligand_atom_feature_dim),
        )
        # residue_h is (hidden_dim + 3); output is 3 translation + 4 quaternion
        # + 5 chi.
        self.res_inference = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 3 + 4 + 5),
        )

        self.cond_dim = cond_dim
        self.topk_prompt = topk_prompt
        self.emb_mlp_aug = nn.Linear(emb_dim + cond_dim, cond_dim)
        self.emb_mlp = nn.Linear(emb_dim + cond_dim * 2, emb_dim)
        self.prompt_protein_mlp = nn.Linear(cond_dim * (topk_prompt + 1), cond_dim)
        self.prompt_ligand_attn = RetAugmentationLinearAttention(
            in_dim=cond_dim, d=cond_dim, context_dim=cond_dim
        )

        # The 5 reverse steps on which the pocket is actually moved
        # (molopt_score_model.py:350-352). At T = 1000: {799, 599, 399, 199, 10}.
        num = num_protein_update_steps
        last = self.num_timesteps - 1
        self.protein_update_steps = {
            int(last * (1 - i / num)) for i in range(1, num)
        } | {10}

    # ---------------------------------------------------------------- #
    # forward
    # ---------------------------------------------------------------- #
    def forward(
        self,
        protein_pos,
        protein_v,
        batch_protein,
        init_ligand_pos,
        init_ligand_v,
        batch_ligand,
        protein_atom_to_aa_group,
        time_step=None,
        return_all=False,
        fix_x=False,
        hbap_protein=None,
        hbap_ligand=None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = batch_protein.max().item() + 1
        init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()
        if self.time_emb_dim > 0:
            if self.time_emb_mode == "simple":
                input_ligand_feat = torch.cat(
                    [
                        init_ligand_v,
                        (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1),
                    ],
                    -1,
                )
            else:
                input_ligand_feat = torch.cat(
                    [init_ligand_v, self.time_emb(time_step)], -1
                )
        else:
            input_ligand_feat = init_ligand_v

        h_protein = self.protein_atom_emb(protein_v)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)

        if hbap_protein is None:
            hbap_protein = torch.zeros(
                [h_protein.shape[0], self.cond_dim], device=h_protein.device
            )
        if hbap_ligand is None:
            hbap_ligand = torch.zeros(
                [init_ligand_h.shape[0], self.cond_dim], device=init_ligand_h.device
            )

        hbap_protein_aug = self.emb_mlp_aug(
            torch.cat([h_protein, hbap_protein.detach()], dim=1)
        )
        hbap_ligand_aug = self.emb_mlp_aug(
            torch.cat([init_ligand_h, hbap_ligand.detach()], dim=1)
        )

        # topk_prompt == 0 in the release, so the retrieval list is empty and
        # prompt_protein_mlp is a plain (cond_dim -> cond_dim) projection.
        hbap_protein_aug = self.prompt_protein_mlp(
            torch.cat([hbap_protein_aug], dim=1)
        )

        hbap_ligand_aug_batch = torch.zeros(
            [batch_size, MAX_LIG_LEN, hbap_ligand_aug.shape[1]],
            device=hbap_ligand_aug.device,
        )
        valid_num_atom_list = []
        for bi in range(batch_size):
            rows = hbap_ligand_aug[batch_ligand == bi]
            num_atom = rows.shape[0]
            if num_atom > MAX_LIG_LEN:
                raise ValueError(
                    f"complex {bi} has {num_atom} ligand atoms, above the "
                    f"hard cap of {MAX_LIG_LEN} (molopt_score_model.py:389)."
                )
            valid_num_atom_list.append(num_atom)
            hbap_ligand_aug_batch[bi, :num_atom] = rows

        # Retrieval branch is dead at topk_prompt=0 -> self-attention.
        hbap_ligand_aug_batch = self.prompt_ligand_attn(
            h=hbap_ligand_aug_batch, h_retrieved=hbap_ligand_aug_batch
        )
        hbap_ligand_aug = torch.cat(
            [
                hbap_ligand_aug_batch[bi][: valid_num_atom_list[bi]]
                for bi in range(batch_size)
            ],
            dim=0,
        )

        h_protein = self.emb_mlp(
            torch.cat([h_protein, hbap_protein, hbap_protein_aug], dim=1)
        )
        init_ligand_h = self.emb_mlp(
            torch.cat([init_ligand_h, hbap_ligand, hbap_ligand_aug], dim=1)
        )

        if self.node_indicator:
            h_protein = torch.cat(
                [h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1
            )
            init_ligand_h = torch.cat(
                [init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1
            )

        outputs = self.refine_net(
            h_protein,
            init_ligand_h,
            protein_pos,
            init_ligand_pos,
            batch_protein,
            batch_ligand,
            protein_atom_to_aa_group,
            return_all=return_all,
            fix_x=fix_x,
        )

        residue_h = outputs["residue_h"]
        final_res_out = self.res_inference(residue_h)

        return {
            "pred_ligand_pos": outputs["ligand_pos"],
            "pred_ligand_v": self.v_inference(outputs["ligand_h"]),
            "pred_residue_tr": final_res_out[:, :3],
            "pred_residue_rot": final_res_out[:, 3:7],
            "pred_residue_chi": final_res_out[:, 7:],
            "final_ligand_h": outputs["ligand_h"],
            "residue_h": residue_h,
        }

    # ---------------------------------------------------------------- #
    # D3PM helpers for the atom-type channel
    # ---------------------------------------------------------------- #
    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        log_alpha_t = extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)
        return log_add_exp(
            log_vt_1 + log_alpha_t, log_1_min_alpha_t - np.log(self.num_classes)
        )

    def q_v_pred(self, log_v0, t, batch):
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)
        log_1_min_cumprod_alpha = extract(
            self.log_one_minus_alphas_cumprod_v, t, batch
        )
        return log_add_exp(
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes),
        )

    def q_v_sample(self, log_v0, t, batch):
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        sample_index = log_sample_categorical(log_qvt_v0)
        return sample_index, index_to_log_onehot(sample_index, self.num_classes)

    def q_v_posterior(self, log_v0, log_vt, t, batch):
        t_minus_1 = torch.clamp(t - 1, min=0)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        unnormed = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)
        return unnormed - torch.logsumexp(unnormed, dim=-1, keepdim=True)

    def _predict_x0_from_eps(self, xt, eps, t, batch):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, batch) * xt
            - extract(self.sqrt_recipm1_alphas_cumprod, t, batch) * eps
        )

    def q_pos_posterior(self, x0, xt, t, batch):
        return (
            extract(self.posterior_mean_c0_coef, t, batch) * x0
            + extract(self.posterior_mean_ct_coef, t, batch) * xt
        )

    def sample_time(self, num_graphs, device, method):
        if method == "importance":
            if not (self.Lt_count > 10).all():
                return self.sample_time(num_graphs, device, method="symmetric")
            lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            lt_sqrt[0] = lt_sqrt[1]
            pt_all = lt_sqrt / lt_sqrt.sum()
            time_step = torch.multinomial(
                pt_all, num_samples=num_graphs, replacement=True
            )
            return time_step, pt_all.gather(dim=0, index=time_step)
        if method == "symmetric":
            time_step = torch.randint(
                0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device
            )
            time_step = torch.cat(
                [time_step, self.num_timesteps - time_step - 1], dim=0
            )[:num_graphs]
            return time_step, torch.ones_like(time_step).float() / self.num_timesteps
        raise ValueError(method)

    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t, batch):  # noqa: N802
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)
        mask = (t == 0).float()[batch]
        return scatter_mean(
            mask * decoder_nll_v + (1.0 - mask) * kl_v, batch, dim=0
        )

    # ---------------------------------------------------------------- #
    # pocket channel
    # ---------------------------------------------------------------- #
    def add_noise_to_quaternion(self, q, noise_scale):
        """Compose ``q`` with a small random rotation of scale ``noise_scale``."""
        axis = torch.randn_like(q[..., 1:])
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        angle = torch.randn(q.shape[0], device=q.device) * noise_scale.squeeze(-1)
        delta_q = axis_angle_to_quaternion(axis * angle.unsqueeze(-1))
        q_noisy = quaternion_product(q, delta_q)
        return q_noisy / q_noisy.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def calculate_quat_loss(self, pred_rot, target_rot, batch):
        """Norm penalty + L2 on the normalised quaternion (``:794-810``)."""
        scale_loss = abs(1.0 - pred_rot.norm(dim=-1, keepdim=True))
        scale_loss = torch.mean(scatter_mean(scale_loss, batch, dim=0))
        pred_rot = pred_rot / pred_rot.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        quat_loss = ((pred_rot - target_rot) ** 2).sum(-1)
        quat_loss = torch.mean(scatter_mean(quat_loss, batch, dim=0)) * 10
        return scale_loss + quat_loss, pred_rot

    # ---------------------------------------------------------------- #
    # training objective
    # ---------------------------------------------------------------- #
    def get_diffusion_loss(
        self,
        net_cond,
        data,
        protein_pos_apo,
        protein_pos_holo,
        protein_v,
        batch_protein,
        ligand_pos,
        ligand_v,
        batch_ligand,
        time_step=None,
    ) -> Dict[str, Any]:
        """One training step's losses (``molopt_score_model.py:614-792``).

        ``data`` is any object exposing the per-residue fields as attributes
        (``protein_translations``, ``protein_rotations``, ``protein_chi_apo``
        / ``_holo`` / ``_mask``, ``protein_translations_batch``,
        ``protein_atom_name``, ``protein_atom_to_aa_name``,
        ``protein_atom_to_aa_group``, ``protein_element_batch``); the task
        wraps the collated dict in a ``SimpleNamespace`` for exactly this.

        Note **the diffusion target for the pocket is HOLO**: the apo pocket
        only supplies the centering offset.
        """
        num_graphs = batch_protein.max().item() + 1
        protein_pos_apo, protein_pos_holo, ligand_pos, offset = center_pos(
            protein_pos_apo, protein_pos_holo, ligand_pos,
            batch_protein, batch_ligand, mode=self.center_pos_mode,
        )

        if time_step is None:
            time_step, _pt = self.sample_time(
                num_graphs, protein_pos_holo.device, self.sample_time_method
            )

        # --- ligand coordinates: Gaussian forward process ---
        a = self.alphas_cumprod.index_select(0, time_step)
        a_pos = a[batch_ligand].unsqueeze(-1)
        pos_noise = torch.randn_like(ligand_pos)
        ligand_pos_perturbed = (
            a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise
        )

        # --- ligand types: D3PM forward process ---
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(
            log_ligand_v0, time_step, batch_ligand
        )

        # --- pocket: interpolate holo -> apo, then jitter ---
        prot_update_batch = data.protein_translations_batch
        beta_update = self.betas.index_select(0, time_step)
        beta_update = beta_update[prot_update_batch].unsqueeze(-1)
        lambdas = self.lambdas.index_select(0, time_step)
        lambdas_update = lambdas[prot_update_batch].unsqueeze(-1)

        prot_tr = data.protein_translations
        prot_rot = data.protein_rotations
        prot_chi_update = data.protein_chi_apo - data.protein_chi_holo
        prot_chi_mask = data.protein_chi_mask

        # The stored translation is in the ORIGINAL frame; re-express it for
        # the pocket-centred frame the model works in.
        prot_rot_mat = quaternion_to_rotation_matrix(prot_rot)
        prot_tr = (
            torch.matmul(prot_rot_mat, offset[prot_update_batch].unsqueeze(-1))
            .squeeze(-1)
            - offset[prot_update_batch]
            + prot_tr
        )
        prot_tr_t = (1 - lambdas_update) * prot_tr
        prot_rot_t = slerp_identity_to_q(prot_rot, lambdas_update)
        prot_chi_t = (1 - lambdas_update) * prot_chi_update
        prot_tr_t = prot_tr_t + torch.randn_like(prot_tr_t) * beta_update.sqrt() * 3
        prot_rot_t = self.add_noise_to_quaternion(prot_rot_t, beta_update.sqrt() * 2)
        prot_chi_t = prot_chi_t + torch.randn_like(prot_chi_t) * beta_update.sqrt()
        prot_chi_t = prot_chi_t * prot_chi_mask

        protein_pos_perturbed = apply_transforms_tensor_batch(
            protein_pos=protein_pos_holo,
            protein_atom_name=data.protein_atom_name,
            protein_atom_to_aa_name=data.protein_atom_to_aa_name,
            protein_atom_to_aa_group=data.protein_atom_to_aa_group,
            protein_element_batch=data.protein_element_batch,
            rotations=prot_rot_t,
            translations=prot_tr_t,
            chi_update=prot_chi_t,
            chi_mask=prot_chi_mask,
            protein_translations_batch=prot_update_batch,
        )

        # --- the frozen interaction prior, evaluated on the CLEAN complex ---
        gt_protein_a_h = torch.argmax(protein_v[:, :6], dim=1)
        gt_protein_r_h = torch.argmax(protein_v[:, 6:26], dim=1)
        if self.model_mean_type != "C0":
            raise NotImplementedError(
                f"model_mean_type={self.model_mean_type!r}: the released "
                "configuration is 'C0'."
            )
        hbap_ligand, hbap_protein = net_cond.extract_features(
            ligand_pos, protein_pos_holo, ligand_v,
            gt_protein_a_h, gt_protein_r_h, batch_ligand, batch_protein,
        )

        preds = self.forward(
            protein_pos=protein_pos_perturbed,
            protein_v=protein_v,
            batch_protein=batch_protein,
            init_ligand_pos=ligand_pos_perturbed,
            init_ligand_v=ligand_v_perturbed,
            batch_ligand=batch_ligand,
            protein_atom_to_aa_group=data.protein_atom_to_aa_group,
            time_step=time_step,
            hbap_protein=hbap_protein,
            hbap_ligand=hbap_ligand,
        )

        pred_ligand_pos = preds["pred_ligand_pos"]
        pred_ligand_v = preds["pred_ligand_v"]
        pred_res_tr = preds["pred_residue_tr"]
        pred_res_rot = preds["pred_residue_rot"]
        pred_res_chi = preds["pred_residue_chi"]

        # --- ligand losses ---
        loss_ligand_pos = torch.mean(
            scatter_mean(
                ((pred_ligand_pos - ligand_pos) ** 2).sum(-1), batch_ligand, dim=0
            )
        )
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(
            log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand
        )
        log_v_true_prob = self.q_v_posterior(
            log_ligand_v0, log_ligand_vt, time_step, batch_ligand
        )
        loss_v = torch.mean(
            self.compute_v_Lt(
                log_v_model_prob=log_v_model_prob,
                log_v0=log_ligand_v0,
                log_v_true_prob=log_v_true_prob,
                t=time_step,
                batch=batch_ligand,
            )
        )

        # --- pocket losses: the target is the INVERSE of the applied noise ---
        inverse_rot = torch.cat(
            [prot_rot_t[:, :1], -prot_rot_t[:, 1:]], dim=-1
        )  # quaternion conjugate
        prot_rot_mat_inv_t = quaternion_to_rotation_matrix(prot_rot_t).transpose(
            -2, -1
        )
        inverse_tr = -torch.matmul(
            prot_rot_mat_inv_t, prot_tr_t.unsqueeze(-1)
        ).squeeze(-1)
        inverse_chi = -prot_chi_t

        loss_prot_tr = torch.mean(
            scatter_mean(
                nn.L1Loss(reduction="none")(pred_res_tr, inverse_tr).sum(-1),
                prot_update_batch,
                dim=0,
            )
        )
        loss_prot_rot, pred_res_rot = self.calculate_quat_loss(
            pred_res_rot, inverse_rot, prot_update_batch
        )
        chi_loss = 1 - (pred_res_chi - inverse_chi).cos()
        chi_loss = (chi_loss * prot_chi_mask).sum(dim=-1) / (
            prot_chi_mask.sum(dim=-1) + 1e-12
        )
        loss_prot_chi = torch.mean(
            scatter_mean(chi_loss, prot_update_batch, dim=0)
        )

        loss = (
            loss_ligand_pos
            + loss_v * self.loss_v_weight
            + loss_prot_tr
            + loss_prot_rot
            + self.loss_chi_weight * loss_prot_chi
        )

        return {
            "loss": loss,
            "loss_ligand_pos": loss_ligand_pos,
            "loss_v": loss_v,
            "loss_protein_tr": loss_prot_tr,
            "loss_protein_rot": loss_prot_rot,
            "loss_protein_chi": loss_prot_chi,
            "x0": ligand_pos,
            "p0": protein_pos_holo,
            "pred_ligand_pos": pred_ligand_pos,
            "pred_ligand_v": pred_ligand_v,
            "perturbed_protein_pos": protein_pos_perturbed,
            "protein_pos_apo": protein_pos_apo,
            "pred_res_tr": pred_res_tr,
            "pred_res_rot": pred_res_rot,
            "pred_res_chi": pred_res_chi,
        }

    # ---------------------------------------------------------------- #
    # sampling
    # ---------------------------------------------------------------- #
    @torch.no_grad()
    def sample_diffusion(
        self,
        data,
        protein_pos_apo,
        protein_pos_holo,
        protein_v,
        batch_protein,
        init_ligand_pos,
        init_ligand_v,
        batch_ligand,
        num_steps: Optional[int] = None,
        center_pos_mode: Optional[str] = None,
        net_cond=None,
        progress: bool = True,
    ) -> Dict[str, Any]:
        """Reverse process (``molopt_score_model.py:812-960``).

        Everything comes back in the **input pocket's frame** -- the pocket
        centroid offset is added back before returning.

        In pocket-only generation the caller sets ``protein_pos_holo =
        protein_pos_apo``, so the returned ``protein_pos_rmsd`` /
        ``protein_pos_tmscore`` measure displacement from the input apo
        structure, NOT accuracy against a true holo structure.
        """
        if net_cond is None:
            raise ValueError(
                "net_cond (the PMINet prior) is required: it is re-evaluated "
                "every reverse step."
            )
        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1

        protein_pos_apo, protein_pos_holo, init_ligand_pos, offset = center_pos(
            protein_pos_apo, protein_pos_holo, init_ligand_pos,
            batch_protein, batch_ligand, mode=center_pos_mode,
        )
        protein_pos = protein_pos_apo

        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v
        gt_protein_a_h = torch.argmax(protein_v[:, :6], dim=1)
        gt_protein_r_h = torch.argmax(protein_v[:, 6:26], dim=1)

        hbap_protein = hbap_ligand = None
        time_seq = list(
            reversed(range(self.num_timesteps - num_steps, self.num_timesteps))
        )

        for i in tqdm(
            time_seq, desc="sampling", total=len(time_seq), disable=not progress
        ):
            t = torch.full(
                size=(num_graphs,), fill_value=i, dtype=torch.long,
                device=protein_pos_apo.device,
            )
            preds = self.forward(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                init_ligand_pos=ligand_pos,
                init_ligand_v=ligand_v,
                batch_ligand=batch_ligand,
                protein_atom_to_aa_group=data.protein_atom_to_aa_group,
                time_step=t,
                hbap_protein=hbap_protein,
                hbap_ligand=hbap_ligand,
            )
            ligand_pos0_from_e = preds["pred_ligand_pos"]
            v0_from_e = preds["pred_ligand_v"]

            pred_ligand_pos = ligand_pos0_from_e.detach()
            pred_lig_a_h = torch.argmax(v0_from_e.detach(), dim=1)
            pred_residue_tr = preds["pred_residue_tr"].detach()
            pred_residue_rot = preds["pred_residue_rot"].detach()
            pred_residue_rot = pred_residue_rot / pred_residue_rot.norm(
                dim=-1, keepdim=True
            )
            pred_residue_chi = preds["pred_residue_chi"].detach()

            ligand_pos_model_mean = self.q_pos_posterior(
                x0=ligand_pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand
            )
            ligand_pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
            ligand_pos = ligand_pos_model_mean + nonzero_mask * (
                0.5 * ligand_pos_log_variance
            ).exp() * torch.randn_like(ligand_pos)

            if int(t[0]) in self.protein_update_steps:
                prot_update_batch = data.protein_translations_batch
                beta_update = self.betas.index_select(0, t)
                beta_update = beta_update[prot_update_batch].unsqueeze(-1)
                step_scale = 1 / t[0] + 1
                residue_tr_t = (
                    step_scale * pred_residue_tr
                    + torch.randn_like(pred_residue_tr) * beta_update.sqrt() * 3
                )
                residue_chi_t = (
                    step_scale * pred_residue_chi
                    + torch.randn_like(pred_residue_chi) * beta_update.sqrt()
                )
                residue_rot_t = slerp_identity_to_q(
                    pred_residue_rot, (1 / t + 1)[prot_update_batch].unsqueeze(-1)
                )
                residue_rot_t = self.add_noise_to_quaternion(
                    residue_rot_t, beta_update.sqrt() * 2
                )
                pred_protein_pos = apply_transforms_tensor_batch(
                    protein_pos=protein_pos,
                    protein_atom_name=data.protein_atom_name,
                    protein_atom_to_aa_name=data.protein_atom_to_aa_name,
                    protein_atom_to_aa_group=data.protein_atom_to_aa_group,
                    protein_element_batch=data.protein_element_batch,
                    rotations=residue_rot_t,
                    translations=residue_tr_t,
                    chi_update=residue_chi_t,
                    chi_mask=data.protein_chi_mask,
                    protein_translations_batch=prot_update_batch,
                )
                pred_protein_pos = (
                    pred_protein_pos
                    - scatter_mean(pred_protein_pos, batch_protein, dim=0)[
                        batch_protein
                    ]
                )
                protein_pos = pred_protein_pos
            else:
                pred_protein_pos = protein_pos

            log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
            log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes)
            log_model_prob = self.q_v_posterior(
                log_ligand_v_recon, log_ligand_v, t, batch_ligand
            )
            ligand_v = log_sample_categorical(log_model_prob)

            hbap_ligand, hbap_protein = net_cond.extract_features(
                pred_ligand_pos, pred_protein_pos, pred_lig_a_h,
                gt_protein_a_h, gt_protein_r_h, batch_ligand, batch_protein,
            )
            hbap_ligand, hbap_protein = hbap_ligand.detach(), hbap_protein.detach()

        ligand_pos = ligand_pos + offset[batch_ligand]
        protein_pos = protein_pos + offset[batch_protein]
        protein_pos_holo = protein_pos_holo + offset[batch_protein]

        rmsds, tmscores = [], []
        for i in range(int(batch_protein.max()) + 1):
            mask = batch_protein == i
            pred_i, ref_i = protein_pos[mask], protein_pos_holo[mask]
            rmsds.append(torch.sqrt(((pred_i - ref_i) ** 2).sum(-1)).mean())
            try:
                tmscores.append(calculate_tm_score(pred_i, ref_i))
            except ImportError:
                logger.warning(
                    "biopython is not installed; skipping the TM-score "
                    "diagnostic. Install the [bio] extra to enable it."
                )
                break

        return {
            "ligand_pos": ligand_pos,
            "protein_pos": protein_pos,
            "protein_pos_rmsd": rmsds,
            "protein_pos_tmscore": tmscores,
            "v": ligand_v,
        }
