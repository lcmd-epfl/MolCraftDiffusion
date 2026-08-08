"""KGDiff's score network: pocket-conditioned hybrid diffusion + value head.

Ported from KGDiff ``models/molopt_score_model.py`` (commit ``ad893fc``).
KGDiff is TargetDiff plus a per-atom affinity ("value") head that is used as
its own classifier guide at sampling time.

Three coupled channels:

* **Coordinates** -- continuous Gaussian DDPM, ``sigmoid`` beta schedule,
  ``model_mean_type='C0'`` (the net predicts x0). Positions are centred on
  the *pocket* centroid, not zero-CoM.
* **Atom types** -- D3PM-style multinomial diffusion in log space over a
  13-class ``(element, is_aromatic)`` vocabulary, cosine schedule.
* **Affinity** -- MSE against the normalised Vina score.

Loss is ``loss_pos + 100 * loss_v + 1 * loss_exp``.

Deliberately **not** ported (see the integration plan's scope list): the
``vina`` guide mode (needs AutoDockTools/vina at sampling time), the
``valuenet*`` / ``target_diff`` guide modes (each needs a *second*
pretrained checkpoint), the PDBBind ``pdbbind_random`` mode, and the
``calc_atom_dis`` debugging routine. What remains is ``guide_mode='joint'``
(the paper's headline, self-guided from one checkpoint) and ``'wo'`` (the
unguided ablation).

Upstream took a single ``EasyDict`` config object; this takes explicit
keyword arguments so the Hydra task config is the single source of truth and
no third-party config class is needed to unpickle anything. Every parameter
and buffer name is unchanged, so the released checkpoint maps across with a
pure prefix add.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch_scatter import scatter_mean
from tqdm.auto import tqdm

from MolecularDiffusion.modules.models.kgdiff.common import (
    ShiftedSoftplus,
    compose_context,
)
from MolecularDiffusion.modules.models.kgdiff.uni_transformer import (
    UniTransformerO2TwoUpdateGeneral,
)

#: Guide modes this port supports. See the module docstring for why the
#: others are out of scope.
GUIDE_MODES = ("joint", "wo")


def get_beta_schedule(
    beta_schedule: str,
    *,
    beta_start: float,
    beta_end: float,
    num_diffusion_timesteps: int,
) -> np.ndarray:
    """Position-channel beta schedule (upstream supports several)."""
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
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
        x = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = (1 / (np.exp(-x) + 1)) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> np.ndarray:
    """Cosine alpha schedule, sqrt-ed (the atom-type channel uses this)."""
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    alphas = np.clip(alphas, a_min=0.001, a_max=1.0)
    return np.sqrt(alphas)


def to_torch_const(x: np.ndarray) -> nn.Parameter:
    """Frozen ``nn.Parameter`` -- upstream's way of pinning schedule tables.

    Kept as a ``Parameter`` (not a buffer) purely for state-dict key parity
    with the released checkpoint.
    """
    return nn.Parameter(torch.from_numpy(x).float(), requires_grad=False)


def center_pos(
    protein_pos, ligand_pos, batch_protein, batch_ligand, mode="protein"
):
    """Shift both clouds so the pocket centroid sits at the origin."""
    if mode == "none":
        return protein_pos, ligand_pos, 0.0
    if mode != "protein":
        raise NotImplementedError(mode)
    offset = scatter_mean(protein_pos, batch_protein, dim=0)
    return (
        protein_pos - offset[batch_protein],
        ligand_pos - offset[batch_ligand],
        offset,
    )


def index_to_log_onehot(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    assert x.max().item() < num_classes, f"{x.max().item()} >= {num_classes}"
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


def log_sample_categorical(logits: torch.Tensor) -> torch.Tensor:
    """Gumbel-perturbed logits; ``argmax`` of the result is a sample."""
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    return gumbel_noise + logits


def log_1_min_a(a: np.ndarray) -> np.ndarray:
    return np.log(1 - np.exp(a) + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(coef, t, batch):
    return coef[t][batch].unsqueeze(-1)


class ScorePosNet3D(nn.Module):
    """Pocket-conditioned hybrid diffusion model with a value head."""

    def __init__(
        self,
        protein_atom_feature_dim: int = 27,
        ligand_atom_feature_dim: int = 13,
        model_mean_type: str = "C0",
        beta_schedule: str = "sigmoid",
        beta_start: float = 1.0e-7,
        beta_end: float = 2.0e-3,
        pos_beta_s: float = 0.01,
        v_beta_schedule: str = "cosine",
        v_beta_s: float = 0.01,
        num_diffusion_timesteps: int = 1000,
        loss_v_weight: float = 100.0,
        loss_exp_weight: float = 1.0,
        sample_time_method: str = "symmetric",
        use_classifier_guide: bool = True,
        time_emb_dim: int = 0,
        time_emb_mode: str = "simple",
        center_pos_mode: str = "protein",
        node_indicator: bool = True,
        model_type: str = "uni_o2",
        num_blocks: int = 1,
        num_layers: int = 9,
        hidden_dim: int = 128,
        n_heads: int = 16,
        edge_feat_dim: int = 4,
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
        pred_exp_from_all: bool = False,
    ) -> None:
        super().__init__()
        self.model_mean_type = model_mean_type
        self.loss_v_weight = loss_v_weight
        self.loss_exp_weight = loss_exp_weight
        self.sample_time_method = sample_time_method
        self.use_classifier_guide = use_classifier_guide
        self.pred_exp_from_all = pred_exp_from_all
        self.node_indicator = node_indicator

        # --- position channel schedule ---
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
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.betas = to_torch_const(betas)
        self.num_timesteps = self.betas.size(0)
        self.alphas_cumprod = to_torch_const(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)
        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(
            np.sqrt(1.0 - alphas_cumprod)
        )
        self.sqrt_recip_alphas_cumprod = to_torch_const(
            np.sqrt(1.0 / alphas_cumprod)
        )
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

        # --- atom-type channel schedule (log space) ---
        if v_beta_schedule != "cosine":
            raise NotImplementedError(v_beta_schedule)
        alphas_v = cosine_beta_schedule(self.num_timesteps, v_beta_s)
        log_alphas_v = np.log(alphas_v)
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)
        self.log_alphas_v = to_torch_const(log_alphas_v)
        self.log_one_minus_alphas_v = to_torch_const(log_1_min_a(log_alphas_v))
        self.log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)
        self.log_one_minus_alphas_cumprod_v = to_torch_const(
            log_1_min_a(log_alphas_cumprod_v)
        )

        self.register_buffer("Lt_history", torch.zeros(self.num_timesteps))
        self.register_buffer("Lt_count", torch.zeros(self.num_timesteps))

        # --- network ---
        self.hidden_dim = hidden_dim
        self.num_classes = ligand_atom_feature_dim
        emb_dim = hidden_dim - 1 if node_indicator else hidden_dim

        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)
        self.center_pos_mode = center_pos_mode

        self.time_emb_dim = time_emb_dim
        self.time_emb_mode = time_emb_mode
        if self.time_emb_dim > 0:
            if self.time_emb_mode != "simple":
                raise NotImplementedError(
                    "Only time_emb_mode='simple' is ported; the released "
                    "checkpoint uses time_emb_dim=0 (no time embedding)."
                )
            self.ligand_atom_emb = nn.Linear(
                ligand_atom_feature_dim + 1, emb_dim
            )
        else:
            self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)

        if model_type != "uni_o2":
            raise ValueError(f"Only model_type='uni_o2' is ported: {model_type!r}")
        self.refine_net_type = model_type
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
        # the value head: per-atom affinity in [0, 1], mean-pooled per ligand
        self.expert_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def _embed(self, protein_v, init_ligand_v, time_step, batch_ligand):
        if init_ligand_v.dim() == 1:
            init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()
        elif init_ligand_v.dim() != 2:
            raise ValueError(f"init_ligand_v has rank {init_ligand_v.dim()}")

        if self.time_emb_dim > 0:
            input_ligand_feat = torch.cat(
                [
                    init_ligand_v,
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1),
                ],
                -1,
            )
        else:
            input_ligand_feat = init_ligand_v

        h_protein = self.protein_atom_emb(protein_v)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)

        if self.node_indicator:
            h_protein = torch.cat(
                [h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1
            )
            init_ligand_h = torch.cat(
                [init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)],
                -1,
            )
        return h_protein, init_ligand_h

    def _heads(self, final_h, final_ligand_h, batch_all, batch_ligand):
        final_ligand_v = self.v_inference(final_ligand_h)
        if self.pred_exp_from_all:
            atom_affinity = self.expert_pred(final_h).squeeze(-1)
            final_exp_pred = scatter_mean(atom_affinity, batch_all)
        else:
            atom_affinity = self.expert_pred(final_ligand_h).squeeze(-1)
            final_exp_pred = scatter_mean(atom_affinity, batch_ligand)
        return final_ligand_v, atom_affinity, final_exp_pred

    def forward(
        self,
        protein_pos,
        protein_v,
        batch_protein,
        init_ligand_pos,
        init_ligand_v,
        batch_ligand,
        time_step=None,
        return_all=False,
        fix_x=False,
    ) -> dict:
        h_protein, init_ligand_h = self._embed(
            protein_v, init_ligand_v, time_step, batch_ligand
        )
        h_all, pos_all, batch_all, mask_ligand = compose_context(
            h_protein=h_protein,
            h_ligand=init_ligand_h,
            pos_protein=protein_pos,
            pos_ligand=init_ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        outputs = self.refine_net(
            h_all, pos_all, mask_ligand, batch_all,
            return_all=return_all, fix_x=fix_x,
        )
        final_pos, final_h = outputs["x"], outputs["h"]
        final_ligand_pos = final_pos[mask_ligand]
        final_ligand_h = final_h[mask_ligand]
        final_ligand_v, atom_affinity, final_exp_pred = self._heads(
            final_h, final_ligand_h, batch_all, batch_ligand
        )

        preds = {
            "pred_ligand_pos": final_ligand_pos,
            "pred_ligand_v": final_ligand_v,
            "final_h": final_h,
            "final_ligand_h": final_ligand_h,
            "atom_affinity": atom_affinity,
            "final_exp_pred": final_exp_pred,
            "batch_all": batch_all,
            "mask_ligand": mask_ligand,
        }
        if return_all:
            preds.update(
                {
                    "layer_pred_ligand_pos": [
                        pos[mask_ligand] for pos in outputs["all_x"]
                    ],
                    "layer_pred_ligand_v": [
                        self.v_inference(h[mask_ligand]) for h in outputs["all_h"]
                    ],
                }
            )
        return preds

    # ------------------------------------------------------------------ #
    # atom-type diffusion process
    # ------------------------------------------------------------------ #
    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        """``q(v_t | v_{t-1})``."""
        log_alpha_t = extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)
        return log_add_exp(
            log_vt_1 + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes),
        )

    def q_v_pred(self, log_v0, t, batch):
        """``q(v_t | v_0)``."""
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
        sample_index = log_sample_categorical(log_qvt_v0).argmax(dim=-1)
        return sample_index, index_to_log_onehot(sample_index, self.num_classes)

    def q_v_posterior(self, log_v0, log_vt, t, batch):
        """``q(v_{t-1} | v_t, v_0)``."""
        t_minus_1 = torch.where(t - 1 < 0, torch.zeros_like(t), t - 1)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        unnormed = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)
        return unnormed - torch.logsumexp(unnormed, dim=-1, keepdim=True)

    # ------------------------------------------------------------------ #
    # position diffusion process
    # ------------------------------------------------------------------ #
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
                return self.sample_time(num_graphs, device, "symmetric")
            lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            lt_sqrt[0] = lt_sqrt[1]  # overwrite decoder term with L1
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
            pt = torch.ones_like(time_step).float() / self.num_timesteps
            return time_step, pt
        raise ValueError(method)

    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t, batch):  # noqa: N802
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)
        assert kl_v.shape == decoder_nll_v.shape
        mask = (t == 0).float()[batch]
        return scatter_mean(
            mask * decoder_nll_v + (1.0 - mask) * kl_v, batch, dim=0
        )

    # ------------------------------------------------------------------ #
    # training objective
    # ------------------------------------------------------------------ #
    def get_diffusion_loss(
        self,
        protein_pos,
        protein_v,
        affinity,
        batch_protein,
        ligand_pos,
        ligand_v,
        batch_ligand,
        time_step=None,
    ) -> dict:
        """``loss_pos + loss_v_weight * loss_v + loss_exp_weight * loss_exp``."""
        num_graphs = batch_protein.max().item() + 1
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand,
            mode=self.center_pos_mode,
        )

        if time_step is None:
            time_step, _pt = self.sample_time(
                num_graphs, protein_pos.device, self.sample_time_method
            )
        a = self.alphas_cumprod.index_select(0, time_step)

        # perturb positions:  x_t = sqrt(a) x_0 + sqrt(1-a) eps
        a_pos = a[batch_ligand].unsqueeze(-1)
        pos_noise = torch.randn_like(ligand_pos)
        ligand_pos_perturbed = (
            a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise
        )
        # perturb atom types:  v_t = a v_0 + (1-a)/K
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(
            log_ligand_v0, time_step, batch_ligand
        )

        preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            init_ligand_pos=ligand_pos_perturbed,
            init_ligand_v=ligand_v_perturbed,
            batch_ligand=batch_ligand,
            time_step=time_step,
        )
        pred_ligand_pos = preds["pred_ligand_pos"]
        pred_ligand_v = preds["pred_ligand_v"]
        pred_pos_noise = pred_ligand_pos - ligand_pos_perturbed

        if self.model_mean_type == "C0":
            target, pred = ligand_pos, pred_ligand_pos
        elif self.model_mean_type == "noise":
            target, pred = pos_noise, pred_pos_noise
        else:
            raise ValueError(self.model_mean_type)
        loss_pos = scatter_mean(
            ((pred - target) ** 2).sum(-1), batch_ligand, dim=0
        ).mean()

        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(
            log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand
        )
        log_v_true_prob = self.q_v_posterior(
            log_ligand_v0, log_ligand_vt, time_step, batch_ligand
        )
        loss_v = self.compute_v_Lt(
            log_v_model_prob=log_v_model_prob,
            log_v0=log_ligand_v0,
            log_v_true_prob=log_v_true_prob,
            t=time_step,
            batch=batch_ligand,
        ).mean()

        loss_exp = F.mse_loss(preds["final_exp_pred"], affinity)

        loss = loss_pos + loss_v * self.loss_v_weight
        if self.use_classifier_guide:
            loss = loss + loss_exp * self.loss_exp_weight

        return {
            "loss": loss,
            "loss_pos": loss_pos,
            "loss_v": loss_v,
            "loss_exp": loss_exp,
            "pred_ligand_pos": pred_ligand_pos,
            "pred_ligand_v": pred_ligand_v,
            "pred_exp": preds["final_exp_pred"],
        }

    # ------------------------------------------------------------------ #
    # self-guided sampling
    # ------------------------------------------------------------------ #
    def pv_joint_guide(
        self,
        ligand_v_index,
        ligand_pos,
        protein_v,
        protein_pos,
        batch_protein,
        batch_ligand,
    ):
        """One denoiser pass that also returns d(affinity)/d(type, pos).

        This is KGDiff's headline trick: the value head is differentiated
        w.r.t. the atom-type one-hot and the coordinates of the *same*
        network, so guidance needs no second model. Note there is no time
        embedding here -- the released checkpoint has ``time_emb_dim=0``,
        so ``t`` never enters the network.
        """
        with torch.enable_grad():
            ligand_v = (
                F.one_hot(ligand_v_index, self.num_classes)
                .float()
                .detach()
                .requires_grad_(True)
            )
            ligand_pos = ligand_pos.detach().requires_grad_(True)

            init_h_protein = self.protein_atom_emb(protein_v)
            init_ligand_h = self.ligand_atom_emb(ligand_v)
            h_protein = torch.cat(
                [
                    init_h_protein,
                    torch.zeros(len(init_h_protein), 1).to(init_h_protein),
                ],
                -1,
            )
            ligand_h = torch.cat(
                [
                    init_ligand_h,
                    torch.ones(len(init_ligand_h), 1).to(init_ligand_h),
                ],
                -1,
            )

            h_all, pos_all, batch_all, mask_ligand = compose_context(
                h_protein=h_protein,
                h_ligand=ligand_h,
                pos_protein=protein_pos,
                pos_ligand=ligand_pos,
                batch_protein=batch_protein,
                batch_ligand=batch_ligand,
            )

            outputs = self.refine_net(h_all, pos_all, mask_ligand, batch_all)
            final_pos, final_h = outputs["x"], outputs["h"]
            final_ligand_pos = final_pos[mask_ligand]
            final_ligand_h = final_h[mask_ligand]

            if self.pred_exp_from_all:
                atom_affinity = self.expert_pred(final_h).squeeze(-1)
                pred_affinity = scatter_mean(atom_affinity, batch_all)
            else:
                atom_affinity = self.expert_pred(final_ligand_h).squeeze(-1)
                pred_affinity = scatter_mean(atom_affinity, batch_ligand)

            ones = torch.ones_like(pred_affinity)
            type_grad = torch.autograd.grad(
                pred_affinity, ligand_v, grad_outputs=ones, retain_graph=True
            )[0]
            # position guidance uses log(affinity): upstream's choice, it
            # rescales the step by 1/affinity so weak binders move further.
            pos_grad = torch.autograd.grad(
                pred_affinity.log(),
                ligand_pos,
                grad_outputs=ones,
                retain_graph=True,
            )[0]

        final_ligand_v = self.v_inference(final_ligand_h)
        preds = {
            "pred_ligand_pos": final_ligand_pos,
            "pred_ligand_v": final_ligand_v,
            "atom_affinity": atom_affinity,
            "final_h": final_h,
            "final_ligand_h": final_ligand_h,
            "final_exp_pred": pred_affinity,
            "batch_all": batch_all,
            "mask_ligand": mask_ligand,
        }
        return preds, type_grad, pos_grad

    def sample_diffusion(
        self,
        guide_mode: str,
        type_grad_weight: float,
        pos_grad_weight: float,
        protein_pos,
        protein_v,
        batch_protein,
        init_ligand_pos,
        init_ligand_v,
        batch_ligand,
        num_steps: Optional[int] = None,
        center_pos_mode: Optional[str] = None,
        progress: bool = True,
        **_ignored: Any,
    ) -> dict:
        """Reverse process. Returns flat ``pos``/``v`` in the INPUT frame.

        ``guide_mode='joint'`` is the self-guided KGDiff sampler;
        ``'wo'`` is the same loop with guidance switched off.
        """
        if guide_mode not in GUIDE_MODES:
            raise ValueError(
                f"guide_mode={guide_mode!r} is not supported by this "
                f"integration; ported modes are {GUIDE_MODES}. The "
                "valuenet*/target_diff modes need a second pretrained "
                "checkpoint and 'vina' needs AutoDockTools -- see the "
                "integration plan's scope list."
            )
        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1

        protein_pos, init_ligand_pos, offset = center_pos(
            protein_pos,
            init_ligand_pos,
            batch_protein,
            batch_ligand,
            mode=center_pos_mode or self.center_pos_mode,
        )

        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v
        exp_pred = None
        time_seq = list(
            reversed(range(self.num_timesteps - num_steps, self.num_timesteps))
        )
        iterator = (
            tqdm(time_seq, desc="sampling", total=len(time_seq))
            if progress
            else time_seq
        )

        for i in iterator:
            t = torch.full(
                size=(num_graphs,),
                fill_value=i,
                dtype=torch.long,
                device=protein_pos.device,
            )

            if guide_mode == "joint":
                preds, type_grad, pos_grad = self.pv_joint_guide(
                    ligand_v, ligand_pos, protein_v, protein_pos,
                    batch_protein, batch_ligand,
                )
            else:
                with torch.no_grad():
                    preds = self(
                        protein_pos=protein_pos,
                        protein_v=protein_v,
                        batch_protein=batch_protein,
                        init_ligand_pos=ligand_pos,
                        init_ligand_v=ligand_v,
                        batch_ligand=batch_ligand,
                        time_step=t,
                    )

            if self.model_mean_type == "noise":
                pred_pos_noise = preds["pred_ligand_pos"] - ligand_pos
                pos0_from_e = self._predict_x0_from_eps(
                    xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand
                )
            elif self.model_mean_type == "C0":
                pos0_from_e = preds["pred_ligand_pos"]
            else:
                raise ValueError(self.model_mean_type)
            v0_from_e = preds["pred_ligand_v"]

            pos_model_mean = self.q_pos_posterior(
                x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand
            )
            pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
            log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
            log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes)
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)

            if guide_mode == "joint":
                exp_pred = preds["final_exp_pred"]
                pos_model_mean = pos_model_mean + pos_grad_weight * (
                    0.5 * pos_log_variance
                ).exp() * pos_grad
                log_ligand_v = log_ligand_v + type_grad_weight * type_grad

            with torch.no_grad():
                ligand_pos = pos_model_mean + nonzero_mask * (
                    0.5 * pos_log_variance
                ).exp() * torch.randn_like(ligand_pos)
                ligand_pos = ligand_pos.detach()
                log_model_prob = self.q_v_posterior(
                    log_ligand_v_recon, log_ligand_v, t, batch_ligand
                )
                ligand_v = log_sample_categorical(log_model_prob).argmax(dim=-1)

        ligand_pos = ligand_pos + offset[batch_ligand]
        return {
            "pos": ligand_pos,
            "v": ligand_v,
            "exp": exp_pred.detach() if exp_pred is not None else None,
        }
