"""DiffSMol hybrid diffusion score model, bond-free.

Ported from DiffSMol ``source/models/molopt_score_model.py``
(+ ``source/models/diffusion.py`` for the beta schedules).

Hybrid objective, both kept:
  * continuous Gaussian DDPM on positions (``sigmoid`` beta schedule),
  * discrete D3PM categorical diffusion on atom types (``cosine`` schedule,
    uniform or absorbing-mask transition kernel).

Dropped per the approved INTEGRATION_PLAN: the bond-type CE loss, the
bond-distance loss, the bond-angle loss and the torsion-angle loss (the last
two were already switched off by upstream's own shipped config). The loss
therefore reduces to ``loss_pos + loss_v_weight * loss_v``.

Kept: shape conditioning and classifier-free guidance. Training zeroes the
whole ``(B, 128, 3)`` latent with probability ``cond_mask_prob``; that is
what creates the unconditional branch, and it is why ``sample()`` can fall
back to a zero latent when no reference shape is supplied.

Frame convention: ``center_pos_mode`` is ``'none'``. Coordinates are assumed
to already be in the *surface-centered* frame (the caller subtracts the
precomputed ``shape_center``), which is the frame the shape latent lives in.
Do not re-center here.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from MolecularDiffusion.modules.models.diffsmol.common import (
    MLP,
    GaussianSmearing,
    ShiftedSoftplus,
    SinusoidalPosEmb,
)
from MolecularDiffusion.modules.models.diffsmol.uni_transformer import (
    UniTransformerO2TwoUpdateGeneral,
)


# --------------------------------------------------------------------- #
# beta schedules                                                        #
# --------------------------------------------------------------------- #
def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> np.ndarray:
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


def get_beta_schedule(
    beta_schedule: str, num_diffusion_timesteps: int, **kwargs: Any
) -> np.ndarray:
    kwargs = {k: float(v) for k, v in kwargs.items()}
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                kwargs["beta_start"] ** 0.5,
                kwargs["beta_end"] ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            kwargs["beta_start"],
            kwargs["beta_end"],
            num_diffusion_timesteps,
            dtype=np.float64,
        )
    elif beta_schedule == "sigmoid":
        s = kwargs.get("s", 3)
        betas = np.linspace(-s, s, num_diffusion_timesteps)
        betas = 1 / (np.exp(-betas) + 1) * (
            kwargs["beta_end"] - kwargs["beta_start"]
        ) + kwargs["beta_start"]
    elif beta_schedule == "cosine":
        betas = cosine_beta_schedule(
            num_diffusion_timesteps, s=kwargs.get("s", 0.008)
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


# --------------------------------------------------------------------- #
# categorical-diffusion helpers                                         #
# --------------------------------------------------------------------- #
def to_torch_const(x: np.ndarray) -> nn.Parameter:
    return nn.Parameter(torch.from_numpy(x).float(), requires_grad=False)


def extract(coef: torch.Tensor, t: torch.Tensor, batch: torch.Tensor):
    return coef[t][batch].unsqueeze(-1)


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
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    return (gumbel_noise + logits).argmax(dim=-1)


def log_1_min_a(a: np.ndarray) -> np.ndarray:
    return np.log(1 - np.exp(a) + 1e-40)


def log_add_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    maximum = torch.max(a, b)
    return maximum + torch.log(
        torch.exp(a - maximum) + torch.exp(b - maximum)
    )


# --------------------------------------------------------------------- #
# classifier-free-guidance thresholding                                 #
# --------------------------------------------------------------------- #
def threshold_cfg(
    x0: torch.Tensor,
    x0_cond: torch.Tensor,
    threshold_type: Optional[str],
    p: Optional[float] = None,
) -> torch.Tensor:
    if threshold_type is None:
        return x0
    if threshold_type == "reference_threshold":
        s = torch.max(torch.abs(x0_cond)) * (1.1 if p is None else p)
        return torch.clip(x0, min=-s, max=s)
    if threshold_type == "dynamic_threshold":
        s = torch.quantile(x0, 0.995 if p is None else p)
        return torch.clip(x0, min=-s, max=s)
    if threshold_type == "rescale":
        p = 0.7 if p is None else p
        ratio = torch.std(x0_cond) / torch.std(x0)
        return p * (x0 * ratio) + (1 - p) * x0
    raise ValueError(
        "undefined thresholding strategy: expect one of "
        "(reference_threshold, dynamic_threshold, rescale, null), "
        f"got {threshold_type}"
    )


DEFAULT_CONFIG: Dict[str, Any] = {
    "model_mean_type": "C0",
    "schedule_pos": {
        "beta_schedule": "sigmoid",
        "beta_start": 1.0e-7,
        "beta_end": 0.01,
        "s": 6,
    },
    "schedule_v": {"beta_schedule": "cosine", "s": 0.01},
    "num_diffusion_timesteps": 1000,
    "loss_v_weight": 200.0,
    "v_mode": "uniform",
    "v_net_type": "mlp",
    "loss_pos_type": "mse",
    "sample_time_method": "symmetric",
    "loss_weight_type": "noise_level",
    "loss_pos_min_weight": 0,
    "loss_pos_max_weight": 10,
    "time_emb_dim": 8,
    "num_blocks": 1,
    "num_layers": 8,
    "scalar_hidden_dim": 128,
    "vec_hidden_dim": 32,
    "n_heads": 16,
    "edge_feat_dim": 5,
    "num_r_gaussian": 20,
    "knn": 8,
    "act_fn": "relu",
    "norm": True,
    "r_max": 10.0,
    "shape_dim": 128,
    "shape_latent_dim": 128,
    "cond_mask_prob": 0.1,
    "use_shape_vec_mul": False,
    "use_residue": True,
    # Opt-in bond machinery. False reproduces the bond-free port exactly;
    # True restores upstream's b_inference head and its predicted-bond-type
    # edge-feature feedback loop, which is what the released diffusion.pt
    # weights were trained with. See uni_transformer's module docstring.
    "pred_bond_type": False,
    # Per-class aromaticity flags for the 15-class ``add_aromatic`` vocab.
    # None => plain element vocab (the default), and the backbone keeps using
    # the platform's own bond-length tables.
    "aromatic_flags": None,
}


class ScorePosNet3D(nn.Module):
    """Shape-conditioned hybrid (Gaussian pos + D3PM type) score model."""

    #: Class-level default so models pickled before the toggle existed still
    #: resolve it after unpickling.
    pred_bond_type = False

    def __init__(
        self,
        config: Dict[str, Any],
        atom_vocab: list[str],
    ) -> None:
        super().__init__()
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.config = SimpleNamespace(**cfg)
        self.atom_vocab = list(atom_vocab)
        self.num_classes = len(self.atom_vocab)

        self.model_mean_type = cfg["model_mean_type"]
        self.loss_v_weight = cfg["loss_v_weight"]
        self.loss_weight_type = cfg["loss_weight_type"]
        self.v_mode = cfg["v_mode"]
        self.v_net_type = cfg["v_net_type"]
        self.sample_time_method = cfg["sample_time_method"]
        self.loss_pos_type = cfg["loss_pos_type"]
        self.cond_mask_prob = cfg["cond_mask_prob"]
        self.shape_dim = cfg["shape_dim"]
        self.pred_bond_type = cfg["pred_bond_type"]

        # --- position (Gaussian) schedule ---
        betas = get_beta_schedule(
            num_diffusion_timesteps=cfg["num_diffusion_timesteps"],
            **cfg["schedule_pos"],
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        if self.loss_weight_type == "noise_level":
            snr = alphas_cumprod / (1 - alphas_cumprod)
            self.loss_pos_step_weight = to_torch_const(
                np.clip(
                    cfg["loss_pos_min_weight"] + snr,
                    None,
                    cfg["loss_pos_max_weight"],
                )
            )

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
            (1.0 - alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - alphas_cumprod)
        )
        self.posterior_var = to_torch_const(posterior_variance)
        self.posterior_logvar = to_torch_const(
            np.log(
                np.append(posterior_variance[1], posterior_variance[1:])
            )
        )

        # --- atom-type (D3PM) schedule ---
        betas_v = get_beta_schedule(
            num_diffusion_timesteps=cfg["num_diffusion_timesteps"],
            **cfg["schedule_v"],
        )
        log_alphas_v = np.log(1.0 - betas_v)
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)
        self.log_alphas_v = to_torch_const(log_alphas_v)
        self.log_one_minus_alphas_v = to_torch_const(log_1_min_a(log_alphas_v))
        self.log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)
        self.log_one_minus_alphas_cumprod_v = to_torch_const(
            log_1_min_a(log_alphas_cumprod_v)
        )

        # --- network ---
        self.scalar_hidden_dim = cfg["scalar_hidden_dim"]
        self.vec_hidden_dim = cfg["vec_hidden_dim"]
        self.time_emb_dim = cfg["time_emb_dim"]
        n_extra = int(self.v_mode == "tomask")

        if self.time_emb_dim > 0:
            self.time_emb = nn.Sequential(
                SinusoidalPosEmb(self.time_emb_dim),
                nn.Linear(self.time_emb_dim, self.time_emb_dim * 2),
                nn.SiLU(),
                nn.Linear(self.time_emb_dim * 2, self.time_emb_dim),
            )
            in_dim = self.num_classes + self.time_emb_dim + n_extra
        else:
            in_dim = self.num_classes + n_extra
        self.ligand_atom_emb = nn.Linear(in_dim, self.scalar_hidden_dim)

        self.refine_net = UniTransformerO2TwoUpdateGeneral(
            num_blocks=cfg["num_blocks"],
            num_layers=cfg["num_layers"],
            scalar_hidden_dim=cfg["scalar_hidden_dim"],
            vec_hidden_dim=cfg["vec_hidden_dim"],
            shape_dim=cfg["shape_dim"],
            shape_latent_dim=cfg["shape_latent_dim"],
            atom_vocab=self.atom_vocab,
            n_heads=cfg["n_heads"],
            k=cfg["knn"],
            num_r_gaussian=cfg["num_r_gaussian"],
            edge_feat_dim=cfg["edge_feat_dim"],
            act_fn=cfg["act_fn"],
            norm=cfg["norm"],
            r_max=cfg["r_max"],
            use_shape_vec_mul=cfg["use_shape_vec_mul"],
            use_residue=cfg["use_residue"],
            pred_bond_type=cfg["pred_bond_type"],
            aromatic_flags=cfg["aromatic_flags"],
        )

        if self.v_net_type == "mlp":
            self.v_inference = nn.Sequential(
                nn.Linear(self.scalar_hidden_dim, self.scalar_hidden_dim),
                ShiftedSoftplus(),
                nn.Linear(
                    self.scalar_hidden_dim, self.num_classes + n_extra
                ),
            )
        elif self.v_net_type == "attention":
            self.v_distance_expansion = GaussianSmearing(
                0.0, 10.0, num_gaussians=cfg["num_r_gaussian"]
            )
            kv_input_dim = (
                self.scalar_hidden_dim * 2 + cfg["num_r_gaussian"]
            )
            self.vk_func = MLP(
                kv_input_dim, self.scalar_hidden_dim, self.scalar_hidden_dim
            )
            self.vv_func = MLP(
                kv_input_dim, self.scalar_hidden_dim, self.scalar_hidden_dim
            )
            self.vq_func = MLP(
                self.scalar_hidden_dim,
                self.scalar_hidden_dim,
                self.scalar_hidden_dim,
            )
            self.v_inference = nn.Sequential(
                nn.Linear(self.scalar_hidden_dim * 2, self.scalar_hidden_dim),
                ShiftedSoftplus(),
                nn.Linear(
                    self.scalar_hidden_dim, self.num_classes + n_extra
                ),
            )
        else:
            raise NotImplementedError(self.v_net_type)

    # ----------------------------------------------------------------- #
    # forward                                                           #
    # ----------------------------------------------------------------- #
    def forward(
        self,
        ligand_pos_perturbed: torch.Tensor,
        ligand_v_perturbed: torch.Tensor,
        batch_ligand: torch.Tensor,
        ligand_shape: torch.Tensor,
        time_step: Optional[torch.Tensor] = None,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Predict ``(x0, v0)`` from the noisy sample at step ``t``.

        ``ligand_shape``: ``(B, shape_dim, 3)`` equivariant latent, in the
        same frame as ``ligand_pos_perturbed``.
        """
        n_extra = int(self.v_mode == "tomask")
        v_onehot = F.one_hot(
            ligand_v_perturbed, self.num_classes + n_extra
        ).float()

        if self.time_emb_dim > 0:
            ligand_feat = torch.cat(
                [v_onehot, self.time_emb(time_step)[batch_ligand]], -1
            )
        else:
            ligand_feat = v_onehot

        ligand_emb = self.ligand_atom_emb(ligand_feat)
        outputs = self.refine_net(
            v_onehot,
            ligand_emb,
            ligand_pos_perturbed,
            batch_ligand,
            ligand_shape,
            return_all=return_all,
        )
        preds = {
            "pred_ligand_pos": outputs["x"],
            "pred_ligand_h": outputs["h"],
            "pred_ligand_v": self.v_inference(outputs["h"]),
        }
        if return_all:
            preds.update(
                {
                    "layer_pred_ligand_pos": list(outputs["all_vec"]),
                    "layer_pred_ligand_v": [
                        self.v_inference(h) for h in outputs["all_h"]
                    ],
                }
            )
        return preds

    # ----------------------------------------------------------------- #
    # D3PM kernels                                                      #
    # ----------------------------------------------------------------- #
    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        """q(v_t | v_{t-1})."""
        log_alpha_t = extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)
        if self.v_mode == "uniform":
            return log_add_exp(
                log_vt_1 + log_alpha_t,
                log_1_min_alpha_t - np.log(self.num_classes),
            )
        if self.v_mode == "tomask":
            log_probs = log_vt_1 + log_alpha_t
            log_probs[:, -1] = log_1_min_alpha_t.squeeze(1)
            return log_probs
        raise ValueError(f"undefined v_mode: {self.v_mode}")

    def q_v_pred(self, log_v0, t, batch):
        """q(v_t | v_0)."""
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)
        log_1_min_cumprod_alpha = extract(
            self.log_one_minus_alphas_cumprod_v, t, batch
        )
        if self.v_mode == "uniform":
            return log_add_exp(
                log_v0 + log_cumprod_alpha_t,
                log_1_min_cumprod_alpha - np.log(self.num_classes),
            )
        if self.v_mode == "tomask":
            log_probs = log_v0 + log_cumprod_alpha_t
            log_probs[:, -1] = log_1_min_cumprod_alpha.squeeze(1)
            return log_probs
        raise ValueError(f"undefined v_mode: {self.v_mode}")

    def q_v_sample(self, log_v0, t, batch, num_classes):
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        sample_index = log_sample_categorical(log_qvt_v0)
        return sample_index, index_to_log_onehot(sample_index, num_classes)

    def q_v_posterior(self, log_v0, log_vt, t, batch):
        """q(v_{t-1} | v_t, v_0)."""
        t_minus_1 = torch.clamp(t - 1, min=0)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        unnormed = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)
        return unnormed - torch.logsumexp(unnormed, dim=-1, keepdim=True)

    def q_pos_posterior(self, x0, xt, t, batch):
        return (
            extract(self.posterior_mean_c0_coef, t, batch) * x0
            + extract(self.posterior_mean_ct_coef, t, batch) * xt
        )

    def sample_time(self, num_graphs: int, device):
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device
        )
        time_step = torch.cat(
            [time_step, self.num_timesteps - time_step - 1], dim=0
        )[:num_graphs]
        pt = torch.ones_like(time_step).float() / self.num_timesteps
        return time_step, pt

    def compute_v_Lt(
        self, log_v_model_prob, log_v0, log_v_true_prob, t, batch
    ):
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)
        mask = (t == 0).float()[batch]
        return scatter_mean(
            mask * decoder_nll_v + (1.0 - mask) * kl_v, batch, dim=0
        )

    # ----------------------------------------------------------------- #
    # training objective                                                #
    # ----------------------------------------------------------------- #
    def get_diffusion_loss(
        self,
        ligand_pos: torch.Tensor,
        ligand_v: torch.Tensor,
        batch_ligand: torch.Tensor,
        ligand_shape: torch.Tensor,
        time_step: Optional[torch.Tensor] = None,
        eval_mode: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Bond-free ``loss_pos + loss_v_weight * loss_v``.

        ``ligand_pos`` must already be surface-centered; no re-centering is
        applied (``center_pos_mode: none`` upstream).
        """
        num_graphs = int(batch_ligand.max().item()) + 1

        if time_step is None:
            time_step, _ = self.sample_time(num_graphs, ligand_pos.device)

        a_pos = self.alphas_cumprod.index_select(0, time_step)[
            batch_ligand
        ].unsqueeze(-1)
        pos_noise = torch.randn_like(ligand_pos)
        ligand_pos_perturbed = (
            a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise
        )

        n_extra = int(self.v_mode == "tomask")
        log_ligand_v0 = index_to_log_onehot(
            ligand_v, self.num_classes + n_extra
        )
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(
            log_ligand_v0, time_step, batch_ligand,
            self.num_classes + n_extra,
        )

        ligand_shape = ligand_shape.view(num_graphs, -1, 3)
        if not eval_mode and self.cond_mask_prob > 0:
            # Classifier-free guidance: drop the whole latent for a random
            # subset of molecules. This is what trains the uncond branch.
            keep = torch.bernoulli(
                torch.full(
                    (num_graphs,),
                    1 - self.cond_mask_prob,
                    device=ligand_shape.device,
                )
            ).view(num_graphs, 1, 1)
            ligand_shape = keep * ligand_shape

        preds = self(
            ligand_pos_perturbed=ligand_pos_perturbed,
            ligand_v_perturbed=ligand_v_perturbed,
            batch_ligand=batch_ligand,
            ligand_shape=ligand_shape,
            time_step=time_step,
        )
        pred_ligand_pos = preds["pred_ligand_pos"]
        pred_ligand_v = preds["pred_ligand_v"]

        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(
            log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand
        )
        log_v_true_prob = self.q_v_posterior(
            log_ligand_v0, log_ligand_vt, time_step, batch_ligand
        )
        kl_v = self.compute_v_Lt(
            log_v_model_prob, log_ligand_v0, log_v_true_prob,
            time_step, batch_ligand,
        )

        loss_pos = scatter_mean(
            ((pred_ligand_pos - ligand_pos) ** 2).sum(-1),
            batch_ligand, dim=0,
        )
        if self.loss_weight_type == "noise_level":
            loss_pos_weight = self.loss_pos_step_weight.index_select(
                0, time_step
            )
            loss_pos = torch.mean(loss_pos_weight * loss_pos)
        else:
            loss_pos = torch.mean(loss_pos)

        loss_v = torch.mean(kl_v)
        loss = loss_pos + loss_v * self.loss_v_weight

        return {
            "loss": loss,
            "loss_pos": loss_pos,
            "loss_v": loss_v,
            "x0": ligand_pos,
            "ligand_pos_perturbed": ligand_pos_perturbed,
            "ligand_v_perturbed": ligand_v_perturbed,
            "pred_ligand_pos": pred_ligand_pos,
            "pred_ligand_v": pred_ligand_v,
        }

    # ----------------------------------------------------------------- #
    # sampling                                                          #
    # ----------------------------------------------------------------- #
    @torch.no_grad()
    def sample_diffusion(
        self,
        init_ligand_pos: torch.Tensor,
        init_ligand_v: torch.Tensor,
        batch_ligand: torch.Tensor,
        ligand_shape: torch.Tensor,
        num_steps: Optional[int] = None,
        guide_stren: float = 0.0,
        threshold_type: Optional[str] = None,
        threshold_p: Optional[float] = None,
        progress: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Reverse diffusion with classifier-free guidance.

        ``guide_stren`` is the CFG weight ``w``: the guided prediction is
        ``uncond + (1 + w) * (cond - uncond)``. ``0`` means conditional-only
        (no second pass), matching upstream's shipped no-guidance setting.
        """
        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = int(batch_ligand.max().item()) + 1
        ligand_shape = ligand_shape.view(num_graphs, -1, 3)

        n_extra = int(self.v_mode == "tomask")
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v

        time_seq = list(
            reversed(range(self.num_timesteps - num_steps, self.num_timesteps))
        )
        if progress:
            from tqdm.auto import tqdm

            time_seq = tqdm(time_seq, desc="diffsmol sampling")

        for i in time_seq:
            t = torch.full(
                size=(num_graphs,),
                fill_value=i,
                dtype=torch.long,
                device=ligand_pos.device,
            )
            preds_cond = self(
                ligand_pos_perturbed=ligand_pos,
                ligand_v_perturbed=ligand_v,
                batch_ligand=batch_ligand,
                ligand_shape=ligand_shape,
                time_step=t,
            )

            if self.cond_mask_prob > 0 and guide_stren != 0.0:
                preds_uncond = self(
                    ligand_pos_perturbed=ligand_pos,
                    ligand_v_perturbed=ligand_v,
                    batch_ligand=batch_ligand,
                    ligand_shape=torch.zeros_like(ligand_shape),
                    time_step=t,
                )
                pos0 = preds_uncond["pred_ligand_pos"] + (
                    1 + guide_stren
                ) * (
                    preds_cond["pred_ligand_pos"]
                    - preds_uncond["pred_ligand_pos"]
                )
                pos0 = threshold_cfg(
                    pos0, preds_cond["pred_ligand_pos"],
                    threshold_type, threshold_p,
                )
                v0 = preds_uncond["pred_ligand_v"] + (1 + guide_stren) * (
                    preds_cond["pred_ligand_v"]
                    - preds_uncond["pred_ligand_v"]
                )
            else:
                pos0 = preds_cond["pred_ligand_pos"]
                v0 = preds_cond["pred_ligand_v"]

            if self.v_mode == "tomask":
                v0 = v0.clone()
                v0[:, -1] = -1.0e5

            pos_model_mean = self.q_pos_posterior(
                x0=pos0, xt=ligand_pos, t=t, batch=batch_ligand
            )
            pos_log_variance = extract(
                self.posterior_logvar, t, batch_ligand
            )
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
            ligand_pos = pos_model_mean + nonzero_mask * (
                0.5 * pos_log_variance
            ).exp() * torch.randn_like(ligand_pos)

            log_ligand_v_recon = F.log_softmax(v0, dim=-1)
            log_ligand_v = index_to_log_onehot(
                ligand_v, self.num_classes + n_extra
            )
            log_model_prob = self.q_v_posterior(
                log_ligand_v_recon, log_ligand_v, t, batch_ligand
            )
            ligand_v = log_sample_categorical(log_model_prob)

        return {"pos": ligand_pos, "v": ligand_v}
