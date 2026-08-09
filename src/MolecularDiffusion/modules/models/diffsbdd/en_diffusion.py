"""DiffSBDD's E(3)-equivariant DDPMs over a ligand + protein-pocket system.

Ported from ``equivariant_diffusion/en_diffusion.py`` and
``equivariant_diffusion/conditional_model.py``. Two selectable models:

* :class:`EnVariationalDiffusion` (``mode: joint``) -- diffuses ligand *and*
  pocket together. Its :meth:`~EnVariationalDiffusion.inpaint` is also how
  upstream does pocket-conditioned generation in joint mode
  (``lightning_modules.py:814-834``): a zero ligand with ``lig_fixed=0`` and
  ``pocket_fixed=1``. De-novo generation and fragment-fixing are the same
  call with a different ``lig_fixed``.
* :class:`ConditionalDDPM` (``mode: pocket_conditioning``) -- diffuses the
  ligand only; the pocket is clean, never-noised context, and the system is
  kept translation-invariant by re-centring on the ligand CoM each step. It
  has both :meth:`~ConditionalDDPM.sample_given_pocket` and its own
  :meth:`~ConditionalDDPM.inpaint`.

Continuous Gaussian noise on coordinates **and** the one-hot feature block,
epsilon parametrisation, mean-free coordinate subspace.

Not ported, all unused by every shipped DiffSBDD config or explicitly out of
scope in the integration plan: ``GammaNetwork``/``PositiveLinear``
(``noise_schedule='learned'``), virtual nodes, the auxiliary Lennard-Jones
term and the ``xh_lig_hat`` it consumed, ``SimpleConditionalDDPM``,
``diversify()``/``partially_noised_ligand()``.

``forward()`` returns a dict rather than upstream's 13-tuple: the task module
in ``modules/tasks/diffusion_diffsbdd.py`` is the only consumer, and both
models return the *same* keys -- which is why ``mode`` costs the task zero
extra code (``en_diffusion.py:465`` vs ``conditional_model.py:326`` return
matching tuples upstream, the conditional one substituting 0.0 for the pocket
error).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_add, scatter_mean

# --------------------------------------------------------------------------- #
# Noise schedules
# --------------------------------------------------------------------------- #


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> np.ndarray:
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = np.clip(1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]), 0, 0.999)
    return np.cumprod(1.0 - betas, axis=0)


def clip_noise_schedule(
    alphas2: np.ndarray, clip_value: float = 0.001
) -> np.ndarray:
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)
    alphas_step = np.clip(alphas2[1:] / alphas2[:-1], a_min=clip_value, a_max=1.0)
    return np.cumprod(alphas_step, axis=0)


def polynomial_schedule(
    timesteps: int, s: float = 1e-4, power: float = 3.0
) -> np.ndarray:
    """``1 - x^power``; ``polynomial_2`` with s=5e-4 is DiffSBDD's default."""
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power)) ** 2
    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)
    return (1 - 2 * s) * alphas2 + s


class PredefinedNoiseSchedule(nn.Module):
    """Lookup table of ``gamma`` for a non-learned schedule."""

    def __init__(
        self, noise_schedule: str, timesteps: int, precision: float
    ) -> None:
        super().__init__()
        self.timesteps = timesteps
        if noise_schedule == "cosine":
            alphas2 = cosine_beta_schedule(timesteps)
        elif "polynomial" in noise_schedule:
            _, power = noise_schedule.split("_")
            alphas2 = polynomial_schedule(
                timesteps, s=precision, power=float(power)
            )
        else:
            raise ValueError(
                f"unknown noise_schedule {noise_schedule!r}; the port supports "
                "'cosine' and 'polynomial_<power>' (learned schedules are not "
                "used by any shipped DiffSBDD config)."
            )
        sigmas2 = 1 - alphas2
        log_alphas2_to_sigmas2 = np.log(alphas2) - np.log(sigmas2)
        # Parameter, not buffer: matches upstream so the released checkpoints'
        # `gamma.gamma` key maps without a remap.
        self.gamma = nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(), requires_grad=False
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.gamma[torch.round(t * self.timesteps).long()]


# --------------------------------------------------------------------------- #
# Joint size prior
# --------------------------------------------------------------------------- #


class DistributionNodes:
    """2D joint histogram over ``(n_ligand_atoms, n_pocket_nodes)``.

    Same semantics as upstream ``en_diffusion.py:958``, but the 178k-entry
    ``n_nodes_to_idx`` dict and the eager per-column ``Categorical`` lists are
    replaced by row-major arithmetic and ``torch.multinomial``. Identical
    maths, built in milliseconds instead of seconds.
    """

    def __init__(self, histogram: torch.Tensor) -> None:
        hist = torch.as_tensor(histogram).float() + 1e-3
        self.prob = hist / hist.sum()
        self.max_n_lig, self.max_n_pocket = self.prob.shape

    def _check(self, n_lig: torch.Tensor, n_pocket: torch.Tensor) -> None:
        if int(n_lig.max()) >= self.max_n_lig or int(n_pocket.max()) >= self.max_n_pocket:
            raise ValueError(
                f"complex size outside the size prior's support: got "
                f"n_lig<={int(n_lig.max())}, n_pocket<={int(n_pocket.max())} but "
                f"the histogram is {self.prob.shape}. Raise tasks.max_n_lig / "
                "tasks.max_n_pocket (note this changes the checkpoint's buffer "
                "shape)."
            )

    def sample(self, n_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = torch.multinomial(self.prob.view(-1), n_samples, replacement=True)
        return idx // self.max_n_pocket, idx % self.max_n_pocket

    def sample_conditional(
        self, n1: Optional[torch.Tensor] = None, n2: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Draw one axis given the other. ``n2=pocket_sizes`` is the live path."""
        if (n1 is None) == (n2 is None):
            raise ValueError("exactly one of n1 / n2 must be given")
        cond = n2 if n2 is not None else n1
        cond_cpu = cond.detach().cpu().long()
        # columns (ligand size | pocket size) or rows (pocket size | ligand size)
        weights = (
            self.prob[:, cond_cpu].T if n2 is not None else self.prob[cond_cpu, :]
        )
        out = torch.multinomial(weights, 1).squeeze(1)
        return out.to(cond.device)

    def log_prob(
        self, n_lig: torch.Tensor, n_pocket: torch.Tensor
    ) -> torch.Tensor:
        """log p(N_lig, N_pocket) -- the joint model's ``log_pN``."""
        self._check(n_lig, n_pocket)
        p = self.prob.to(n_lig.device)
        return torch.log(p[n_lig.long(), n_pocket.long()])

    def log_prob_n1_given_n2(
        self, n1: torch.Tensor, n2: torch.Tensor
    ) -> torch.Tensor:
        """log p(N_lig | N_pocket) -- the conditional model's ``log_pN``."""
        self._check(n1, n2)
        p = self.prob.to(n1.device)
        col = p[:, n2.long()]
        return torch.log(p[n1.long(), n2.long()] / col.sum(dim=0))


# --------------------------------------------------------------------------- #
# Joint model
# --------------------------------------------------------------------------- #


class EnVariationalDiffusion(nn.Module):
    """``mode: joint`` -- ligand and pocket are diffused together."""

    def __init__(
        self,
        dynamics: nn.Module,
        atom_nf: int,
        residue_nf: int,
        n_dims: int = 3,
        size_histogram: Optional[torch.Tensor] = None,
        max_n_lig: int = 107,
        max_n_pocket: int = 1671,
        timesteps: int = 500,
        noise_schedule: str = "polynomial_2",
        noise_precision: float = 5e-4,
        norm_values: Tuple[float, float] = (1.0, 4.0),
        norm_biases: Tuple[Optional[float], float] = (None, 0.0),
    ) -> None:
        super().__init__()
        self.gamma = PredefinedNoiseSchedule(
            noise_schedule, timesteps=timesteps, precision=noise_precision
        )
        self.dynamics = dynamics
        self.atom_nf = atom_nf
        self.residue_nf = residue_nf
        self.n_dims = n_dims
        self.num_classes = atom_nf
        self.T = timesteps
        self.norm_values = tuple(norm_values)
        self.norm_biases = tuple(norm_biases)
        self.register_buffer("buffer", torch.zeros(1))

        # The joint size histogram is a genuine dataset statistic needed at
        # BOTH train (log_pN) and generate time, so it lives in the state dict
        # -- cli/generate.py rebuilds the task with train_set=None. Shape is
        # fixed by config so checkpoint shapes never depend on which db was
        # used; (107, 1671) is the released CrossDocked histogram's shape.
        hist = torch.zeros(max_n_lig, max_n_pocket)
        if size_histogram is not None:
            src = torch.as_tensor(size_histogram).float()
            n1, n2 = min(src.shape[0], max_n_lig), min(src.shape[1], max_n_pocket)
            hist[:n1, :n2] = src[:n1, :n2]
        self.register_buffer("size_histogram", hist)
        self._size_dist: Optional[DistributionNodes] = None
        self._size_dist_key: Optional[float] = None

        self.check_issues_norm_values()

    # -- size prior ------------------------------------------------------ #
    @property
    def size_distribution(self) -> DistributionNodes:
        """Rebuilt lazily, so ``load_state_dict`` on the buffer takes effect."""
        key = float(self.size_histogram.sum())
        if self._size_dist is None or self._size_dist_key != key:
            self._size_dist = DistributionNodes(self.size_histogram.cpu())
            self._size_dist_key = key
        return self._size_dist

    def log_pN(self, n_lig: torch.Tensor, n_pocket: torch.Tensor) -> torch.Tensor:
        return self.size_distribution.log_prob(n_lig, n_pocket)

    # -- schedule helpers ------------------------------------------------ #
    def check_issues_norm_values(self, num_stdevs: int = 8) -> None:
        zeros = torch.zeros((1, 1))
        sigma_0 = self.sigma(self.gamma(zeros), target_tensor=zeros).item()
        norm_value = self.norm_values[1]
        if sigma_0 * num_stdevs > 1.0 / norm_value:
            raise ValueError(
                f"normalize_factors[1]={norm_value} is too large for sigma_0="
                f"{sigma_0:.5f} (1/norm_value={1.0 / norm_value})."
            )

    @staticmethod
    def inflate_batch_array(
        array: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return array.view((array.size(0),) + (1,) * (len(target.size()) - 1))

    def sigma(self, gamma: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        return self.inflate_batch_array(
            torch.sqrt(torch.sigmoid(gamma)), target_tensor
        )

    def alpha(self, gamma: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        return self.inflate_batch_array(
            torch.sqrt(torch.sigmoid(-gamma)), target_tensor
        )

    @staticmethod
    def SNR(gamma: torch.Tensor) -> torch.Tensor:  # noqa: N802 - upstream name
        return torch.exp(-gamma)

    def sigma_and_alpha_t_given_s(
        self,
        gamma_t: torch.Tensor,
        gamma_s: torch.Tensor,
        target_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma2_t_given_s = self.inflate_batch_array(
            -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t)), target_tensor
        )
        log_alpha2_t_given_s = F.logsigmoid(-gamma_t) - F.logsigmoid(-gamma_s)
        alpha_t_given_s = self.inflate_batch_array(
            torch.exp(0.5 * log_alpha2_t_given_s), target_tensor
        )
        return sigma2_t_given_s, torch.sqrt(sigma2_t_given_s), alpha_t_given_s

    # -- normalisation --------------------------------------------------- #
    def normalize(
        self,
        ligand: Optional[Dict[str, torch.Tensor]] = None,
        pocket: Optional[Dict[str, torch.Tensor]] = None,
    ):
        for node_set in (ligand, pocket):
            if node_set is not None:
                node_set["x"] = node_set["x"] / self.norm_values[0]
                node_set["one_hot"] = (
                    node_set["one_hot"].float() - self.norm_biases[1]
                ) / self.norm_values[1]
        return ligand, pocket

    def unnormalize(
        self, x: torch.Tensor, h_cat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return x * self.norm_values[0], h_cat * self.norm_values[1] + self.norm_biases[1]

    def unnormalize_z(
        self, z_lig: torch.Tensor, z_pocket: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_lig, h_lig = self.unnormalize(
            z_lig[:, : self.n_dims], z_lig[:, self.n_dims :]
        )
        x_pocket, h_pocket = self.unnormalize(
            z_pocket[:, : self.n_dims], z_pocket[:, self.n_dims :]
        )
        return (
            torch.cat([x_lig, h_lig], dim=1),
            torch.cat([x_pocket, h_pocket], dim=1),
        )

    def subspace_dimensionality(self, input_size: torch.Tensor) -> torch.Tensor:
        return (input_size - 1) * self.n_dims

    def delta_log_px(self, num_nodes: torch.Tensor) -> torch.Tensor:
        return -self.subspace_dimensionality(num_nodes) * np.log(self.norm_values[0])

    # -- static maths ---------------------------------------------------- #
    @staticmethod
    def gaussian_KL(  # noqa: N802 - upstream name
        q_mu_minus_p_mu_squared: torch.Tensor,
        q_sigma: torch.Tensor,
        p_sigma: torch.Tensor,
        d,
    ) -> torch.Tensor:
        return (
            d * torch.log(p_sigma / q_sigma)
            + 0.5 * (d * q_sigma**2 + q_mu_minus_p_mu_squared) / (p_sigma**2)
            - 0.5 * d
        )

    @staticmethod
    def remove_mean_batch(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return x - scatter_mean(x, indices, dim=0)[indices]

    @staticmethod
    def assert_mean_zero_with_mask(
        x: torch.Tensor, node_mask: torch.Tensor, eps: float = 1e-10
    ) -> None:
        largest = x.abs().max().item()
        error = scatter_add(x, node_mask, dim=0).abs().max().item()
        if error / (largest + eps) >= 1e-2:
            raise AssertionError(
                f"coordinates are not mean-zero (relative error "
                f"{error / (largest + eps):.4f})"
            )

    @staticmethod
    def sum_except_batch(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return scatter_add(x.sum(-1), indices, dim=0)

    @staticmethod
    def cdf_standard_gaussian(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))

    @staticmethod
    def sample_gaussian(size, device) -> torch.Tensor:
        return torch.randn(size, device=device)

    @staticmethod
    def sample_center_gravity_zero_gaussian_batch(
        size, lig_indices: torch.Tensor, pocket_indices: torch.Tensor
    ) -> torch.Tensor:
        x = torch.randn(size, device=lig_indices.device)
        return EnVariationalDiffusion.remove_mean_batch(
            x, torch.cat((lig_indices, pocket_indices))
        )

    # -- noise ----------------------------------------------------------- #
    def sample_combined_position_feature_noise(
        self, lig_indices: torch.Tensor, pocket_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_x = self.sample_center_gravity_zero_gaussian_batch(
            size=(len(lig_indices) + len(pocket_indices), self.n_dims),
            lig_indices=lig_indices,
            pocket_indices=pocket_indices,
        )
        z_h_lig = self.sample_gaussian(
            (len(lig_indices), self.atom_nf), lig_indices.device
        )
        z_h_pocket = self.sample_gaussian(
            (len(pocket_indices), self.residue_nf), pocket_indices.device
        )
        return (
            torch.cat([z_x[: len(lig_indices)], z_h_lig], dim=1),
            torch.cat([z_x[len(lig_indices) :], z_h_pocket], dim=1),
        )

    def sample_normal(
        self,
        mu_lig: torch.Tensor,
        mu_pocket: torch.Tensor,
        sigma: torch.Tensor,
        lig_mask: torch.Tensor,
        pocket_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eps_lig, eps_pocket = self.sample_combined_position_feature_noise(
            lig_mask, pocket_mask
        )
        return (
            mu_lig + sigma[lig_mask] * eps_lig,
            mu_pocket + sigma[pocket_mask] * eps_pocket,
        )

    def noised_representation(
        self,
        xh_lig: torch.Tensor,
        xh_pocket: torch.Tensor,
        lig_mask: torch.Tensor,
        pocket_mask: torch.Tensor,
        gamma_t: torch.Tensor,
    ):
        alpha_t = self.alpha(gamma_t, xh_lig)
        sigma_t = self.sigma(gamma_t, xh_lig)
        eps_lig, eps_pocket = self.sample_combined_position_feature_noise(
            lig_mask, pocket_mask
        )
        z_t_lig = alpha_t[lig_mask] * xh_lig + sigma_t[lig_mask] * eps_lig
        z_t_pocket = (
            alpha_t[pocket_mask] * xh_pocket + sigma_t[pocket_mask] * eps_pocket
        )
        return z_t_lig, z_t_pocket, eps_lig, eps_pocket

    def compute_x_pred(
        self,
        net_out: torch.Tensor,
        zt: torch.Tensor,
        gamma_t: torch.Tensor,
        batch_mask: torch.Tensor,
    ) -> torch.Tensor:
        sigma_t = self.sigma(gamma_t, target_tensor=net_out)
        alpha_t = self.alpha(gamma_t, target_tensor=net_out)
        return 1.0 / alpha_t[batch_mask] * (zt - sigma_t[batch_mask] * net_out)

    def log_constants_p_x_given_z0(
        self, n_nodes: torch.Tensor, device
    ) -> torch.Tensor:
        batch_size = len(n_nodes)
        dof_x = self.subspace_dimensionality(n_nodes)
        gamma_0 = self.gamma(torch.zeros((batch_size, 1), device=device))
        log_sigma_x = 0.5 * gamma_0.view(batch_size)
        return dof_x * (-log_sigma_x - 0.5 * np.log(2 * np.pi))

    # -- loss ------------------------------------------------------------ #
    def kl_prior_with_pocket(
        self,
        xh_lig: torch.Tensor,
        xh_pocket: torch.Tensor,
        mask_lig: torch.Tensor,
        mask_pocket: torch.Tensor,
        num_nodes: torch.Tensor,
    ) -> torch.Tensor:
        ones = torch.ones((len(num_nodes), 1), device=xh_lig.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, xh_lig)

        mu_T_lig = alpha_T[mask_lig] * xh_lig
        mu_T_lig_x, mu_T_lig_h = mu_T_lig[:, : self.n_dims], mu_T_lig[:, self.n_dims :]
        mu_T_pocket = alpha_T[mask_pocket] * xh_pocket
        mu_T_pocket_x = mu_T_pocket[:, : self.n_dims]
        mu_T_pocket_h = mu_T_pocket[:, self.n_dims :]

        sigma_T_x = self.sigma(gamma_T, mu_T_lig_x).squeeze()
        sigma_T_h = self.sigma(gamma_T, mu_T_lig_h).squeeze()

        mu_norm2_h = self.sum_except_batch(
            mu_T_lig_h**2, mask_lig
        ) + self.sum_except_batch(mu_T_pocket_h**2, mask_pocket)
        kl_h = self.gaussian_KL(
            mu_norm2_h, sigma_T_h, torch.ones_like(sigma_T_h), d=1
        )

        mu_norm2_x = self.sum_except_batch(
            mu_T_lig_x**2, mask_lig
        ) + self.sum_except_batch(mu_T_pocket_x**2, mask_pocket)
        kl_x = self.gaussian_KL(
            mu_norm2_x,
            sigma_T_x,
            torch.ones_like(sigma_T_x),
            self.subspace_dimensionality(num_nodes),
        )
        return kl_x + kl_h

    def log_pxh_given_z0_without_constants(
        self,
        ligand,
        z_0_lig,
        eps_lig,
        net_out_lig,
        pocket,
        z_0_pocket,
        eps_pocket,
        net_out_pocket,
        gamma_0,
        epsilon: float = 1e-10,
    ):
        sigma_0_cat = self.sigma(gamma_0, target_tensor=z_0_lig) * self.norm_values[1]

        log_p_x_lig = -0.5 * self.sum_except_batch(
            (eps_lig[:, : self.n_dims] - net_out_lig[:, : self.n_dims]) ** 2,
            ligand["mask"],
        )
        log_p_x_pocket = -0.5 * self.sum_except_batch(
            (eps_pocket[:, : self.n_dims] - net_out_pocket[:, : self.n_dims]) ** 2,
            pocket["mask"],
        )

        log_ph = 0.0
        for node_set, z_0, key in (
            (ligand, z_0_lig, "mask"),
            (pocket, z_0_pocket, "mask"),
        ):
            onehot = node_set["one_hot"] * self.norm_values[1] + self.norm_biases[1]
            estimated = (
                z_0[:, self.n_dims :] * self.norm_values[1] + self.norm_biases[1]
            )
            centered = estimated - 1
            log_p_prop = torch.log(
                self.cdf_standard_gaussian(
                    (centered + 0.5) / sigma_0_cat[node_set[key]]
                )
                - self.cdf_standard_gaussian(
                    (centered - 0.5) / sigma_0_cat[node_set[key]]
                )
                + epsilon
            )
            log_probs = log_p_prop - torch.logsumexp(log_p_prop, dim=1, keepdim=True)
            log_ph = log_ph + self.sum_except_batch(
                log_probs * onehot, node_set[key]
            )

        return log_p_x_lig, log_p_x_pocket, log_ph

    def forward(self, ligand: Dict, pocket: Dict) -> Dict[str, Any]:
        """Returns the loss terms; see the module docstring."""
        ligand, pocket = self.normalize(ligand, pocket)
        delta_log_px = self.delta_log_px(ligand["size"] + pocket["size"])

        lowest_t = 0 if self.training else 1
        t_int = torch.randint(
            lowest_t,
            self.T + 1,
            size=(ligand["size"].size(0), 1),
            device=ligand["x"].device,
        ).float()
        s_int = t_int - 1
        t_is_zero = (t_int == 0).float()
        t_is_not_zero = 1 - t_is_zero
        s, t = s_int / self.T, t_int / self.T

        gamma_s = self.inflate_batch_array(self.gamma(s), ligand["x"])
        gamma_t = self.inflate_batch_array(self.gamma(t), ligand["x"])

        xh_lig = torch.cat([ligand["x"], ligand["one_hot"]], dim=1)
        xh_pocket = torch.cat([pocket["x"], pocket["one_hot"]], dim=1)

        z_t_lig, z_t_pocket, eps_t_lig, eps_t_pocket = self.noised_representation(
            xh_lig, xh_pocket, ligand["mask"], pocket["mask"], gamma_t
        )
        net_out_lig, net_out_pocket = self.dynamics(
            z_t_lig, z_t_pocket, t, ligand["mask"], pocket["mask"]
        )

        error_t_lig = self.sum_except_batch(
            (eps_t_lig - net_out_lig) ** 2, ligand["mask"]
        )
        error_t_pocket = self.sum_except_batch(
            (eps_t_pocket - net_out_pocket) ** 2, pocket["mask"]
        )
        SNR_weight = (1 - self.SNR(gamma_s - gamma_t)).squeeze(1)  # noqa: N806

        neg_log_constants = -self.log_constants_p_x_given_z0(
            n_nodes=ligand["size"] + pocket["size"], device=error_t_lig.device
        )
        kl_prior = self.kl_prior_with_pocket(
            xh_lig,
            xh_pocket,
            ligand["mask"],
            pocket["mask"],
            ligand["size"] + pocket["size"],
        )

        if self.training:
            log_p_x_lig, log_p_x_pocket, log_ph = (
                self.log_pxh_given_z0_without_constants(
                    ligand,
                    z_t_lig,
                    eps_t_lig,
                    net_out_lig,
                    pocket,
                    z_t_pocket,
                    eps_t_pocket,
                    net_out_pocket,
                    gamma_t,
                )
            )
            loss_0_x_ligand = -log_p_x_lig * t_is_zero.squeeze()
            loss_0_x_pocket = -log_p_x_pocket * t_is_zero.squeeze()
            loss_0_h = -log_ph * t_is_zero.squeeze()
            error_t_lig = error_t_lig * t_is_not_zero.squeeze()
            error_t_pocket = error_t_pocket * t_is_not_zero.squeeze()
        else:
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), ligand["x"])
            z_0_lig, z_0_pocket, eps_0_lig, eps_0_pocket = (
                self.noised_representation(
                    xh_lig, xh_pocket, ligand["mask"], pocket["mask"], gamma_0
                )
            )
            net_out_0_lig, net_out_0_pocket = self.dynamics(
                z_0_lig, z_0_pocket, t_zeros, ligand["mask"], pocket["mask"]
            )
            loss_0_x_ligand, loss_0_x_pocket, log_ph = (
                self.log_pxh_given_z0_without_constants(
                    ligand,
                    z_0_lig,
                    eps_0_lig,
                    net_out_0_lig,
                    pocket,
                    z_0_pocket,
                    eps_0_pocket,
                    net_out_0_pocket,
                    gamma_0,
                )
            )
            loss_0_x_ligand = -loss_0_x_ligand
            loss_0_x_pocket = -loss_0_x_pocket
            loss_0_h = -log_ph

        return {
            "delta_log_px": delta_log_px,
            "error_t_lig": error_t_lig,
            "error_t_pocket": error_t_pocket,
            "SNR_weight": SNR_weight,
            "loss_0_x_ligand": loss_0_x_ligand,
            "loss_0_x_pocket": loss_0_x_pocket,
            "loss_0_h": loss_0_h,
            "neg_log_constants": neg_log_constants,
            "kl_prior": kl_prior,
            "log_pN": self.log_pN(ligand["size"], pocket["size"]),
        }

    # -- sampling -------------------------------------------------------- #
    def sample_p_zt_given_zs(
        self, zs_lig, zs_pocket, ligand_mask, pocket_mask, gamma_t, gamma_s
    ):
        _, sigma_t_given_s, alpha_t_given_s = self.sigma_and_alpha_t_given_s(
            gamma_t, gamma_s, zs_lig
        )
        zt_lig, zt_pocket = self.sample_normal(
            alpha_t_given_s[ligand_mask] * zs_lig,
            alpha_t_given_s[pocket_mask] * zs_pocket,
            sigma_t_given_s,
            ligand_mask,
            pocket_mask,
        )
        return self._recentre(zt_lig, zt_pocket, ligand_mask, pocket_mask)

    def _recentre(self, z_lig, z_pocket, ligand_mask, pocket_mask):
        z_x = self.remove_mean_batch(
            torch.cat(
                (z_lig[:, : self.n_dims], z_pocket[:, : self.n_dims]), dim=0
            ),
            torch.cat((ligand_mask, pocket_mask)),
        )
        n = len(ligand_mask)
        return (
            torch.cat((z_x[:n], z_lig[:, self.n_dims :]), dim=1),
            torch.cat((z_x[n:], z_pocket[:, self.n_dims :]), dim=1),
        )

    def sample_p_zs_given_zt(
        self, s, t, zt_lig, zt_pocket, ligand_mask, pocket_mask
    ):
        gamma_s, gamma_t = self.gamma(s), self.gamma(t)
        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = (
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt_lig)
        )
        sigma_s = self.sigma(gamma_s, target_tensor=zt_lig)
        sigma_t = self.sigma(gamma_t, target_tensor=zt_lig)

        eps_t_lig, eps_t_pocket = self.dynamics(
            zt_lig, zt_pocket, t, ligand_mask, pocket_mask
        )
        scale = sigma2_t_given_s / alpha_t_given_s / sigma_t
        mu_lig = zt_lig / alpha_t_given_s[ligand_mask] - scale[ligand_mask] * eps_t_lig
        mu_pocket = (
            zt_pocket / alpha_t_given_s[pocket_mask]
            - scale[pocket_mask] * eps_t_pocket
        )
        zs_lig, zs_pocket = self.sample_normal(
            mu_lig,
            mu_pocket,
            sigma_t_given_s * sigma_s / sigma_t,
            ligand_mask,
            pocket_mask,
        )
        return self._recentre(zs_lig, zs_pocket, ligand_mask, pocket_mask)

    def sample_p_xh_given_z0(
        self, z0_lig, z0_pocket, lig_mask, pocket_mask, batch_size
    ):
        t_zeros = torch.zeros(size=(batch_size, 1), device=z0_lig.device)
        gamma_0 = self.gamma(t_zeros)
        sigma_x = self.SNR(-0.5 * gamma_0)
        net_out_lig, net_out_pocket = self.dynamics(
            z0_lig, z0_pocket, t_zeros, lig_mask, pocket_mask
        )
        mu_x_lig = self.compute_x_pred(net_out_lig, z0_lig, gamma_0, lig_mask)
        mu_x_pocket = self.compute_x_pred(
            net_out_pocket, z0_pocket, gamma_0, pocket_mask
        )
        xh_lig, xh_pocket = self.sample_normal(
            mu_x_lig, mu_x_pocket, sigma_x, lig_mask, pocket_mask
        )
        x_lig, h_lig = self.unnormalize(
            xh_lig[:, : self.n_dims], z0_lig[:, self.n_dims :]
        )
        x_pocket, h_pocket = self.unnormalize(
            xh_pocket[:, : self.n_dims], z0_pocket[:, self.n_dims :]
        )
        h_lig = F.one_hot(torch.argmax(h_lig, dim=1), self.atom_nf)
        h_pocket = F.one_hot(torch.argmax(h_pocket, dim=1), self.residue_nf)
        return x_lig, h_lig, x_pocket, h_pocket

    @staticmethod
    def get_repaint_schedule(resamplings: int, jump_length: int, timesteps: int):
        """RePaint jump schedule: denoising steps before each jump back."""
        schedule: list = []
        curr_t = 0
        while curr_t < timesteps:
            if curr_t + jump_length < timesteps:
                if schedule:
                    schedule[-1] += jump_length
                    schedule.extend([jump_length] * (resamplings - 1))
                else:
                    schedule.extend([jump_length] * resamplings)
                curr_t += jump_length
            else:
                residual = timesteps - curr_t
                if schedule:
                    schedule[-1] += residual
                else:
                    schedule.append(residual)
                curr_t += residual
        return list(reversed(schedule))

    @torch.no_grad()
    def inpaint(
        self,
        ligand: Dict,
        pocket: Dict,
        lig_fixed: torch.Tensor,
        pocket_fixed: torch.Tensor,
        resamplings: int = 1,
        jump_length: int = 1,
        timesteps: Optional[int] = None,
        **_kwargs: Any,
    ):
        """RePaint sampling with parts of the system held fixed.

        In joint mode this is ALSO the de-novo pocket-conditioned path:
        ``lig_fixed=0`` / ``pocket_fixed=1`` reproduces upstream's
        ``lightning_modules.py:814-834``. Lugmayr et al., CVPR 2022.
        """
        timesteps = self.T if timesteps is None else timesteps
        if lig_fixed.dim() == 1:
            lig_fixed = lig_fixed.unsqueeze(1)
        if pocket_fixed.dim() == 1:
            pocket_fixed = pocket_fixed.unsqueeze(1)

        ligand, pocket = self.normalize(ligand, pocket)
        n_samples = len(ligand["size"])
        combined_mask = torch.cat((ligand["mask"], pocket["mask"]))
        xh0_lig = torch.cat([ligand["x"], ligand["one_hot"]], dim=1)
        xh0_pocket = torch.cat([pocket["x"], pocket["one_hot"]], dim=1)

        lig_known = lig_fixed.bool().view(-1)
        pocket_known = pocket_fixed.bool().view(-1)
        mean_known = scatter_mean(
            torch.cat((ligand["x"][lig_known], pocket["x"][pocket_known])),
            torch.cat(
                (ligand["mask"][lig_known], pocket["mask"][pocket_known])
            ),
            dim=0,
        )
        xh0_lig[:, : self.n_dims] -= mean_known[ligand["mask"]]
        xh0_pocket[:, : self.n_dims] -= mean_known[pocket["mask"]]

        z_lig, z_pocket = self.sample_combined_position_feature_noise(
            ligand["mask"], pocket["mask"]
        )

        schedule = self.get_repaint_schedule(resamplings, jump_length, timesteps)
        s = timesteps - 1
        for i, n_denoise_steps in enumerate(schedule):
            for j in range(n_denoise_steps):
                s_array = torch.full(
                    (n_samples, 1), fill_value=s, device=z_lig.device
                )
                t_array = (s_array + 1) / timesteps
                s_array = s_array / timesteps

                gamma_s = self.inflate_batch_array(
                    self.gamma(s_array), ligand["x"]
                )
                z_lig_known, z_pocket_known, _, _ = self.noised_representation(
                    xh0_lig, xh0_pocket, ligand["mask"], pocket["mask"], gamma_s
                )
                z_lig_unknown, z_pocket_unknown = self.sample_p_zs_given_zt(
                    s_array, t_array, z_lig, z_pocket, ligand["mask"], pocket["mask"]
                )

                known_mask = torch.cat(
                    (ligand["mask"][lig_known], pocket["mask"][pocket_known])
                )
                com_noised = scatter_mean(
                    torch.cat(
                        (
                            z_lig_known[:, : self.n_dims][lig_known],
                            z_pocket_known[:, : self.n_dims][pocket_known],
                        )
                    ),
                    known_mask,
                    dim=0,
                )
                com_denoised = scatter_mean(
                    torch.cat(
                        (
                            z_lig_unknown[:, : self.n_dims][lig_known],
                            z_pocket_unknown[:, : self.n_dims][pocket_known],
                        )
                    ),
                    known_mask,
                    dim=0,
                )
                dx = com_denoised - com_noised
                z_lig_known[:, : self.n_dims] += dx[ligand["mask"]]
                z_pocket_known[:, : self.n_dims] += dx[pocket["mask"]]

                z_lig = z_lig_known * lig_fixed + z_lig_unknown * (1 - lig_fixed)
                z_pocket = z_pocket_known * pocket_fixed + z_pocket_unknown * (
                    1 - pocket_fixed
                )
                self.assert_mean_zero_with_mask(
                    torch.cat(
                        (z_lig[:, : self.n_dims], z_pocket[:, : self.n_dims]),
                        dim=0,
                    ),
                    combined_mask,
                )

                if j == n_denoise_steps - 1 and i < len(schedule) - 1:
                    t = s + jump_length
                    t_array = (
                        torch.full(
                            (n_samples, 1), fill_value=t, device=z_lig.device
                        )
                        / timesteps
                    )
                    gamma_s = self.inflate_batch_array(
                        self.gamma(s_array), ligand["x"]
                    )
                    gamma_t = self.inflate_batch_array(
                        self.gamma(t_array), ligand["x"]
                    )
                    z_lig, z_pocket = self.sample_p_zt_given_zs(
                        z_lig,
                        z_pocket,
                        ligand["mask"],
                        pocket["mask"],
                        gamma_t,
                        gamma_s,
                    )
                    s = t
                s -= 1

        x_lig, h_lig, x_pocket, h_pocket = self.sample_p_xh_given_z0(
            z_lig, z_pocket, ligand["mask"], pocket["mask"], n_samples
        )
        x = torch.cat((x_lig, x_pocket))
        if scatter_add(x, combined_mask, dim=0).abs().max().item() > 5e-2:
            x = self.remove_mean_batch(x, combined_mask)
            x_lig, x_pocket = x[: len(x_lig)], x[len(x_lig) :]

        return (
            torch.cat([x_lig, h_lig], dim=1),
            torch.cat([x_pocket, h_pocket], dim=1),
            ligand["mask"],
            pocket["mask"],
        )


# --------------------------------------------------------------------------- #
# Conditional model
# --------------------------------------------------------------------------- #


class ConditionalDDPM(EnVariationalDiffusion):
    """``mode: pocket_conditioning`` -- only the ligand is diffused."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.dynamics.update_pocket_coords:
            raise ValueError(
                "ConditionalDDPM requires update_pocket_coords=False; build "
                "the dynamics with update_pocket_coords=(mode == 'joint')."
            )

    # -- overrides ------------------------------------------------------- #
    @classmethod
    def remove_mean_batch(  # type: ignore[override]
        cls, x_lig, x_pocket, lig_indices, pocket_indices
    ):
        """Subtract only the *sampled* part's CoM (the pocket rides along)."""
        mean = scatter_mean(x_lig, lig_indices, dim=0)
        return x_lig - mean[lig_indices], x_pocket - mean[pocket_indices]

    def log_pN(self, n_lig: torch.Tensor, n_pocket: torch.Tensor) -> torch.Tensor:
        return self.size_distribution.log_prob_n1_given_n2(n_lig, n_pocket)

    def delta_log_px(self, num_nodes: torch.Tensor) -> torch.Tensor:
        return -self.subspace_dimensionality(num_nodes) * np.log(self.norm_values[0])

    def sample_normal(self, *args: Any, **kwargs: Any):  # noqa: D102
        raise NotImplementedError("replaced by sample_normal_zero_com()")

    def sample_combined_position_feature_noise(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("use sample_normal_zero_com() instead")

    def sample_normal_zero_com(
        self, mu_lig, xh0_pocket, sigma, lig_mask, pocket_mask
    ):
        eps_lig = self.sample_gaussian(
            (len(lig_mask), self.n_dims + self.atom_nf), lig_mask.device
        )
        out_lig = mu_lig + sigma[lig_mask] * eps_lig
        xh_pocket = xh0_pocket.detach().clone()
        out_lig[:, : self.n_dims], xh_pocket[:, : self.n_dims] = (
            self.remove_mean_batch(
                out_lig[:, : self.n_dims],
                xh0_pocket[:, : self.n_dims],
                lig_mask,
                pocket_mask,
            )
        )
        return out_lig, xh_pocket

    def noised_representation(  # type: ignore[override]
        self, xh_lig, xh0_pocket, lig_mask, pocket_mask, gamma_t
    ):
        alpha_t = self.alpha(gamma_t, xh_lig)
        sigma_t = self.sigma(gamma_t, xh_lig)
        eps_lig = self.sample_gaussian(
            (len(lig_mask), self.n_dims + self.atom_nf), lig_mask.device
        )
        z_t_lig = alpha_t[lig_mask] * xh_lig + sigma_t[lig_mask] * eps_lig
        xh_pocket = xh0_pocket.detach().clone()
        z_t_lig[:, : self.n_dims], xh_pocket[:, : self.n_dims] = (
            self.remove_mean_batch(
                z_t_lig[:, : self.n_dims],
                xh_pocket[:, : self.n_dims],
                lig_mask,
                pocket_mask,
            )
        )
        return z_t_lig, xh_pocket, eps_lig

    def kl_prior(self, xh_lig, mask_lig, num_nodes):
        ones = torch.ones((len(num_nodes), 1), device=xh_lig.device)
        gamma_T = self.gamma(ones)
        mu_T_lig = self.alpha(gamma_T, xh_lig)[mask_lig] * xh_lig
        mu_T_lig_x = mu_T_lig[:, : self.n_dims]
        mu_T_lig_h = mu_T_lig[:, self.n_dims :]

        sigma_T_x = self.sigma(gamma_T, mu_T_lig_x).squeeze()
        sigma_T_h = self.sigma(gamma_T, mu_T_lig_h).squeeze()

        kl_h = self.gaussian_KL(
            self.sum_except_batch(mu_T_lig_h**2, mask_lig),
            sigma_T_h,
            torch.ones_like(sigma_T_h),
            d=1,
        )
        kl_x = self.gaussian_KL(
            self.sum_except_batch(mu_T_lig_x**2, mask_lig),
            sigma_T_x,
            torch.ones_like(sigma_T_x),
            self.subspace_dimensionality(num_nodes),
        )
        return kl_x + kl_h

    def log_pxh_given_z0_without_constants(  # type: ignore[override]
        self, ligand, z_0_lig, eps_lig, net_out_lig, gamma_0, epsilon: float = 1e-10
    ):
        sigma_0_cat = self.sigma(gamma_0, target_tensor=z_0_lig) * self.norm_values[1]
        log_p_x = -0.5 * self.sum_except_batch(
            (eps_lig[:, : self.n_dims] - net_out_lig[:, : self.n_dims]) ** 2,
            ligand["mask"],
        )
        onehot = ligand["one_hot"] * self.norm_values[1] + self.norm_biases[1]
        estimated = (
            z_0_lig[:, self.n_dims :] * self.norm_values[1] + self.norm_biases[1]
        )
        centered = estimated - 1
        log_p_prop = torch.log(
            self.cdf_standard_gaussian(
                (centered + 0.5) / sigma_0_cat[ligand["mask"]]
            )
            - self.cdf_standard_gaussian(
                (centered - 0.5) / sigma_0_cat[ligand["mask"]]
            )
            + epsilon
        )
        log_probs = log_p_prop - torch.logsumexp(log_p_prop, dim=1, keepdim=True)
        return log_p_x, self.sum_except_batch(log_probs * onehot, ligand["mask"])

    def forward(self, ligand: Dict, pocket: Dict) -> Dict[str, Any]:
        ligand, pocket = self.normalize(ligand, pocket)
        delta_log_px = self.delta_log_px(ligand["size"])

        lowest_t = 0 if self.training else 1
        t_int = torch.randint(
            lowest_t,
            self.T + 1,
            size=(ligand["size"].size(0), 1),
            device=ligand["x"].device,
        ).float()
        s_int = t_int - 1
        t_is_zero = (t_int == 0).float()
        t_is_not_zero = 1 - t_is_zero
        s, t = s_int / self.T, t_int / self.T

        gamma_s = self.inflate_batch_array(self.gamma(s), ligand["x"])
        gamma_t = self.inflate_batch_array(self.gamma(t), ligand["x"])

        xh0_lig = torch.cat([ligand["x"], ligand["one_hot"]], dim=1)
        xh0_pocket = torch.cat([pocket["x"], pocket["one_hot"]], dim=1)
        xh0_lig[:, : self.n_dims], xh0_pocket[:, : self.n_dims] = (
            self.remove_mean_batch(
                xh0_lig[:, : self.n_dims],
                xh0_pocket[:, : self.n_dims],
                ligand["mask"],
                pocket["mask"],
            )
        )

        z_t_lig, xh_pocket, eps_t_lig = self.noised_representation(
            xh0_lig, xh0_pocket, ligand["mask"], pocket["mask"], gamma_t
        )
        net_out_lig, _ = self.dynamics(
            z_t_lig, xh_pocket, t, ligand["mask"], pocket["mask"]
        )

        error_t_lig = self.sum_except_batch(
            (eps_t_lig - net_out_lig) ** 2, ligand["mask"]
        )
        SNR_weight = (1 - self.SNR(gamma_s - gamma_t)).squeeze(1)  # noqa: N806
        neg_log_constants = -self.log_constants_p_x_given_z0(
            n_nodes=ligand["size"], device=error_t_lig.device
        )
        kl_prior = self.kl_prior(xh0_lig, ligand["mask"], ligand["size"])

        if self.training:
            log_p_x, log_ph = self.log_pxh_given_z0_without_constants(
                ligand, z_t_lig, eps_t_lig, net_out_lig, gamma_t
            )
            loss_0_x_ligand = -log_p_x * t_is_zero.squeeze()
            loss_0_h = -log_ph * t_is_zero.squeeze()
            error_t_lig = error_t_lig * t_is_not_zero.squeeze()
        else:
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), ligand["x"])
            z_0_lig, xh_pocket_0, eps_0_lig = self.noised_representation(
                xh0_lig, xh0_pocket, ligand["mask"], pocket["mask"], gamma_0
            )
            net_out_0_lig, _ = self.dynamics(
                z_0_lig, xh_pocket_0, t_zeros, ligand["mask"], pocket["mask"]
            )
            log_p_x, log_ph = self.log_pxh_given_z0_without_constants(
                ligand, z_0_lig, eps_0_lig, net_out_0_lig, gamma_0
            )
            loss_0_x_ligand = -log_p_x
            loss_0_h = -log_ph

        zero = torch.zeros_like(error_t_lig)
        return {
            "delta_log_px": delta_log_px,
            "error_t_lig": error_t_lig,
            # constant 0: the pocket is never noised in this mode. A joint run
            # yields a real number here -- that is the smoke test's proof that
            # `mode` took effect.
            "error_t_pocket": zero,
            "SNR_weight": SNR_weight,
            "loss_0_x_ligand": loss_0_x_ligand,
            "loss_0_x_pocket": zero,
            "loss_0_h": loss_0_h,
            "neg_log_constants": neg_log_constants,
            "kl_prior": kl_prior,
            "log_pN": self.log_pN(ligand["size"], pocket["size"]),
        }

    # -- sampling -------------------------------------------------------- #
    def sample_p_zt_given_zs(  # type: ignore[override]
        self, zs_lig, xh0_pocket, ligand_mask, pocket_mask, gamma_t, gamma_s
    ):
        _, sigma_t_given_s, alpha_t_given_s = self.sigma_and_alpha_t_given_s(
            gamma_t, gamma_s, zs_lig
        )
        return self.sample_normal_zero_com(
            alpha_t_given_s[ligand_mask] * zs_lig,
            xh0_pocket,
            sigma_t_given_s,
            ligand_mask,
            pocket_mask,
        )

    def sample_p_zs_given_zt(  # type: ignore[override]
        self, s, t, zt_lig, xh0_pocket, ligand_mask, pocket_mask
    ):
        gamma_s, gamma_t = self.gamma(s), self.gamma(t)
        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = (
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt_lig)
        )
        sigma_s = self.sigma(gamma_s, target_tensor=zt_lig)
        sigma_t = self.sigma(gamma_t, target_tensor=zt_lig)

        eps_t_lig, _ = self.dynamics(
            zt_lig, xh0_pocket, t, ligand_mask, pocket_mask
        )
        scale = sigma2_t_given_s / alpha_t_given_s / sigma_t
        mu_lig = zt_lig / alpha_t_given_s[ligand_mask] - scale[ligand_mask] * eps_t_lig
        zs_lig, xh0_pocket = self.sample_normal_zero_com(
            mu_lig,
            xh0_pocket,
            sigma_t_given_s * sigma_s / sigma_t,
            ligand_mask,
            pocket_mask,
        )
        self.assert_mean_zero_with_mask(zt_lig[:, : self.n_dims], ligand_mask)
        return zs_lig, xh0_pocket

    def sample_p_xh_given_z0(  # type: ignore[override]
        self, z0_lig, xh0_pocket, lig_mask, pocket_mask, batch_size
    ):
        t_zeros = torch.zeros(size=(batch_size, 1), device=z0_lig.device)
        gamma_0 = self.gamma(t_zeros)
        sigma_x = self.SNR(-0.5 * gamma_0)
        net_out_lig, _ = self.dynamics(
            z0_lig, xh0_pocket, t_zeros, lig_mask, pocket_mask
        )
        mu_x_lig = self.compute_x_pred(net_out_lig, z0_lig, gamma_0, lig_mask)
        xh_lig, xh0_pocket = self.sample_normal_zero_com(
            mu_x_lig, xh0_pocket, sigma_x, lig_mask, pocket_mask
        )
        x_lig, h_lig = self.unnormalize(
            xh_lig[:, : self.n_dims], z0_lig[:, self.n_dims :]
        )
        x_pocket, h_pocket = self.unnormalize(
            xh0_pocket[:, : self.n_dims], xh0_pocket[:, self.n_dims :]
        )
        h_lig = F.one_hot(torch.argmax(h_lig, dim=1), self.atom_nf)
        return x_lig, h_lig, x_pocket, h_pocket

    def sample(self, *args: Any, **kwargs: Any):  # noqa: D102
        raise NotImplementedError(
            "the conditional model cannot sample without a pocket"
        )

    @torch.no_grad()
    def sample_given_pocket(
        self,
        pocket: Dict,
        num_nodes_lig: torch.Tensor,
        timesteps: Optional[int] = None,
        **_kwargs: Any,
    ):
        """De-novo ligand generation inside a fixed pocket."""
        timesteps = self.T if timesteps is None else timesteps
        n_samples = len(pocket["size"])
        device = pocket["x"].device

        _, pocket = self.normalize(pocket=pocket)
        xh0_pocket = torch.cat([pocket["x"], pocket["one_hot"]], dim=1)
        lig_mask = torch.repeat_interleave(
            torch.arange(n_samples, device=device), num_nodes_lig.to(device)
        )

        mu_lig_x = scatter_mean(pocket["x"], pocket["mask"], dim=0)
        mu_lig_h = torch.zeros((n_samples, self.atom_nf), device=device)
        mu_lig = torch.cat((mu_lig_x, mu_lig_h), dim=1)[lig_mask]
        sigma = torch.ones_like(pocket["size"]).unsqueeze(1)

        z_lig, xh_pocket = self.sample_normal_zero_com(
            mu_lig, xh0_pocket, sigma, lig_mask, pocket["mask"]
        )
        self.assert_mean_zero_with_mask(z_lig[:, : self.n_dims], lig_mask)

        for s in reversed(range(timesteps)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=device)
            t_array = (s_array + 1) / timesteps
            s_array = s_array / timesteps
            z_lig, xh_pocket = self.sample_p_zs_given_zt(
                s_array, t_array, z_lig, xh_pocket, lig_mask, pocket["mask"]
            )

        x_lig, h_lig, x_pocket, h_pocket = self.sample_p_xh_given_z0(
            z_lig, xh_pocket, lig_mask, pocket["mask"], n_samples
        )
        if scatter_add(x_lig, lig_mask, dim=0).abs().max().item() > 5e-2:
            x_lig, x_pocket = self.remove_mean_batch(
                x_lig, x_pocket, lig_mask, pocket["mask"]
            )
        return (
            torch.cat([x_lig, h_lig], dim=1),
            torch.cat([x_pocket, h_pocket], dim=1),
            lig_mask,
            pocket["mask"],
        )

    @torch.no_grad()
    def inpaint(  # type: ignore[override]
        self,
        ligand: Dict,
        pocket: Dict,
        lig_fixed: torch.Tensor,
        resamplings: int = 1,
        timesteps: Optional[int] = None,
        center: str = "ligand",
        **_kwargs: Any,
    ):
        """RePaint sampling with ligand atoms held fixed (``inpaint.py:147``)."""
        timesteps = self.T if timesteps is None else timesteps
        if lig_fixed.dim() == 1:
            lig_fixed = lig_fixed.unsqueeze(1)

        n_samples = len(ligand["size"])
        device = pocket["x"].device
        ligand, pocket = self.normalize(ligand, pocket)

        xh0_pocket = torch.cat([pocket["x"], pocket["one_hot"]], dim=1)
        com_pocket_0 = scatter_mean(pocket["x"], pocket["mask"], dim=0)
        xh_ligand = torch.cat([ligand["x"], ligand["one_hot"]], dim=1)

        known = lig_fixed.bool().view(-1)
        if center == "ligand":
            mean_known = scatter_mean(
                ligand["x"][known], ligand["mask"][known], dim=0
            )
        elif center == "pocket":
            mean_known = com_pocket_0
        else:
            raise ValueError(f"center must be 'ligand' or 'pocket', got {center!r}")

        mu_lig_h = torch.zeros((n_samples, self.atom_nf), device=device)
        mu_lig = torch.cat((mean_known, mu_lig_h), dim=1)[ligand["mask"]]
        sigma = torch.ones_like(pocket["size"]).unsqueeze(1)
        z_lig, xh_pocket = self.sample_normal_zero_com(
            mu_lig, xh0_pocket, sigma, ligand["mask"], pocket["mask"]
        )

        for s in reversed(range(timesteps)):
            for u in range(resamplings):
                s_array = torch.full((n_samples, 1), fill_value=s, device=device)
                t_array = (s_array + 1) / timesteps
                s_array = s_array / timesteps
                gamma_t = self.gamma(t_array)
                gamma_s = self.gamma(s_array)

                z_lig_unknown, xh_pocket = self.sample_p_zs_given_zt(
                    s_array, t_array, z_lig, xh_pocket, ligand["mask"], pocket["mask"]
                )

                com_pocket = scatter_mean(
                    xh_pocket[:, : self.n_dims], pocket["mask"], dim=0
                )
                xh_ligand[:, : self.n_dims] = (
                    ligand["x"] + (com_pocket - com_pocket_0)[ligand["mask"]]
                )
                z_lig_known, xh_pocket, _ = self.noised_representation(
                    xh_ligand, xh_pocket, ligand["mask"], pocket["mask"], gamma_s
                )

                com_noised = scatter_mean(
                    z_lig_known[known][:, : self.n_dims],
                    ligand["mask"][known],
                    dim=0,
                )
                com_denoised = scatter_mean(
                    z_lig_unknown[known][:, : self.n_dims],
                    ligand["mask"][known],
                    dim=0,
                )
                dx = com_denoised - com_noised
                z_lig_known[:, : self.n_dims] += dx[ligand["mask"]]
                xh_pocket[:, : self.n_dims] += dx[pocket["mask"]]

                z_lig = z_lig_known * lig_fixed + z_lig_unknown * (1 - lig_fixed)

                if u < resamplings - 1:
                    z_lig, xh_pocket = self.sample_p_zt_given_zs(
                        z_lig,
                        xh_pocket,
                        ligand["mask"],
                        pocket["mask"],
                        gamma_t,
                        gamma_s,
                    )

        x_lig, h_lig, x_pocket, h_pocket = self.sample_p_xh_given_z0(
            z_lig, xh_pocket, ligand["mask"], pocket["mask"], n_samples
        )
        return (
            torch.cat([x_lig, h_lig], dim=1),
            torch.cat([x_pocket, h_pocket], dim=1),
            ligand["mask"],
            pocket["mask"],
        )
