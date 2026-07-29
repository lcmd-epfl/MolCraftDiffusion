"""Pocket-conditioned DDPM for DiffPharma.

Port of ``ConditionalDDPM`` from
``others/DiffPharma/equivariant_diffusion/conditional_model.py`` (EDM /
DiffSBDD lineage): epsilon parametrisation, predefined polynomial noise
schedule, only the ligand is noised while pocket and both pharmacophore
node sets stay clean and act as fixed context.

Dropped from the upstream file (all dead or broken in this repo's config):
``SimpleConditionalDDPM``, ``GammaNetwork`` (learned schedule),
``partially_noised_ligand`` / ``sample_p_zt_given_zs`` /
``sample_combined_position_feature_noise`` /
``sample_center_gravity_zero_gaussian_batch`` -- all of which call
pre-DiffPharma 2-node-set signatures and would raise on entry.

One upstream bug is fixed: the CoM-drift correction at the end of
``sample_given_pocket`` called ``remove_mean_batch`` with 4 of its 8
arguments, i.e. it crashed whenever it triggered.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_add, scatter_mean

from MolecularDiffusion.modules.models.diffpharma.distributions import (
    DistributionNodes,
)


def clip_noise_schedule(alphas2, clip_value=0.001):
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)
    alphas_step = alphas2[1:] / alphas2[:-1]
    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    return np.cumprod(alphas_step, axis=0)


def polynomial_schedule(timesteps: int, s=1e-4, power=3.0):
    """Noise schedule ``1 - x^power``."""
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power)) ** 2
    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)
    precision = 1 - 2 * s
    return precision * alphas2 + s


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas_cumprod = np.cumprod(1.0 - betas, axis=0)
    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)
    return alphas_cumprod


class PredefinedNoiseSchedule(nn.Module):
    """Lookup table for a non-learned noise schedule."""

    def __init__(self, noise_schedule, timesteps, precision):
        super().__init__()
        self.timesteps = timesteps

        if noise_schedule == "cosine":
            alphas2 = cosine_beta_schedule(timesteps)
        elif "polynomial" in noise_schedule:
            splits = noise_schedule.split("_")
            assert len(splits) == 2
            alphas2 = polynomial_schedule(
                timesteps, s=precision, power=float(splits[1])
            )
        else:
            raise ValueError(noise_schedule)

        log_alphas2_to_sigmas2 = np.log(alphas2) - np.log(1 - alphas2)
        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(), requires_grad=False
        )

    def forward(self, t):
        return self.gamma[torch.round(t * self.timesteps).long()]


class ConditionalDDPM(nn.Module):
    """Conditional diffusion module (pocket + pharmacophore particles as context)."""

    def __init__(
        self,
        dynamics,
        atom_nf,
        residue_nf,
        interh_nf,
        interhp_nf,
        n_dims,
        size_histogram,
        timesteps=1000,
        parametrization="eps",
        noise_schedule="polynomial_2",
        noise_precision=1e-4,
        loss_type="l2",
        norm_values=(1.0, 1.0),
        norm_biases=(None, 0.0),
        virtual_node_idx=None,
    ):
        super().__init__()
        assert parametrization == "eps", "Only the eps parametrization is supported"
        if noise_schedule == "learned":
            raise ValueError(
                "The learned (GammaNetwork) noise schedule is not ported; "
                "DiffPharma's released config uses 'polynomial_2'."
            )

        self.dynamics = dynamics
        self.atom_nf = atom_nf
        self.residue_nf = residue_nf
        self.interh_nf = interh_nf
        self.interhp_nf = interhp_nf
        self.n_dims = n_dims
        self.T = timesteps
        self.parametrization = parametrization
        self.loss_type = loss_type
        self.norm_values = norm_values
        self.norm_biases = norm_biases

        self.gamma = PredefinedNoiseSchedule(
            noise_schedule, timesteps=timesteps, precision=noise_precision
        )
        self.size_distribution = DistributionNodes(size_histogram)
        self.vnode_idx = virtual_node_idx
        self.check_issues_norm_values()

    def check_issues_norm_values(self, num_stdevs=8):
        zeros = torch.zeros((1, 1))
        sigma_0 = self.sigma(self.gamma(zeros), target_tensor=zeros).item()
        norm_value = self.norm_values[1]
        if sigma_0 * num_stdevs > 1.0 / norm_value:
            raise ValueError(
                f"Value for normalization value {norm_value} probably too "
                f"large with sigma_0 {sigma_0:.5f} and "
                f"1 / norm_value = {1.0 / norm_value}"
            )

    # -- schedule helpers ------------------------------------------------ #
    @staticmethod
    def inflate_batch_array(array, target):
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def sigma(self, gamma, target_tensor):
        return self.inflate_batch_array(
            torch.sqrt(torch.sigmoid(gamma)), target_tensor
        )

    def alpha(self, gamma, target_tensor):
        return self.inflate_batch_array(
            torch.sqrt(torch.sigmoid(-gamma)), target_tensor
        )

    @staticmethod
    def SNR(gamma):
        return torch.exp(-gamma)

    def sigma_and_alpha_t_given_s(self, gamma_t, gamma_s, target_tensor):
        sigma2_t_given_s = self.inflate_batch_array(
            -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t)), target_tensor
        )
        log_alpha2_t_given_s = F.logsigmoid(-gamma_t) - F.logsigmoid(-gamma_s)
        alpha_t_given_s = self.inflate_batch_array(
            torch.exp(0.5 * log_alpha2_t_given_s), target_tensor
        )
        return sigma2_t_given_s, torch.sqrt(sigma2_t_given_s), alpha_t_given_s

    # -- normalisation --------------------------------------------------- #
    def normalize(self, ligand=None, pocket=None):
        if ligand is not None:
            ligand["x"] = ligand["x"] / self.norm_values[0]
            ligand["one_hot"] = (
                ligand["one_hot"].float() - self.norm_biases[1]
            ) / self.norm_values[1]
        if pocket is not None:
            pocket["x"] = pocket["x"] / self.norm_values[0]
            pocket["one_hot"] = (
                pocket["one_hot"].float() - self.norm_biases[1]
            ) / self.norm_values[1]
        return ligand, pocket

    def unnormalize(self, x, h_cat):
        return x * self.norm_values[0], h_cat * self.norm_values[1] + self.norm_biases[1]

    def unnormalize_z(self, z_lig, z_pocket):
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

    def subspace_dimensionality(self, input_size):
        return (input_size - 1) * self.n_dims

    def delta_log_px(self, num_nodes):
        return -self.subspace_dimensionality(num_nodes) * np.log(self.norm_values[0])

    # -- static helpers -------------------------------------------------- #
    @staticmethod
    def gaussian_KL(q_mu_minus_p_mu_squared, q_sigma, p_sigma, d):
        return (
            d * torch.log(p_sigma / q_sigma)
            + 0.5 * (d * q_sigma**2 + q_mu_minus_p_mu_squared) / (p_sigma**2)
            - 0.5 * d
        )

    @staticmethod
    def sum_except_batch(x, indices):
        return scatter_add(x.sum(-1), indices, dim=0)

    @staticmethod
    def cdf_standard_gaussian(x):
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))

    @staticmethod
    def sample_gaussian(size, device):
        return torch.randn(size, device=device)

    @staticmethod
    def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
        largest_value = x.abs().max().item()
        error = scatter_add(x, node_mask, dim=0).abs().max().item()
        assert error / (largest_value + eps) < 1e-2, (
            f"Mean is not zero, relative_error {error / (largest_value + eps)}"
        )

    @classmethod
    def remove_mean_batch(
        cls,
        x_lig,
        x_pocket,
        x_interh,
        x_interhp,
        lig_indices,
        pocket_indices,
        interh_indices,
        interhp_indices,
    ):
        """Subtract the ligand centre of mass from every node set."""
        mean = scatter_mean(x_lig, lig_indices, dim=0)
        x_lig = x_lig - mean[lig_indices]
        x_pocket = x_pocket - mean[pocket_indices]
        if len(x_interh) != 0:
            x_interh = x_interh - mean[interh_indices]
        if len(x_interhp) != 0:
            x_interhp = x_interhp - mean[interhp_indices]
        return x_lig, x_pocket, x_interh, x_interhp

    # -- diffusion ------------------------------------------------------- #
    def kl_prior(self, xh_lig, mask_lig, num_nodes):
        batch_size = len(num_nodes)
        ones = torch.ones((batch_size, 1), device=xh_lig.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, xh_lig)

        mu_T_lig = alpha_T[mask_lig] * xh_lig
        mu_T_lig_x = mu_T_lig[:, : self.n_dims]
        mu_T_lig_h = mu_T_lig[:, self.n_dims :]

        sigma_T_x = self.sigma(gamma_T, mu_T_lig_x).squeeze()
        sigma_T_h = self.sigma(gamma_T, mu_T_lig_h).squeeze()

        mu_norm2 = self.sum_except_batch(mu_T_lig_h**2, mask_lig)
        kl_distance_h = self.gaussian_KL(
            mu_norm2, sigma_T_h, torch.ones_like(sigma_T_h), d=1
        )

        mu_norm2 = self.sum_except_batch(mu_T_lig_x**2, mask_lig)
        kl_distance_x = self.gaussian_KL(
            mu_norm2,
            sigma_T_x,
            torch.ones_like(sigma_T_x),
            self.subspace_dimensionality(num_nodes),
        )
        return kl_distance_x + kl_distance_h

    def compute_x_pred(self, net_out, zt, gamma_t, batch_mask):
        sigma_t = self.sigma(gamma_t, target_tensor=net_out)
        alpha_t = self.alpha(gamma_t, target_tensor=net_out)
        return 1.0 / alpha_t[batch_mask] * (zt - sigma_t[batch_mask] * net_out)

    def log_constants_p_x_given_z0(self, n_nodes, device):
        batch_size = len(n_nodes)
        degrees_of_freedom_x = self.subspace_dimensionality(n_nodes)
        gamma_0 = self.gamma(torch.zeros((batch_size, 1), device=device))
        log_sigma_x = 0.5 * gamma_0.view(batch_size)
        return degrees_of_freedom_x * (-log_sigma_x - 0.5 * np.log(2 * np.pi))

    def log_pxh_given_z0_without_constants(
        self, ligand, z_0_lig, eps_lig, net_out_lig, gamma_0, epsilon=1e-10
    ):
        z_h_lig = z_0_lig[:, self.n_dims :]
        eps_lig_x = eps_lig[:, : self.n_dims]
        net_lig_x = net_out_lig[:, : self.n_dims]

        sigma_0 = self.sigma(gamma_0, target_tensor=z_0_lig)
        sigma_0_cat = sigma_0 * self.norm_values[1]

        squared_error = (eps_lig_x - net_lig_x) ** 2
        if self.vnode_idx is not None:
            squared_error[
                ligand["one_hot"][:, self.vnode_idx].bool(), : self.n_dims
            ] = 0
        log_p_x_given_z0_without_constants_ligand = -0.5 * (
            self.sum_except_batch(squared_error, ligand["mask"])
        )

        ligand_onehot = ligand["one_hot"] * self.norm_values[1] + self.norm_biases[1]
        estimated_ligand_onehot = z_h_lig * self.norm_values[1] + self.norm_biases[1]
        centered_ligand_onehot = estimated_ligand_onehot - 1

        log_ph_cat_proportional_ligand = torch.log(
            self.cdf_standard_gaussian(
                (centered_ligand_onehot + 0.5) / sigma_0_cat[ligand["mask"]]
            )
            - self.cdf_standard_gaussian(
                (centered_ligand_onehot - 0.5) / sigma_0_cat[ligand["mask"]]
            )
            + epsilon
        )
        log_Z = torch.logsumexp(log_ph_cat_proportional_ligand, dim=1, keepdim=True)
        log_probabilities_ligand = log_ph_cat_proportional_ligand - log_Z
        log_ph_given_z0_ligand = self.sum_except_batch(
            log_probabilities_ligand * ligand_onehot, ligand["mask"]
        )
        return log_p_x_given_z0_without_constants_ligand, log_ph_given_z0_ligand

    def sample_normal_zero_com(
        self,
        mu_lig,
        xh0_pocket,
        xh0_interh,
        xh0_interhp,
        sigma,
        lig_mask,
        pocket_mask,
        interh_mask,
        interhp_mask,
        fix_noise=False,
    ):
        if fix_noise:
            raise NotImplementedError("fix_noise option isn't implemented yet")

        eps_lig = self.sample_gaussian(
            size=(len(lig_mask), self.n_dims + self.atom_nf), device=lig_mask.device
        )
        out_lig = mu_lig + sigma[lig_mask] * eps_lig

        xh_pocket = xh0_pocket.detach().clone()
        xh_interh = xh0_interh.detach().clone()
        xh_interhp = xh0_interhp.detach().clone()
        (
            out_lig[:, : self.n_dims],
            xh_pocket[:, : self.n_dims],
            xh_interh[:, : self.n_dims],
            xh_interhp[:, : self.n_dims],
        ) = self.remove_mean_batch(
            out_lig[:, : self.n_dims],
            xh0_pocket[:, : self.n_dims],
            xh_interh[:, : self.n_dims],
            xh_interhp[:, : self.n_dims],
            lig_mask,
            pocket_mask,
            interh_mask,
            interhp_mask,
        )
        return out_lig, xh_pocket, xh_interh, xh_interhp

    def noised_representation(
        self,
        xh_lig,
        xh0_pocket,
        xh0_interh,
        xh0_interhp,
        lig_mask,
        pocket_mask,
        interh_mask,
        interhp_mask,
        gamma_t,
    ):
        alpha_t = self.alpha(gamma_t, xh_lig)
        sigma_t = self.sigma(gamma_t, xh_lig)

        eps_lig = self.sample_gaussian(
            size=(len(lig_mask), self.n_dims + self.atom_nf), device=lig_mask.device
        )
        z_t_lig = alpha_t[lig_mask] * xh_lig + sigma_t[lig_mask] * eps_lig

        xh_pocket = xh0_pocket.detach().clone()
        xh_interh = xh0_interh.detach().clone()
        xh_interhp = xh0_interhp.detach().clone()
        (
            z_t_lig[:, : self.n_dims],
            xh_pocket[:, : self.n_dims],
            xh_interh[:, : self.n_dims],
            xh_interhp[:, : self.n_dims],
        ) = self.remove_mean_batch(
            z_t_lig[:, : self.n_dims],
            xh_pocket[:, : self.n_dims],
            xh_interh[:, : self.n_dims],
            xh_interhp[:, : self.n_dims],
            lig_mask,
            pocket_mask,
            interh_mask,
            interhp_mask,
        )
        return z_t_lig, xh_pocket, xh_interh, xh_interhp, eps_lig

    def xh_given_zt_and_epsilon(self, z_t, epsilon, gamma_t, batch_mask):
        """Equation (7) in the EDM paper."""
        alpha_t = self.alpha(gamma_t, z_t)
        sigma_t = self.sigma(gamma_t, z_t)
        return (
            z_t / alpha_t[batch_mask]
            - epsilon * sigma_t[batch_mask] / alpha_t[batch_mask]
        )

    def log_pN(self, N_lig, N_pocket):
        return self.size_distribution.log_prob_n1_given_n2(N_lig, N_pocket)

    def forward(self, ligand, pocket, interh, interhp, return_info=False):
        """Compute the loss terms (see the task module for how they combine)."""
        ligand, pocket = self.normalize(ligand, pocket)
        _, interh = self.normalize(pocket=interh)
        _, interhp = self.normalize(pocket=interhp)
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

        s = s_int / self.T
        t = t_int / self.T

        gamma_s = self.inflate_batch_array(self.gamma(s), ligand["x"])
        gamma_t = self.inflate_batch_array(self.gamma(t), ligand["x"])

        xh0_lig = torch.cat([ligand["x"], ligand["one_hot"]], dim=1)
        xh0_pocket = torch.cat([pocket["x"], pocket["one_hot"]], dim=1)
        xh0_interh = torch.cat([interh["x"], interh["one_hot"]], dim=1)
        xh0_interhp = torch.cat([interhp["x"], interhp["one_hot"]], dim=1)

        (
            xh0_lig[:, : self.n_dims],
            xh0_pocket[:, : self.n_dims],
            xh0_interh[:, : self.n_dims],
            xh0_interhp[:, : self.n_dims],
        ) = self.remove_mean_batch(
            xh0_lig[:, : self.n_dims],
            xh0_pocket[:, : self.n_dims],
            xh0_interh[:, : self.n_dims],
            xh0_interhp[:, : self.n_dims],
            ligand["mask"],
            pocket["mask"],
            interh["mask"],
            interhp["mask"],
        )

        z_t_lig, xh_pocket, xh_interh, xh_interhp, eps_t_lig = (
            self.noised_representation(
                xh0_lig,
                xh0_pocket,
                xh0_interh,
                xh0_interhp,
                ligand["mask"],
                pocket["mask"],
                interh["mask"],
                interhp["mask"],
                gamma_t,
            )
        )

        net_out_lig, _ = self.dynamics(
            z_t_lig,
            xh_pocket,
            xh_interh,
            xh_interhp,
            t,
            ligand["mask"],
            pocket["mask"],
            interh["mask"],
            interhp["mask"],
        )

        xh_lig_hat = self.xh_given_zt_and_epsilon(
            z_t_lig, net_out_lig, gamma_t, ligand["mask"]
        )

        squared_error = (eps_t_lig - net_out_lig) ** 2
        if self.vnode_idx is not None:
            squared_error[
                ligand["one_hot"][:, self.vnode_idx].bool(), : self.n_dims
            ] = 0
        error_t_lig = self.sum_except_batch(squared_error, ligand["mask"])
        SNR_weight = (1 - self.SNR(gamma_s - gamma_t)).squeeze(1)

        neg_log_constants = -self.log_constants_p_x_given_z0(
            n_nodes=ligand["size"], device=error_t_lig.device
        )
        kl_prior = self.kl_prior(xh0_lig, ligand["mask"], ligand["size"])

        if self.training:
            (
                log_p_x_given_z0_without_constants_ligand,
                log_ph_given_z0,
            ) = self.log_pxh_given_z0_without_constants(
                ligand, z_t_lig, eps_t_lig, net_out_lig, gamma_t
            )
            loss_0_x_ligand = (
                -log_p_x_given_z0_without_constants_ligand * t_is_zero.squeeze()
            )
            loss_0_h = -log_ph_given_z0 * t_is_zero.squeeze()
            error_t_lig = error_t_lig * t_is_not_zero.squeeze()
        else:
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), ligand["x"])
            z_0_lig, xh_pocket, xh_interh, xh_interhp, eps_0_lig = (
                self.noised_representation(
                    xh0_lig,
                    xh0_pocket,
                    xh0_interh,
                    xh0_interhp,
                    ligand["mask"],
                    pocket["mask"],
                    interh["mask"],
                    interhp["mask"],
                    gamma_0,
                )
            )
            net_out_0_lig, _ = self.dynamics(
                z_0_lig,
                xh_pocket,
                xh_interh,
                xh_interhp,
                t_zeros,
                ligand["mask"],
                pocket["mask"],
                interh["mask"],
                interhp["mask"],
            )
            (
                log_p_x_given_z0_without_constants_ligand,
                log_ph_given_z0,
            ) = self.log_pxh_given_z0_without_constants(
                ligand, z_0_lig, eps_0_lig, net_out_0_lig, gamma_0
            )
            loss_0_x_ligand = -log_p_x_given_z0_without_constants_ligand
            loss_0_h = -log_ph_given_z0

        log_pN = self.log_pN(ligand["size"], pocket["size"])
        alpha_t = self.alpha(gamma_t, z_t_lig)

        info = {
            "eps_hat_lig_x": scatter_mean(
                net_out_lig[:, : self.n_dims].abs().mean(1), ligand["mask"], dim=0
            ).mean(),
            "eps_hat_lig_h": scatter_mean(
                net_out_lig[:, self.n_dims :].abs().mean(1), ligand["mask"], dim=0
            ).mean(),
        }
        loss_terms = (
            delta_log_px,
            error_t_lig,
            torch.tensor(0.0),
            SNR_weight,
            loss_0_x_ligand,
            torch.tensor(0.0),
            loss_0_h,
            neg_log_constants,
            kl_prior,
            log_pN,
            t_int.squeeze(),
            xh_lig_hat,
            alpha_t,
        )
        return (*loss_terms, info) if return_info else loss_terms

    # -- sampling -------------------------------------------------------- #
    def sample_p_zs_given_zt(
        self,
        s,
        t,
        zt_lig,
        xh0_pocket,
        xh0_interh,
        xh0_interhp,
        ligand_mask,
        pocket_mask,
        interh_mask,
        interhp_mask,
        fix_noise=False,
    ):
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)
        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = (
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt_lig)
        )
        sigma_s = self.sigma(gamma_s, target_tensor=zt_lig)
        sigma_t = self.sigma(gamma_t, target_tensor=zt_lig)

        eps_t_lig, _ = self.dynamics(
            zt_lig,
            xh0_pocket,
            xh0_interh,
            xh0_interhp,
            t,
            ligand_mask,
            pocket_mask,
            interh_mask,
            interhp_mask,
        )

        mu_lig = (
            zt_lig / alpha_t_given_s[ligand_mask]
            - (sigma2_t_given_s / alpha_t_given_s / sigma_t)[ligand_mask] * eps_t_lig
        )
        sigma = sigma_t_given_s * sigma_s / sigma_t

        zs_lig, xh0_pocket, xh0_interh, xh0_interhp = self.sample_normal_zero_com(
            mu_lig,
            xh0_pocket,
            xh0_interh,
            xh0_interhp,
            sigma,
            ligand_mask,
            pocket_mask,
            interh_mask,
            interhp_mask,
            fix_noise,
        )
        self.assert_mean_zero_with_mask(zt_lig[:, : self.n_dims], ligand_mask)
        return zs_lig, xh0_pocket, xh0_interh, xh0_interhp

    def sample_p_xh_given_z0(
        self,
        z0_lig,
        xh0_pocket,
        xh0_interh,
        xh0_interhp,
        lig_mask,
        pocket_mask,
        interh_mask,
        interhp_mask,
        batch_size,
        fix_noise=False,
    ):
        t_zeros = torch.zeros(size=(batch_size, 1), device=z0_lig.device)
        gamma_0 = self.gamma(t_zeros)
        sigma_x = self.SNR(-0.5 * gamma_0)

        net_out_lig, _ = self.dynamics(
            z0_lig,
            xh0_pocket,
            xh0_interh,
            xh0_interhp,
            t_zeros,
            lig_mask,
            pocket_mask,
            interh_mask,
            interhp_mask,
        )
        mu_x_lig = self.compute_x_pred(net_out_lig, z0_lig, gamma_0, lig_mask)
        xh_lig, xh0_pocket, xh0_interh, xh0_interhp = self.sample_normal_zero_com(
            mu_x_lig,
            xh0_pocket,
            xh0_interh,
            xh0_interhp,
            sigma_x,
            lig_mask,
            pocket_mask,
            interh_mask,
            interhp_mask,
            fix_noise,
        )

        x_lig, h_lig = self.unnormalize(
            xh_lig[:, : self.n_dims], z0_lig[:, self.n_dims :]
        )
        x_pocket, h_pocket = self.unnormalize(
            xh0_pocket[:, : self.n_dims], xh0_pocket[:, self.n_dims :]
        )
        x_interh, h_interh = self.unnormalize(
            xh0_interh[:, : self.n_dims], xh0_interh[:, self.n_dims :]
        )
        x_interhp, h_interhp = self.unnormalize(
            xh0_interhp[:, : self.n_dims], xh0_interhp[:, self.n_dims :]
        )
        h_lig = F.one_hot(torch.argmax(h_lig, dim=1), self.atom_nf)
        return (
            x_lig,
            h_lig,
            x_pocket,
            h_pocket,
            x_interh,
            h_interh,
            x_interhp,
            h_interhp,
        )

    def sample(self, *args, **kwargs):
        raise NotImplementedError(
            "Conditional model does not support sampling without given pocket."
        )

    @torch.no_grad()
    def sample_given_pocket(
        self, pocket, interh, interhp, num_nodes_lig, return_frames=1, timesteps=None
    ):
        """Reverse-diffuse a ligand inside ``pocket``.

        Returns ``(out_lig, out_pocket, lig_mask, pocket_mask)`` -- flat,
        scatter-masked tensors, NOT ``(B, N, .)``.
        """
        timesteps = self.T if timesteps is None else timesteps
        assert 0 < return_frames <= timesteps
        assert timesteps % return_frames == 0
        n_samples = len(pocket["size"])
        device = pocket["x"].device

        _, pocket = self.normalize(pocket=pocket)
        _, interh = self.normalize(pocket=interh)
        _, interhp = self.normalize(pocket=interhp)

        xh0_pocket = torch.cat([pocket["x"], pocket["one_hot"]], dim=1)
        xh0_interh = torch.cat([interh["x"], interh["one_hot"]], dim=1)
        xh0_interhp = torch.cat([interhp["x"], interhp["one_hot"]], dim=1)

        lig_mask = torch.repeat_interleave(
            torch.arange(n_samples, device=device), num_nodes_lig.to(device)
        )

        # prior centred on the pocket centroid
        mu_lig_x = scatter_mean(pocket["x"], pocket["mask"], dim=0)
        mu_lig_h = torch.zeros((n_samples, self.atom_nf), device=device)
        mu_lig = torch.cat((mu_lig_x, mu_lig_h), dim=1)[lig_mask]
        sigma = torch.ones_like(pocket["size"]).unsqueeze(1)

        z_lig, xh_pocket, xh_interh, xh_interhp = self.sample_normal_zero_com(
            mu_lig,
            xh0_pocket,
            xh0_interh,
            xh0_interhp,
            sigma,
            lig_mask,
            pocket["mask"],
            interh["mask"],
            interhp["mask"],
        )
        self.assert_mean_zero_with_mask(z_lig[:, : self.n_dims], lig_mask)

        out_lig = torch.zeros((return_frames,) + z_lig.size(), device=device)
        out_pocket = torch.zeros((return_frames,) + xh_pocket.size(), device=device)

        for s in reversed(range(0, timesteps)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=device)
            t_array = (s_array + 1) / timesteps
            s_array = s_array / timesteps

            z_lig, xh_pocket, xh_interh, xh_interhp = self.sample_p_zs_given_zt(
                s_array,
                t_array,
                z_lig,
                xh_pocket,
                xh_interh,
                xh_interhp,
                lig_mask,
                pocket["mask"],
                interh["mask"],
                interhp["mask"],
            )
            if (s * return_frames) % timesteps == 0:
                idx = (s * return_frames) // timesteps
                out_lig[idx], out_pocket[idx] = self.unnormalize_z(z_lig, xh_pocket)

        (
            x_lig,
            h_lig,
            x_pocket,
            h_pocket,
            x_interh,
            _h_interh,
            x_interhp,
            _h_interhp,
        ) = self.sample_p_xh_given_z0(
            z_lig,
            xh_pocket,
            xh_interh,
            xh_interhp,
            lig_mask,
            pocket["mask"],
            interh["mask"],
            interhp["mask"],
            n_samples,
        )
        self.assert_mean_zero_with_mask(x_lig, lig_mask)

        if return_frames == 1:
            max_cog = scatter_add(x_lig, lig_mask, dim=0).abs().max().item()
            if max_cog > 5e-2:
                print(
                    f"Warning CoG drift with error {max_cog:.3f}. Projecting "
                    f"the positions down."
                )
                # upstream passed 4 of 8 args here and crashed; fixed.
                x_lig, x_pocket, _, _ = self.remove_mean_batch(
                    x_lig,
                    x_pocket,
                    x_interh,
                    x_interhp,
                    lig_mask,
                    pocket["mask"],
                    interh["mask"],
                    interhp["mask"],
                )

        out_lig[0] = torch.cat([x_lig, h_lig], dim=1)
        out_pocket[0] = torch.cat([x_pocket, h_pocket], dim=1)
        return (
            out_lig.squeeze(0),
            out_pocket.squeeze(0),
            lig_mask,
            pocket["mask"],
        )
