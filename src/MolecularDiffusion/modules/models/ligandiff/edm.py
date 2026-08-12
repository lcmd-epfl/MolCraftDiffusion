"""LigandDiff EDM. Verbatim port of the target repo's ``src/edm.py``.

Continuous 3D Gaussian DDPM (epsilon parametrisation) over
``xh = cat(coords, one_hot)``, noised and denoised **only** on the
``ligand_diff`` rows; the ``context`` rows are re-pasted clean at every step.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add

from MolecularDiffusion.modules.models.ligandiff import utils
from MolecularDiffusion.modules.models.ligandiff.egnn import Dynamics
from MolecularDiffusion.modules.models.ligandiff.noise import (
    GammaNetwork,
    PredefinedNoiseSchedule,
)


class EDM(torch.nn.Module):
    """Equivariant diffusion model with two-mask (context/ligand) inpainting."""

    def __init__(
        self,
        dynamics: Dynamics,
        in_node_nf: int,
        n_dims: int,
        timesteps: int = 1000,
        noise_schedule: str = "learned",
        noise_precision: float = 1e-4,
        loss_type: str = "vlb",
        norm_values: tuple = (1.0, 1.0, 1.0),
        norm_biases: tuple = (None, 0.0, 0.0),
    ) -> None:
        super().__init__()
        if noise_schedule == "learned":
            assert loss_type == "vlb", (
                "A noise schedule can only be learned with a vlb objective"
            )
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(
                noise_schedule, timesteps=timesteps, precision=noise_precision
            )

        self.dynamics = dynamics
        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.T = timesteps
        self.norm_values = norm_values
        self.norm_biases = norm_biases

    def noised_representation(
        self, xh, ligand_diff, context, batch_seg, gamma_t
    ):
        alpha_t = self.alpha(gamma_t)
        sigma_t = self.sigma(gamma_t)
        eps_t = self.sample_combined_position_feature_noise(xh, ligand_diff)
        z_t = alpha_t[batch_seg] * xh + sigma_t[batch_seg] * eps_t
        z_t = xh * context + z_t * ligand_diff
        return z_t, eps_t

    def forward(
        self,
        x,
        h,
        context,
        ligand_diff,
        batch_seg,
        batch_size,
        ligand_group=None,
    ):
        x, h = self.normalize(x, h)
        xh = torch.cat([x, h], dim=1)
        delta_log_px = (
            self.n_dims
            * self.inflate_batch_array(ligand_diff, batch_seg)
            * np.log(self.norm_values[0])
        )
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(batch_size, 1), device=x.device
        ).float()
        s_int = t_int - 1
        t = t_int / self.T
        s = s_int / self.T

        t_is_zero = (t_int == 0).float()
        t_is_not_zero = 1 - t_is_zero

        gamma_t = self.gamma(t)
        gamma_s = self.gamma(s)
        z_t, eps_t = self.noised_representation(
            xh, ligand_diff, context, batch_seg, gamma_t
        )
        eps_t_hat = self.dynamics.forward(
            xh=z_t,
            t=t,
            ligand_diff=ligand_diff,
            ligand_group=ligand_group,
            batch_seg=batch_seg,
        )

        eps_t_hat = eps_t_hat * ligand_diff
        squared_error = (eps_t - eps_t_hat) ** 2
        error_t = self.inflate_batch_array(squared_error, batch_seg)
        SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1)
        assert error_t.size() == SNR_weight.size()
        neg_log_constants = -self.log_constant_of_p_x_given_z0(
            ligand_diff, batch_seg, batch_size
        )
        kl_prior = self.kl_prior(xh, ligand_diff, batch_seg)
        if self.training:
            (
                log_p_x_given_z0_without_constants,
                log_ph_given_z0,
            ) = self.log_p_xh_given_z0_without_constants(
                h, z_t, gamma_t, eps_t, eps_t_hat, ligand_diff, batch_seg
            )
            loss_0_x = (
                -log_p_x_given_z0_without_constants * t_is_zero.squeeze()
            )
            loss_0_h = -log_ph_given_z0 * t_is_zero.squeeze()
            error_t = error_t * t_is_not_zero.squeeze()
        else:
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.gamma(t_zeros)
            z_0, eps_0 = self.noised_representation(
                xh, ligand_diff, context, batch_seg, gamma_0
            )
            eps_0_hat = self.dynamics.forward(
                z_0, t_zeros, ligand_diff, ligand_group, batch_seg
            )
            eps_0_hat = eps_0_hat * ligand_diff
            (
                log_p_x_given_z0_without_constants,
                log_ph_given_z0,
            ) = self.log_p_xh_given_z0_without_constants(
                h, z_0, gamma_0, eps_0, eps_0_hat, ligand_diff, batch_seg
            )
            loss_0_x = -log_p_x_given_z0_without_constants
            loss_0_h = -log_ph_given_z0

        return (
            delta_log_px,
            error_t,
            SNR_weight,
            loss_0_x,
            loss_0_h,
            neg_log_constants,
            kl_prior,
        )

    def sample_normal(self, mu_xh, ligand_diff, sigma, batch_seg):
        eps = self.sample_combined_position_feature_noise(mu_xh, ligand_diff)
        return mu_xh + sigma[batch_seg] * eps

    @torch.no_grad()
    def sample_chain(
        self,
        x,
        h,
        context,
        ligand_diff,
        batch_seg,
        batch_size,
        ligand_group,
        keep_frames=None,
        timesteps=None,
    ):
        timesteps = self.T if timesteps is None else timesteps
        assert 0 < keep_frames <= timesteps
        assert timesteps % keep_frames == 0

        x, h = self.normalize(x, h)
        xh = torch.cat([x, h], dim=1)
        mu_x = scatter_add(x * context, batch_seg, dim=0) / scatter_add(
            context, batch_seg, dim=0
        )
        mu_h = torch.zeros((batch_size, self.in_node_nf), device=x.device)
        mu_xh = torch.cat([mu_x, mu_h], dim=1)[batch_seg]
        sigma = torch.ones((batch_size, 1), device=x.device)
        z = self.sample_normal(mu_xh, ligand_diff, sigma, batch_seg)
        z = xh * context + z * ligand_diff

        chain = torch.zeros((keep_frames,) + z.size(), device=z.device)

        for s in reversed(range(0, timesteps)):
            s_array = torch.full(
                (batch_size, 1), fill_value=s, device=z.device
            )
            t_array = s_array + 1
            s_array = s_array / timesteps
            t_array = t_array / timesteps
            z = self.sample_p_zs_given_zt_only_ligandDiff(
                s=s_array,
                t=t_array,
                z_t=z,
                context=context,
                ligand_diff=ligand_diff,
                batch_seg=batch_seg,
                ligand_group=ligand_group,
            )
            if (s * keep_frames) % timesteps == 0:
                write_index = (s * keep_frames) // timesteps
                chain[write_index] = self.unnormalize_z(z)

        x, h = self.sample_p_xh_given_z0_only_ligandDiff(
            z_0=z,
            context=context,
            ligand_diff=ligand_diff,
            batch_size=batch_size,
            batch_seg=batch_seg,
            ligand_group=ligand_group,
        )

        if keep_frames == 1:
            max_cog = scatter_add(x, batch_seg, dim=0).abs().max().item()
            if max_cog > 5e-2:
                x = utils.remove_partial_mean_with_mask(
                    x, ligand_diff, batch_seg
                )

        chain[0] = torch.cat([x, h], dim=1)

        return chain

    def sample_p_zs_given_zt_only_ligandDiff(  # noqa: N802
        self, s, t, z_t, context, ligand_diff, batch_seg, ligand_group
    ):
        """Sample ``z_s ~ p(z_s | z_t)``, ligand rows only."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        (
            sigma2_t_given_s,
            sigma_t_given_s,
            alpha_t_given_s,
        ) = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s)
        sigma_s = self.sigma(gamma_s)
        sigma_t = self.sigma(gamma_t)

        eps_hat = self.dynamics.forward(
            xh=z_t,
            t=t,
            ligand_diff=ligand_diff,
            ligand_group=ligand_group,
            batch_seg=batch_seg,
        )
        eps_hat = eps_hat * ligand_diff

        mu = (
            z_t / alpha_t_given_s[batch_seg]
            - (sigma2_t_given_s / alpha_t_given_s / sigma_t)[batch_seg]
            * eps_hat
        )
        sigma = sigma_t_given_s * sigma_s / sigma_t

        z_s = self.sample_normal(mu, ligand_diff, sigma, batch_seg)
        return z_t * context + z_s * ligand_diff

    def sample_p_xh_given_z0_only_ligandDiff(  # noqa: N802
        self, z_0, context, ligand_diff, batch_size, batch_seg, ligand_group
    ):
        """Sample ``x, h ~ p(x, h | z_0)``, ligand rows only."""
        zeros = torch.zeros(size=(batch_size, 1), device=z_0.device)
        gamma_0 = self.gamma(zeros)

        sigma_x = self.SNR(-0.5 * gamma_0)
        eps_hat = self.dynamics.forward(
            xh=z_0,
            t=zeros,
            ligand_diff=ligand_diff,
            ligand_group=ligand_group,
            batch_seg=batch_seg,
        )
        eps_hat = eps_hat * ligand_diff

        mu_x = self.compute_x_pred(
            eps_t=eps_hat, z_t=z_0, gamma_t=gamma_0, batch_seg=batch_seg
        )
        xh = self.sample_normal(mu_x, ligand_diff, sigma_x, batch_seg)
        xh = z_0 * context + xh * ligand_diff
        x, h = xh[:, : self.n_dims], xh[:, self.n_dims :]
        x, h = self.unnormalize(x, h)
        h = F.one_hot(torch.argmax(h, dim=1), self.in_node_nf)

        return x, h

    def compute_x_pred(self, eps_t, z_t, gamma_t, batch_seg):
        """Most likely prediction of x."""
        sigma_t = self.sigma(gamma_t)
        alpha_t = self.alpha(gamma_t)
        return 1.0 / alpha_t[batch_seg] * (z_t - sigma_t[batch_seg] * eps_t)

    def kl_prior(self, xh, mask, batch_seg):
        """KL between ``q(z_1 | x)`` and the prior ``N(0, 1)``."""
        batch_size = torch.max(batch_seg) + 1
        ones = torch.ones((batch_size, 1), device=xh.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T)
        mu_T = alpha_T[batch_seg].view(-1, 1) * xh
        mu_T_x, mu_T_h = mu_T[:, : self.n_dims], mu_T[:, self.n_dims :]
        sigma_T_x = self.sigma(gamma_T).squeeze(1)
        sigma_T_h = self.sigma(gamma_T).squeeze(1)

        zeros, ones = torch.zeros_like(mu_T_h), torch.ones_like(sigma_T_h)
        mu_norm2 = self.inflate_batch_array(
            (mu_T_h - zeros) ** 2 * mask, batch_seg
        )
        kl_distance_h = self.gaussian_kl(mu_norm2, sigma_T_h, ones, d=1)

        zeros, ones = torch.zeros_like(mu_T_x), torch.ones_like(sigma_T_x)
        mu_norm2 = self.inflate_batch_array(
            (mu_T_x - zeros) ** 2 * mask, batch_seg
        )
        d = self.n_dims * (self.inflate_batch_array(mask, batch_seg) - 1)
        kl_distance_x = self.gaussian_kl(mu_norm2, sigma_T_x, ones, d)
        return kl_distance_x + kl_distance_h

    def log_constant_of_p_x_given_z0(self, mask, batch_seg, batch_size):
        """Normalising constant of ``p(x | z_0)``."""
        degrees_of_freedom_x = self.n_dims * (
            self.inflate_batch_array(mask, batch_seg) - 1
        )
        zeros = torch.zeros((batch_size, 1), device=mask.device)
        gamma_0 = self.gamma(zeros)

        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (-log_sigma_x - 0.5 * np.log(2 * np.pi))

    def log_p_xh_given_z0_without_constants(
        self, h, z_0, gamma_0, eps, eps_hat, mask, batch_seg, epsilon=1e-10
    ):
        """Reconstruction terms for coordinates and atom types at t=0."""
        z_h = z_0[:, self.n_dims :]

        eps_x = eps[:, : self.n_dims]
        eps_hat_x = eps_hat[:, : self.n_dims]

        sigma_0 = self.sigma(gamma_0) * self.norm_values[1]

        squared_error = (eps_x - eps_hat_x) ** 2
        log_p_x_given_z_without_constants = -0.5 * self.inflate_batch_array(
            squared_error, batch_seg
        )

        h = h * self.norm_values[1] + self.norm_biases[1]
        estimated_h = z_h * self.norm_values[1] + self.norm_biases[1]

        centered_h = estimated_h - 1

        log_p_h_proportional = torch.log(
            self.cdf_standard_gaussian(
                (centered_h + 0.5) / sigma_0[batch_seg]
            )
            - self.cdf_standard_gaussian(
                (centered_h - 0.5) / sigma_0[batch_seg]
            )
            + epsilon
        )

        log_Z = torch.logsumexp(log_p_h_proportional, dim=1, keepdim=True)
        log_probabilities = log_p_h_proportional - log_Z

        log_p_h_given_z = self.inflate_batch_array(
            log_probabilities * h * mask, batch_seg
        )

        return log_p_x_given_z_without_constants, log_p_h_given_z

    def sample_combined_position_feature_noise(self, x, ligand_diff):
        """Gaussian noise on coordinates + features, ligand rows only."""
        z_x = torch.randn(x.shape[0], self.n_dims, device=x.device)
        z_h = torch.randn(x.shape[0], self.in_node_nf, device=x.device)
        return torch.cat([z_x, z_h], dim=1) * ligand_diff

    def normalize(self, x, h):
        """Scale coordinates and features to the model's working range."""
        new_x = x / self.norm_values[0]
        new_h = (h.float() - self.norm_biases[1]) / self.norm_values[1]
        return new_x, new_h

    def unnormalize(self, x, h):
        """Inverse of :meth:`normalize`."""
        new_x = x * self.norm_values[0]
        new_h = h * self.norm_values[1] + self.norm_biases[1]
        return new_x, new_h

    def unnormalize_z(self, z):
        """Unnormalize a concatenated ``[x | h]`` latent."""
        assert z.size(1) == self.n_dims + self.in_node_nf
        x, h = z[:, : self.n_dims], z[:, self.n_dims :]
        x, h = self.unnormalize(x, h)
        return torch.cat([x, h], dim=1)

    def sigma(self, gamma):
        """Compute sigma given gamma."""
        return torch.sqrt(torch.sigmoid(gamma))

    def alpha(self, gamma):
        """Compute alpha given gamma."""
        return torch.sqrt(torch.sigmoid(-gamma))

    def SNR(self, gamma):  # noqa: N802
        """Signal-to-noise ratio ``alpha^2 / sigma^2`` given gamma."""
        return torch.exp(-gamma)

    def sigma_and_alpha_t_given_s(
        self, gamma_t: torch.Tensor, gamma_s: torch.Tensor
    ):
        """``alpha_t|s = alpha_t / alpha_s`` and the matching sigma."""
        sigma2_t_given_s = -torch.expm1(
            F.softplus(gamma_s) - F.softplus(gamma_t)
        )

        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s
        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    @staticmethod
    def inflate_batch_array(x, batch_seg):
        """Sum a per-atom quantity into a per-molecule vector."""
        return scatter_add(x.sum(-1), batch_seg, dim=0)

    @staticmethod
    def cdf_standard_gaussian(x):
        """Standard-normal CDF."""
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))

    @staticmethod
    def gaussian_kl(q_mu_minus_p_mu_squared, q_sigma, p_sigma, d):
        """KL distance between two isotropic normal distributions."""
        return (
            d * torch.log(p_sigma / q_sigma)
            + 0.5
            * (d * q_sigma**2 + q_mu_minus_p_mu_squared)
            / (p_sigma**2)
            - 0.5 * d
        )
