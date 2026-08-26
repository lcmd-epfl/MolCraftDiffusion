"""EDM / AlphaFold3 atom diffusion for ChefNMR (MIT, (c) 2025 Ziyu Xiong).

Upstream: ``src/model/modules/diffusion.py``, which itself started from
Boltz (``jwohlwend/boltz``, MIT, (c) 2024 Wohlwend/Corso/Passaro).

References: Karras et al. 2022 (EDM); Abramson et al. 2024 (AlphaFold3).

Ported unchanged except for two things that are packaging, not maths:

* ``score_model_args`` is the denoiser's kwargs dict directly, instead of
  upstream's ``{model_name: str, <model_name>: {...}}`` indirection --
  there is exactly one score model and Hydra already picks the task.
* ``einops.rearrange(sigma, "b -> b 1 1")`` is ``sigma[:, None, None]``.
  ``einops`` is not installed here and this was its only site in this file.

``sigma_data`` enters the preconditioning (``c_in``/``c_sigma``/``d_sigma``),
so it is **baked into the weights**: a checkpoint must be sampled with the
same value it was trained with, which is why it is a task-config key rather
than something derived from whatever dataset happens to be attached.
"""

from __future__ import annotations

from math import sqrt
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn import Module

from MolecularDiffusion.modules.models.chefnmr.score_models import (
    DiffusionModuleTransformer,
)
from MolecularDiffusion.modules.models.chefnmr.utils import (
    center_random_augmentation,
    default,
    log,
    smooth_lddt_loss,
)

_DEFAULT_EDM_ARGS = {
    "sigma_data": 3.0,
    "rho": 7,
    "use_heun_solver": True,
    "gamma_0": 0.8,
}
_DEFAULT_TRAIN_SIGMA_ARGS = {"edm_P_mean": -1.2, "edm_P_std": 1.3}


class AtomDiffusion(Module):
    """Forward noising + reverse (Heun/EDM) sampling over atom coordinates."""

    def __init__(  # noqa: PLR0913
        self,
        score_model_args: dict,
        train_sigma_distribution_type: str = "af3",
        sample_sigma_schedule_type: str = "edm",
        sample_gamma_schedule_type: str = "edm",
        num_sampling_steps: int = 50,
        sigma_min: float = 0.0004,
        sigma_max: float = 80.0,
        gamma_min: float = 1.0,
        noise_scale: float = 1.0,
        step_scale: float = 1.0,
        guidance_scale: float = 0.0,
        synchronize_sigmas: bool = False,
        coordinate_transformation_when_training: str = "centering_rotation_translation",
        edm_args: Optional[Dict] = None,
        train_sigma_args: Optional[Dict] = None,
        **kwargs,  # noqa: ARG002
    ) -> None:
        super().__init__()
        self.score_model = DiffusionModuleTransformer(**dict(score_model_args))

        self.train_sigma_distribution_type = train_sigma_distribution_type
        self.sample_sigma_schedule_type = sample_sigma_schedule_type
        self.sample_gamma_schedule_type = sample_gamma_schedule_type
        self.num_sampling_steps = num_sampling_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.guidance_scale = guidance_scale
        self.synchronize_sigmas = synchronize_sigmas
        self.coordinate_transformation_when_training = (
            coordinate_transformation_when_training
        )

        self.edm_args = SimpleNamespace(
            **{**_DEFAULT_EDM_ARGS, **dict(edm_args or {})}
        )
        self.train_sigma_args = SimpleNamespace(
            **{**_DEFAULT_TRAIN_SIGMA_ARGS, **dict(train_sigma_args or {})}
        )
        if self.edm_args.sigma_data is None:
            msg = (
                "edm_args.sigma_data is None. It enters the EDM "
                "preconditioning and is baked into the weights -- set it to "
                "the value the checkpoint was trained with (2.67 for the "
                "released USPTO models)."
            )
            raise ValueError(msg)
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

    @property
    def device(self) -> torch.device:
        return next(self.score_model.parameters()).device

    # --- shape helpers ----------------------------------------------------
    def float_to_tensor(
        self, value: Union[float, torch.Tensor], batch_size: int, device
    ) -> torch.Tensor:
        if isinstance(value, float):
            value = torch.full((batch_size,), value, device=device)
        return value

    def pad_sigma(
        self, sigma: torch.Tensor, batch_size: int, device
    ) -> torch.Tensor:
        sigma = self.float_to_tensor(sigma, batch_size, device)
        return sigma[:, None, None]  # einops-free: "b -> b 1 1"

    # --- Karras preconditioning ------------------------------------------
    def a_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(sigma)

    def b_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma

    def interpolate(self, atom_coords, noise, sigma):
        padded_sigma = self.pad_sigma(sigma, atom_coords.shape[0], atom_coords.device)
        return (
            self.a_sigma(padded_sigma) * atom_coords
            + self.b_sigma(padded_sigma) * noise
        )

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        return 1 / (torch.sqrt(sigma**2 + self.edm_args.sigma_data**2))

    def noised_coords_in_network(self, atom_coords, sigma):
        padded_sigma = self.pad_sigma(sigma, atom_coords.shape[0], atom_coords.device)
        return self.c_in(padded_sigma) * atom_coords

    def sigma_in_network(self, sigma: torch.Tensor) -> torch.Tensor:
        return log(sigma) * 0.25

    def c_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma / (
            self.edm_args.sigma_data * torch.sqrt(sigma**2 + self.edm_args.sigma_data**2)
        )

    def d_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return -self.edm_args.sigma_data / torch.sqrt(
            sigma**2 + self.edm_args.sigma_data**2
        )

    def net_target(self, atom_coords, noise, sigma):
        padded_sigma = self.pad_sigma(sigma, atom_coords.shape[0], atom_coords.device)
        return (
            self.c_sigma(padded_sigma) * atom_coords
            + self.d_sigma(padded_sigma) * noise
        )

    # --- schedules --------------------------------------------------------
    def sample_sigma_schedule(self, num_sampling_steps=None) -> torch.Tensor:
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        if self.sample_sigma_schedule_type not in ("edm", "af3"):
            msg = (
                "Unknown sample_sigma_schedule_type: "
                f"{self.sample_sigma_schedule_type}"
            )
            raise ValueError(msg)
        inv_rho = 1 / self.edm_args.rho
        steps = torch.arange(num_sampling_steps, device=self.device, dtype=torch.float32)
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (num_sampling_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.edm_args.rho
        if self.sample_sigma_schedule_type == "af3":
            sigmas = sigmas * self.edm_args.sigma_data
        return F.pad(sigmas, (0, 1), value=0.0)  # final step is sigma 0

    def sample_gamma_schedule(self, sigmas: torch.Tensor) -> torch.Tensor:
        return torch.where(sigmas > self.gamma_min, self.edm_args.gamma_0, 0.0)

    # --- sampling ---------------------------------------------------------
    def sample(
        self,
        model_inputs: Dict[str, torch.Tensor],
        num_sampling_steps: Optional[int] = None,
        multiplicity: int = 1,
        n_chain_frames: int = 1,
        guidance_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reverse diffusion.

        ``guidance_scale=None`` uses ``self.guidance_scale`` (the model's own
        default). It is an explicit argument rather than a mutated attribute
        so a generator can vary it per run without touching the module.
        """
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        w = self.guidance_scale if guidance_scale is None else float(guidance_scale)

        if num_sampling_steps < n_chain_frames:
            save_chain_indices = torch.arange(0, num_sampling_steps, device=self.device)
        else:
            save_chain_indices = (
                torch.arange(0, num_sampling_steps * n_chain_frames, num_sampling_steps)
                // n_chain_frames
            )
        save_chain_indices = save_chain_indices.tolist()
        atom_coords_chains = torch.tensor([], device=self.device)

        atom_mask = model_inputs["atom_mask"].repeat_interleave(multiplicity, 0)
        shape = (*atom_mask.shape, 3)

        sigmas = self.sample_sigma_schedule(num_sampling_steps)
        gammas = self.sample_gamma_schedule(sigmas)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        atom_coords = sigmas[0] * torch.randn(shape, device=self.device)

        for i, (sigma_tm, sigma_t, gamma) in enumerate(sigmas_and_gammas):
            if i in save_chain_indices:
                atom_coords_chains = torch.cat(
                    [atom_coords_chains, atom_coords[:, None, ...]], dim=1
                )
            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()
            eps = torch.randn(shape, device=self.device)
            atom_coords = self._sample_one_step_edm(
                model_inputs=model_inputs,
                multiplicity=multiplicity,
                atom_coords=atom_coords,
                sigma_tm=sigma_tm,
                sigma_t=sigma_t,
                gamma=gamma,
                eps=eps,
                guidance_scale=w,
            )

        atom_coords_chains = torch.cat(
            [atom_coords_chains, atom_coords[:, None, ...]], dim=1
        )
        return atom_coords, atom_coords_chains

    def _sample_one_step_edm(  # noqa: PLR0913
        self,
        model_inputs,
        multiplicity,
        atom_coords,
        sigma_tm,
        sigma_t,
        gamma,
        eps,
        guidance_scale,
    ):
        # 1. stochastic churn
        t_hat = sigma_tm * (1 + gamma)
        eps = eps * self.noise_scale * sqrt(t_hat**2 - sigma_tm**2)
        noisy_atom_coords = atom_coords + eps

        # 2. denoise
        with torch.no_grad():
            net_out = self.neural_network_forward(
                noisy_atom_coords,
                t_hat,
                network_condition_kwargs={
                    "multiplicity": multiplicity,
                    "model_inputs": model_inputs,
                    "guidance_scale": guidance_scale,
                },
            )
        denoised_atom_coords = self.predict_denoised_atom_coords(
            noisy_atom_coords, net_out, t_hat
        )

        # 3-4. velocity + Euler step
        velocity = self.predict_velocity(
            noisy_atom_coords=noisy_atom_coords,
            net_out=net_out,
            sigma=t_hat,
            denoised_atom_coords=denoised_atom_coords,
        )
        atom_coords_next = (
            noisy_atom_coords + self.step_scale * (sigma_t - t_hat) * velocity
        )

        # 5. Heun 2nd-order correction (a second network call per step)
        if self.edm_args.use_heun_solver and sigma_t > 0:
            with torch.no_grad():
                net_out = self.neural_network_forward(
                    atom_coords_next,
                    sigma_t,
                    network_condition_kwargs={
                        "multiplicity": multiplicity,
                        "model_inputs": model_inputs,
                        "guidance_scale": guidance_scale,
                    },
                )
            denoised_atom_coords = self.predict_denoised_atom_coords(
                atom_coords_next, net_out, sigma_t
            )
            velocity_next = self.predict_velocity(
                noisy_atom_coords=atom_coords_next,
                net_out=net_out,
                sigma=sigma_t,
                denoised_atom_coords=denoised_atom_coords,
            )
            atom_coords_next = (
                noisy_atom_coords
                + 0.5 * self.step_scale * (sigma_t - t_hat) * velocity
                + 0.5 * self.step_scale * (sigma_t - t_hat) * velocity_next
            )

        return atom_coords_next

    def neural_network_forward(
        self,
        noisy_atom_coords: torch.Tensor,
        sigma: Union[float, torch.Tensor],
        network_condition_kwargs: dict,
    ) -> Dict[str, torch.Tensor]:
        batch_size, device = noisy_atom_coords.shape[0], noisy_atom_coords.device
        sigma = self.float_to_tensor(sigma, batch_size, device)
        return self.score_model(
            r_noisy=self.noised_coords_in_network(noisy_atom_coords, sigma),
            times=self.sigma_in_network(sigma),
            **network_condition_kwargs,
        )

    def predict_denoised_atom_coords(self, noisy_atom_coords, net_out, sigma):
        batch_size, device = noisy_atom_coords.shape[0], noisy_atom_coords.device
        padded_sigma = self.pad_sigma(sigma, batch_size, device)
        denoised = self.d_sigma(padded_sigma) * noisy_atom_coords - self.b_sigma(
            padded_sigma
        ) * net_out["r_update"]
        return denoised * -self.edm_args.sigma_data * self.c_in(padded_sigma)

    def predict_velocity(
        self, noisy_atom_coords, net_out, sigma, denoised_atom_coords=None
    ):
        batch_size, device = noisy_atom_coords.shape[0], noisy_atom_coords.device
        padded_sigma = self.pad_sigma(sigma, batch_size, device)
        if denoised_atom_coords is None:
            denoised_atom_coords = self.predict_denoised_atom_coords(
                noisy_atom_coords, net_out, sigma
            )
        return (noisy_atom_coords - denoised_atom_coords) / padded_sigma

    # --- training ---------------------------------------------------------
    def train_sigma_distribution(self, batch_size: int) -> torch.Tensor:
        if self.train_sigma_distribution_type not in ("edm", "af3"):
            msg = (
                "Unknown train_sigma_distribution_type: "
                f"{self.train_sigma_distribution_type}"
            )
            raise ValueError(msg)
        sigmas = (
            self.train_sigma_args.edm_P_mean
            + self.train_sigma_args.edm_P_std
            * torch.randn((batch_size,), device=self.device)
        ).exp()
        if self.train_sigma_distribution_type == "af3":
            sigmas = sigmas * self.edm_args.sigma_data
        return sigmas

    def forward(
        self,
        model_inputs: Dict[str, torch.Tensor],
        atom_coords: torch.Tensor,
        multiplicity: int = 1,
    ) -> Dict[str, torch.Tensor]:
        batch_size = atom_coords.shape[0]

        if self.synchronize_sigmas:
            sigmas = self.train_sigma_distribution(batch_size).repeat_interleave(
                multiplicity, 0
            )
        else:
            sigmas = self.train_sigma_distribution(batch_size * multiplicity)

        atom_coords = atom_coords.repeat_interleave(multiplicity, 0)
        atom_mask = model_inputs["atom_mask"].repeat_interleave(multiplicity, 0)

        if self.coordinate_transformation_when_training == (
            "centering_rotation_translation"
        ):
            atom_coords = center_random_augmentation(
                atom_coords, atom_mask, centering=True, augmentation=True
            )

        noise = torch.randn_like(atom_coords)
        noisy_atom_coords = self.interpolate(atom_coords, noise, sigmas)

        net_out = self.neural_network_forward(
            noisy_atom_coords,
            sigmas,
            network_condition_kwargs={
                "model_inputs": model_inputs,
                "multiplicity": multiplicity,
                "guidance_scale": 0.0,  # never guided during training
            },
        )
        denoised_atom_coords = self.predict_denoised_atom_coords(
            noisy_atom_coords, net_out, sigmas
        )

        return {
            "noisy_atom_coords": noisy_atom_coords,
            "denoised_atom_coords": denoised_atom_coords,
            "sigmas": sigmas,
            "aligned_true_atom_coords": atom_coords,
            "net_out": net_out["r_update"],
            "noise": noise,
        }

    def compute_loss(
        self,
        model_inputs: Dict[str, torch.Tensor],
        dict_out: Dict[str, torch.Tensor],
        multiplicity: int = 1,
        add_smooth_lddt_loss: bool = True,
        lddt_loss_threshold: Optional[list] = None,
    ) -> Dict[str, Any]:
        lddt_loss_threshold = lddt_loss_threshold or [0.5, 1.0, 2.0, 4.0]
        denoised_atom_coords = dict_out["denoised_atom_coords"]
        sigmas = dict_out["sigmas"]
        atom_mask = model_inputs["atom_mask"].repeat_interleave(multiplicity, 0)
        # Every atom weighs 1 in the small-molecule case.
        align_weights = denoised_atom_coords.new_ones(denoised_atom_coords.shape[:2])

        atom_coords_aligned_ground_truth = dict_out["aligned_true_atom_coords"].to(
            denoised_atom_coords
        )

        net_target = self.net_target(
            atom_coords_aligned_ground_truth, dict_out["noise"], sigmas
        )
        mse_loss = ((dict_out["net_out"] - net_target) ** 2).sum(dim=-1)
        mse_loss = torch.sum(mse_loss * align_weights * atom_mask, dim=-1) / torch.sum(
            3 * align_weights * atom_mask, dim=-1
        )
        mse_loss = mse_loss.mean()

        total_loss = mse_loss
        lddt_loss = self.zero
        if add_smooth_lddt_loss:
            lddt_loss = smooth_lddt_loss(
                pred_coords=denoised_atom_coords,
                true_coords=dict_out["aligned_true_atom_coords"],
                is_nucleotide=torch.zeros_like(atom_mask),
                coords_mask=atom_mask,
                lddt_loss_threshold=lddt_loss_threshold,
                multiplicity=multiplicity,
            )
            total_loss = total_loss + lddt_loss

        return {
            "loss": total_loss,
            "loss_breakdown": {"mse_loss": mse_loss, "smooth_lddt_loss": lddt_loss},
        }
