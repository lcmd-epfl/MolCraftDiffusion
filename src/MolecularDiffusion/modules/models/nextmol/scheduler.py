"""Continuous-time VP noise schedule (``NoiseScheduleVPV2``, DPM-Solver).

Ported from ``others/NExT-Mol/data_provider/diffusion_scheduler.py:24``.

Only the ``cosine`` and ``linear`` branches are here. The ``discrete`` /
``discrete_poly`` branches need a 1000-step table and an interpolation helper,
and every released DMT checkpoint was trained with ``noise_scheduler: cosine``
and ``discrete_schedule: False`` -- so porting them would be dead code today.
Note the cosine schedule's ``T = 0.9946``, not 1.0: t=1 has numerical issues.
"""

from __future__ import annotations

import math

import torch

__all__ = ["NoiseScheduleVPV2"]


class NoiseScheduleVPV2:
    """``marginal_prob(t) -> (alpha_t, sigma_t)`` for a VP forward SDE."""

    def __init__(
        self,
        schedule: str = "cosine",
        continuous_beta_0: float = 0.1,
        continuous_beta_1: float = 20.0,
        discrete_mode: bool = False,
    ) -> None:
        if schedule not in ("linear", "cosine"):
            msg = (
                f"Unsupported noise schedule {schedule!r}. Only 'cosine' and "
                f"'linear' are ported; see this module's docstring."
            )
            raise ValueError(msg)
        self.schedule = schedule
        self.discrete_mode = discrete_mode
        self.total_N = 1000
        self.beta_0 = continuous_beta_0
        self.beta_1 = continuous_beta_1
        self.cosine_s = 0.008
        self.cosine_log_alpha_0 = math.log(
            math.cos(self.cosine_s / (1.0 + self.cosine_s) * math.pi / 2.0)
        )
        # For the cosine schedule T = 1 has numerical issues; upstream ends at
        # 0.9946 and every released checkpoint was sampled that way.
        self.T = 0.9946 if schedule == "cosine" else 1.0

    def marginal_log_mean_coeff(self, t):
        if self.schedule == "linear":
            return -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        log_alpha = torch.log(
            torch.cos((t + self.cosine_s) / (1.0 + self.cosine_s) * math.pi / 2.0)
        )
        return log_alpha - self.cosine_log_alpha_0

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        return torch.sqrt(1.0 - torch.exp(2.0 * self.marginal_log_mean_coeff(t)))

    def marginal_prob(self, t):
        if self.discrete_mode:
            t = torch.floor(t * self.total_N) / self.total_N
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        return (
            torch.exp(log_mean_coeff),
            torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff)),
        )
