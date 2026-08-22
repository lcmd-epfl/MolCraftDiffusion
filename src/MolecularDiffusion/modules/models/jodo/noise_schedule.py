"""JODO's VP noise schedule.

Ported from others/JODO/diffusion/noise_schedule.py (itself from DPM-Solver).
Only the two continuous schedules JODO's own configs use are kept -- `cosine`
(every released JODO config) and `linear`. The `discrete`/`discrete_poly`
branches and the DPM-Solver-only helpers (`inverse_lambda`,
`numerical_clip_alpha`, `interpolate_fn`) are dropped with the fast sampler
they exist for; ancestral sampling needs only `marginal_prob`.
"""

import math

import torch


class NoiseScheduleVP:
    """Continuous-time VP forward SDE, `alpha_t` / `sigma_t` of a label t."""

    def __init__(
        self,
        schedule: str = "cosine",
        continuous_beta_0: float = 0.1,
        continuous_beta_1: float = 20.0,
    ) -> None:
        if schedule not in ("linear", "cosine"):
            msg = f"Unsupported noise schedule {schedule!r}"
            raise ValueError(msg)
        self.schedule = schedule
        self.total_N = 1000
        self.beta_0 = continuous_beta_0
        self.beta_1 = continuous_beta_1
        self.cosine_s = 0.008
        self.cosine_log_alpha_0 = math.log(
            math.cos(self.cosine_s / (1.0 + self.cosine_s) * math.pi / 2.0)
        )
        # For the cosine schedule T = 1 is numerically unstable; upstream pins
        # the end time at 0.9946 (noise_schedule.py:48-52).
        self.T = 0.9946 if schedule == "cosine" else 1.0

    def marginal_log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """log(alpha_t)."""
        if self.schedule == "linear":
            return (
                -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
            )
        log_alpha_t = torch.log(
            torch.cos((t + self.cosine_s) / (1.0 + self.cosine_s) * math.pi / 2.0)
        )
        return log_alpha_t - self.cosine_log_alpha_0

    def marginal_alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1.0 - torch.exp(2.0 * self.marginal_log_mean_coeff(t)))

    def marginal_prob(self, t: torch.Tensor):
        """(alpha_t, sigma_t)."""
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        return (
            torch.exp(log_mean_coeff),
            torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff)),
        )
