"""Interpolant schedule for FlowMol flow-matching.

Ported verbatim from FlowMol (``flowmol/models/interpolant_scheduler.py``).
Governs the per-feature interpolation weights between prior (t=0) and data
(t=1) and the time-dependent loss weights.
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


class InterpolantScheduler(nn.Module):

    supported_schedule_types = ["cosine", "linear"]

    def __init__(
        self,
        canonical_feat_order: list,
        schedule_type: Union[str, Dict[str, str]] = "cosine",
        cosine_params: dict = {},
    ):
        super().__init__()

        self.feats = canonical_feat_order
        self.n_feats = len(self.feats)

        if not isinstance(schedule_type, (str, dict)):
            raise ValueError("schedule_type must be a string or a dictionary")

        if isinstance(schedule_type, str):
            if schedule_type not in self.supported_schedule_types:
                raise ValueError(f"unsupported schedule_type: {schedule_type}")
            self.schedule_dict = {feat: schedule_type for feat in self.feats}
        else:
            for feat in self.feats:
                if feat not in schedule_type:
                    raise ValueError(
                        f"must specify schedule_type for feature {feat}"
                    )
            self.schedule_dict = schedule_type

        for feat, sch in self.schedule_dict.items():
            if sch == "cosine" and feat not in cosine_params:
                raise ValueError(
                    f"must specify cosine_params for feature {feat}"
                )

        self.schedule_types = list(set(self.schedule_dict.values()))

        if "cosine" in self.schedule_types:
            for feat in cosine_params:
                cosine_params[feat] = torch.tensor(
                    cosine_params[feat]
                ).unsqueeze(0)

        self.cosine_params = cosine_params
        self.clamp_t = True

    def update_device(self, t):
        # Move the per-feature cosine `nu` tensors onto t's device on every
        # call. cosine_params is a plain dict (not registered buffers), so
        # module.to(device) never moves it; doing it unconditionally here also
        # avoids a stale device cache surviving a checkpoint round-trip (which
        # previously made the guard skip the move -> cuda/cpu mismatch at
        # sampling). ponytail: register_buffer per nu if this ever hot-paths.
        if "cosine" in self.schedule_types:
            for key in self.cosine_params:
                self.cosine_params[key] = self.cosine_params[key].to(t.device)

    def interpolant_weights(
        self, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Weights for x_0 and x_1 in the interpolation between them."""
        self.update_device(t)
        alpha_t = self.alpha_t(t)
        weights = (1 - alpha_t, alpha_t)
        return weights

    def loss_weights(self, t: torch.Tensor):
        alpha_t = self.alpha_t(t)
        weights = alpha_t / (1 - alpha_t)
        weights = torch.clamp(weights, min=0.05, max=1.5)
        return weights

    def alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        self.update_device(t)
        per_feat_alpha = []
        for feat in self.feats:
            schedule_type = self.schedule_dict[feat]
            if schedule_type == "cosine":
                alpha_t = self.cosine_alpha_t(t, nu=self.cosine_params[feat])
            elif schedule_type == "linear":
                alpha_t = self.linear_alpha_t(t)
            per_feat_alpha.append(alpha_t)
        return torch.cat(per_feat_alpha, dim=1)

    def alpha_t_prime(self, t: torch.Tensor) -> torch.Tensor:
        self.update_device(t)
        per_feat_alpha_prime = []
        for feat in self.feats:
            schedule_type = self.schedule_dict[feat]
            if schedule_type == "cosine":
                alpha_t_prime = self.cosine_alpha_t_prime(
                    t, nu=self.cosine_params[feat]
                )
            elif schedule_type == "linear":
                alpha_t_prime = self.linear_alpha_t_prime(t)
            per_feat_alpha_prime.append(alpha_t_prime)
        return torch.cat(per_feat_alpha_prime, dim=1)

    def cosine_alpha_t(self, t: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(-1)
        alpha_t = 1 - torch.cos(torch.pi * 0.5 * torch.pow(t, nu)).square()
        return alpha_t

    def cosine_alpha_t_prime(
        self, t: torch.Tensor, nu: torch.Tensor
    ) -> torch.Tensor:
        if self.clamp_t:
            t = torch.clamp_(t, min=1e-9)
        t = t.unsqueeze(-1)
        sin_input = torch.pi * torch.pow(t, nu)
        alpha_t_prime = (
            torch.pi * 0.5 * torch.sin(sin_input) * nu * torch.pow(t, nu - 1)
        )
        return alpha_t_prime

    def linear_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(-1)

    def linear_alpha_t_prime(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t).unsqueeze(-1)
