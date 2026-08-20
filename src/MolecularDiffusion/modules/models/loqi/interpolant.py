"""LoQI coordinate interpolants: VDM diffusion and continuous flow matching.

Ported from ``others/LoQI/src/megalodon/interpolant/`` (NVIDIA, Apache-2.0),
narrowed to the two paths both LoQI configs actually take:

* ``ContinuousDiffusionInterpolant`` with ``diffusion_type: vdm`` -- the
  discrete-time, cosine-adaptive VDM schedule of ``loqi.yaml``. The ``ddpm``
  branch is not ported (no LoQI config selects it).
* ``ContinuousFlowMatchingInterpolant`` with a ``linear`` continuous-time
  schedule, ``prediction_type: velocity`` and ``optimal_transport: 'rigid'`` --
  ``loqi_flow.yaml``. The ``vpe`` schedule and the permutation OT branch are
  ported because they cost three lines and share code paths, but no LoQI config
  reaches them.

The **discrete** interpolants are deliberately absent: LoQI marks ``h``,
``edge_attr`` and ``charges`` as ``discrete_null``, i.e. supplied un-noised as
conditioning, so nothing here ever noises a categorical variable.

Buffer names are load-bearing -- they are the ``interpolants.x.*`` keys of the
released checkpoints.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch_scatter import scatter_mean

# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


class _Schedule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("alphas", self.compute_alphas())
        log_alphas = torch.log(self.alphas)
        self.register_buffer("alphas_bar", torch.exp(torch.cumsum(log_alphas, dim=0)))

    def get_alphas_and_betas(self):
        return self.alphas, 1 - self.alphas

    def compute_alphas(self) -> torch.Tensor:
        raise NotImplementedError


class LinearSchedule(_Schedule):
    def __init__(self, num_diffusion_timesteps: int, **kwargs) -> None:  # noqa: ARG002
        self.num_diffusion_timesteps = num_diffusion_timesteps
        super().__init__()

    def compute_alphas(self) -> torch.Tensor:
        alphas = torch.linspace(1, 0, self.num_diffusion_timesteps + 1)[:-1]
        return alphas.clip(min=0.001, max=1.0)


class CosineSchedule(_Schedule):
    """MiDi-style adaptive cosine schedule (``nu`` reshapes the time axis)."""

    def __init__(  # noqa: PLR0913
        self,
        num_diffusion_timesteps: int,
        s: float = 0.008,
        sqrt: bool = False,
        nu: float = 1.0,
        clip: bool = True,
        cut: bool = False,
        **kwargs,  # noqa: ARG002
    ) -> None:
        self.s = s
        self.nu = nu
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.sqrt = sqrt
        self.clip = clip
        self.cut = cut
        super().__init__()

    @staticmethod
    def _clip_noise_schedule_np(alphas2, clip_value: float = 0.001):
        alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)
        alphas_step = alphas2[1:] / alphas2[:-1]
        alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
        return np.cumprod(alphas_step, axis=0)

    def compute_alphas(self) -> torch.Tensor:
        steps = self.num_diffusion_timesteps + 2
        x = np.linspace(0, steps, steps)
        alphas_cumprod = (
            np.cos(0.5 * np.pi * (((x / steps) ** self.nu) + self.s) / (1 + self.s)) ** 2
        )
        alphas_cumprod_new = alphas_cumprod / alphas_cumprod[0]
        alphas_cumprod_new = self._clip_noise_schedule_np(
            alphas_cumprod_new, clip_value=0.05
        )
        alphas = alphas_cumprod_new[1:] / alphas_cumprod_new[:-1]
        alphas = alphas.clip(min=0.001)
        betas = torch.clip(torch.from_numpy(1 - alphas), 0.0, 0.999).squeeze().float()
        return 1.0 - betas[1:] if self.cut else 1.0 - betas


def build_scheduler(  # noqa: PLR0913
    scheduler_type: str,
    num_diffusion_timesteps: int,
    s: float = 0.008,
    sqrt: bool = False,
    nu: float = 1.0,
    clip: bool = True,
    cut: bool = True,
):
    if scheduler_type == "cosine_adaptive":
        return CosineSchedule(num_diffusion_timesteps, s, sqrt, nu, clip, cut)
    if scheduler_type == "linear":
        return LinearSchedule(num_diffusion_timesteps)
    msg = f"Scheduler '{scheduler_type}' is not implemented"
    raise NotImplementedError(msg)


# ---------------------------------------------------------------------------
# Rigid optimal transport (Kabsch)
# ---------------------------------------------------------------------------


def rigid_alignment(x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
    """Kabsch-align ``x_0`` onto ``x_1``; reflection-safe (``det(R) = 1``)."""
    d = x_0.shape[1]
    if x_0.shape != x_1.shape:
        msg = "x_0 and x_1 must have the same shape"
        raise ValueError(msg)
    x_0_c = x_0 - x_0.mean(dim=0, keepdim=True)
    x_1_mean = x_1.mean(dim=0, keepdim=True)
    x_1_c = x_1 - x_1_mean

    H = x_0_c.T.mm(x_1_c)
    U, _S, V = torch.svd(H)
    D = torch.eye(d, device=x_0.device)
    D[-1, -1] = torch.det(V.mm(U.T)).sign()
    R = V.mm(D).mm(U.T)
    return x_0_c.mm(R.T) + x_1_mean


def align_prior(
    prior_feat: torch.Tensor,
    dst_feat: torch.Tensor,
    permutation: bool = False,
    rigid_body: bool = False,
    n_alignments: int = 1,
) -> torch.Tensor:
    for _ in range(n_alignments):
        if permutation:
            from scipy.optimize import linear_sum_assignment

            cost_mat = torch.cdist(dst_feat, prior_feat, p=2).cpu().detach().numpy()
            _, prior_idx = linear_sum_assignment(cost_mat)
            prior_feat = prior_feat[prior_idx]
        if rigid_body:
            prior_feat = rigid_alignment(prior_feat, dst_feat)
    return prior_feat


# ---------------------------------------------------------------------------
# Interpolants
# ---------------------------------------------------------------------------


class Interpolant(nn.Module):
    def __init__(
        self,
        prior_type: str,
        solver_type: str = "sde",
        timesteps: int = 500,
        time_type: str = "discrete",
    ) -> None:
        super().__init__()
        self.prior_type = prior_type
        self.timesteps = timesteps
        self.solver_type = solver_type
        self.time_type = time_type

    def sample_time(  # noqa: PLR0913
        self,
        num_samples: int,
        method: str = "uniform",
        device: str | torch.device = "cpu",
        mean: float = 0.0,
        scale: float = 0.81,
        min_t: float = 0.0,
    ) -> torch.Tensor:
        """Only ``uniform`` is ported -- both LoQI configs use it."""
        if method != "uniform":
            msg = f"sample_time method '{method}' is not ported (LoQI uses uniform)"
            raise NotImplementedError(msg)
        if self.time_type == "continuous":
            t = torch.rand(num_samples)
            if min_t > 0:
                t = t * (1 - 2 * min_t) + min_t
        else:
            t = torch.randint(0, self.timesteps, size=(num_samples,))
        return t.to(device)


class ContinuousDiffusionInterpolant(Interpolant):
    """VDM continuous Gaussian diffusion over coordinates, discrete time."""

    def __init__(  # noqa: PLR0913
        self,
        prior_type: str = "gaussian",
        diffusion_type: str = "vdm",
        solver_type: str = "sde",
        timesteps: int = 500,
        time_type: str = "discrete",
        num_classes: int = 3,
        scheduler_type: str = "cosine_adaptive",
        s: float = 0.008,
        sqrt: bool = False,
        nu: float = 1.0,
        clip: bool = True,
        com_free: bool = True,
        cut: bool = False,
    ) -> None:
        super().__init__(prior_type, solver_type, timesteps, time_type)
        if diffusion_type != "vdm":
            msg = f"Only the 'vdm' diffusion_type is ported, got '{diffusion_type}'"
            raise NotImplementedError(msg)
        self.num_classes = num_classes
        self.diffusion_type = diffusion_type
        self.com_free = com_free
        self._init_schedulers(timesteps, scheduler_type, s, sqrt, nu, clip, cut)

    def _init_schedulers(self, timesteps, scheduler_type, s, sqrt, nu, clip, cut):  # noqa: PLR0913
        self.scheduler = build_scheduler(scheduler_type, timesteps, s, sqrt, nu, clip, cut)
        alphas, betas = self.scheduler.get_alphas_and_betas()
        if cut:
            msg = "vdm requires the uncut schedule (alphas.shape == T + 1)"
            raise ValueError(msg)

        log_alpha_bar = torch.cumsum(torch.log(alphas), dim=0)
        alpha_bar = torch.exp(log_alpha_bar)
        self.register_buffer("alphas", alphas[1:])
        self.register_buffer("betas", betas[1:])
        self.register_buffer("alpha_bar", alpha_bar[1:])
        sigma2_bar = -torch.expm1(2 * log_alpha_bar)
        sigma_bar = torch.sqrt(sigma2_bar)
        self.register_buffer("sigma_bar", sigma_bar[1:])
        self.register_buffer("forward_data_schedule", alpha_bar[1:])
        self.register_buffer("forward_noise_schedule", sigma_bar[1:])

        s_time = list(range(self.timesteps))
        t_time = list(range(1, 1 + self.timesteps))
        s2_s = -torch.expm1(2 * log_alpha_bar[s_time])
        s2_t = -torch.expm1(2 * log_alpha_bar[t_time])
        sigma_sq_ratio = torch.exp(torch.log(s2_s) - torch.log(s2_t)).float()
        self.register_buffer("sigma_sq_ratio", sigma_sq_ratio)

        alpha_pos_ts_sq = torch.exp(2 * log_alpha_bar[t_time] - 2 * log_alpha_bar[s_time])
        sigma2_t_s = sigma2_bar[t_time] - sigma2_bar[s_time] * alpha_pos_ts_sq
        noise_prefactor = torch.sqrt(sigma2_t_s * sigma_sq_ratio)
        z_t_prefactor = (
            torch.exp(log_alpha_bar[t_time] - log_alpha_bar[s_time]).float()
            * sigma_sq_ratio
        )
        x_prefactor = (alpha_bar[s_time] * (1 - alpha_pos_ts_sq * sigma_sq_ratio)).float()

        self.register_buffer("reverse_data_schedule", x_prefactor)
        self.register_buffer("reverse_noise_schedule", z_t_prefactor)
        self.register_buffer("log_var", 2 * torch.log(noise_prefactor))

    def forward_schedule(self, batch, time):
        t_idx = self.timesteps - 1 - time
        return (
            self.forward_data_schedule[t_idx].unsqueeze(1)[batch],
            self.forward_noise_schedule[t_idx].unsqueeze(1)[batch],
        )

    def reverse_schedule(self, batch, time):
        t_idx = self.timesteps - 1 - time
        return (
            self.reverse_data_schedule[t_idx].unsqueeze(1)[batch],
            self.reverse_noise_schedule[t_idx].unsqueeze(1)[batch],
            self.log_var[t_idx].unsqueeze(1)[batch],
        )

    def interpolate(self, batch, x1, time):
        x0 = self.prior(batch, x1.shape, x1.device)
        data_scale, noise_scale = self.forward_schedule(batch, time)
        return x1, data_scale * x1 + noise_scale * x0, x0

    def prior(self, batch, shape, device, x1=None):  # noqa: ARG002
        if self.prior_type not in ("gaussian", "normal"):
            msg = "Only a Gaussian prior is supported"
            raise ValueError(msg)
        x0 = torch.randn(shape, device=device)
        if self.com_free:
            x0 = x0 - scatter_mean(x0, batch, dim=0)[batch]
        return x0

    def step(self, batch, xt, x_hat, x0=None, time=None, dt=None):  # noqa: ARG002, PLR0913
        if self.solver_type != "sde":
            msg = "Only the SDE solver is implemented"
            raise ValueError(msg)
        data_scale, noise_scale, log_var = self.reverse_schedule(batch, time)
        mean = data_scale * x_hat + noise_scale * xt
        x_next = mean + (0.5 * log_var).exp() * self.prior(batch, xt.shape, xt.device)
        if self.com_free:
            x_next = x_next - scatter_mean(x_next, batch, dim=0)[batch]
        return x_next

    def snr(self, time):
        abar = self.alpha_bar[self.timesteps - 1 - time]
        return abar / (1 - abar)

    def loss_weight_t(self, time):
        return torch.clamp(self.snr(time), min=0.05, max=1.5)


class ContinuousFlowMatchingInterpolant(Interpolant):
    """Linear continuous-time flow matching with velocity prediction + rigid OT."""

    def __init__(  # noqa: PLR0913
        self,
        prior_type: str = "gaussian",
        vector_field_type: str = "standard",
        solver_type: str = "ode",
        timesteps: int = 500,
        min_t: float = 1e-2,
        time_type: str = "continuous",
        num_classes: int = 3,
        scheduler_type: str = "linear",
        s: float = 0.008,  # noqa: ARG002
        sqrt: bool = False,  # noqa: ARG002
        nu: float = 1.0,  # noqa: ARG002
        clip: bool = True,  # noqa: ARG002
        com_free: bool = True,
        noise_sigma: float = 0.0,
        optimal_transport: str | None = None,
        clip_t: float = 0.9,
        loss_weight_type: str = "uniform",
        loss_t_scale: float = 0.1,
        inference_noise_sigma: float | None = None,
        prediction_type: str = "data",
    ) -> None:
        super().__init__(prior_type, solver_type, timesteps, time_type)
        self.num_classes = num_classes
        self.vector_field_type = vector_field_type
        self.min_t = min_t
        self.com_free = com_free
        self.noise_sigma = noise_sigma
        self.optimal_transport = optimal_transport
        self.schedule_type = scheduler_type
        if scheduler_type != "linear":
            msg = f"Only the 'linear' FM schedule is ported, got '{scheduler_type}'"
            raise NotImplementedError(msg)
        time = torch.linspace(self.min_t, 1, self.timesteps)
        self.register_buffer("time", time)
        self.register_buffer("forward_data_schedule", time)
        self.register_buffer("forward_noise_schedule", 1.0 - time)
        self.max_t = 1.0 - min_t
        self.clip_t = clip_t
        self.loss_weight_type = loss_weight_type
        self.loss_t_scale = loss_t_scale
        self.prediction_type = prediction_type
        self.inference_noise_sigma = (
            inference_noise_sigma if inference_noise_sigma is not None else noise_sigma
        )

    def loss_weight_t(self, time):
        """``None`` means "no per-molecule weighting"; the loss function treats
        a ``None`` batch weight as 1 (upstream ``InterpolantLossFunction``
        line 175). ``loss_weight_type`` defaults to ``'standard'`` through
        upstream's builder, which falls through every branch and returns
        ``None`` -- reproduced here rather than silently changed."""
        if self.loss_weight_type == "uniform":
            return torch.ones_like(time).to(time.device)
        if self.loss_weight_type == "frameflow":
            t = torch.clamp(time, self.min_t, self.clip_t)
            return (self.loss_t_scale * (1 / (1 - t))) ** 2
        if self.loss_weight_type == "snr":
            t = torch.clamp(time, self.min_t, self.clip_t)
            return t / (1 - t)
        return None

    def update_weight(self, t):
        if self.vector_field_type == "endpoint":
            return torch.ones_like(t).to(t.device)
        return 1 / (1 - torch.clamp(t, self.min_t, self.max_t))

    def forward_schedule(self, batch, time):
        return time[batch].unsqueeze(1), (1.0 - time)[batch].unsqueeze(1)

    @torch.no_grad()
    def equivariant_ot_prior(self, batch, data_chunk, permutation: bool = True):
        aligned_prior = self.prior_func(batch, data_chunk.shape, data_chunk.device)
        for i in range(int(batch.max()) + 1):
            mask = batch == i
            aligned_prior[mask] = align_prior(
                aligned_prior[mask],
                data_chunk[mask],
                permutation=permutation,
                rigid_body=True,
            )
        return aligned_prior

    def interpolate(self, batch, x1, time):
        if self.optimal_transport in ("equivariant_ot", "scale_ot"):
            x0 = self.equivariant_ot_prior(batch, x1, permutation=True)
        elif self.optimal_transport == "rigid":
            x0 = self.equivariant_ot_prior(batch, x1, permutation=False)
        else:
            x0 = self.prior_func(batch, x1.shape, x1.device)

        data_scale, noise_scale = self.forward_schedule(batch, time)
        interp_noise = (
            self.prior_func(batch, x1.shape, x1.device) * self.noise_sigma
            if self.noise_sigma > 0
            else 0
        )
        x_t = data_scale * x1 + noise_scale * x0 + interp_noise
        target = x1 - x0 if self.prediction_type == "velocity" else x1
        return target, x_t, x0

    def vector_field(self, batch, x1, xt, time):
        vf = (x1 - xt) / (1.0 - time[batch].unsqueeze(-1))
        return vf + torch.randn_like(x1) * self.inference_noise_sigma

    def prior_func(self, batch, shape, device, x1=None):  # noqa: ARG002
        if self.prior_type not in ("gaussian", "normal"):
            msg = "Only a Gaussian prior is supported"
            raise ValueError(msg)
        x0 = torch.randn(shape, device=device)
        if self.com_free:
            x0 = (
                x0 - scatter_mean(x0, batch, dim=0)[batch]
                if batch is not None
                else x0 - x0.mean(0)
            )
        return x0

    def prior(self, batch, shape, device, x1=None):
        sample = self.prior_func(batch, shape, device, x1)
        if self.optimal_transport == "scale_ot":
            _, counts = torch.unique(batch, return_counts=True)
            sample = sample * (0.2 * torch.log(counts + 1).unsqueeze(1))[batch]
        return sample

    def step(self, batch, xt, x_hat, x0=None, time=None, dt=None):  # noqa: PLR0913
        if self.prediction_type == "velocity":
            x_next = xt + dt * x_hat
        elif self.vector_field_type == "standard":
            x_next = xt + dt * self.vector_field(batch, x_hat, xt, time)
        elif self.vector_field_type == "endpoint":
            data_scale = (self.update_weight(time[batch]) * dt).unsqueeze(1)
            x_next = xt + data_scale * (x_hat - x0)
        else:
            msg = f"{self.vector_field_type} is not a recognized vector_field_type"
            raise ValueError(msg)

        if self.com_free:
            batch_size = int(batch.max()) + 1
            x_next = x_next - scatter_mean(x_next, batch, dim=0, dim_size=batch_size)[batch]
        return x_next


def build_interpolant(interpolant_type: str, **kwargs):
    """The two coordinate interpolants LoQI selects, by config string."""
    if interpolant_type == "continuous_diffusion":
        allowed = {
            "prior_type", "diffusion_type", "solver_type", "timesteps", "time_type",
            "num_classes", "scheduler_type", "s", "sqrt", "nu", "clip", "com_free",
            "cut",
        }
        return ContinuousDiffusionInterpolant(
            **{k: v for k, v in kwargs.items() if k in allowed}
        )
    if interpolant_type == "continuous_flow_matching":
        allowed = {
            "prior_type", "vector_field_type", "solver_type", "timesteps", "min_t",
            "time_type", "num_classes", "scheduler_type", "com_free", "noise_sigma",
            "optimal_transport", "clip_t", "loss_weight_type", "loss_t_scale",
            "inference_noise_sigma", "prediction_type",
        }
        kwargs.setdefault("solver_type", "ode")
        return ContinuousFlowMatchingInterpolant(
            **{k: v for k, v in kwargs.items() if k in allowed}
        )
    msg = f"Interpolant not supported: {interpolant_type}"
    raise NotImplementedError(msg)
