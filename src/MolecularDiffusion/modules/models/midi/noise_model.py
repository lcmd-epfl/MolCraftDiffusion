"""MiDi's joint noise process: D3PM on the categoricals, VP-SDE on positions.

Ported from ``midi/diffusion/noise_model.py`` with one deliberate change: the
constructor takes plain arguments instead of the upstream Hydra ``cfg``
object, so nothing here depends on MiDi's config tree.

This is **not** an ``nn.Module``. The transition matrices and marginals are
plain tensors derived from dataset statistics, which is why they are absent
from every released MiDi checkpoint and must be rebuilt at construction time
from ``train_set.graph3d_stats``.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F  # noqa: N812

from . import diffusion_utils
from .placeholder import Dims, PlaceHolder, remove_mean_with_mask

#: Modality order of the per-modality schedule exponents.
MODALITIES = ("p", "x", "c", "e", "y")


class NoiseModel:
    """Forward/reverse noise process shared by both transition variants."""

    def __init__(
        self,
        nu: dict[str, float],
        diffusion_steps: int = 500,
        noise_schedule: str = "cosine",
    ) -> None:
        self.mapping = list(MODALITIES)
        self.inverse_mapping = {m: i for i, m in enumerate(self.mapping)}
        self.nu_arr = [float(nu[m]) for m in self.mapping]

        # Filled in by the subclasses.
        self.Px: torch.Tensor | None = None
        self.Pe: torch.Tensor | None = None
        self.Py: torch.Tensor | None = None
        self.Pcharges: torch.Tensor | None = None
        self.X_classes = 0
        self.charges_classes = 0
        self.E_classes = 0
        self.y_classes = 0
        self.X_marginals: torch.Tensor | None = None
        self.charges_marginals: torch.Tensor | None = None
        self.E_marginals: torch.Tensor | None = None
        self.y_marginals: torch.Tensor | None = None

        self.noise_schedule = noise_schedule
        self.timesteps = diffusion_steps
        self.T = diffusion_steps

        if noise_schedule != "cosine":
            raise NotImplementedError(noise_schedule)
        betas = diffusion_utils.cosine_beta_schedule_discrete(
            self.timesteps, self.nu_arr
        )

        self._betas = torch.from_numpy(betas)
        self._alphas = 1 - torch.clamp(self._betas, min=0, max=0.9999)
        log_alpha_bar = torch.cumsum(torch.log(self._alphas), dim=0)
        self._log_alpha_bar = log_alpha_bar
        self._alphas_bar = torch.exp(log_alpha_bar)
        self._sigma2_bar = -torch.expm1(2 * log_alpha_bar)
        self._sigma_bar = torch.sqrt(self._sigma2_bar)
        self._gamma = (
            torch.log(-torch.special.expm1(2 * log_alpha_bar))
            - 2 * log_alpha_bar
        )

    # -- device / schedule lookups -----------------------------------------

    def move_P_device(self, tensor: torch.Tensor) -> PlaceHolder:  # noqa: N802
        """Transition matrices on ``tensor``'s device."""
        return PlaceHolder(
            X=self.Px.float().to(tensor.device),
            charges=self.Pcharges.float().to(tensor.device),
            E=self.Pe.float().to(tensor.device),
            y=self.Py.float().to(tensor.device),
            pos=None,
        )

    def _lookup(
        self,
        table: torch.Tensor,
        t_int: torch.Tensor | None,
        t_normalized: torch.Tensor | None,
        key: str | None,
    ) -> torch.Tensor:
        if (t_int is None) == (t_normalized is None):
            msg = "pass exactly one of t_int / t_normalized"
            raise ValueError(msg)
        if t_int is None:
            t_int = torch.round(t_normalized * self.T)
        value = table.to(t_int.device)[t_int.long()]
        if key is None:
            return value.float()
        return value[..., self.inverse_mapping[key]].float()

    def get_beta(
        self,
        t_normalized: torch.Tensor | None = None,
        t_int: torch.Tensor | None = None,
        key: str | None = None,
    ) -> torch.Tensor:
        """``beta_t`` for one modality."""
        return self._lookup(self._betas, t_int, t_normalized, key)

    def get_alpha_bar(
        self,
        t_normalized: torch.Tensor | None = None,
        t_int: torch.Tensor | None = None,
        key: str | None = None,
    ) -> torch.Tensor:
        """``alpha_bar_t`` for one modality."""
        return self._lookup(self._alphas_bar, t_int, t_normalized, key)

    def get_sigma_bar(
        self,
        t_normalized: torch.Tensor | None = None,
        t_int: torch.Tensor | None = None,
        key: str | None = None,
    ) -> torch.Tensor:
        """``sigma_bar_t`` for one modality."""
        return self._lookup(self._sigma_bar, t_int, t_normalized, key)

    def get_sigma2_bar(
        self,
        t_normalized: torch.Tensor | None = None,
        t_int: torch.Tensor | None = None,
        key: str | None = None,
    ) -> torch.Tensor:
        """``sigma_bar_t^2`` for one modality."""
        return self._lookup(self._sigma2_bar, t_int, t_normalized, key)

    def get_gamma(
        self,
        t_normalized: torch.Tensor | None = None,
        t_int: torch.Tensor | None = None,
        key: str | None = None,
    ) -> torch.Tensor:
        """``gamma_t`` (log SNR) for one modality."""
        return self._lookup(self._gamma, t_int, t_normalized, key)

    # -- transition matrices ------------------------------------------------

    def get_Qt(self, t_int: torch.Tensor) -> PlaceHolder:  # noqa: N802
        """One-step transition matrices from ``t-1`` to ``t``."""
        p = self.move_P_device(t_int)
        kwargs = {"device": t_int.device, "dtype": torch.float32}

        bx = self.get_beta(t_int=t_int, key="x").unsqueeze(1)
        q_x = bx * p.X + (1 - bx) * torch.eye(
            self.X_classes, **kwargs
        ).unsqueeze(0)

        bc = self.get_beta(t_int=t_int, key="c").unsqueeze(1)
        q_c = bc * p.charges + (1 - bc) * torch.eye(
            self.charges_classes, **kwargs
        ).unsqueeze(0)

        be = self.get_beta(t_int=t_int, key="e").unsqueeze(1)
        q_e = be * p.E + (1 - be) * torch.eye(
            self.E_classes, **kwargs
        ).unsqueeze(0)

        by = self.get_beta(t_int=t_int, key="y").unsqueeze(1)
        q_y = by * p.y + (1 - by) * torch.eye(
            self.y_classes, **kwargs
        ).unsqueeze(0)

        return PlaceHolder(X=q_x, charges=q_c, E=q_e, y=q_y, pos=None)

    def get_Qt_bar(self, t_int: torch.Tensor) -> PlaceHolder:  # noqa: N802
        """Cumulative transition matrices from ``0`` to ``t``."""
        a_x = self.get_alpha_bar(t_int=t_int, key="x").unsqueeze(1)
        a_c = self.get_alpha_bar(t_int=t_int, key="c").unsqueeze(1)
        a_e = self.get_alpha_bar(t_int=t_int, key="e").unsqueeze(1)
        a_y = self.get_alpha_bar(t_int=t_int, key="y").unsqueeze(1)

        p = self.move_P_device(t_int)
        dev = t_int.device
        q_x = a_x * torch.eye(self.X_classes, device=dev).unsqueeze(0) + (
            1 - a_x
        ) * p.X
        q_c = a_c * torch.eye(self.charges_classes, device=dev).unsqueeze(0) + (
            1 - a_c
        ) * p.charges
        q_e = a_e * torch.eye(self.E_classes, device=dev).unsqueeze(0) + (
            1 - a_e
        ) * p.E
        q_y = a_y * torch.eye(self.y_classes, device=dev).unsqueeze(0) + (
            1 - a_y
        ) * p.y

        if not ((q_x.sum(dim=2) - 1.0).abs() < 1e-4).all():
            msg = "atom-type transition matrix rows do not sum to 1"
            raise ValueError(msg)
        if not ((q_e.sum(dim=2) - 1.0).abs() < 1e-4).all():
            msg = "bond transition matrix rows do not sum to 1"
            raise ValueError(msg)

        return PlaceHolder(X=q_x, charges=q_c, E=q_e, y=q_y, pos=None)

    # -- continuous (position) schedule -------------------------------------

    def get_alpha_pos_ts(
        self, t_int: torch.Tensor, s_int: torch.Tensor
    ) -> torch.Tensor:
        """``alpha_t / alpha_s`` for positions."""
        log_a_bar = self._log_alpha_bar[..., self.inverse_mapping["p"]].to(
            t_int.device
        )
        return torch.exp(log_a_bar[t_int] - log_a_bar[s_int]).float()

    def get_alpha_pos_ts_sq(
        self, t_int: torch.Tensor, s_int: torch.Tensor
    ) -> torch.Tensor:
        """``(alpha_t / alpha_s)^2`` for positions."""
        log_a_bar = self._log_alpha_bar[..., self.inverse_mapping["p"]].to(
            t_int.device
        )
        return torch.exp(2 * log_a_bar[t_int] - 2 * log_a_bar[s_int]).float()

    def get_sigma_pos_sq_ratio(
        self, s_int: torch.Tensor, t_int: torch.Tensor
    ) -> torch.Tensor:
        """``sigma_s^2 / sigma_t^2`` for positions."""
        log_a_bar = self._log_alpha_bar[..., self.inverse_mapping["p"]].to(
            t_int.device
        )
        s2_s = -torch.expm1(2 * log_a_bar[s_int])
        s2_t = -torch.expm1(2 * log_a_bar[t_int])
        return torch.exp(torch.log(s2_s) - torch.log(s2_t)).float()

    def get_x_pos_prefactor(
        self, s_int: torch.Tensor, t_int: torch.Tensor
    ) -> torch.Tensor:
        """``a_s (s_t^2 - a_ts^2 s_s^2) / s_t^2``."""
        a_s = self.get_alpha_bar(t_int=s_int, key="p")
        alpha_ratio_sq = self.get_alpha_pos_ts_sq(t_int=t_int, s_int=s_int)
        sigma_ratio_sq = self.get_sigma_pos_sq_ratio(s_int=s_int, t_int=t_int)
        return (a_s * (1 - alpha_ratio_sq * sigma_ratio_sq)).float()

    # -- forward / reverse --------------------------------------------------

    def apply_noise(self, dense_data: PlaceHolder) -> PlaceHolder:
        """Sample ``t`` and return the noised batch ``z_t``."""
        device = dense_data.X.device
        t_int = torch.randint(
            1, self.T + 1, size=(dense_data.X.size(0), 1), device=device
        )
        t_float = t_int.float() / self.T

        qtb = self.get_Qt_bar(t_int=t_int)

        prob_x = dense_data.X @ qtb.X  # (bs, n, dx_out)
        prob_charges = dense_data.charges @ qtb.charges
        prob_e = dense_data.E @ qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(
            probX=prob_x,
            probE=prob_e,
            prob_charges=prob_charges,
            node_mask=dense_data.node_mask,
        )

        x_t = F.one_hot(sampled_t.X, num_classes=self.X_classes).float()
        e_t = F.one_hot(sampled_t.E, num_classes=self.E_classes).float()
        charges_t = F.one_hot(
            sampled_t.charges, num_classes=self.charges_classes
        ).float()

        noise_pos = torch.randn(dense_data.pos.shape, device=device)
        noise_pos = noise_pos * dense_data.node_mask.unsqueeze(-1)
        noise_pos = remove_mean_with_mask(noise_pos, dense_data.node_mask)

        a = self.get_alpha_bar(t_int=t_int, key="p").unsqueeze(-1)
        s = self.get_sigma_bar(t_int=t_int, key="p").unsqueeze(-1)
        pos_t = a * dense_data.pos + s * noise_pos

        return PlaceHolder(
            X=x_t,
            charges=charges_t,
            E=e_t,
            y=dense_data.y,
            pos=pos_t,
            t_int=t_int,
            t=t_float,
            node_mask=dense_data.node_mask,
        ).mask()

    def get_limit_dist(self) -> PlaceHolder:
        """Smoothed marginals used as the ``t = T`` prior."""
        x_marginals = self.X_marginals + 1e-7
        x_marginals = x_marginals / torch.sum(x_marginals)
        e_marginals = self.E_marginals + 1e-7
        e_marginals = e_marginals / torch.sum(e_marginals)
        charges_marginals = self.charges_marginals + 1e-7
        charges_marginals = charges_marginals / torch.sum(charges_marginals)
        return PlaceHolder(
            X=x_marginals,
            E=e_marginals,
            charges=charges_marginals,
            y=None,
            pos=None,
        )

    def sample_limit_dist(self, node_mask: torch.Tensor) -> PlaceHolder:
        """Draw ``z_T`` from the limit (marginal) distribution."""
        bs, n_max = node_mask.shape
        device = node_mask.device
        x_limit = self.X_marginals.to(device).expand(bs, n_max, -1)
        e_limit = (
            self.E_marginals.to(device)[None, None, None, :]
            .expand(bs, n_max, n_max, -1)
        )
        charges_limit = self.charges_marginals.to(device).expand(bs, n_max, -1)

        u_x = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max)
        u_c = (
            charges_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max)
        )
        u_e = (
            e_limit.flatten(end_dim=-2)
            .multinomial(1)
            .reshape(bs, n_max, n_max)
        )
        u_y = torch.zeros((bs, 0), device=device)

        u_x = F.one_hot(u_x, num_classes=x_limit.shape[-1]).float()
        u_e = F.one_hot(u_e, num_classes=e_limit.shape[-1]).float()
        u_c = F.one_hot(u_c, num_classes=charges_limit.shape[-1]).float()

        # Keep the strict upper triangle, then mirror it: symmetry by
        # construction rather than by hope.
        upper_triangular_mask = torch.zeros_like(u_e)
        indices = torch.triu_indices(
            row=u_e.size(1), col=u_e.size(2), offset=1
        )
        upper_triangular_mask[:, indices[0], indices[1], :] = 1
        u_e = u_e * upper_triangular_mask
        u_e = u_e + torch.transpose(u_e, 1, 2)
        if not (u_e == torch.transpose(u_e, 1, 2)).all():
            msg = "prior edge tensor is not symmetric"
            raise ValueError(msg)

        pos = torch.randn(bs, n_max, 3, device=device)
        pos = pos * node_mask.unsqueeze(-1)
        pos = remove_mean_with_mask(pos, node_mask)

        t_array = pos.new_ones((bs, 1))
        return PlaceHolder(
            X=u_x,
            charges=u_c,
            E=u_e,
            y=u_y,
            pos=pos,
            t_int=self.T * t_array.long(),
            t=t_array,
            node_mask=node_mask,
        ).mask(node_mask)

    def sample_zs_from_zt_and_pred(  # noqa: PLR0914
        self, z_t: PlaceHolder, pred: PlaceHolder, s_int: torch.Tensor
    ) -> PlaceHolder:
        """One reverse step: ``z_s ~ p(z_s | z_t)`` given the denoiser output."""
        bs, n, _dxs = z_t.X.shape
        node_mask = z_t.node_mask
        t_int = z_t.t_int

        qtb = self.get_Qt_bar(t_int=t_int)
        qsb = self.get_Qt_bar(t_int=s_int)
        qt = self.get_Qt(t_int)

        # Positions: Gaussian posterior.
        sigma_sq_ratio = self.get_sigma_pos_sq_ratio(s_int=s_int, t_int=t_int)
        z_t_prefactor = (
            self.get_alpha_pos_ts(t_int=t_int, s_int=s_int) * sigma_sq_ratio
        ).unsqueeze(-1)
        x_prefactor = self.get_x_pos_prefactor(
            s_int=s_int, t_int=t_int
        ).unsqueeze(-1)

        mu = z_t_prefactor * z_t.pos + x_prefactor * pred.pos  # bs, n, 3

        sampled_pos = torch.randn(
            z_t.pos.shape, device=z_t.pos.device
        ) * node_mask.unsqueeze(-1)
        noise = remove_mean_with_mask(sampled_pos, node_mask=node_mask)

        prefactor1 = self.get_sigma2_bar(t_int=t_int, key="p")
        prefactor2 = self.get_sigma2_bar(
            t_int=s_int, key="p"
        ) * self.get_alpha_pos_ts_sq(t_int=t_int, s_int=s_int)
        noise_prefactor = torch.sqrt(
            (prefactor1 - prefactor2) * sigma_sq_ratio
        ).unsqueeze(-1)

        pos = mu + noise_prefactor * noise  # bs, n, 3

        # Categoricals: D3PM posterior, marginalized over the x_0 prediction.
        pred_x = F.softmax(pred.X, dim=-1)
        pred_e = F.softmax(pred.E, dim=-1)
        pred_charges = F.softmax(pred.charges, dim=-1)

        p_s_and_t_given_0_x = (
            diffusion_utils.compute_batched_over0_posterior_distribution(
                X_t=z_t.X, Qt=qt.X, Qsb=qsb.X, Qtb=qtb.X
            )
        )
        p_s_and_t_given_0_e = (
            diffusion_utils.compute_batched_over0_posterior_distribution(
                X_t=z_t.E, Qt=qt.E, Qsb=qsb.E, Qtb=qtb.E
            )
        )
        p_s_and_t_given_0_c = (
            diffusion_utils.compute_batched_over0_posterior_distribution(
                X_t=z_t.charges,
                Qt=qt.charges,
                Qsb=qsb.charges,
                Qtb=qtb.charges,
            )
        )

        prob_x = _marginalize(pred_x, p_s_and_t_given_0_x)
        prob_c = _marginalize(pred_charges, p_s_and_t_given_0_c)

        pred_e = pred_e.reshape((bs, -1, pred_e.shape[-1]))
        prob_e = _marginalize(pred_e, p_s_and_t_given_0_e)
        prob_e = prob_e.reshape(bs, n, n, pred_e.shape[-1])

        sampled_s = diffusion_utils.sample_discrete_features(
            prob_x, prob_e, prob_c, node_mask=node_mask
        )

        x_s = F.one_hot(sampled_s.X, num_classes=self.X_classes).float()
        charges_s = F.one_hot(
            sampled_s.charges, num_classes=self.charges_classes
        ).float()
        e_s = F.one_hot(sampled_s.E, num_classes=self.E_classes).float()

        return PlaceHolder(
            X=x_s,
            charges=charges_s,
            E=e_s,
            y=torch.zeros(z_t.y.shape[0], 0, device=x_s.device),
            pos=pos,
            t_int=s_int,
            t=s_int / self.T,
            node_mask=node_mask,
        ).mask(node_mask)


def _marginalize(pred: torch.Tensor, posterior: torch.Tensor) -> torch.Tensor:
    """Sum the ``x_0``-conditioned posterior against the predicted ``x_0``."""
    weighted = pred.unsqueeze(-1) * posterior  # bs, N, d0, d_t-1
    unnormalized = weighted.sum(dim=-2)  # bs, N, d_t-1
    unnormalized[torch.sum(unnormalized, dim=-1) == 0] = 1e-5
    prob = unnormalized / torch.sum(unnormalized, dim=-1, keepdim=True)
    if not ((prob.sum(dim=-1) - 1).abs() < 1e-4).all():
        msg = "posterior does not normalize"
        raise ValueError(msg)
    return prob


class DiscreteUniformTransition(NoiseModel):
    """Uniform limit distribution over every categorical modality."""

    def __init__(
        self,
        output_dims: Dims,
        nu: dict[str, float],
        diffusion_steps: int = 500,
        noise_schedule: str = "cosine",
    ) -> None:
        super().__init__(
            nu=nu,
            diffusion_steps=diffusion_steps,
            noise_schedule=noise_schedule,
        )
        self.X_classes = output_dims.X
        self.charges_classes = output_dims.charges
        self.E_classes = output_dims.E
        self.y_classes = output_dims.y
        self.X_marginals = torch.ones(self.X_classes) / self.X_classes
        self.charges_marginals = (
            torch.ones(self.charges_classes) / self.charges_classes
        )
        self.E_marginals = torch.ones(self.E_classes) / self.E_classes
        self.y_marginals = torch.ones(self.y_classes) / max(self.y_classes, 1)
        self.Px = (
            torch.ones(1, self.X_classes, self.X_classes) / self.X_classes
        )
        self.Pcharges = (
            torch.ones(1, self.charges_classes, self.charges_classes)
            / self.charges_classes
        )
        self.Pe = torch.ones(1, self.E_classes, self.E_classes) / self.E_classes
        self.Py = torch.ones(1, self.y_classes, self.y_classes) / max(
            self.y_classes, 1
        )


class MarginalUniformTransition(NoiseModel):
    """Limit distribution = the dataset's own class marginals.

    This is what every released MiDi config uses (``transition: marginal``),
    and the marginals come from ``train_set.graph3d_stats`` -- not from the
    checkpoint, which holds ``nn.Module`` weights only.
    """

    def __init__(  # noqa: PLR0913
        self,
        x_marginals: torch.Tensor,
        e_marginals: torch.Tensor,
        charges_marginals: torch.Tensor,
        y_classes: int,
        nu: dict[str, float],
        diffusion_steps: int = 500,
        noise_schedule: str = "cosine",
    ) -> None:
        super().__init__(
            nu=nu,
            diffusion_steps=diffusion_steps,
            noise_schedule=noise_schedule,
        )
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.charges_classes = len(charges_marginals)
        self.y_classes = y_classes
        self.X_marginals = x_marginals
        self.E_marginals = e_marginals
        self.charges_marginals = charges_marginals
        self.y_marginals = torch.ones(y_classes) / max(y_classes, 1)

        self.Px = x_marginals.unsqueeze(0).expand(
            self.X_classes, -1
        ).unsqueeze(0)
        self.Pe = e_marginals.unsqueeze(0).expand(
            self.E_classes, -1
        ).unsqueeze(0)
        self.Pcharges = charges_marginals.unsqueeze(0).expand(
            self.charges_classes, -1
        ).unsqueeze(0)
        self.Py = torch.ones(1, y_classes, y_classes) / max(y_classes, 1)
