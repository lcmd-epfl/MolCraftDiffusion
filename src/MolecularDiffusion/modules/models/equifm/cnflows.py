"""EquiFM continuous normalizing flow -- the generative core.

Ported from ``others/MolFM/sampling/cnf_models.py`` (``Cnflows``) with two
deliberate differences, both mandated by the approved integration plan:

* **The training objective is a reconstruction of the paper's Algorithm 1.**
  MolFM's release is sampling-only: its ``Cnflows`` has no ``forward``, no
  loss, and no EOT solver -- the ctor stores ``loss_type``/``cat_loss``/
  ``angle_penalty`` and never reads them. ``compute_loss`` below is
  reconstructed from Algorithm 1 (p. 14) and Algorithm 3 (p. 18) of
  arXiv:2312.07168. It is **not** the authors' released objective: their
  ``args.pickle`` shows the released QM9 weights were trained with
  ``angle_penalty=True``, ``cat_loss='l2_masked_mean'`` and
  ``ode_regularization=0.001``, none of which appear anywhere in the paper.
  A model trained here will therefore not reproduce the paper's Table 1
  numbers, and any gap must not be reported as a reproduction failure.

* **The ODE solver is fixed-step RK4, not ``torchdiffeq``'s adaptive dopri5.**
  ``torchdiffeq`` is not installed and is not a declared dependency. Paper
  Fig. 3 (p. 9) benchmarks EquiFM with Euler and midpoint integrators and shows
  both reaching ~0.87-0.88 molecule stability against the dopri5 headline of
  0.883, so a fixed-step solver costs essentially nothing -- and unlike an
  adaptive solver it can honour the platform's ``num_steps``.
  ponytail: fixed-step RK4; wire torchdiffeq dopri5 behind an optional import
  only if exact paper NFE numbers are ever needed.

The EGNN backbone is *not* vendored: ``modules/models/geoldm/networks.py``'s
``EGNN_dynamics_QM9`` was verified token-identical to MolFM's, and the released
checkpoint's ``dynamics.egnn.*`` keys already match it.

Time convention: ``t = 1`` is noise, ``t = 0`` is data (MolFM's convention).
"""

from typing import Optional

import torch
import torch.nn as nn

from MolecularDiffusion.utils import (
    remove_mean_with_mask,
    sample_center_gravity_zero_gaussian_with_mask,
)

from .eot import solve_eot
from .paths import m_para, ot_interpolate, vp_interpolate

DISCRETE_PATHS = ("OT_path", "HB_path", "VP_path")


class Cnflows(nn.Module):
    """E(n)-equivariant CNF with hybrid (OT on x, VP on h) probability transport."""

    def __init__(
        self,
        dynamics: nn.Module,
        in_node_nf: int,
        n_dims: int = 3,
        include_charges: bool = True,
        norm_values=(1.0, 4.0, 10.0),
        norm_biases=(None, 0.0, 0.0),
        discrete_path: str = "HB_path",
        sigma_min: float = 1e-4,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        use_eot: bool = True,
        eot_max_iters: int = 20,
    ):
        super().__init__()
        if discrete_path not in DISCRETE_PATHS:
            raise ValueError(
                f"discrete_path must be one of {DISCRETE_PATHS}, got {discrete_path!r}"
            )
        self.dynamics = dynamics
        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.include_charges = int(include_charges)
        self.num_classes = in_node_nf - self.include_charges
        self.norm_values = tuple(norm_values)
        self.norm_biases = tuple(norm_biases)
        self.discrete_path = discrete_path
        self.sigma_min = sigma_min
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.use_eot = use_eot
        self.eot_max_iters = eot_max_iters
        # Present in the released checkpoint; kept so the key maps 1:1.
        self.register_buffer("buffer", torch.zeros(1))

    # ------------------------------------------------------------------ #
    # normalization (identical to EDM/GeoLDM)                            #
    # ------------------------------------------------------------------ #
    def normalize(self, x, h_cat, h_int, node_mask):
        x = x / self.norm_values[0]
        h_cat = (h_cat.float() - self.norm_biases[1]) / self.norm_values[1] * node_mask
        h_int = (h_int.float() - self.norm_biases[2]) / self.norm_values[2]
        if self.include_charges:
            h_int = h_int * node_mask
        return x, h_cat, h_int

    def unnormalize(self, x, h_cat, h_int, node_mask):
        x = x * self.norm_values[0]
        h_cat = (h_cat * self.norm_values[1] + self.norm_biases[1]) * node_mask
        h_int = h_int * self.norm_values[2] + self.norm_biases[2]
        if self.include_charges:
            h_int = h_int * node_mask
        return x, h_cat, h_int

    # ------------------------------------------------------------------ #
    # vector field + hybrid reweighting                                  #
    # ------------------------------------------------------------------ #
    def phi(self, t, xh, node_mask, edge_mask, context):
        return self.dynamics._forward(t, xh, node_mask, edge_mask, context)

    def reweight_factor(self, t: torch.Tensor, dims: int) -> torch.Tensor:
        """Per-channel multiplier applied to the network output at sampling time.

        This is the ``M_para`` reweighting of MolFM's sampler
        (cnf_models.py:261-270). **It is also divided out of the training target**
        in :meth:`compute_loss`, which is the whole point of exposing
        ``discrete_path`` as a config field.

        The paper and the release disagree here, and the disagreement is not
        cosmetic: Algorithm 1 line 9 regresses ``v_theta^h`` onto the *full* VP
        velocity, which already contains the ``alpha'_t / (1 - alpha_t^2)``
        prefactor; the released sampler then multiplies the network's ``h``
        output by ``M_para`` (and ``alpha'_t/(1-alpha_t^2) == M_para * alpha_t``),
        double-counting it. Rather than pick a winner and bake it in silently,
        both conventions are supported and kept self-consistent by construction:

        * ``OT_path``  -- factor 1. The network emits the full velocity, so this
          is the verbatim Algorithm-1 model. Use it for anything trained here.
        * ``HB_path``  -- factor ``M_para`` on the ``h`` channels only. The
          network emits ``velocity / M_para``. This is what the released QM9
          checkpoint was trained to emit (``args.pickle``:
          ``discrete_path='HB_path'``), so the converted checkpoint MUST be
          sampled with it.
        * ``VP_path``  -- factor ``M_para`` on every channel. Included for the
          Table 3 ablation row; no released weights use it.
        """
        factor = torch.ones(t.size(0), 1, dims, device=t.device, dtype=t.dtype)
        if self.discrete_path == "OT_path":
            return factor
        mp = m_para(t, self.beta_min, self.beta_max).view(-1, 1, 1)
        if self.discrete_path == "VP_path":
            return factor * mp
        factor[:, :, self.n_dims :] = mp
        return factor

    # ------------------------------------------------------------------ #
    # training -- paper Algorithm 1 (RECONSTRUCTED, see module docstring) #
    # ------------------------------------------------------------------ #
    def compute_loss(self, x_0, h_cat, h_int, node_mask, edge_mask, context=None):
        """One Algorithm-1 training step. Returns ``(loss, stats)``."""
        bsz, n_nodes, _ = x_0.shape
        device = x_0.device

        x_0, h_cat, h_int = self.normalize(x_0, h_cat, h_int, node_mask)
        h_0 = torch.cat([h_cat, h_int], dim=2) if self.include_charges else h_cat

        # Alg. 1 line 4-5: t ~ U(0,1); eps ~ N(0,I) with eps_x centre-of-gravity
        # free. The clamp keeps the VP target finite at t -> 0.
        t = torch.rand(bsz, 1, device=device).clamp_(1e-4, 1.0)
        t_b = t.view(bsz, 1, 1)
        eps_x = sample_center_gravity_zero_gaussian_with_mask(
            (bsz, n_nodes, self.n_dims), device, node_mask
        )
        eps_h = torch.randn(bsz, n_nodes, h_0.size(2), device=device) * node_mask

        # Alg. 1 line 6: Equivariant Optimal Transport plan (Algorithm 3).
        if self.use_eot:
            eps_x = solve_eot(eps_x, x_0, node_mask, max_iters=self.eot_max_iters)

        # Alg. 1 lines 7-8.
        x_t, u_x = ot_interpolate(x_0, eps_x, t_b, self.sigma_min)
        h_t, u_h = vp_interpolate(h_0, eps_h, t_b, self.beta_min, self.beta_max)
        x_t = remove_mean_with_mask(x_t * node_mask, node_mask)
        h_t = h_t * node_mask

        net_out = self.phi(
            t, torch.cat([x_t, h_t], dim=2), node_mask, edge_mask, context
        )

        # Alg. 1 line 9. The target is divided by the sampling-time reweighting
        # so that `reweight_factor(t) * net_out` is the true velocity for every
        # `discrete_path` -- see reweight_factor's docstring.
        target = torch.cat([u_x, u_h], dim=2) / self.reweight_factor(
            t, self.n_dims + h_0.size(2)
        )
        err = ((net_out - target) ** 2) * node_mask
        n_real = node_mask.sum() + 1e-8
        loss_x = err[:, :, : self.n_dims].sum() / (n_real * self.n_dims)
        loss_h = err[:, :, self.n_dims :].sum() / (n_real * h_0.size(2))
        loss = loss_x + loss_h
        return loss, {
            "loss_x": loss_x.detach(),
            "loss_h": loss_h.detach(),
            "loss": loss.detach(),
        }

    # ------------------------------------------------------------------ #
    # sampling                                                           #
    # ------------------------------------------------------------------ #
    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        z_x = sample_center_gravity_zero_gaussian_with_mask(
            (n_samples, n_nodes, self.n_dims), node_mask.device, node_mask
        )
        z_h = torch.randn(
            n_samples, n_nodes, self.in_node_nf, device=node_mask.device
        ) * node_mask
        return torch.cat([z_x, z_h], dim=2)

    def decode(self, z, node_mask, edge_mask, context, num_steps: int):
        """Fixed-step RK4 integration of ``dz/dt`` from ``t = 1`` down to ``t = 0``.

        ~15 lines instead of ``torchdiffeq.odeint(method='dopri5')``; see the
        module docstring for why. 4 network evaluations per step.
        """
        bsz = z.size(0)
        dims = z.size(2)
        dt = -1.0 / num_steps

        def f(t_val: float, state: torch.Tensor) -> torch.Tensor:
            t = torch.full((bsz, 1), t_val, device=z.device, dtype=z.dtype)
            out = self.phi(t, state, node_mask, edge_mask, context)
            return out * self.reweight_factor(t, dims)

        t_val = 1.0
        for _ in range(num_steps):
            k1 = f(t_val, z)
            k2 = f(t_val + 0.5 * dt, z + 0.5 * dt * k1)
            k3 = f(t_val + 0.5 * dt, z + 0.5 * dt * k2)
            k4 = f(t_val + dt, z + dt * k3)
            z = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t_val += dt
        return z

    def sample_p_xh_given_z0(self, z0, node_mask):
        """Unnormalize, then invert the uniform dequantization (a plain round).

        MolFM routes this through ``UniformDequantizer.reverse``
        (cnf_models.py:679-682), which is exactly ``torch.round`` on both
        channel groups -- not worth a class.
        """
        x = z0[:, :, : self.n_dims]
        h_cat = z0[:, :, self.n_dims : self.n_dims + self.num_classes]
        h_int = (
            z0[:, :, self.n_dims + self.num_classes :]
            if self.include_charges
            else torch.zeros(0, device=z0.device)
        )
        x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)
        return x, torch.round(h_cat), torch.round(h_int)

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        n_nodes: int,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        num_steps: int = 250,
    ):
        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)
        z = self.decode(z, node_mask, edge_mask, context, num_steps)
        x, h_cat, h_int = self.sample_p_xh_given_z0(z, node_mask)
        x = remove_mean_with_mask(x * node_mask, node_mask)
        return x, h_cat, h_int
