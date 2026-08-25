"""React-OT's E(n) Schrodinger bridge over reactant / TS / product.

Vendored from ``reactot/diffusion/en_sb.py`` of
https://github.com/deepprinciple/react-ot at commit 6dfccd0.

Duan, C.; Liu, G.-H.; Du, Y.; ...; Kulik, H. J., *Optimal transport for
generating transition states in chemical reactions*, Nature Machine
Intelligence (2025), doi:10.1038/s42256-025-01010-0. Preprint (a **different
title**, same work): arXiv:2404.13430, *React-OT: Optimal Transport for
Generating Transition State in Chemical Reactions*.

This is **not** a diffusion model and not flow matching. It is a Schrodinger
bridge between two fixed endpoints, solved -- on the released settings --
as a deterministic optimal-transport ODE:

* ``x1`` is the **midpoint of the reactant and product coordinates**,
  ``(r_pos + p_pos) / 2``;
* ``x0`` is the transition state;
* training draws ``t ~ U[0, T)``, interpolates
  ``xt = mu_x0[t] * x0 + mu_x1[t] * x1``, writes ``xt`` into the TS object's
  position columns, runs the network once and regresses
  ``(xt - x0) / std_fwd[t]`` under plain ``F.mse_loss``;
* ``ot_ode=True`` and ``sigma=0.0`` mean **no Gaussian noise is drawn
  anywhere**, in training or in sampling. Run it twice, get the same answer.

There is no ELBO, no KL prior, no ``loss_0`` term and no per-object
``scales`` weighting -- ``scales`` reaches only upstream's dead ``DDPMModule``
path, never ``EnSB``. That is why
:class:`~MolecularDiffusion.modules.tasks.diffusion_reactot.ReactOTTask`'s
``forward`` is four lines where OA-ReactDiff's is sixty.

What is reused rather than re-vendored
--------------------------------------

The **network is the same object-aware LEFTNet** OA-ReactDiff already ships,
verified by ``diff -u``: identical constructor (14 kwargs, same order, same
defaults) and a forward that differs only by two periodic-boundary kwargs
whose ``pbc=False`` path is the code already in the tree. So ``LEFTNet``,
``EGNNDynamics``, ``Normalizer``/``FEATURE_MAPPING`` and every graph helper
are imported from ``modules/models/oareactdiff/``; nothing there is edited.
What is genuinely new -- the bridge and its schedule -- is what lives here.

Scope, and what was stripped when vendoring
-------------------------------------------

* **Only ``mapping="R+P->TS"`` with ``mapping_initial="RP"``.** The other
  three mappings and five other initialisers have no released weights and no
  data path; they raise rather than silently running.
* **No ``ei`` solver** (upstream's ``en_sb.py:552-666``): a research variant,
  absent from the README, needing a 10,000-sample Monte-Carlo quadrature per
  Adams-Bashforth coefficient. ``ddpm`` and ``ode`` cover every documented
  use.
* **No ``torchdiffeq``.** :meth:`EnSB.ode_sampling` vendors the single
  midpoint step -- see its ``ponytail:`` comment.
* Upstream's monkey-patched ``self.opt`` namespace (set from outside at
  ``evaluation.py:56``) is replaced by real named arguments; ``ipdb``,
  ``colored_traceback``, in-sampler ``tqdm`` (the platform's
  ``PocketGenerator.run`` owns the progress bar) and the dead
  ``return_timesteps=True`` branch are gone.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from MolecularDiffusion.modules.models.oareactdiff.dynamics import (
    EGNNDynamics,
)
from MolecularDiffusion.modules.models.oareactdiff.graph_tools import (
    get_edges_index,
    get_mask_for_frag,
    get_n_frag_switch,
    get_subgraph_mask,
    remove_mean_batch,
)
from MolecularDiffusion.modules.models.oareactdiff.normalizer import (
    FEATURE_MAPPING,
    Normalizer,
)
from MolecularDiffusion.modules.models.reactot.schedule import (
    SBSchedule,
    compute_gaussian_product_coef,
    space_indices,
)

#: The only mapping this port implements. See the module docstring.
MAPPING = "R+P->TS"
#: The only initialiser: the reactant/product midpoint.
MAPPING_INITIAL = "RP"
#: Index of the generated object inside ``representations``.
TS_IDX = 1


def compute_scaled_err(x: Tensor, y: Tensor) -> Tensor:
    """Mean absolute error scaled by the label's own magnitude.

    Upstream's checkpoint callback monitors ``val_ep_scaled_err``, not the
    loss, so this travels alongside it (``en_sb.py:26-28``).

    Args:
        x: prediction.
        y: label.

    Returns:
        0-dim scalar.
    """
    max_y = torch.max(torch.abs(y))
    return torch.mean(torch.abs((x - y) / max_y))


class EnSB(nn.Module):
    """The E(n) Schrodinger bridge module.

    Attributes:
        dynamics: the object-aware ``EGNNDynamics`` wrapper around LEFTNet.
        schedule: the bridge's :class:`SBSchedule`. **Not** in the state
            dict -- it is a plain Python object.
        normalizer: identity at the released ``(1,1,1)/(0,0,0)`` settings,
            kept because it is on the path upstream.
        T: ``schedule.timesteps`` (3000 for the released weights).
    """

    def __init__(
        self,
        dynamics: EGNNDynamics,
        schedule: SBSchedule,
        normalizer: Normalizer,
        size_histogram: Optional[Dict[str, Any]] = None,
        loss_type: str = "l2",
        pos_only: bool = False,
        fixed_idx: Optional[List[int]] = None,
        mapping: str = MAPPING,
        mapping_initial: str = MAPPING_INITIAL,
        sigma: float = 0.0,
        ts_guess: Optional[Any] = None,
        idx: int = TS_IDX,
    ) -> None:
        """Assemble the bridge.

        Args:
            dynamics: object-aware denoiser.
            schedule: bridge schedule. Upstream spells this parameter
                ``schdule``; the typo is not preserved, because unlike
                OA-ReactDiff's ``EnVariationalDiffusion`` this class is not
                a verbatim vendoring and nothing constructs it by that name.
            normalizer: feature normaliser.
            size_histogram: accepted and ignored, exactly as upstream --
                it is passed ``None`` (``pl_trainer.py:813``) and never read,
                which is why no ``train_set`` is needed at build time.
            loss_type: ``"vlb"`` or ``"l2"``; asserted, then unread.
            pos_only: coordinates only. The released weights are ``True``.
            fixed_idx: **inert**, recorded for fidelity. Upstream stores it
                (``en_sb.py:61``) and never reads it: freezing R and P is
                structural, since only ``xh_t[idx]``'s position columns are
                ever overwritten.
            mapping: must be ``"R+P->TS"``.
            mapping_initial: must be ``"RP"``.
            sigma: endpoint jitter. ``0.0`` on the released weights, and the
                code paths that would use it are commented out upstream.
            ts_guess: must be falsy -- ``ts_guess`` conditioning is out of
                scope and unreachable on the released checkpoint, whose
                ``hyper_parameters`` record ``ts_guess = None``.
            idx: which object is generated; must be 1 (the TS).

        Raises:
            ValueError: for any out-of-scope mapping / initialiser /
                ``ts_guess``.
        """
        super().__init__()
        if loss_type not in {"vlb", "l2"}:
            raise ValueError(f"loss_type {loss_type!r} not in {{vlb, l2}}")
        if mapping != MAPPING or mapping_initial != MAPPING_INITIAL:
            raise ValueError(
                f"this port implements mapping={MAPPING!r} with "
                f"mapping_initial={MAPPING_INITIAL!r} only, not "
                f"{mapping!r}/{mapping_initial!r}. The other three mappings "
                "(R->P, R+TS->P, TS+P->R) and the five other initialisers "
                "(GUESS, R, P, Gaussian, Zeros) have no released weights and "
                "no data path in this integration."
            )
        if idx != TS_IDX:
            raise ValueError(
                f"idx must be {TS_IDX} (the transition state) for "
                f"mapping={MAPPING!r}, not {idx}."
            )
        if ts_guess:
            raise ValueError(
                "ts_guess conditioning is out of scope: the released "
                "checkpoint records ts_guess=None, so its GUESS branch is "
                "unreachable and `conditions` stays a plain (B, 1) zero "
                "tensor. The `ts_guess_NEBCI-xtb-ema` string in upstream's "
                "evaluation.py:28 is a run-directory name, not a data "
                "requirement."
            )

        self.dynamics = dynamics
        self.schedule = schedule
        self.normalizer = normalizer
        self.size_histogram = size_histogram
        self.loss_type = loss_type
        self.pos_only = pos_only
        self.fixed_idx = list(fixed_idx) if fixed_idx else []

        self.pos_dim = dynamics.pos_dim
        self.node_nfs = dynamics.node_nfs
        self.fragment_names = dynamics.fragment_names
        self.T = schedule.timesteps
        self.norm_values = normalizer.norm_values
        self.norm_biases = normalizer.norm_biases

        self.mapping = mapping
        self.mapping_initial = mapping_initial
        self.sigma = sigma
        self.ts_guess = ts_guess
        self.idx = idx

    # ------ forward pass -------------------------------------------------

    def sample_batch(
        self,
        representations: List[Dict[str, Tensor]],
        conditions: Tensor,  # noqa: ARG002 - kept for signature parity
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor], Tensor, Tensor]:
        """Endpoints and graph structure for one batch of reactions.

        Args:
            representations: the three per-object dicts, in ``[R, TS, P]``
                order, as :func:`~MolecularDiffusion.data.component.
                oareactdiff_data.oareactdiff_collate` builds them.
            conditions: accepted and unused. Upstream reads
                ``conditions["ts_guess"]`` here only on the out-of-scope
                ``GUESS`` initialiser; on this path the channel is the
                constant-zero ``(B, 1)`` tensor the released weights saw.

        Returns:
            ``(x0, x1, cond, x0_size, x0_other)`` -- the transition state,
            the reactant/product midpoint (CoM-removed per sample), a dict
            of graph tensors, the TS atom counts and the TS's non-positional
            columns.
        """

        def parse(
            features: Dict[str, Tensor],
        ) -> Tuple[Tensor, Tensor, Tensor]:
            other = torch.cat(
                [features["one_hot"], features["charge"]], dim=1
            ).float()
            return features["pos"], features["size"], other

        r_pos, r_size, r_other = parse(representations[0])
        t_pos, t_size, t_other = parse(representations[1])
        p_pos, p_size, p_other = parse(representations[2])

        # mapping_initial == "RP": the bridge starts at the midpoint of the
        # two endpoint geometries (en_sb.py:128).
        x1 = (r_pos + p_pos) / 2
        x0 = t_pos
        cond: Dict[str, Tensor] = {
            "hs": torch.stack([r_other, t_other, p_other]),
            "r_pos": r_pos.detach(),
            "p_pos": p_pos.detach(),
        }
        fragments_nodes = [r_size, t_size, p_size]
        x0_size, x0_other = t_size, t_other

        fragments_masks = [
            get_mask_for_frag(natm_nodes) for natm_nodes in fragments_nodes
        ]
        combined_mask = torch.cat(fragments_masks)
        edge_index = get_edges_index(combined_mask, remove_self_edge=True)
        n_frag_switch = get_n_frag_switch(fragments_nodes)
        cond["edge_index"] = edge_index
        cond["subgraph_mask"] = get_subgraph_mask(edge_index, n_frag_switch)
        cond["ts_mask"] = fragments_masks[self.idx]

        x1 = remove_mean_batch(x1, cond["ts_mask"]).to(r_pos.device)
        return x0, x1, cond, x0_size, x0_other

    def q_sample(
        self,
        step: Tensor,
        x0: Tensor,
        x1: Tensor,
        ot_ode: bool = True,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Sample ``q(x_t | x_0, x_1)`` -- upstream's eq. 11.

        Args:
            step: per-row integer timestep, already inflated to ``(n, 1)``.
            x0: target endpoint (the transition state).
            x1: source endpoint (the R/P midpoint).
            ot_ode: ``True`` (the released setting) skips the Gaussian term
                entirely, which is what makes training deterministic.
            mask: per-row scatter index; when given, the result is
                CoM-removed per sample.

        Returns:
            ``x_t``.
        """
        if x0.shape != x1.shape:
            raise ValueError(f"x0 {x0.shape} != x1 {x1.shape}")
        device = self.schedule.mu_x0.device
        step = step.to(device)

        mu_x0 = self.schedule.mu_x0[step].to(x0.device)
        mu_x1 = self.schedule.mu_x1[step].to(x0.device)
        xt = mu_x0 * x0 + mu_x1 * x1
        if not ot_ode:
            std_sb = self.schedule.std_sb[step].to(x0.device)
            xt = xt + std_sb * torch.randn_like(xt)
        if mask is not None:
            xt = remove_mean_batch(xt, mask)
        return xt

    def compute_label(self, step: Tensor, x0: Tensor, xt: Tensor) -> Tensor:
        """The regression target ``(x_t - x_0) / std_fwd[t]`` (eq. 12).

        Args:
            step: per-row integer timestep.
            x0: target endpoint.
            xt: current bridge state.

        Returns:
            The label the network is regressed onto.
        """
        std_fwd = self.schedule.get_std_fwd(step, xdim=x0.shape[1:])
        return (xt - x0) / std_fwd.to(x0.device)

    def compute_pred_x0(
        self,
        step: Tensor,
        xt: Tensor,
        net_out: Tensor,
        clip_denoise: bool = False,
        val: float = 10.0,
    ) -> Tensor:
        """Recover ``x_0`` from a network output -- the inverse of eq. 12.

        Args:
            step: per-row integer timestep.
            xt: current bridge state.
            net_out: what the network predicted at ``xt``.
            clip_denoise: clamp the result into ``[-val, val]`` angstroms.
            val: the clamp bound.

        Returns:
            The predicted transition state.
        """
        std_fwd = self.schedule.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd.to(xt.device) * net_out
        if clip_denoise:
            pred_x0.clamp_(-val, val)
        return pred_x0

    def forward(
        self,
        representations: List[Dict[str, Tensor]],
        conditions: Tensor,
        ot_ode: bool = True,
    ) -> Dict[str, Tensor]:
        """One training step's loss terms.

        Args:
            representations: the three per-object dicts.
            conditions: the constant-zero ``(B, 1)`` channel.
            ot_ode: ``True`` => no noise is drawn.

        Returns:
            ``{"loss", "scaled_err", "pred", "label"}``. ``loss`` is already
            a 0-dim scalar (``F.mse_loss`` reduces by default), so the task
            does not reduce again.
        """
        num_sample = representations[0]["size"].size(0)
        device = representations[0]["pos"].device
        masks = [repre["mask"] for repre in representations]
        combined_mask = torch.cat(masks)
        edge_index = get_edges_index(combined_mask, remove_self_edge=True)
        fragments_nodes = [repre["size"] for repre in representations]
        n_frag_switch = get_n_frag_switch(fragments_nodes)

        representations = self.normalizer.normalize(representations)

        lowest_t = 0 if self.training else 1
        t_int = torch.randint(
            lowest_t, self.T, size=(num_sample, 1), device=device
        )
        t = t_int / self.T

        x0, x1, cond, x0_size, _x0_other = self.sample_batch(
            representations, conditions
        )

        timestep = torch.repeat_interleave(t_int, x0_size)
        timestep = self.schedule.inflate_batch_array(
            timestep, representations[0]["pos"]
        )
        xt = self.q_sample(
            timestep, x0, x1, ot_ode=ot_ode, mask=cond["ts_mask"]
        )

        xh_t = [
            torch.cat(
                [repre[feature_type] for feature_type in FEATURE_MAPPING],
                dim=1,
            )
            for repre in representations
        ]
        # The whole conditioning mechanism: reactant and product stay at
        # their true coordinates in the graph, and only the TS object's
        # position columns carry the bridge state. No RePaint, no re-noising
        # of the fixed objects.
        xh_t[self.idx][:, : self.pos_dim] = xt

        net_eps_xh, _ = self.dynamics(
            xh=xh_t,
            edge_index=edge_index,
            t=t,
            conditions=conditions,
            n_frag_switch=n_frag_switch,
            combined_mask=combined_mask,
            edge_attr=None,
        )
        pred = net_eps_xh[self.idx][:, : self.pos_dim]
        label = self.compute_label(timestep.squeeze(), x0, xt)
        return {
            "loss": F.mse_loss(pred, label),
            "scaled_err": compute_scaled_err(pred, label),
            "pred": pred,
            "label": label,
        }

    # ------ inverse pass -------------------------------------------------

    def p_posterior(
        self,
        nprev: int,
        n: int,
        x_n: Tensor,
        x0: Tensor,
        ot_ode: bool = True,
    ) -> Tensor:
        """Sample ``p(x_{nprev} | x_n, x_0)`` -- upstream's eq. 4.

        Args:
            nprev: the earlier grid index.
            n: the current grid index.
            x_n: state at ``n``.
            x0: predicted endpoint.
            ot_ode: ``True`` => deterministic, no Gaussian term.

        Returns:
            State at ``nprev``.
        """
        if nprev >= n:
            raise ValueError(f"nprev {nprev} must be < n {n}")
        std_n = self.schedule.std_fwd[n]
        std_nprev = self.schedule.std_fwd[nprev]
        std_delta = (std_n**2 - std_nprev**2).sqrt()
        mu_x0, mu_xn, var = compute_gaussian_product_coef(
            std_nprev, std_delta
        )
        mu_x0 = mu_x0.to(x0.device)
        mu_xn = mu_xn.to(x_n.device)

        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if not ot_ode and nprev > 0:
            std = var.to(xt_prev.device).sqrt()
            xt_prev = xt_prev + std * torch.randn_like(xt_prev)
        return xt_prev

    @staticmethod
    def _stack_bwd(frames: List[Tensor]) -> Tensor:
        """``(sum n, n_logged, 3)``, earliest-first (i.e. index 0 = final)."""
        return torch.flip(torch.stack(frames, dim=1), dims=(1,))

    def ddpm_sampling(
        self,
        steps: Sequence[int],
        pred_x0_fn: Callable[[Tensor, int], Tensor],
        x1: Tensor,
        ot_ode: bool = True,
        log_steps: Optional[Sequence[int]] = None,
        cog_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Walk the grid backwards with the Gaussian-product posterior.

        Deterministic at ``ot_ode=True`` (the released setting): the
        ``randn_like`` in :meth:`p_posterior` is skipped. This is what
        upstream's shipped FastAPI service runs (``service_ot.py:49``) and
        what its train script defaults to (``train_rpsb_ts1x.py:32``).

        Args:
            steps: ascending grid indices, ``steps[0] == 0``.
            pred_x0_fn: ``(xt, step) -> predicted x0``.
            x1: the starting structure (the R/P midpoint).
            ot_ode: ``True`` => deterministic.
            log_steps: which grid indices to record a frame at.
            cog_mask: per-row scatter index for CoM removal.

        Returns:
            ``(xs, pred_x0s)``, each ``(sum n, n_logged, 3)``.
        """
        xt = x1.detach()
        xs: List[Tensor] = []
        pred_x0s: List[Tensor] = []
        log_steps = log_steps if log_steps is not None else steps
        descending = list(steps)[::-1]
        if descending[-1] != 0:
            raise ValueError("the sampling grid must reach step 0")

        for prev_step, step in zip(descending[1:], descending[:-1]):
            pred_x0 = pred_x0_fn(xt, step)
            xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)
            if cog_mask is not None:
                xt = remove_mean_batch(xt, cog_mask)
            if step in log_steps or prev_step == 0:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())
        return self._stack_bwd(xs), self._stack_bwd(pred_x0s)

    @torch.no_grad()
    def ode_sampling(
        self,
        steps: Sequence[int],
        net_out_fn: Callable[[Tensor, Tensor], Tensor],
        x1: Tensor,
        t_size: int,
        method: str = "midpoint",
        atol: float = 1e-2,  # noqa: ARG002 - dead for a fixed-grid method
        rtol: float = 1e-2,  # noqa: ARG002 - dead for a fixed-grid method
        log_steps: Optional[Sequence[int]] = None,
        cog_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Integrate the probability-flow ODE backwards, one RK2 step a rung.

        The drift is ``f(t, x) = net_out(x, t) * sqrt(beta / t)`` with
        ``beta = betas[0] * T``, which is only well defined because the beta
        schedule is constant -- asserted below, exactly as upstream does
        (``en_sb.py:411``). See :mod:`~MolecularDiffusion.modules.models.
        reactot.schedule` for why it is constant and what breaks it.

        Args:
            steps: ascending grid indices, ``steps[0] == 0``.
            net_out_fn: ``(xt, t) -> network output`` with ``t`` of shape
                ``(B, 1)``.
            x1: the starting structure (the R/P midpoint).
            t_size: batch size, i.e. how wide ``t`` must be broadcast.
            method: only ``"midpoint"``; anything else raises.
            atol: accepted for signature fidelity and **unused** --
                torchdiffeq ignores tolerances for fixed-grid solvers, and
                upstream threads ``1e-2`` through to no effect.
            rtol: as ``atol``.
            log_steps: which grid indices to record a frame at.
            cog_mask: per-row scatter index for CoM removal.

        Returns:
            ``(xs, xs)`` -- upstream returns the trajectory twice here,
            because the ODE path never computes ``pred_x0``.

        Raises:
            ValueError: for a non-``midpoint`` method, or a grid that does
                not reach 0.
            AssertionError: if the beta schedule is not constant.
        """
        if method != "midpoint":
            raise ValueError(
                f"method={method!r} is not available: this port vendors the "
                "single fixed-step midpoint (RK2) update inline rather than "
                "depending on torchdiffeq, and upstream never selects any "
                "other method (train_rpsb_ts1x.py:32, evaluation.py:96, "
                "appmain.py:60 all default to 'midpoint'). For an adaptive "
                "solver: pip install torchdiffeq and restore the odeint call "
                "in ode_sampling."
            )
        xt = x1.detach()
        xs: List[Tensor] = []
        log_steps = log_steps if log_steps is not None else steps
        descending = list(steps)[::-1]
        if descending[-1] != 0:
            raise ValueError("the sampling grid must reach step 0")

        if not torch.allclose(
            self.schedule.betas[:-1], self.schedule.betas[1:]
        ):
            raise AssertionError(
                "ode_sampling requires a CONSTANT beta schedule. It is "
                "constant only at timesteps=3000 / beta_max=0.3, where "
                "beta_max/timesteps equals make_beta_schedule's hardcoded "
                "linear_start of 1e-4. Use solver='ddpm' at other settings."
            )
        beta = self.schedule.betas[0] * self.T

        def drift(t_val: float, x: Tensor) -> Tensor:
            t_scalar = torch.as_tensor(t_val, dtype=x.dtype, device=x.device)
            tt = t_scalar.repeat(t_size).reshape(-1, 1)
            net_out = net_out_fn(x, tt)
            return net_out * torch.sqrt(beta.to(x) / t_scalar)

        for prev_step, step in zip(descending[1:], descending[:-1]):
            prev_t = max(1e-5, prev_step / self.T)
            t = step / self.T
            if not prev_t < t:
                raise ValueError(f"prev_t={prev_t} must be < t={t}")

            # ponytail: vendored single midpoint (RK2) step ==
            # torchdiffeq Midpoint._step_func, which computes
            #   dy = dt * f(t0 + dt/2, y0 + (dt/2) * f(t0, y0)).
            # Upstream calls odeint with exactly two time points, no
            # options={'step_size': ...} and method='midpoint', so a
            # fixed-grid solver takes the supplied t AS its grid: one step
            # per outer iteration is the whole behaviour. Adaptive methods:
            # pip install torchdiffeq and restore odeint here.
            h = prev_t - t  # negative; we integrate backwards
            k1 = drift(t, xt)
            k2 = drift(t + h / 2, xt + (h / 2) * k1)
            xt = xt + h * k2

            if cog_mask is not None:
                xt = remove_mean_batch(xt, cog_mask)
            if step in log_steps or prev_step == 0:
                xs.append(xt.detach().cpu())

        stacked = self._stack_bwd(xs)
        return stacked, stacked

    @torch.no_grad()
    def sample(
        self,
        representations: List[Dict[str, Tensor]],
        conditions: Tensor,
        clip_denoise: bool = True,
        nfe: Optional[int] = None,
        log_count: int = 10,
        ot_ode: bool = True,
        solver: str = "ode",
        method: str = "midpoint",
        atol: float = 1e-2,
        rtol: float = 1e-2,
    ) -> Tuple[Tensor, Tensor]:
        """Generate transition states for a batch of reactions.

        **``nfe`` is the number of network evaluations**, not a fraction of
        a 3000-step schedule: ``space_indices(T, nfe + 1)`` picks ``nfe + 1``
        rungs out of the grid and the loop walks them backwards. The README's
        published command is ``--solver ode --nfe 10``.

        Upstream's ``x1`` parameter is not reproduced: it is shadowed at
        ``en_sb.py:477`` by :meth:`sample_batch`'s own return, so the caller
        never had a say.

        Args:
            representations: the three per-object dicts. **Object 1's
                positions must already be the R/P midpoint, not a reference
                transition state** -- see
                :meth:`~MolecularDiffusion.modules.tasks.diffusion_reactot.
                ReactOTTask.sample`, which substitutes it.
            conditions: the constant-zero ``(B, 1)`` channel.
            clip_denoise: clamp each predicted ``x0`` into +/-10 A.
            nfe: network evaluations; ``None`` => ``T - 1``, which is 2999
                and almost certainly not what you want.
            log_count: how many trajectory frames to keep. Index 0 of the
                returned tensors is always the final structure.
            ot_ode: ``True`` => deterministic.
            solver: ``"ode"`` (the published default) or ``"ddpm"``.
            method: ODE method; only ``"midpoint"``.
            atol: dead for a fixed-grid method; threaded for fidelity.
            rtol: as ``atol``.

        Returns:
            ``(xs, pred_x0)``, each ``(sum n, n_logged, 3)`` -- **positions
            only**, not the 9-wide concat. The generated structure is
            ``xs[:, 0, :]``.

        Raises:
            ValueError: for an unknown solver or an out-of-range ``nfe``.
        """
        nfe = nfe or self.T - 1
        if not 0 < nfe < self.T:
            raise ValueError(f"nfe must be in (0, {self.T}), got {nfe}")
        steps = space_indices(self.T, nfe + 1)
        log_count = min(len(steps) - 1, log_count)
        picked = space_indices(len(steps) - 1, log_count)
        log_steps = [steps[i] for i in picked]

        masks = [repre["mask"] for repre in representations]
        combined_mask = torch.cat(masks)
        edge_index = get_edges_index(combined_mask, remove_self_edge=True)
        fragments_nodes = [repre["size"] for repre in representations]
        n_frag_switch = get_n_frag_switch(fragments_nodes)

        _x0, x1, _cond, _x0_size, _x0_other = self.sample_batch(
            representations, conditions
        )
        xh_t = [
            torch.cat(
                [repre[feature_type] for feature_type in FEATURE_MAPPING],
                dim=1,
            )
            for repre in representations
        ]

        def net_out_fn(xt: Tensor, t: Tensor) -> Tensor:
            xh_t[self.idx][:, : self.pos_dim] = xt
            net_eps_xh, _ = self.dynamics(
                xh=xh_t,
                edge_index=edge_index,
                t=t,
                conditions=conditions,
                n_frag_switch=n_frag_switch,
                combined_mask=combined_mask,
                edge_attr=None,
            )
            return net_eps_xh[self.idx][:, : self.pos_dim]

        def pred_x0_fn(xt: Tensor, step: int) -> Tensor:
            step_t = torch.full(
                (representations[self.idx]["size"].size(0),),
                step,
                dtype=torch.long,
                device=xt.device,
            ).unsqueeze(1)
            timestep = torch.repeat_interleave(
                step_t, representations[self.idx]["size"]
            )
            timestep = self.schedule.inflate_batch_array(
                timestep, representations[0]["pos"]
            )
            out = net_out_fn(xt, step_t / self.T)
            return self.compute_pred_x0(
                timestep.squeeze(), xt, out, clip_denoise=clip_denoise
            )

        cog_mask = representations[self.idx]["mask"]
        if solver == "ddpm":
            return self.ddpm_sampling(
                steps,
                pred_x0_fn,
                x1,
                ot_ode=ot_ode,
                log_steps=log_steps,
                cog_mask=cog_mask,
            )
        if solver == "ode":
            return self.ode_sampling(
                steps,
                net_out_fn,
                x1,
                t_size=representations[self.idx]["size"].size(0),
                method=method,
                atol=atol,
                rtol=rtol,
                log_steps=log_steps,
                cog_mask=cog_mask,
            )
        raise ValueError(
            f"solver={solver!r} is not available: this port implements "
            "'ode' (the README's published --solver ode --nfe 10) and "
            "'ddpm' (upstream's train-script default and what their FastAPI "
            "service runs). Upstream's third, 'ei', is an unpublished "
            "exponential-integrator variant and is out of scope."
        )
