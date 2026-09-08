"""CTMC (Continuous-Time Markov Chain) discrete flow matching for FlowMol3.

Ported from FlowMol (``flowmol/models/ctmc_vector_field.py``). Coordinates stay
continuous flow matching; ``a``/``c``/``e`` follow a **masking** interpolant from
an all-mask prior to the data, integrated with Campbell-style unmask/re-mask
steps. CTMC for flow matching was introduced in arXiv:2402.04997; FlowMol
interpolates along a per-modality progress coordinate ``alpha_t`` and does purity
sampling per batched graph rather than per molecule.

**Not ported** (out of scope per the approved plan):

- ``gat_step`` and its ``forward_weight_schedule``. These serve
  ``dfm_type='gat'``, which no released config selects; ``campbell`` is the
  default and the only mode in scope. Passing ``dfm_type='gat'`` raises rather
  than silently doing something else.

The categorical temperature function **is** ported: ``step`` applies it
unconditionally (upstream ``:355-356``), so dropping it would silently change
the sampler.
"""

from collections.abc import Callable

import dgl
import torch
import torch.nn.functional as F  # noqa: N812
from torch.distributions.categorical import Categorical
from torch.nn.functional import one_hot

from MolecularDiffusion.modules.models.flowmol_graph3d.ctmc_utils import (
    purity_sampling,
)
from MolecularDiffusion.modules.models.flowmol_graph3d.graph_utils import (
    get_edge_batch_idxs,
)
from MolecularDiffusion.modules.models.flowmol_graph3d.vector_field import (
    EndpointVectorField,
)

__all__ = ["CTMCVectorField"]


class CTMCVectorField(EndpointVectorField):
    """Masking-interpolant discrete flow matching over ``a``/``c``/``e``."""

    def __init__(  # noqa: PLR0913
        self,
        *args,
        stochasticity: float = 0.0,
        high_confidence_threshold: float = 0.0,
        dfm_type: str = "campbell",
        cat_temperature_schedule: str | Callable | float = 0.05,
        cat_temp_decay_max: float = 0.8,
        cat_temp_decay_a: float = 2,
        fake_atoms: bool = False,
        **kwargs,
    ) -> None:
        # has_mask=True: every categorical token embedding gains one extra row
        # for the mask token. It is an internal noise state, never a chemistry
        # class, and never appears in an output head.
        super().__init__(*args, has_mask=True, **kwargs)

        self.eta = stochasticity
        self.hc_thresh = high_confidence_threshold
        self.dfm_type = dfm_type
        self.fake_atoms = fake_atoms

        self.cat_temperature_schedule = cat_temperature_schedule
        self.cat_temp_decay_max = cat_temp_decay_max
        self.cat_temp_decay_a = cat_temp_decay_a
        self.cat_temp_func = self.build_cat_temp_schedule(
            cat_temperature_schedule, cat_temp_decay_max, cat_temp_decay_a
        )

        if self.dfm_type != "campbell":
            msg = (
                f"dfm_type={self.dfm_type!r} is not supported by this port. "
                "Only 'campbell' is in scope -- upstream's 'gat' step and its "
                "forward-weight schedule were deliberately not ported (no "
                "released FlowMol config selects them)."
            )
            raise ValueError(msg)

        #: per-modality mask-token index (== the number of real classes)
        self.mask_idxs = {
            "a": self.n_atom_types,
            "c": self.n_charges,
            "e": self.n_bond_types,
        }

    @staticmethod
    def build_cat_temp_schedule(
        cat_temperature_schedule, cat_temp_decay_max, cat_temp_decay_a
    ) -> Callable:
        """Temperature applied to the predicted categorical distribution."""
        if cat_temperature_schedule == "decay":
            return lambda t: (
                cat_temp_decay_max * torch.pow(1 - t, cat_temp_decay_a)
            )
        if isinstance(cat_temperature_schedule, (float, int)):
            return lambda t: cat_temperature_schedule  # noqa: ARG005
        if callable(cat_temperature_schedule):
            return cat_temperature_schedule
        msg = f"Invalid cat_temperature_schedule: {cat_temperature_schedule}"
        raise ValueError(msg)

    # -- training-time interpolation ----------------------------------------

    def sample_conditional_path(
        self,
        g: dgl.DGLGraph,
        t: torch.Tensor,
        node_batch_idx: torch.Tensor,
        edge_batch_idx: torch.Tensor,
        upper_edge_mask: torch.Tensor,
    ) -> dgl.DGLGraph:
        """Sample ``p(g_t | g_0, g_1)``: linear for ``x``, masking for ``a/c/e``.

        Each categorical position is replaced by the mask token with probability
        ``1 - alpha_t``. The ground-truth one-hots arrive without a mask column,
        so they are argmaxed to indices and re-one-hotted one class wider.
        """
        _, alpha_t = self.interpolant_scheduler.interpolant_weights(t)
        num_nodes = g.num_nodes()
        device = g.device

        x_idx = self.canonical_feat_order.index("x")
        dst_weight = alpha_t[:, x_idx][node_batch_idx].unsqueeze(-1)
        g.ndata["x_t"] = (1 - dst_weight) * g.ndata[
            "x_0"
        ] + dst_weight * g.ndata["x_1_true"]

        for feat in ("a", "c"):
            feat_idx = self.canonical_feat_order.index(feat)
            xt = g.ndata[f"{feat}_1_true"].argmax(-1)
            alpha_t_feat = alpha_t[:, feat_idx][node_batch_idx]
            xt[torch.rand(num_nodes, device=device) < 1 - alpha_t_feat] = (
                self.mask_idxs[feat]
            )
            g.ndata[f"{feat}_t"] = one_hot(
                xt, num_classes=self.n_cat_feats[feat] + 1
            ).float()

        e_idx = self.canonical_feat_order.index("e")
        num_upper_edges = int(g.num_edges() / 2)
        alpha_t_e = alpha_t[:, e_idx][edge_batch_idx][upper_edge_mask]
        et_upper = g.edata["e_1_true"][upper_edge_mask].argmax(-1)
        et_upper[
            torch.rand(num_upper_edges, device=device) < 1 - alpha_t_e
        ] = self.mask_idxs["e"]

        n, d = g.edata["e_1_true"].shape
        e_t = torch.zeros((n, d + 1), dtype=torch.float32, device=device)
        et_upper_onehot = one_hot(
            et_upper, num_classes=self.n_cat_feats["e"] + 1
        ).float()
        e_t[upper_edge_mask] = et_upper_onehot
        e_t[~upper_edge_mask] = et_upper_onehot
        g.edata["e_t"] = e_t

        return g

    # -- integration --------------------------------------------------------

    def integrate(  # noqa: PLR0913
        self,
        g: dgl.DGLGraph,
        node_batch_idx: torch.Tensor,
        upper_edge_mask: torch.Tensor,
        n_timesteps: int,
        stochasticity: float = 8.0,
        high_confidence_threshold: float = 0.9,
        cat_temp_func: Callable = None,
        tspan: torch.Tensor = None,
        condition: torch.Tensor | None = None,
        negative_condition: torch.Tensor | None = None,
        cfg_scale: float = 0.0,
        cfg_scale_schedule: str | None = None,
        **kwargs,
    ) -> dgl.DGLGraph:
        """Integrate from the all-mask prior to a molecule in ``n_timesteps``."""
        if cat_temp_func is None:
            cat_temp_func = self.cat_temp_func

        edge_batch_idx = get_edge_batch_idxs(g)

        t = (
            torch.linspace(0, 1, n_timesteps, device=g.device)
            if tspan is None
            else tspan
        )
        alpha_t = self.interpolant_scheduler.alpha_t(t)
        alpha_t_prime = self.interpolant_scheduler.alpha_t_prime(t)

        for feat in self.canonical_feat_order:
            data_src = g.edata if feat == "e" else g.ndata
            data_src[f"{feat}_t"] = data_src[f"{feat}_0"]

        dst_dict = None
        for s_idx in range(1, t.shape[0]):
            step_cfg_scale = (
                self._scale_schedule(t[s_idx - 1], cfg_scale, cfg_scale_schedule)
                if cfg_scale_schedule else cfg_scale
            )
            g, dst_dict = self.step(
                g,
                t[s_idx],
                t[s_idx - 1],
                alpha_t[s_idx - 1],
                alpha_t[s_idx],
                alpha_t_prime[s_idx - 1],
                node_batch_idx,
                edge_batch_idx,
                upper_edge_mask,
                cat_temp_func=cat_temp_func,
                stochasticity=stochasticity,
                high_confidence_threshold=high_confidence_threshold,
                last_step=(s_idx == t.shape[0] - 1),
                prev_dst_dict=dst_dict,
                condition=condition,
                negative_condition=negative_condition,
                cfg_scale=step_cfg_scale,
                **kwargs,
            )

        for feat in self.canonical_feat_order:
            data_src = g.edata if feat == "e" else g.ndata
            data_src[f"{feat}_1"] = data_src[f"{feat}_t"]

        return g

    def step(  # noqa: PLR0913
        self,
        g: dgl.DGLGraph,
        s_i: torch.Tensor,
        t_i: torch.Tensor,
        alpha_t_i: torch.Tensor,
        alpha_s_i: torch.Tensor,  # noqa: ARG002 - signature parity with parent
        alpha_t_prime_i: torch.Tensor,
        node_batch_idx: torch.Tensor,
        edge_batch_idx: torch.Tensor = None,
        upper_edge_mask: torch.Tensor = None,
        cat_temp_func: Callable = None,
        prev_dst_dict: dict = None,
        stochasticity: float = 8.0,
        high_confidence_threshold: float = 0.9,
        last_step: bool = False,
        inv_temp_func: Callable = None,
        condition: torch.Tensor | None = None,
        negative_condition: torch.Tensor | None = None,
        cfg_scale: float = 0.0,
        **kwargs,  # noqa: ARG002
    ):
        """One CTMC step: Euler on ``x``, Campbell unmask/re-mask on ``a/c/e``."""
        eta = self.eta if stochasticity is None else stochasticity
        hc_thresh = (
            self.hc_thresh
            if high_confidence_threshold is None
            else high_confidence_threshold
        )
        if cat_temp_func is None:
            cat_temp_func = self.cat_temp_func
        if inv_temp_func is None:
            inv_temp_func = lambda t: 1.0  # noqa: E731, ARG005
        if edge_batch_idx is None:
            edge_batch_idx = get_edge_batch_idxs(g)

        t_full = torch.full((g.batch_size,), t_i, device=g.device)

        if condition is not None and cfg_scale != 0.0:
            # Two sequential forward passes on the same (un-doubled) graph,
            # swapping g.ndata["cond"] between calls, rather than DGL
            # graph-batch doubling -- g already carries per-step evolving
            # state that must NOT be duplicated, only the velocity
            # prediction needs the cond/uncond combination. prev_dst_dict is
            # a (prev_cond, prev_uncond) tuple across CFG steps so
            # self-conditioning stays consistent per branch (never mixing
            # conditional info into the null branch's self-conditioning).
            prev_cond = prev_dst_dict[0] if isinstance(prev_dst_dict, tuple) else None
            prev_uncond = prev_dst_dict[1] if isinstance(prev_dst_dict, tuple) else None

            g.ndata["cond"] = condition
            dst_cond = self(
                g, t=t_full, node_batch_idx=node_batch_idx, upper_edge_mask=upper_edge_mask,
                apply_softmax=True, remove_com=True, prev_dst_dict=prev_cond,
            )
            g.ndata["cond"] = (
                negative_condition if negative_condition is not None else torch.zeros_like(condition)
            )
            dst_uncond = self(
                g, t=t_full, node_batch_idx=node_batch_idx, upper_edge_mask=upper_edge_mask,
                apply_softmax=True, remove_com=True, prev_dst_dict=prev_uncond,
            )

            # dst_cond/dst_uncond are already-softmaxed probabilities (needed
            # so self-conditioning continuity above stays in the format it
            # expects). Combining them linearly would not generally produce a
            # valid simplex for campbell_step's Categorical(...).sample()
            # below. softmax((1+w)*log(p_cond) - w*log(p_uncond)) is
            # mathematically identical to the textbook logit-space
            # combination softmax((1+w)*logit_cond - w*logit_uncond) --
            # softmax is invariant to the per-row (not per-class) additive
            # constant the two branches' missing logsumexp terms leave
            # behind -- so this recovers proper logit-space CFG from
            # probabilities alone, with no separate un-softmaxed pass needed.
            eps = 1e-12
            dst_dict = {}
            for feat in dst_cond:
                if feat == "x":
                    dst_dict[feat] = (1 + cfg_scale) * dst_cond[feat] - cfg_scale * dst_uncond[feat]
                else:
                    combined_logp = (
                        (1 + cfg_scale) * torch.log(dst_cond[feat].clamp_min(eps))
                        - cfg_scale * torch.log(dst_uncond[feat].clamp_min(eps))
                    )
                    dst_dict[feat] = torch.softmax(combined_logp, dim=-1)
            next_prev_dst_dict = (dst_cond, dst_uncond)
        else:
            if condition is not None:
                g.ndata["cond"] = condition
            dst_dict = self(
                g,
                t=t_full,
                node_batch_idx=node_batch_idx,
                upper_edge_mask=upper_edge_mask,
                apply_softmax=True,
                remove_com=True,
                prev_dst_dict=prev_dst_dict,
            )
            next_prev_dst_dict = dst_dict

        dt = s_i - t_i

        # continuous step for positions
        x_t = g.ndata["x_t"]
        vf = self.vector_field(
            x_t, dst_dict["x"], alpha_t_i[0], alpha_t_prime_i[0]
        )
        g.ndata["x_t"] = x_t + dt * vf * inv_temp_func(t_i)

        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if feat == "x":
                continue

            data_src = g.edata if feat == "e" else g.ndata
            xt = data_src[f"{feat}_t"].argmax(-1)
            if feat == "e":
                xt = xt[upper_edge_mask]

            p_s_1 = dst_dict[feat]
            p_s_1 = F.softmax(torch.log(p_s_1) / cat_temp_func(t_i), dim=-1)

            xt, _x_1_sampled = self.campbell_step(
                p_1_given_t=p_s_1,
                xt=xt,
                stochasticity=eta,
                hc_thresh=hc_thresh,
                alpha_t=alpha_t_i[feat_idx],
                alpha_t_prime=alpha_t_prime_i[feat_idx],
                dt=dt,
                batch_size=g.batch_size,
                batch_num_nodes=(
                    g.batch_num_edges() // 2
                    if feat == "e"
                    else g.batch_num_nodes()
                ),
                n_classes=self.n_cat_feats[feat] + 1,
                mask_index=self.mask_idxs[feat],
                last_step=last_step,
                batch_idx=(
                    edge_batch_idx[upper_edge_mask]
                    if feat == "e"
                    else node_batch_idx
                ),
            )

            if feat == "e":
                # mirror the upper-triangle state onto both edge directions
                e_t = torch.zeros_like(g.edata["e_t"])
                e_t[upper_edge_mask] = xt
                e_t[~upper_edge_mask] = xt
                xt = e_t

            data_src[f"{feat}_t"] = xt

        return g, next_prev_dst_dict

    @staticmethod
    def campbell_step(  # noqa: PLR0913
        p_1_given_t: torch.Tensor,
        xt: torch.Tensor,
        stochasticity: float,
        hc_thresh: float,
        alpha_t: torch.Tensor,
        alpha_t_prime: torch.Tensor,
        dt: torch.Tensor,
        batch_size: int,
        batch_num_nodes: torch.Tensor,
        n_classes: int,
        mask_index: int,
        last_step: bool,
        batch_idx: torch.Tensor,
    ):
        """Unmask some masked positions, re-mask some unmasked ones."""
        x1 = Categorical(p_1_given_t).sample()

        unmask_prob = (
            dt * (alpha_t_prime + stochasticity * alpha_t) / (1 - alpha_t)
        )
        mask_prob = dt * stochasticity

        unmask_prob = torch.clamp(unmask_prob, min=0, max=1)
        mask_prob = torch.clamp(mask_prob, min=0, max=1)

        if hc_thresh > 0:
            will_unmask = purity_sampling(
                xt=xt,
                x1=x1,
                x1_probs=p_1_given_t,
                unmask_prob=unmask_prob,
                mask_index=mask_index,
                batch_size=batch_size,
                batch_num_nodes=batch_num_nodes,
                node_batch_idx=batch_idx,
                hc_thresh=hc_thresh,
                device=xt.device,
            )
        else:
            will_unmask = (
                torch.rand(xt.shape[0], device=xt.device) < unmask_prob
            )
            will_unmask = will_unmask * (xt == mask_index)

        if not last_step:
            will_mask = torch.rand(xt.shape[0], device=xt.device) < mask_prob
            will_mask = will_mask * (xt != mask_index)
            xt[will_mask] = mask_index

        xt[will_unmask] = x1[will_unmask]

        xt = one_hot(xt, num_classes=n_classes).float()
        x1 = one_hot(x1, num_classes=n_classes).float()
        return xt, x1
