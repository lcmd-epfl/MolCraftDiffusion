"""IPDiff's score network: KGDiff/TargetDiff's, plus an interaction prior.

Ported from IPDiff's ``models/molopt_score_model.py`` (commit ``00ed078``).
That file is TargetDiff's ``ScorePosNet3D`` with four additions, so rather
than re-porting ~700 lines this subclasses the already-in-tree
:class:`~...models.kgdiff.score_model.ScorePosNet3D` (itself a TargetDiff
port) and overrides only what genuinely differs:

======================  =====================================================
Member                  IPDiff's change
======================  =====================================================
``__init__``            no affinity head; adds ``cond_dim`` / ``emb_mlp`` /
                        ``shift_t_mlp_pos`` and the ``k_t`` buffer
``_embed``              concatenates IPNet's features into the protein and
                        ligand token embeddings before the node indicator
``_heads``              the affinity head is gone (see ``__init__``)
``q_pos_posterior``     the matching ``-k_t*shift_t`` / ``+k_t(t-1)*
                        shift_{t-1}`` reverse correction
``get_diffusion_loss``  ``h_bap`` from the GROUND-TRUTH complex, and the
                        forward noising gains ``+ k_t * shift``
``sample_diffusion``    ``h_bap`` recomputed from the current predicted x0
                        EVERY reverse step (self-conditioning on the prior)
======================  =====================================================

Everything else -- the beta schedules, the whole D3PM ``q_v_*`` family,
``_predict_x0_from_eps``, ``sample_time``, ``compute_v_Lt`` and ``forward``
-- is inherited unchanged, because IPDiff did not change it.

**The two mechanisms, in one place.** Let ``a_bar`` be the cumulative alpha
and ``s_t = shift_t_mlp_pos([h_bap_ligand, t])``:

* *prior conditioning* -- ``h = emb_mlp([h, h_bap])`` on every token, so the
  denoiser sees the interaction representation directly;
* *prior shifting* -- the forward process becomes
  ``x_t = sqrt(a_bar) x0 + sqrt(1-a_bar) eps + k_t s_t`` with
  ``k_t = sqrt(a_bar)(1 - sqrt(a_bar))``, i.e. the noising trajectory itself
  bends toward the prior, and the reverse posterior undoes it exactly.

There is no classifier, no CFG, and no gradient of any predictor: unlike
KGDiff, the parent's ``use_classifier_guide`` machinery is switched off and
its ``expert_pred`` head is deleted outright (the released IPDiff checkpoint
has no such tensors).

**Passing h_bap into an inherited ``forward``.** The parent's ``forward``
calls ``self._embed(...)`` with a fixed argument list, so the conditioning
features are handed over on ``self`` (:attr:`hbap_protein` /
:attr:`hbap_ligand`) immediately before each call rather than threaded
through the signature. ``None`` means "no prior" and reproduces upstream's
zero-fill (``molopt_score_model.py:321-324``).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch_scatter import scatter_mean
from tqdm.auto import tqdm

from MolecularDiffusion.modules.models.kgdiff.score_model import (
    ScorePosNet3D,
    center_pos,
    extract,
    index_to_log_onehot,
    log_sample_categorical,
    to_torch_const,
)


class IPDiffScorePosNet3D(ScorePosNet3D):
    """TargetDiff's denoiser conditioned on a frozen interaction prior."""

    def __init__(self, cond_dim: int = 128, **kwargs: Any) -> None:
        kwargs.pop("use_classifier_guide", None)
        kwargs.pop("loss_exp_weight", None)
        kwargs.pop("pred_exp_from_all", None)
        super().__init__(use_classifier_guide=False, **kwargs)

        # IPDiff has no affinity head, and the released checkpoint has no
        # tensors for one. Deleting it keeps the state dict exactly the
        # upstream key set, so the converter can assert zero drops.
        del self.expert_pred

        self.cond_dim = cond_dim
        emb_dim = self.hidden_dim - 1 if self.node_indicator else self.hidden_dim
        self.emb_mlp = nn.Linear(emb_dim + cond_dim, emb_dim)
        # +1 is the (unnormalised) timestep, appended to h_bap.
        self.shift_t_mlp_pos = nn.Sequential(nn.Linear(cond_dim + 1, 3))

        alphas_cumprod = self.alphas_cumprod.detach().cpu().numpy().astype(
            np.float64
        )
        self.k_t = to_torch_const(
            np.sqrt(alphas_cumprod) * (1 - np.sqrt(alphas_cumprod))
        )

        #: Set by the callers below just before each ``forward``; ``None``
        #: reproduces upstream's zero-fill. Plain attributes on purpose --
        #: they are transient activations, not state.
        self.hbap_protein: Optional[torch.Tensor] = None
        self.hbap_ligand: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    # embedding: fold the prior into every token
    # ------------------------------------------------------------------ #
    def _embed(self, protein_v, init_ligand_v, time_step, batch_ligand):
        if init_ligand_v.dim() == 1:
            init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()
        elif init_ligand_v.dim() != 2:
            raise ValueError(f"init_ligand_v has rank {init_ligand_v.dim()}")

        if self.time_emb_dim > 0:
            input_ligand_feat = torch.cat(
                [
                    init_ligand_v,
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1),
                ],
                -1,
            )
        else:
            input_ligand_feat = init_ligand_v

        h_protein = self.protein_atom_emb(protein_v)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)

        hbap_protein = self.hbap_protein
        if hbap_protein is None:
            hbap_protein = h_protein.new_zeros(len(h_protein), self.cond_dim)
        hbap_ligand = self.hbap_ligand
        if hbap_ligand is None:
            hbap_ligand = init_ligand_h.new_zeros(
                len(init_ligand_h), self.cond_dim
            )

        h_protein = self.emb_mlp(torch.cat([h_protein, hbap_protein], dim=1))
        init_ligand_h = self.emb_mlp(
            torch.cat([init_ligand_h, hbap_ligand], dim=1)
        )

        if self.node_indicator:
            h_protein = torch.cat(
                [h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1
            )
            init_ligand_h = torch.cat(
                [init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)],
                -1,
            )
        return h_protein, init_ligand_h

    def _heads(self, final_h, final_ligand_h, batch_all, batch_ligand):  # noqa: ARG002
        """Type logits only -- there is no affinity head to run."""
        return self.v_inference(final_ligand_h), None, None

    def _condition(self, net_cond, ligand_pos, protein_pos, ligand_v,
                   protein_v, batch_ligand, batch_protein):
        """IPNet features for one complex, in the argument form it wants.

        The three index arguments are derived exactly as upstream does
        (``molopt_score_model.py:484-486``): the ligand's 13-class index as
        is, and the pocket's element / amino-acid indices as the argmax of
        the 27-dim protein feature's first 6 and next 20 columns.
        """
        return net_cond.extract_features(
            ligand_pos,
            protein_pos,
            ligand_v,
            torch.argmax(protein_v[:, :6], dim=1),
            torch.argmax(protein_v[:, 6:26], dim=1),
            batch_ligand,
            batch_protein,
        )

    def _shift(self, hbap_ligand, time_step, batch_ligand):
        """``shift_t_mlp_pos([h_bap_ligand, t])`` -> a per-atom 3-vector."""
        return self.shift_t_mlp_pos(
            torch.cat(
                [hbap_ligand, time_step[batch_ligand].unsqueeze(-1)], -1
            )
        )

    # ------------------------------------------------------------------ #
    # reverse posterior, with the prior-shift correction
    # ------------------------------------------------------------------ #
    def q_pos_posterior(  # type: ignore[override]
        self,
        x0,
        xt,
        t,
        batch,
        t_minus1=None,
        shift=None,
        shift_minus1=None,
    ):
        """TargetDiff's posterior mean, un-shifted at ``t`` and re-shifted
        at ``t-1``.

        With no shift (the first reverse step, and ``t == 0``) this is
        exactly the parent's expression. Upstream's middle branch
        (``molopt_score_model.py:415-417``) is unreachable -- the outer
        condition already excludes ``shift is None`` -- so it is not ported.
        """
        mean = extract(self.posterior_mean_c0_coef, t, batch) * x0
        if shift is None or t_minus1 is None:
            return mean + extract(self.posterior_mean_ct_coef, t, batch) * xt
        return (
            mean
            + extract(self.posterior_mean_ct_coef, t, batch)
            * (xt - extract(self.k_t, t, batch) * shift)
            + extract(self.k_t, t_minus1, batch) * shift_minus1
        )

    # ------------------------------------------------------------------ #
    # training objective
    # ------------------------------------------------------------------ #
    def get_diffusion_loss(  # type: ignore[override]
        self,
        net_cond,
        protein_pos,
        protein_v,
        batch_protein,
        ligand_pos,
        ligand_v,
        batch_ligand,
        time_step=None,
    ) -> dict:
        """``loss_pos + loss_v_weight * loss_v``, with prior shifting.

        ``h_bap`` is computed ONCE, from the ground-truth complex -- the
        training-time counterpart of the sampler's per-step recomputation.
        """
        num_graphs = batch_protein.max().item() + 1
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand,
            mode=self.center_pos_mode,
        )
        if self.model_mean_type != "C0":
            raise ValueError(
                "IPDiff's prior shifting is only defined for "
                f"model_mean_type='C0', not {self.model_mean_type!r}."
            )

        hbap_ligand, hbap_protein = self._condition(
            net_cond, ligand_pos, protein_pos, ligand_v, protein_v,
            batch_ligand, batch_protein,
        )

        if time_step is None:
            time_step, _pt = self.sample_time(
                num_graphs, protein_pos.device, self.sample_time_method
            )
        a_pos = self.alphas_cumprod.index_select(0, time_step)[
            batch_ligand
        ].unsqueeze(-1)
        k_t_pos = self.k_t.index_select(0, time_step)[batch_ligand].unsqueeze(-1)

        # x_t = sqrt(a) x0 + sqrt(1-a) eps + k_t * shift_t   <- the IPDiff term
        pos_noise = torch.randn_like(ligand_pos)
        shift_cond_t = self._shift(hbap_ligand, time_step, batch_ligand)
        ligand_pos_perturbed = (
            a_pos.sqrt() * ligand_pos
            + (1.0 - a_pos).sqrt() * pos_noise
            + k_t_pos * shift_cond_t
        )
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(
            log_ligand_v0, time_step, batch_ligand
        )

        self.hbap_protein, self.hbap_ligand = hbap_protein, hbap_ligand
        preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            init_ligand_pos=ligand_pos_perturbed,
            init_ligand_v=ligand_v_perturbed,
            batch_ligand=batch_ligand,
            time_step=time_step,
        )
        pred_ligand_pos = preds["pred_ligand_pos"]
        pred_ligand_v = preds["pred_ligand_v"]

        loss_pos = scatter_mean(
            ((pred_ligand_pos - ligand_pos) ** 2).sum(-1), batch_ligand, dim=0
        ).mean()

        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        loss_v = self.compute_v_Lt(
            log_v_model_prob=self.q_v_posterior(
                log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand
            ),
            log_v0=log_ligand_v0,
            log_v_true_prob=self.q_v_posterior(
                log_ligand_v0, log_ligand_vt, time_step, batch_ligand
            ),
            t=time_step,
            batch=batch_ligand,
        ).mean()

        return {
            "loss": loss_pos + loss_v * self.loss_v_weight,
            "loss_pos": loss_pos,
            "loss_v": loss_v,
            "pred_ligand_pos": pred_ligand_pos,
            "pred_ligand_v": pred_ligand_v,
        }

    # ------------------------------------------------------------------ #
    # sampling
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample_diffusion(  # type: ignore[override]
        self,
        protein_pos,
        protein_v,
        batch_protein,
        init_ligand_pos,
        init_ligand_v,
        batch_ligand,
        net_cond=None,
        num_steps: Optional[int] = None,
        center_pos_mode: Optional[str] = None,
        progress: bool = True,
        **_ignored: Any,
    ) -> dict:
        """Ancestral DDPM reverse loop with per-step prior re-conditioning.

        Each step predicts x0, takes the shifted posterior, then re-runs
        IPNet on ``(predicted x0, the real pocket)`` so the next step's
        conditioning reflects the molecule as it currently stands. That
        second IPNet pass is the expensive part -- it builds a fully
        connected complex graph every step (see ``bapnet.py``).

        Trajectories (``pos_traj``/``v_traj``) are deliberately not
        accumulated: the platform's GIF path is out of scope for this port
        and keeping them costs ``num_steps`` copies of the cloud.
        """
        if net_cond is None:
            raise ValueError(
                "IPDiff sampling needs the frozen IPNet prior; pass "
                "net_cond=<BAPNet>. Without it there is no conditioning "
                "signal and the model is not the model."
            )
        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1
        device = protein_pos.device

        protein_pos, init_ligand_pos, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand,
            mode=center_pos_mode or self.center_pos_mode,
        )
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v

        # Step T starts from a zero prior (nothing has been predicted yet),
        # exactly as upstream (molopt_score_model.py:574-579).
        hbap_protein = torch.zeros(
            len(batch_protein), self.cond_dim, device=device
        )
        hbap_ligand = torch.zeros(
            len(batch_ligand), self.cond_dim, device=device
        )
        time_seq = list(
            reversed(range(self.num_timesteps - num_steps, self.num_timesteps))
        )
        shift_cond_t = None
        t_start = torch.full(
            (num_graphs,), len(time_seq), dtype=torch.long, device=device
        )
        shift_cond_t_minus1 = self._shift(hbap_ligand, t_start, batch_ligand)

        iterator = (
            tqdm(time_seq, desc="sampling", total=len(time_seq))
            if progress
            else time_seq
        )
        for i in iterator:
            t = torch.full(
                (num_graphs,), i, dtype=torch.long, device=device
            )
            t_minus1 = (
                torch.full((num_graphs,), i - 1, dtype=torch.long, device=device)
                if i >= 1
                else None
            )

            self.hbap_protein, self.hbap_ligand = hbap_protein, hbap_ligand
            preds = self(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                init_ligand_pos=ligand_pos,
                init_ligand_v=ligand_v,
                batch_ligand=batch_ligand,
                time_step=t,
            )
            if self.model_mean_type == "C0":
                pos0_from_e = preds["pred_ligand_pos"]
            else:
                pos0_from_e = self._predict_x0_from_eps(
                    xt=ligand_pos,
                    eps=preds["pred_ligand_pos"] - ligand_pos,
                    t=t,
                    batch=batch_ligand,
                )
            v0_from_e = preds["pred_ligand_v"]

            pos_model_mean = self.q_pos_posterior(
                x0=pos0_from_e,
                xt=ligand_pos,
                t=t,
                batch=batch_ligand,
                t_minus1=t_minus1,
                shift=shift_cond_t,
                shift_minus1=shift_cond_t_minus1,
            )
            pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
            ligand_pos = pos_model_mean + nonzero_mask * (
                0.5 * pos_log_variance
            ).exp() * torch.randn_like(ligand_pos)

            # re-condition on the CURRENT predicted x0 (self-conditioning)
            hbap_ligand, hbap_protein = self._condition(
                net_cond,
                pos0_from_e.detach(),
                protein_pos.detach(),
                torch.argmax(v0_from_e.detach(), dim=1),
                protein_v.detach(),
                batch_ligand,
                batch_protein,
            )
            shift_cond_t = shift_cond_t_minus1
            if t_minus1 is not None:
                shift_cond_t_minus1 = self._shift(
                    hbap_ligand, t_minus1, batch_ligand
                )

            log_model_prob = self.q_v_posterior(
                F.log_softmax(v0_from_e, dim=-1),
                index_to_log_onehot(ligand_v, self.num_classes),
                t,
                batch_ligand,
            )
            ligand_v = log_sample_categorical(log_model_prob).argmax(dim=-1)

        self.hbap_protein = self.hbap_ligand = None
        return {"pos": ligand_pos + offset[batch_ligand], "v": ligand_v}
