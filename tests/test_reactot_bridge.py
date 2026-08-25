"""Checks for React-OT's vendored bridge maths.

Two things in `modules/models/reactot/` are numerics we wrote rather than
imported, and both fail silently if they break:

* the **single midpoint (RK2) step** `ode_sampling` vendors in place of
  `torchdiffeq.odeint(..., method="midpoint")`, and
* the **constant beta schedule** that step asserts on, which is constant only
  by a coincidence of the released settings (`beta_max / timesteps` equals
  `make_beta_schedule`'s hardcoded `linear_start`).

There is also one behavioural property the integration exists to guarantee:
`ReactOTTask.sample` must not read the reference transition state out of
`representations[1]`. On CPU that is exactly checkable.
"""

import math

import pytest
import torch

from MolecularDiffusion.modules.models.reactot import EnSB, SBSchedule
from MolecularDiffusion.modules.models.reactot.schedule import space_indices


def test_released_settings_give_a_constant_beta_schedule():
    schedule = SBSchedule(timesteps=3000, beta_max=0.3, power=0.5, inv_power=1)
    assert schedule.timesteps == 3000
    assert torch.allclose(schedule.betas[:-1], schedule.betas[1:])
    # beta_max / timesteps / 2, the value the renormalisation lands on.
    assert schedule.betas[0].item() == pytest.approx(5e-5, rel=1e-6)


def test_space_indices_picks_nfe_plus_one_rungs():
    steps = space_indices(3000, 11)
    assert len(steps) == 11
    assert steps[0] == 0
    assert steps[-1] == 2999
    assert steps == sorted(steps)


class _StubBridge:
    """Just enough of EnSB for ode_sampling: a schedule and a T."""

    def __init__(self):
        self.schedule = SBSchedule(3000, 0.3, 0.5, 1)
        self.T = 3000

    ode_sampling = EnSB.ode_sampling
    # staticmethod(): EnSB._stack_bwd unwraps to a plain function on
    # attribute access, and rebinding it here would make it take `self`.
    _stack_bwd = staticmethod(EnSB._stack_bwd)


def test_vendored_midpoint_step_matches_the_closed_form():
    """One RK2 step against `dy = dt * f(t0 + dt/2, y0 + (dt/2) * f(t0, y0))`.

    With a constant network output the drift is `sqrt(beta / t)`, which does
    not depend on the state, so `k2 = sqrt(beta / (t + h/2))` and the whole
    update has a closed form. That is enough to catch a wrong `h`, a wrong
    sign, a single-evaluation Euler step, or an evaluation at the wrong time.
    """
    bridge = _StubBridge()
    x1 = torch.zeros(4, 3)
    xs, _ = bridge.ode_sampling(
        steps=[0, 2999],
        net_out_fn=lambda x, _t: torch.ones_like(x),
        x1=x1,
        t_size=4,
    )
    beta = float(bridge.schedule.betas[0]) * bridge.T
    t = 2999 / 3000
    h = max(1e-5, 0.0) - t  # negative: we integrate backwards
    want = h * math.sqrt(beta / (t + h / 2))
    assert xs.shape == (4, 1, 3)
    assert xs[:, 0, :].numpy() == pytest.approx(want, rel=1e-6)


def test_a_non_midpoint_method_raises_rather_than_diverging():
    bridge = _StubBridge()
    with pytest.raises(ValueError, match="torchdiffeq"):
        bridge.ode_sampling(
            steps=[0, 2999],
            net_out_fn=lambda x, _t: x,
            x1=torch.zeros(1, 3),
            t_size=1,
            method="rk4",
        )


def test_the_ode_solver_refuses_a_non_constant_schedule():
    """The assertion is the only thing standing between a changed
    `timesteps`/`beta_max` and a silently wrong sampler."""
    bridge = _StubBridge()
    bridge.schedule = SBSchedule(timesteps=1000, beta_max=0.3, power=1.0)
    with pytest.raises(AssertionError, match="CONSTANT beta schedule"):
        bridge.ode_sampling(
            steps=[0, 999],
            net_out_fn=lambda x, _t: x,
            x1=torch.zeros(1, 3),
            t_size=1,
        )


def _tiny_task():
    from MolecularDiffusion.modules.tasks.diffusion_reactot import (
        ModelTaskFactory,
    )

    return ModelTaskFactory(
        model_config={
            "pos_require_grad": False,
            "cutoff": 10.0,
            "num_layers": 1,
            "hidden_channels": 16,
            "num_radial": 8,
            "in_hidden_channels": 8,
            "in_node_nf": 8,
            "reflect_equiv": True,
            "legacy": True,
            "update": True,
            "pos_grad": False,
            "single_layer_output": True,
            "object_aware": True,
            "act_fn": "swish",
        }
    ).build()


def _one_reaction_batch(n_atoms=5):
    torch.manual_seed(0)
    size = torch.tensor([n_atoms])
    mask = torch.zeros(n_atoms, dtype=torch.int64)
    one_hot = torch.nn.functional.one_hot(
        torch.tensor([1] * n_atoms), num_classes=5
    )
    charge = torch.full((n_atoms, 1), 6, dtype=torch.int64)
    reps = [
        {
            "size": size,
            "pos": torch.randn(n_atoms, 3),
            "one_hot": one_hot.clone(),
            "charge": charge.clone(),
            "mask": mask.clone(),
        }
        for _ in range(3)
    ]
    return {"representations": reps, "conditions": torch.zeros(1, 1)}


def test_sample_never_reads_the_reference_transition_state():
    """The whole point of the midpoint substitution.

    Object 1's positions are the *reference* TS when the batch comes from a
    corpus. Randomising them must change nothing -- otherwise a benchmark
    number is measuring a leak. Checked on CPU, where `torch_scatter` is
    deterministic; on CUDA its atomic adds put ~5e-3 A of noise on any two
    runs, leaked or not.
    """
    task = _tiny_task().cpu().eval()
    batch = _one_reaction_batch()

    clean = task.sample(batch=batch, num_steps=3)[2]
    again = task.sample(batch=batch, num_steps=3)[2]
    batch["representations"][1]["pos"] = torch.randn(5, 3) * 5.0
    corrupted = task.sample(batch=batch, num_steps=3)[2]

    assert torch.equal(clean, again), "sampling is not deterministic on CPU"
    assert torch.equal(clean, corrupted), (
        "the reference transition state reached the network: the midpoint "
        "substitution in ReactOTTask.sample is not doing its job"
    )


def test_sample_refuses_a_resized_transition_state():
    task = _tiny_task().cpu().eval()
    batch = _one_reaction_batch()
    with pytest.raises(ValueError, match="cannot be resized"):
        task.sample(batch=batch, nodesxsample=torch.tensor([7]), num_steps=2)
