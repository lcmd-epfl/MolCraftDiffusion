"""GoFlow's conditional-flow-matching algorithm: the perturbation, the loss,
and the sampling-time ODE stepper.

Ported from ``flow_matching/flow_module.py`` and ``flow_matching/utils.py``
(commit ``3ec00a09``) -- the algorithm, not upstream's ``pl.LightningModule``
wrapper (training-loop plumbing this platform's own engine already owns).

One shape adaptation, recorded in ``INTEGRATION_PLAN.md``'s Data adapters
section: upstream's ``Data.pos`` is a fixed ``(N, 3, 3)`` ``[R, TS, P]``
stack, read by ``FlowModule`` only at index 1 (the transition state).
``goflow_collate`` never materialises the R/P slots inside the training/
sampling batch -- it carries a bare ``ts_pos (N, 3)`` instead (present for
training and corpus-driven generation, simply absent for a blind R/P-only
query). :func:`get_perturbed_flow_point_and_time` below reads ``batch.ts_pos``
where upstream reads ``batch.pos[:, 1, :]``; nothing else about the
algorithm changes.

``euler_integrate`` replaces the single ``torchdiffeq.odeint(ode_func,
x_init, t_grid, method='euler')`` call at ``flow_module.py:134`` with a
five-line fixed-step forward-Euler loop (`x_{i+1} = x_i + (t_{i+1}-t_i) *
f(t_i, x_i)`) -- the exact math of that call, since ``method='euler'`` on a
fixed grid is nothing else. Unlike upstream's ``odeint`` call, this returns
only the final position: upstream stores the whole trajectory
(``pos_gen_traj_S_T_N_3``) only to feed ``align_and_rotate_samples``'s
GT-anchored median-consensus ensembling, which this integration does not
port (see ``INTEGRATION_PLAN.md``, Explicitly out of scope: "Trajectory
frames"). ponytail: fixed-step euler only, no adaptive step-size control --
`pip install torchdiffeq` and restore the original call is the upgrade path
if another method is ever wanted (same precedent as React-OT's vendored
midpoint stepper).
"""

from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Batch


def rmsd_loss(pred_N_3: Tensor, gt_N_3: Tensor) -> Tensor:
    """Verbatim from ``flow_matching/utils.py:137-138``: a single scalar
    over the whole batch, not a per-molecule mean averaged afterwards."""
    return torch.sqrt(torch.mean((pred_N_3 - gt_N_3) ** 2))


def get_shortest_path_fast_batched_x_1(x_0_N_3: Tensor, x_1_N_3: Tensor, batch: Batch) -> Tensor:
    """Batched Kabsch rotation of ``x_1`` onto ``x_0``'s frame, per graph.

    Verbatim from ``flow_matching/utils.py:140-188``. Takes the PyG
    ``Batch`` itself (reads ``batch.batch``) rather than a bare tensor, so
    it needs no adaptation for this port's PyG-native collate.
    """
    device = x_0_N_3.device
    n_graphs = int(batch.batch.max().item() + 1)

    counts = torch.bincount(batch.batch, minlength=n_graphs).to(x_0_N_3.dtype)

    centers_x0 = torch.zeros((n_graphs, 3), device=device).index_add(0, batch.batch, x_0_N_3)
    centers_x1 = torch.zeros((n_graphs, 3), device=device).index_add(0, batch.batch, x_1_N_3)
    centers_x0 = centers_x0 / counts.unsqueeze(1)
    centers_x1 = centers_x1 / counts.unsqueeze(1)

    x0_centered = x_0_N_3 - centers_x0[batch.batch]
    x1_centered = x_1_N_3 - centers_x1[batch.batch]

    prod = x1_centered.unsqueeze(2) * x0_centered.unsqueeze(1)
    m = torch.zeros((n_graphs, 3, 3), device=device).index_add(0, batch.batch, prod)

    u, _s, vt = torch.linalg.svd(m)

    det = torch.det(torch.bmm(u, vt))
    d = torch.eye(3, device=device).unsqueeze(0).repeat(n_graphs, 1, 1)
    d[det < 0, 2, 2] = -1
    r_opt = torch.bmm(u, torch.bmm(d, vt))

    x1_rotated = torch.bmm(x1_centered.unsqueeze(1), r_opt[batch.batch]).squeeze(1)
    return x1_rotated + centers_x0[batch.batch]


def get_perturbed_flow_point_and_time(
    batch: Batch, device: torch.device
) -> Tuple[Tensor, Tensor, Tensor]:
    """Draw ``x_0``, interpolate to a random time, and return the target
    velocity.

    Adapted from ``FlowModule.get_perturbed_flow_point_and_time``
    (``flow_module.py:79-92``): reads ``batch.ts_pos`` where upstream reads
    ``batch.pos[:, 1, :]`` (see the module docstring).

    Args:
        batch: the PyG batch from ``goflow_collate``; must carry ``ts_pos``.
        device: where to draw the Gaussian noise and the per-graph times.

    Returns:
        ``(x_t_N_3, dx_dt_N_3, t_G)``: the interpolated point, the target
        straight-line velocity, and the per-graph flow time.
    """
    x_1_n_3 = batch.ts_pos
    x_0_n_3 = torch.randn_like(x_1_n_3, device=device)

    t_g = torch.rand(batch.num_graphs, 1, device=device)
    t_n = t_g[batch.batch]

    x_1_aligned_n_3 = get_shortest_path_fast_batched_x_1(x_0_n_3, x_1_n_3, batch)
    x_t_n_3 = (1 - t_n) * x_0_n_3 + t_n * x_1_aligned_n_3
    dx_dt_n_3 = x_1_aligned_n_3 - x_0_n_3

    return x_t_n_3, dx_dt_n_3, t_g


def euler_integrate(ode_func: Callable[[float, Tensor], Tensor], x_init: Tensor, t_grid: Tensor) -> Tensor:
    """Fixed-step forward-Euler integration; returns only the final point.

    See the module docstring for why this replaces
    ``torchdiffeq.odeint(ode_func, x_init, t_grid, method='euler')`` and why
    only the final position (not the trajectory) is returned.

    Args:
        ode_func: ``(t, x) -> dx/dt``.
        x_init: ``(N, 3)`` starting point, ``t_grid[0]``.
        t_grid: ``(num_steps,)`` ascending time points, e.g.
            ``torch.linspace(0, 1, num_steps)``.

    Returns:
        ``(N, 3)`` the position at ``t_grid[-1]``.
    """
    x = x_init
    for i in range(t_grid.numel() - 1):
        dt = t_grid[i + 1] - t_grid[i]
        x = x + dt * ode_func(t_grid[i], x)
    return x
