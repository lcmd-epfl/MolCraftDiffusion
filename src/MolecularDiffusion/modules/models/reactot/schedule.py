"""The Schrodinger-bridge schedule React-OT replaces DDPM's noise with.

Vendored from ``reactot/diffusion/_schedule.py`` and
``reactot/diffusion/_utils.py`` of
https://github.com/deepprinciple/react-ot at commit 6dfccd0.

:class:`SBSchedule` is a **plain Python class, not an** ``nn.Module``. It
holds no buffers and nothing of it reaches a state dict -- which is exactly
why React-OT's released checkpoint has 246 tensors where OA-ReactDiff's has
248 (OA carries two gamma buffers of 5001 + 151 values, and
10,651,063 - 10,645,911 = 5,152).

A load-bearing degeneracy in the released settings
--------------------------------------------------

At ``timesteps=3000`` / ``beta_max=0.3`` / ``power=0.5`` the schedule is
**constant**::

    linear_end = beta_max / timesteps = 1e-4 = make_beta_schedule's own
    hardcoded linear_start, so the linspace is degenerate; ** 0.5 keeps it
    constant; the mirror-concatenate keeps it constant; and the final
    renormalisation lands every entry on 5e-5.

:meth:`~MolecularDiffusion.modules.models.reactot.en_sb.EnSB.ode_sampling`
*asserts* that constancy (``en_sb.py:411`` upstream). Changing ``timesteps``
or ``beta_max`` therefore breaks the ODE solver with an assertion, and
changes the DDPM solver's meaning silently. Both must stay at 3000 / 0.3;
see the plan's Hyperparameter Provenance table.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


def make_beta_schedule(
    n_timestep: int = 1000,
    linear_start: float = 1e-4,
    linear_end: float = 2e-2,
    power: float = 1.0,
    inv_power: float = 1.0,
) -> np.ndarray:
    """Betas for the bridge, as upstream builds them.

    Args:
        n_timestep: length of the grid.
        linear_start: first endpoint, before ``inv_power`` / ``power``.
        linear_end: last endpoint; the caller passes ``beta_max/timesteps``.
        power: exponent applied to the whole linspace.
        inv_power: exponent applied to each endpoint first.

    Returns:
        ``(n_timestep,)`` float64 array.
    """
    betas = (
        torch.linspace(
            linear_start**inv_power,
            linear_end**inv_power,
            n_timestep,
            dtype=torch.float64,
        )
        ** power
    )
    return betas.numpy()


def compute_gaussian_product_coef(
    sigma1: Any, sigma2: Any
) -> Tuple[Any, Any, Any]:
    """Coefficients of the product of two Gaussians.

    Given ``p1 = N(x_t | x_0, sigma1**2)`` and ``p2 = N(x_t | x_1,
    sigma2**2)``, return ``(coef1, coef2, var)`` such that
    ``p1 * p2 = N(x_t | coef1 * x0 + coef2 * x1, var)``.

    Deliberately untyped in its operands: it is called both with numpy
    arrays (schedule construction) and with torch tensors
    (:meth:`EnSB.p_posterior`), and does nothing either cannot do.

    Args:
        sigma1: first standard deviation.
        sigma2: second standard deviation.

    Returns:
        ``(coef1, coef2, var)``.
    """
    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var


def space_indices(num_steps: int, count: int) -> List[int]:
    """``count`` indices spread evenly over ``range(num_steps)``.

    This is what turns React-OT's 3000-step grid into an ``nfe``-step
    sampling schedule: ``space_indices(3000, nfe + 1)`` picks the ``nfe + 1``
    grid points the sampler walks backwards through, so the number of
    network evaluations is ``nfe``.

    Args:
        num_steps: size of the grid to sample from.
        count: how many indices to take.

    Returns:
        Ascending list of ``count`` indices, starting at 0.
    """
    if count > num_steps:
        raise ValueError(f"count {count} > num_steps {num_steps}")
    frac_stride = 1.0 if count <= 1 else (num_steps - 1) / (count - 1)
    cur_idx = 0.0
    taken_steps = []
    for _ in range(count):
        taken_steps.append(round(cur_idx))
        cur_idx += frac_stride
    return taken_steps


def unsqueeze_xdim(z: Tensor, xdim: Sequence[int]) -> Tensor:
    """Append ``len(xdim)`` trailing singleton axes to ``z``.

    Args:
        z: tensor to broadcast.
        xdim: the trailing shape it must broadcast against.

    Returns:
        ``z`` viewed with the extra axes.
    """
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]


class SBSchedule:
    """The bridge's variance schedule.

    Not an ``nn.Module``: see the module docstring. Everything is derived
    from ``betas``, so a task rebuilds it from four numbers at construction
    and the checkpoint carries none of it.

    Attributes:
        timesteps: grid length, taken from the built array rather than the
            argument (the mirror-concatenate makes them equal for even
            ``timesteps``, which 3000 is).
        betas: ``(T,)`` per-step variance increments.
        std_fwd: ``sqrt(cumsum(betas))``.
        std_bwd: ``sqrt(reverse cumsum(betas))``.
        std_sb: ``sqrt(var)`` of the bridge marginal.
        mu_x0: weight on the endpoint ``x0`` in ``q(x_t | x_0, x_1)``.
        mu_x1: weight on the endpoint ``x1``.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_max: float = 0.3,
        power: float = 1.0,
        inv_power: float = 1.0,
    ) -> None:
        """Build the schedule.

        Args:
            timesteps: grid length. **3000 for the released weights.**
            beta_max: total variance. **0.3 for the released weights.**
            power: exponent on the linspace.
            inv_power: exponent on its endpoints.
        """
        betas = make_beta_schedule(
            n_timestep=timesteps,
            linear_end=beta_max / timesteps,
            power=power,
            inv_power=inv_power,
        )
        # Mirror the first half onto the second, then renormalise so the
        # maximum is beta_max / timesteps / 2. Verbatim upstream
        # (_schedule.py:283-284); at the released settings both lines are
        # no-ops on a already-constant array, but they are not no-ops at
        # any other `power`.
        betas = np.concatenate(
            [betas[: timesteps // 2], np.flip(betas[: timesteps // 2])]
        )
        betas = (beta_max / timesteps) / np.max(betas) * betas * 0.5

        self.timesteps = int(betas.shape[0])

        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        def to_torch(array: np.ndarray) -> Tensor:
            return torch.tensor(array, dtype=torch.float32)

        self.betas = to_torch(betas)
        self.std_fwd = to_torch(std_fwd)
        self.std_bwd = to_torch(std_bwd)
        self.std_sb = to_torch(std_sb)
        self.mu_x0 = to_torch(mu_x0)
        self.mu_x1 = to_torch(mu_x1)

    @staticmethod
    def inflate_batch_array(array: Tensor, target: Tensor) -> Tensor:
        """Reshape a per-node vector to broadcast against ``target``.

        Args:
            array: ``(n,)`` (or ``(n, 1, ..., 1)``).
            target: the tensor whose rank it must match.

        Returns:
            ``array`` viewed as ``(n, 1, ..., 1)``.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def get_std_fwd(self, step: Tensor, xdim: Any = None) -> Tensor:
        """``std_fwd`` at ``step``, optionally broadcast to ``xdim``.

        Args:
            step: integer index tensor.
            xdim: trailing shape to broadcast against; ``None`` => none.

        Returns:
            The looked-up standard deviations.
        """
        step = step.to(self.std_fwd.device)
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)
