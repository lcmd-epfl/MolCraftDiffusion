"""Flax/e3x initializers, reproduced exactly.

Two of these are parity traps that a naive PyTorch port gets wrong:

* Flax's ``lecun_normal`` is a **truncated** normal with stddev
  ``sqrt(1/fan_in) / 0.87962566103423978`` (0.8796... is the stddev of a
  standard normal truncated to ``(-2, 2)``), truncated at ``±2·stddev``.
  Neither ``normal_(std=sqrt(1/fan_in))`` nor ``nn.Linear``'s own default
  ``U(-1/√fan_in, 1/√fan_in)`` matches it.
* ``tensor_lecun_normal`` computes ``fan_in`` as the number of **allowed
  ``(l1,l2) -> l3`` coupling paths** for each output ``(p3, l3)``, not from the
  kernel shape, and hard-masks parity-/triangle-forbidden paths to zero.

Port of ``e3x/nn/initializers.py``.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch

#: stddev of a standard normal truncated to (-2, 2).
TRUNCATED_NORMAL_STDDEV = 0.87962566103423978


def lecun_normal_(tensor: torch.Tensor, fan_in: int) -> torch.Tensor:
    """Flax ``jax.nn.initializers.lecun_normal()``, in place."""
    stddev = (1.0 / fan_in) ** 0.5 / TRUNCATED_NORMAL_STDDEV
    return torch.nn.init.trunc_normal_(
        tensor, mean=0.0, std=stddev, a=-2 * stddev, b=2 * stddev
    )


def _parity_degree_index_parity_list(num_parity: int, num_degree: int):
    """``(p, l, d)`` triples, ``d`` being the physical parity of that slot."""
    if num_parity == 2:
        return [(0, l, 0) for l in range(num_degree)] + [  # noqa: E741
            (1, l, 1) for l in range(num_degree)  # noqa: E741
        ]
    if num_parity == 1:
        return [(0, l, l % 2) for l in range(num_degree)]  # noqa: E741
    msg = f"num_parity should be 1 or 2, received {num_parity}"
    raise ValueError(msg)


def compute_tensor_fans_and_mask(shape):
    """``(fan_in, fan_out, mask)`` for a ``(P1,L1,P2,L2,P3,L3,F)`` kernel.

    A coupling path is forbidden when the parities do not compose
    (``(d1 + d2) % 2 != d3``) or the degrees violate the triangle inequality
    (``not abs(l1-l2) <= l3 <= l1+l2``).
    """
    if len(shape) != 7:
        msg = f"shape should be len=7, received len={len(shape)}"
        raise ValueError(msg)
    fan_in = np.zeros((1, 1, 1, 1, *shape[-3:-1], 1), dtype=np.float64)
    fan_out = np.zeros((*shape[:-3], 1, 1, 1), dtype=np.float64)
    mask = np.ones(shape, dtype=bool)

    all1 = _parity_degree_index_parity_list(shape[0], shape[1])
    all2 = _parity_degree_index_parity_list(shape[2], shape[3])
    all3 = _parity_degree_index_parity_list(shape[4], shape[5])

    for (p1, l1, d1), (p2, l2, d2), (p3, l3, d3) in itertools.product(
        all1, all2, all3
    ):
        if (d1 + d2) % 2 != d3 or not abs(l1 - l2) <= l3 <= l1 + l2:
            mask[p1, l1, p2, l2, p3, l3, :] = False
        else:
            fan_in[0, 0, 0, 0, p3, l3, :] += 1
            fan_out[p1, l1, p2, l2, 0, 0, :] += 1
    return fan_in, fan_out, mask


def tensor_lecun_normal_(tensor: torch.Tensor) -> torch.Tensor:
    """e3x ``tensor_lecun_normal()``: variance ``1/fan_in``, then masked."""
    shape = tuple(tensor.shape)
    fan_in, _fan_out, mask = compute_tensor_fans_and_mask(shape)
    denom = np.where(fan_in > 0, fan_in, 1.0)
    stddev = np.sqrt(1.0 / denom) / TRUNCATED_NORMAL_STDDEV

    with torch.no_grad():
        # Draw a standard normal truncated to (-2, 2), then scale per-element.
        # trunc_normal_ with std=1, a=-2, b=2 is exactly that distribution.
        torch.nn.init.trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0)
        scale = torch.as_tensor(stddev, dtype=tensor.dtype, device=tensor.device)
        tensor.mul_(scale)
        tensor.mul_(torch.as_tensor(mask, device=tensor.device).to(tensor.dtype))
    return tensor


def tensor_product_mask(shape) -> np.ndarray:
    """``e3x/nn/modules._make_tensor_product_mask``: **parity only**.

    Deliberately weaker than :func:`compute_tensor_fans_and_mask`, which also
    applies the triangle rule -- e3x uses the parity-only mask in the forward
    pass and the stronger one in the initializer. Reproduced as-is.
    """
    mask = np.ones((*shape, 1), dtype=np.float64)
    idx1 = _parity_degree_index_parity_list(shape[0], shape[1])
    idx2 = _parity_degree_index_parity_list(shape[2], shape[3])
    idx3 = _parity_degree_index_parity_list(shape[4], shape[5])
    for (p1, l1, d1), (p2, l2, d2), (p3, l3, d3) in itertools.product(
        idx1, idx2, idx3
    ):
        if (d1 + d2) % 2 != d3:
            mask[p1, l1, p2, l2, p3, l3, :] = 0
    return mask
