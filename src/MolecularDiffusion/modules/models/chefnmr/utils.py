"""Geometry / loss helpers ported from ChefNMR (MIT, (c) 2025 Ziyu Xiong).

Upstream: ``src/model/modules/utils.py``, itself started from Boltz
(``jwohlwend/boltz``, MIT) and lucidrains' alphafold3-pytorch (MIT).

Only what the ported forward/sampling path touches is here: the random
SO(3) augmentation that is *how this non-equivariant DiT learns
equivariance*, AlphaFold3's smooth-LDDT auxiliary loss, and the 1-D
sin/cos grid used by the spectra embedder's optional positional encoding.
Upstream's ``ExponentialMovingAverage`` is deliberately NOT ported -- the
platform's engine owns EMA -- but its *layout* is what
``scripts/convert_checkpoint.py`` has to understand, see that file.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def exists(v: object) -> bool:
    return v is not None


def log(t: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return torch.log(t.clamp(min=eps))


def default(v: object, d: object) -> object:
    return v if exists(v) else d


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Rotations given as quaternions (real part first) -> ``(..., 3, 3)``."""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def random_quaternions(
    n: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Uniform random versors with non-negative real part, ``(n, 4)``."""
    if isinstance(device, str):
        device = torch.device(device)
    o = torch.randn((n, 4), dtype=dtype, device=device)
    s = (o * o).sum(1)
    return o / _copysign(torch.sqrt(s), o[:, 0])[:, None]


def random_rotations(
    n: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Uniform random rotation matrices, ``(n, 3, 3)``."""
    return quaternion_to_matrix(random_quaternions(n, dtype=dtype, device=device))


def randomly_rotate(
    coords: torch.Tensor,
    return_second_coords: bool = False,
    second_coords: Optional[torch.Tensor] = None,
):
    rot = random_rotations(len(coords), coords.dtype, coords.device)
    if return_second_coords:
        return torch.einsum("bmd,bds->bms", coords, rot), (
            torch.einsum("bmd,bds->bms", second_coords, rot)
            if second_coords is not None
            else None
        )
    return torch.einsum("bmd,bds->bms", coords, rot)


def center_random_augmentation(  # noqa: PLR0913
    atom_coords: torch.Tensor,
    atom_mask: torch.Tensor,
    s_trans: float = 1.0,
    augmentation: bool = True,
    centering: bool = True,
    return_second_coords: bool = False,
    second_coords: Optional[torch.Tensor] = None,
):
    """Mask-aware centering + random rotation + random translation.

    This is not cosmetic: the DiT backbone has no equivariance built in,
    so this augmentation is the *only* thing that teaches it rotational
    and translational invariance. Dropping it changes what the model
    learns, not just how fast.
    """
    if centering:
        atom_mean = torch.sum(
            atom_coords * atom_mask[:, :, None], dim=1, keepdim=True
        ) / torch.sum(atom_mask[:, :, None], dim=1, keepdim=True)
        atom_coords = atom_coords - atom_mean
        if second_coords is not None:
            second_coords = second_coords - atom_mean

    if augmentation:
        atom_coords, second_coords = randomly_rotate(
            atom_coords, return_second_coords=True, second_coords=second_coords
        )
        random_trans = torch.randn_like(atom_coords[:, 0:1, :]) * s_trans
        atom_coords = atom_coords + random_trans
        if second_coords is not None:
            second_coords = second_coords + random_trans

    if return_second_coords:
        return atom_coords, second_coords
    return atom_coords


def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int, pos: np.ndarray
) -> np.ndarray:
    """``(M, embed_dim)`` sin/cos positional grid."""
    if embed_dim % 2 != 0:
        msg = f"embed_dim must be even, got {embed_dim}"
        raise ValueError(msg)
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def smooth_lddt_loss(  # noqa: PLR0913
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    is_nucleotide: torch.Tensor,
    coords_mask: torch.Tensor,
    lddt_loss_threshold: list,
    nucleic_acid_cutoff: float = 30.0,
    other_cutoff: float = 15.0,
    multiplicity: int = 1,
) -> torch.Tensor:
    """AlphaFold3's smooth-LDDT auxiliary, on all real atom pairs.

    ``is_nucleotide`` is kept in the signature (and passed all-zeros by
    the caller) so the ported maths stays diffable against upstream; for
    small molecules only ``other_cutoff`` is ever active.
    """
    b, n, _ = true_coords.shape
    true_dists = torch.cdist(true_coords, true_coords)
    is_nucleotide = is_nucleotide.repeat_interleave(multiplicity, 0)

    coords_mask = coords_mask.repeat_interleave(multiplicity, 0)
    is_nucleotide_pair = is_nucleotide.unsqueeze(-1).expand(
        -1, -1, is_nucleotide.shape[-1]
    )

    mask = (
        is_nucleotide_pair * (true_dists < nucleic_acid_cutoff).float()
        + (1 - is_nucleotide_pair) * (true_dists < other_cutoff).float()
    )
    mask = mask * (1 - torch.eye(pred_coords.shape[1], device=pred_coords.device))
    mask = mask * (coords_mask.unsqueeze(-1) * coords_mask.unsqueeze(-2))

    pred_dists = torch.cdist(pred_coords, pred_coords)
    dist_diff = torch.abs(true_dists - pred_dists)

    eps = (
        (
            (
                F.sigmoid(lddt_loss_threshold[0] - dist_diff)
                + F.sigmoid(lddt_loss_threshold[1] - dist_diff)
                + F.sigmoid(lddt_loss_threshold[2] - dist_diff)
                + F.sigmoid(lddt_loss_threshold[3] - dist_diff)
            )
            / 4.0
        )
        .view(multiplicity, b // multiplicity, n, n)
        .mean(dim=0)
    )

    eps = eps.repeat_interleave(multiplicity, 0)
    num = (eps * mask).sum(dim=(-1, -2))
    den = mask.sum(dim=(-1, -2)).clamp(min=1)
    return 1.0 - (num / den).mean()
