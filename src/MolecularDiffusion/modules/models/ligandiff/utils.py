"""LigandDiff helpers.

``remove_partial_mean_with_mask`` and ``FoundNaNException`` are verbatim from
the target repo's ``src/utils.py``. ``ligand_groups_from_geometry`` is new --
see its docstring.
"""

from typing import Set

import numpy as np
import torch
from ase.data import covalent_radii
from torch_scatter import scatter_add

# src/const.py:13 idx2metals -- the ten metals LigandDiff supports.
METAL_Z: Set[int] = {24, 25, 26, 27, 28, 29, 30, 44, 46, 78}


def remove_partial_mean_with_mask(x, center_of_mass_mask, batch_seg):
    """Subtract the masked subset's centre of mass from every coordinate."""
    x_masked = x * center_of_mass_mask
    mean = scatter_add(x_masked, batch_seg, dim=0) / scatter_add(
        center_of_mass_mask, batch_seg, dim=0
    ).view(-1, 1)
    return x - mean[batch_seg]


class FoundNaNException(Exception):  # noqa: N818
    """Raised by ``Dynamics.forward`` when the denoiser output goes NaN."""

    def __init__(self, x, h) -> None:
        super().__init__("NaN in denoiser output")
        x_nan_idx = self.find_nan_idx(x)
        h_nan_idx = self.find_nan_idx(h)

        self.x_h_nan_idx = x_nan_idx & h_nan_idx
        self.only_x_nan_idx = x_nan_idx.difference(h_nan_idx)
        self.only_h_nan_idx = h_nan_idx.difference(x_nan_idx)

    @staticmethod
    def find_nan_idx(z):
        return {i for i in range(z.shape[0]) if torch.any(torch.isnan(z[i]))}


def ligand_groups_from_geometry(
    coords: torch.Tensor,
    charges: torch.Tensor,
    n_slots: int = 6,
    scale_factor: float = 1.25,
) -> torch.Tensor:
    """Recover LigandDiff's ``ligand_group`` one-hot from geometry alone.

    Upstream derives this column block with molSimplify's ``ligand_breakdown``
    (``generate.py:81-87``): each atom is tagged with the index of the ligand
    it belongs to, the metal getting an all-zero row. molSimplify is not a
    dependency here and generation input is a bare ``.xyz``, so the same
    decomposition is recovered directly: delete the metal(s), then take the
    connected components of the covalent-radius contact graph -- which is what
    "a ligand" means for a mononuclear complex.

    Args:
        coords: ``(N, 3)`` positions.
        charges: ``(N,)`` atomic numbers (metals identified via ``METAL_Z``).
        n_slots: number of ligand columns (upstream: 6). Components beyond
            this are folded into the last slot.
        scale_factor: covalent-radius multiplier for the contact criterion.

    Returns:
        ``(N, n_slots)`` float one-hot; metal rows are all zero.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    z = charges.detach().cpu().numpy().astype(int)
    pos = coords.detach().cpu().numpy()
    n = len(z)

    is_metal = np.array([zi in METAL_Z for zi in z])
    lig_idx = np.where(~is_metal)[0]

    groups = torch.zeros(n, n_slots, dtype=torch.float32)
    if len(lig_idx) == 0:
        return groups

    p = pos[lig_idx]
    radii = np.array(
        [covalent_radii[zi] if zi < len(covalent_radii) else 0.77 for zi in z]
    )[lig_idx]
    dist = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    cutoff = (radii[:, None] + radii[None, :]) * scale_factor
    adj = (dist < cutoff) & ~np.eye(len(lig_idx), dtype=bool)

    _, labels = connected_components(csr_matrix(adj), directed=False)
    for local, label in enumerate(labels):
        groups[lig_idx[local], min(int(label), n_slots - 1)] = 1.0
    return groups
