"""Geometry helpers for multi-LigandDiff's 20-metal scaffold decomposition.

Everything else multi-LigandDiff needs at runtime -- ``EDM``, ``Dynamics``,
``remove_partial_mean_with_mask``, ``FoundNaNException`` -- is imported
unchanged from ``MolecularDiffusion.modules.models.ligandiff``.

This module exists for one reason: ``models/ligandiff/utils.py`` pins
``METAL_Z`` to LigandDiff's **ten** metals as a module constant, and its
``ligand_groups_from_geometry`` reads it with no override parameter.
multi-LigandDiff supports **twenty** (``src/const.py:15``), so the
decomposition is reimplemented here against the wider set rather than by
editing a file the already-working ``ligandiff`` integration depends on.

``coord_sites_from_geometry`` is genuinely new: multi-LigandDiff adds a
per-atom "coordinates the metal" flag (``coord_site``) that LigandDiff has no
analogue for. Upstream derives it from molSimplify's ``ligcon``
(``generate.py:101-106``); molSimplify is not a dependency here and generation
input is a bare ``.xyz``, so it is recovered from the covalent-contact
criterion instead -- the same substitution ``ligandiff`` already validated for
``ligand_group``.
"""

from typing import List, Set

import numpy as np
import torch
from ase.data import covalent_radii

# src/const.py:15 metals2idx -- the twenty metals multi-LigandDiff supports.
# NOTE the released weights have only ever seen Z in {24..30} (Cr..Zn); the
# other ten are declared for parsing, not for quality. See
# docs/model_integrations/ligandiff_multi/INTEGRATION_PLAN.md.
METAL_Z: Set[int] = {
    22, 23, 24, 25, 26, 27, 28, 29, 30, 40,
    42, 44, 45, 46, 48, 74, 75, 76, 77, 78,
}

# src/const.py:26 cn_oct -- for a context with coordination number k, the
# denticity partitions of the remaining 6 - k octahedral sites among the
# ligands still to be generated. Every partition in bucket k sums to 6 - k;
# `sample()` relies on that (upstream asserts sum(coord_site) == 6 at
# generate.py:210).
CN_OCT: dict = {
    0: [
        [6],
        [5, 1], [4, 2], [3, 3],
        [4, 1, 1], [3, 2, 1], [2, 2, 2],
        [3, 1, 1, 1], [2, 2, 1, 1],
        [2, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ],
    1: [
        [5],
        [4, 1], [3, 2],
        [3, 1, 1], [2, 2, 1],
        [2, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ],
    2: [
        [4],
        [3, 1], [2, 2],
        [2, 1, 1],
        [1, 1, 1, 1],
    ],
    3: [[3], [2, 1], [1, 1, 1]],
    4: [[2], [1, 1]],
    5: [[1]],
}


def _contact_inputs(coords: torch.Tensor, charges: torch.Tensor):
    """Positions, atomic numbers, covalent radii and the metal mask."""
    z = charges.detach().cpu().numpy().astype(int)
    pos = coords.detach().cpu().numpy()
    radii = np.array(
        [covalent_radii[zi] if zi < len(covalent_radii) else 0.77 for zi in z]
    )
    is_metal = np.array([zi in METAL_Z for zi in z])
    return pos, z, radii, is_metal


def ligand_groups_from_geometry(
    coords: torch.Tensor,
    charges: torch.Tensor,
    n_slots: int = 6,
    scale_factor: float = 1.25,
) -> torch.Tensor:
    """Recover the ``ligand_group`` one-hot from geometry alone.

    Same construction as ``models/ligandiff/utils.py``'s function of the same
    name, over the 20-metal ``METAL_Z`` above: delete the metal(s), take the
    connected components of the covalent-radius contact graph, one slot per
    component. Components beyond ``n_slots`` fold into the last slot.

    Returns ``(N, n_slots)`` float one-hot; metal rows are all zero. A
    metal-only scaffold (upstream's ``[]_[...]`` total-generation case)
    correctly yields the all-zero matrix, leaving every slot free.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    pos, z, radii, is_metal = _contact_inputs(coords, charges)
    lig_idx = np.where(~is_metal)[0]

    groups = torch.zeros(len(z), n_slots, dtype=torch.float32)
    if len(lig_idx) == 0:
        return groups

    p = pos[lig_idx]
    r = radii[lig_idx]
    dist = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    cutoff = (r[:, None] + r[None, :]) * scale_factor
    adj = (dist < cutoff) & ~np.eye(len(lig_idx), dtype=bool)

    _, labels = connected_components(csr_matrix(adj), directed=False)
    for local, label in enumerate(labels):
        groups[lig_idx[local], min(int(label), n_slots - 1)] = 1.0
    return groups


def coord_sites_from_geometry(
    coords: torch.Tensor,
    charges: torch.Tensor,
    scale_factor: float = 1.25,
) -> torch.Tensor:
    """Recover the per-atom ``coord_site`` flag from geometry alone.

    ``coord_site[i] == 1`` iff atom ``i`` is a non-metal in covalent contact
    with a metal, i.e. it occupies one of the metal's coordination sites.
    Upstream gets the same set from molSimplify's ``ligcon``
    (``generate.py:101-106``).

    Returns ``(N,)`` float 0/1; metal rows are always 0.
    """
    pos, z, radii, is_metal = _contact_inputs(coords, charges)
    sites = torch.zeros(len(z), dtype=torch.float32)
    m_idx = np.where(is_metal)[0]
    l_idx = np.where(~is_metal)[0]
    if len(m_idx) == 0 or len(l_idx) == 0:
        return sites

    dist = np.linalg.norm(
        pos[l_idx][:, None, :] - pos[m_idx][None, :, :], axis=-1
    )
    cutoff = (radii[l_idx][:, None] + radii[m_idx][None, :]) * scale_factor
    bonded = (dist < cutoff).any(axis=1)
    sites[l_idx[bonded]] = 1.0
    return sites


def distribute_atoms(n_new: int, denticities: List[int]) -> List[int]:
    """Split ``n_new`` new atoms across ligands of the given denticities.

    Each ligand gets at least ``d`` atoms (it has to supply ``d`` donor
    atoms); the surplus is dealt out round-robin. Upstream instead draws each
    ligand's size independently (``generate.py:187-192``) and lets the total
    fall out; the platform fixes the total via ``mol_size`` / the node-size
    distribution, so the split is derived from it rather than the reverse.
    """
    if n_new < sum(denticities):
        raise ValueError(
            f"{n_new} new atoms cannot host denticities {denticities} "
            f"(need at least {sum(denticities)})"
        )
    sizes = list(denticities)
    for i in range(n_new - sum(denticities)):
        sizes[i % len(sizes)] += 1
    return sizes
