"""Pocket-size-conditioned prior over ligand atom counts.

Ported from KGDiff ``utils/evaluation/atom_num.py`` (commit ``ad893fc``).

The prior is a **static table** (:mod:`.atom_num_config`, generated once by
the authors from CrossDocked): pockets are binned by their spatial extent,
and each bin carries an empirical distribution over ligand sizes. Nothing
here depends on the training set, so it behaves identically at train and
generate time and needs no checkpoint buffer -- unlike the histogram priors
the other in-tree pocket models carry.
"""

from __future__ import annotations

import numpy as np

from MolecularDiffusion.modules.models.kgdiff.atom_num_config import CONFIG


def get_space_size(pocket_3d_pos: np.ndarray) -> float:
    """Pocket extent: the median of the 10 largest pairwise distances."""
    from scipy import spatial as sc_spatial

    aa_dist = sc_spatial.distance.pdist(pocket_3d_pos, metric="euclidean")
    aa_dist = np.sort(aa_dist)[::-1]
    return float(np.median(aa_dist[:10]))


def _get_bin_idx(space_size: float) -> int:
    bounds = CONFIG["bounds"]
    for i in range(len(bounds)):
        if bounds[i] > space_size:
            return i
    return len(bounds)


def sample_atom_num(space_size: float) -> int:
    """Draw one ligand atom count for a pocket of this extent."""
    num_atom_list, prob_list = CONFIG["bins"][_get_bin_idx(space_size)]
    return int(np.random.choice(num_atom_list, p=prob_list))


def marginal_size_distribution() -> dict[int, float]:
    """``{n_atoms: probability}`` marginalised over all pocket-size bins.

    Used for the task's ``n_node_dist``, which callers only read to derive
    ``max_atom``; the real, pocket-conditioned draw goes through
    :func:`sample_atom_num`.
    """
    bins = CONFIG["bins"]
    marginal: dict[int, float] = {}
    for num_atom_list, prob_list in bins:
        for n, p in zip(num_atom_list, prob_list):
            marginal[int(n)] = marginal.get(int(n), 0.0) + float(p) / len(bins)
    return marginal
