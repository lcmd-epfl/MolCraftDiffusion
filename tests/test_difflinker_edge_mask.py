"""The two DiffLinker edge-mask sign conventions must not drift.

Released non-pocket DiffLinker weights were trained against the mask
DiffLinker's own collate produces, which is ``-1`` on real edges and ``-2``
on self-loops because ``~torch.eye(torch.int8)`` is a *bitwise* NOT. Feeding
those weights the arithmetically-correct ``1``/``0`` mask makes them diverge
to NaN on the first reverse step. Weights trained by this platform learnt the
``1``/``0`` mask, so the default must stay ``1``/``0``.
"""

import torch

from MolecularDiffusion.modules.tasks.diffusion_difflinker import (
    dense_edge_mask,
)


def test_default_mask_is_one_zero_without_self_loops():
    atom_mask = torch.ones(1, 4, 1)
    m = dense_edge_mask(atom_mask).view(4, 4)
    assert set(m.unique().tolist()) == {0.0, 1.0}
    assert torch.diagonal(m).eq(0).all()
    assert m.sum() == 4 * 3


def test_upstream_int8_mask_negates_edges_and_keeps_self_loops():
    atom_mask = torch.ones(1, 4, 1)
    m = dense_edge_mask(atom_mask, upstream_int8=True).view(4, 4)
    assert torch.diagonal(m).eq(-2).all()
    off_diag = m[~torch.eye(4, dtype=torch.bool)]
    assert off_diag.eq(-1).all()
    # matches upstream src/datasets.py:366-369 exactly
    am = atom_mask.squeeze(-1)
    edges = am[:, None, :] * am[:, :, None]
    diag = ~torch.eye(4, dtype=torch.int8)
    assert torch.equal(m, (edges.to(torch.int8) * diag).float().view(4, 4))


def test_padded_atoms_are_masked_out_in_both_conventions():
    atom_mask = torch.tensor([[[1.0], [1.0], [0.0]]])  # third atom is padding
    for upstream_int8 in (False, True):
        m = dense_edge_mask(atom_mask, upstream_int8=upstream_int8).view(3, 3)
        assert m[2].eq(0).all()
        assert m[:, 2].eq(0).all()
