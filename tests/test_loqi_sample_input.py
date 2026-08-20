"""``sample_input`` accepts a path or an inline SMILES list."""

import pytest

from MolecularDiffusion.modules.tasks.diffusion_loqi import load_conditioning_pool

VOCAB = ["H", "C", "N", "O", "F", "S", "Cl"]


def test_inline_smiles_list() -> None:
    pool = load_conditioning_pool(["CCO", "c1ccccc1", "N[C@@H](C)C(=O)O"], VOCAB)
    assert len(pool) == 3
    item = pool[0]
    assert item.pos.shape[0] == item.atom_idx.shape[0] > 0
    assert item.bond_type.numel() == item.bond_index.shape[1]


def test_inline_smiles_skips_bad_and_honours_limit() -> None:
    assert len(load_conditioning_pool(["CCO", "not_a_smiles"], VOCAB)) == 1
    assert len(load_conditioning_pool(["CCO", "CCC", "CCN"], VOCAB, 2)) == 2


def test_empty_pool_raises() -> None:
    with pytest.raises(ValueError, match="no usable molecules"):
        load_conditioning_pool([], VOCAB)


def test_smi_file_still_works(tmp_path) -> None:
    """Regression: the file path shares the inline loader after the refactor."""
    smi = tmp_path / "m.smi"
    smi.write_text("CCO ethanol\nCCC propane\n")
    assert len(load_conditioning_pool(str(smi), VOCAB)) == 2


def test_unknown_extension_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="expected .sdf"):
        load_conditioning_pool(str(tmp_path / "m.xyz"), VOCAB)
