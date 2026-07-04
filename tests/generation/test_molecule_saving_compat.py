"""Compatibility tests for molecule files written during generation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.generation


def _read_xyz(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return int(lines[0]), lines[1], lines[2:]


def test_save_xyz_file_writes_valid_symbols_and_honors_node_mask(
    monkeypatch, tmp_path
):
    from MolecularDiffusion.utils.geom_utils import save_xyz_file

    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "xyz"
    one_hot = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        ]
    )
    positions = torch.tensor(
        [
            [
                [0.1, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ]
    )
    node_mask = torch.tensor([[1, 1, 0]])

    save_xyz_file(
        out_dir,
        one_hot,
        positions,
        atom_decoder=["H", "C"],
        node_mask=node_mask,
    )

    atom_count, comment, atoms = _read_xyz(out_dir / "molecule_000.xyz")
    assert atom_count == 2
    assert comment == ""
    assert atoms == [
        "H 0.100000001 0.000000000 0.000000000",
        "C 1.000000000 0.000000000 0.000000000",
    ]


def test_save_xyz_file_skips_atoms_near_origin(monkeypatch, tmp_path):
    from MolecularDiffusion.utils.geom_utils import save_xyz_file

    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "xyz"
    one_hot = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    positions = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]])

    save_xyz_file(out_dir, one_hot, positions, atom_decoder=["H", "C"])

    atom_count, _, atoms = _read_xyz(out_dir / "molecule_000.xyz")
    assert atom_count == 1
    assert atoms == ["C 1.000000000 2.000000000 3.000000000"]


def test_save_xyz_file_uses_atomic_number_fallback_for_unknown_token(
    monkeypatch, tmp_path
):
    from MolecularDiffusion.utils.geom_utils import save_xyz_file

    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "xyz"
    one_hot = torch.tensor([[[0.0, 1.0]]])
    positions = torch.tensor([[[0.5, 0.0, 0.0]]])
    atomic_numbers = torch.tensor([[8]])

    save_xyz_file(
        out_dir,
        one_hot,
        positions,
        atom_decoder=["H", "Suisei"],
        atomic_numbers=atomic_numbers,
        use_unknown_fallback=True,
    )

    atom_count, _, atoms = _read_xyz(out_dir / "molecule_000.xyz")
    assert atom_count == 1
    assert atoms == ["O 0.500000000 0.000000000 0.000000000"]


def test_save_xyz_file_atomic_numbers_writes_masked_valid_atoms(tmp_path):
    from MolecularDiffusion.utils.geom_utils import save_xyz_file_atomic_numbers

    positions = torch.tensor(
        [
            [
                [0.1, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ]
    )
    atomic_numbers = torch.tensor([[6, 1, 8]])
    node_mask = torch.tensor([[1, 1, 0]])

    save_xyz_file_atomic_numbers(
        tmp_path,
        positions,
        atomic_numbers,
        node_mask=node_mask,
        idxs=[7],
    )

    atom_count, comment, atoms = _read_xyz(tmp_path / "molecule_007.xyz")
    assert atom_count == 2
    assert comment == ""
    assert atoms == [
        "C 0.100000001 0.000000000 0.000000000",
        "H 1.000000000 0.000000000 0.000000000",
    ]


def test_save_xyz_file_atomic_numbers_rejects_bad_shapes(tmp_path):
    from MolecularDiffusion.utils.geom_utils import save_xyz_file_atomic_numbers

    with pytest.raises(ValueError, match="positions"):
        save_xyz_file_atomic_numbers(
            tmp_path,
            torch.zeros(2, 3),
            torch.zeros(2, dtype=torch.long),
        )

    with pytest.raises(ValueError, match="atomic_numbers"):
        save_xyz_file_atomic_numbers(
            tmp_path,
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, 1, dtype=torch.long),
        )


def test_save_shepherd_outputs_writes_xyz_and_optional_modalities(tmp_path):
    from MolecularDiffusion.utils.geom_utils import save_shepherd_outputs

    structures = [
        {
            "x1": {
                "atoms": np.array([6, 1]),
                "positions": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            },
            "x2": {"positions": np.ones((2, 3))},
            "x3": {"positions": np.ones((2, 3)) * 2, "charges": np.array([0.1, -0.1])},
            "x4": {
                "types": np.array([1]),
                "positions": np.ones((1, 3)) * 3,
                "directions": np.ones((1, 3)),
            },
        }
    ]

    save_shepherd_outputs(tmp_path, structures, idx_offset=5, save_modalities=True)

    atom_count, comment, atoms = _read_xyz(tmp_path / "mol_0005.xyz")
    assert atom_count == 2
    assert comment == "mol_0005"
    assert atoms[0].startswith("C  ")
    assert atoms[1].startswith("H  ")
    assert (tmp_path / "mol_0005_surface.npy").is_file()
    assert (tmp_path / "mol_0005_esp.npz").is_file()
    assert (tmp_path / "mol_0005_pharm.npz").is_file()
    assert (tmp_path / "mol_0005.npz").is_file()
