"""Compatibility tests for data collation, caching, and lightweight loaders."""

from __future__ import annotations

import os

import torch


def _pointcloud_item(natoms: int, max_atoms: int = 4):
    coords = torch.arange(natoms * 3, dtype=torch.float32).view(natoms, 3)
    node_feature = torch.eye(2, dtype=torch.float32)[torch.arange(natoms) % 2]
    charges = torch.arange(1, natoms + 1, dtype=torch.float32)
    edge_mask = torch.ones(max_atoms, max_atoms)
    return {
        "coords": coords,
        "node_feature": node_feature,
        "charges": charges,
        "edge_mask": edge_mask[:natoms, :natoms],
        "node_mask": torch.ones(natoms),
        "natoms": torch.tensor(natoms),
        "energy": float(natoms),
    }


def test_graph_collate_batches_pyg_graphs_and_nested_metadata(tiny_pyg_graphs):
    from MolecularDiffusion.data.dataloader import graph_collate

    batch = graph_collate(
        [
            {"graph": tiny_pyg_graphs[0], "energy": 1.5, "name": "mol-a"},
            {"graph": tiny_pyg_graphs[1], "energy": 2.5, "name": "mol-b"},
        ]
    )

    assert batch["graph"].num_graphs == 2
    assert batch["graph"].x.shape == (5, 2)
    assert batch["graph"].batch.tolist() == [0, 0, 1, 1, 1]
    assert torch.allclose(batch["energy"], torch.tensor([1.5, 2.5]))
    assert batch["name"] == ["mol-a", "mol-b"]


def test_pointcloud_collate_v0_trims_all_zero_padded_nodes():
    from MolecularDiffusion.data.dataloader import pointcloud_collate_v0

    def padded(charges):
        return {
            "coords": torch.randn(4, 3),
            "node_feature": torch.randn(4, 2),
            "charges": torch.tensor(charges, dtype=torch.float32),
            "edge_mask": torch.ones(4, 4),
            "name": "same-container",
        }

    batch = pointcloud_collate_v0(
        [padded([6, 1, 0, 0]), padded([8, 1, 0, 0])]
    )

    assert batch["coords"].shape == (2, 2, 3)
    assert batch["node_feature"].shape == (2, 2, 2)
    assert batch["charges"].shape == (2, 2)
    assert batch["edge_mask"].shape == (2, 2, 2)
    assert batch["name"] == ["same-container", "same-container"]


def test_pointcloud_collate_builds_masks_for_variable_size_items():
    from MolecularDiffusion.data.dataloader import pointcloud_collate

    collate = pointcloud_collate(vram_size=40)
    batch = collate([_pointcloud_item(8), _pointcloud_item(9)])

    assert batch["coords"].shape == (2, 9, 3)
    assert batch["node_feature"].shape == (2, 9, 2)
    assert batch["node_mask"][0].tolist() == [1] * 8 + [0]
    assert batch["node_mask"][1].tolist() == [1] * 9
    assert batch["edge_mask"][:, torch.arange(9), torch.arange(9)].sum() == 0
    assert batch["coords"][0, 8].abs().sum() == 0
    assert torch.allclose(batch["energy"], torch.tensor([8.0, 9.0]))


def test_data_module_split_cache_reuses_indices(tmp_path):
    from MolecularDiffusion.runmodes.train.data import DataModule

    module = DataModule(
        root=str(tmp_path),
        filename="unused.csv",
        task_type="diffusion",
        atom_vocab=["H", "C"],
        with_hydrogen=True,
        train_ratio=0.6,
    )
    dataset = torch.utils.data.TensorDataset(torch.arange(10))
    lengths = [6, 2, 2]

    torch.manual_seed(123)
    first = module._build_or_load_splits(dataset, lengths)
    torch.manual_seed(123)
    second = module._build_or_load_splits(dataset, lengths)

    assert [subset.indices for subset in second] == [subset.indices for subset in first]
    cache_files = list(tmp_path.glob("splits_n10_tr0.60000000_seed123.pt"))
    assert len(cache_files) == 1


def test_lightning_data_module_adjusts_singleton_remainder_batch_size():
    from MolecularDiffusion.data.lightning_data_module import MolecularDiffusionDataModule

    class RawModule:
        train_set = torch.utils.data.TensorDataset(torch.arange(5))
        valid_set = torch.utils.data.TensorDataset(torch.arange(5))
        test_set = torch.utils.data.TensorDataset(torch.arange(5))
        collate_fn = None

    dm = MolecularDiffusionDataModule(RawModule(), batch_size=2, pin_memory=False)
    assert dm.train_dataloader().batch_size == 3
    assert dm.val_dataloader().batch_size == 3
    assert dm.test_dataloader().batch_size == 3


def test_lazy_chunked_dataset_reads_metadata_and_properties(tmp_path):
    from MolecularDiffusion.data.component.dataset import LazyChunkedDataset

    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    chunk_path = chunk_dir / "chunk_000000.pt"
    torch.save(
        {
            "coords_list": [torch.zeros(2, 3), torch.ones(3, 3)],
            "node_mask_list": [torch.ones(2), torch.ones(3)],
            "edge_mask_list": [torch.ones(2, 2), torch.ones(3, 3)],
            "node_feature_list": [torch.eye(2), torch.ones(3, 2)],
            "charges_list": [torch.tensor([6, 1]), torch.tensor([8, 1, 1])],
            "n_atoms": [2, 3],
            "xyzs": ["mol-a.xyz", "mol-b.xyz"],
            "targets": {"energy": [1.0, 2.0]},
        },
        chunk_path,
    )
    torch.save(
        {
            "chunk_paths": [os.fspath(chunk_path)],
            "chunk_sizes": [2],
            "tasks": ["energy"],
            "atom_vocab": ["H", "C", "O"],
            "with_hydrogen": True,
            "smiles_list": ["C", "O"],
            "n_atoms": [2, 3],
        },
        chunk_dir / "meta.pt",
    )

    dataset = LazyChunkedDataset(os.fspath(chunk_dir))

    assert len(dataset) == 2
    assert dataset[1]["coords"].shape == (3, 3)
    assert dataset.targets == {"energy": None}
    assert dataset.num_atoms.tolist() == [2, 3]
    assert torch.allclose(dataset.get_property("energy", indices=[1, 0]), torch.tensor([2.0, 1.0]))
    assert dataset.atom_types() == [1, 6, 8]


def test_compact_pyg_data_reconstructs_public_fields():
    from MolecularDiffusion.data.component.dataset import (
        _compact_build_pyg,
        _compact_expand_pyg,
    )

    x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    compact = {
        "drop_edge_index": True,
        "drop_tags": True,
        "pack_ohe": True,
        "int8_z": True,
    }
    slim, ohe_size = _compact_build_pyg(
        node_features=x,
        coords=torch.zeros(2, 3),
        charges=torch.tensor([6, 1]),
        n_nodes=2,
        smiles="C",
        xyz="mol.xyz",
        edge_index=edge_index,
        edge_type="fully_connected",
        mol_index=4,
        compact=compact,
    )

    expanded = _compact_expand_pyg(slim, compact, ohe_size, "fully_connected", 4)

    assert slim.x.dtype == torch.int8
    assert "edge_index" not in list(slim.keys())
    assert torch.allclose(expanded.x, x)
    assert expanded.atomic_numbers.dtype == torch.long
    assert expanded.tags.tolist() == [4, 4]
    assert expanded.edge_index.shape == (2, 2)


def _tiny_ase_db(tmp_path):
    """Two CH4-like molecules: 1 heavy atom + 4 H each."""
    from ase import Atoms
    from ase.db import connect

    path = str(tmp_path / "tiny.db")
    db = connect(path)
    for symbol in ("C", "N"):
        atoms = Atoms(
            symbols=[symbol, "H", "H", "H", "H"],
            positions=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
        )
        db.write(atoms)
    return path


def test_load_db_honours_with_hydrogen(tmp_path):
    from MolecularDiffusion.data.component.dataset import PointCloudDataset

    path = _tiny_ase_db(tmp_path)

    with_h = PointCloudDataset.__new__(PointCloudDataset)
    with_h.load_db(path, atom_vocab=["H", "C", "N"], with_hydrogen=True)
    assert with_h.n_atoms == [5, 5]

    heavy = PointCloudDataset.__new__(PointCloudDataset)
    heavy.load_db(path, atom_vocab=["C", "N"], with_hydrogen=False)
    assert heavy.n_atoms == [1, 1]
    assert torch.cat(heavy.charges_list).tolist() == [6, 7]
    assert heavy.node_feature_list[0].shape == (1, 2)


def test_load_db_raises_when_every_entry_is_discarded(tmp_path):
    import pytest

    from MolecularDiffusion.data.component.dataset import PointCloudDataset

    dataset = PointCloudDataset.__new__(PointCloudDataset)
    with pytest.raises(ValueError, match="No entries could be loaded"):
        dataset.load_db(_tiny_ase_db(tmp_path), atom_vocab=["Xe"])
