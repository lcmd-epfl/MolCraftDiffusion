"""Model-specific compatibility tests for TABASCO integration."""

from __future__ import annotations

import random

import pytest
import torch

pytestmark = pytest.mark.tabasco


def test_tabasco_node_distribution_samples_from_histogram_keys():
    from MolecularDiffusion.modules.tasks.diffusion_tabasco import TabascoNodeDistribution

    random.seed(3)
    dist = TabascoNodeDistribution({"num_atoms_histogram": {4: 10, 7: 1}})

    samples = dist.sample(20)

    assert samples.dtype == torch.long
    assert set(samples.tolist()).issubset({4, 7})
    assert dist.n_node_dist == {4: 10, 7: 1}


def test_tabasco_pointcloud_adapter_prefers_precomputed_node_features():
    from MolecularDiffusion.modules.tasks.diffusion_tabasco import (
        PointCloudToTensorDictAdapter,
    )

    node_feature = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]
    )
    batch = {
        "coords": torch.zeros(1, 3, 3),
        "node_mask": torch.tensor([[1, 1, 0]]),
        "charges": torch.tensor([[2, 1, 0]]),
        "node_feature": node_feature,
        "natoms": torch.tensor([2]),
    }

    td = PointCloudToTensorDictAdapter(num_atom_types=3)(batch)

    assert td.batch_size == torch.Size([1])
    assert torch.equal(td["coords"], batch["coords"])
    assert torch.equal(td["atomics"], node_feature)
    assert td["padding_mask"].tolist() == [[False, False, True]]


def test_tabasco_pointcloud_adapter_falls_back_to_charge_one_hot():
    from MolecularDiffusion.modules.tasks.diffusion_tabasco import (
        PointCloudToTensorDictAdapter,
    )

    batch = {
        "coords": torch.zeros(1, 2, 3),
        "node_mask": torch.tensor([[1, 0]]),
        "charges": torch.tensor([[2, 0]]),
        "natoms": torch.tensor([1]),
    }

    td = PointCloudToTensorDictAdapter(num_atom_types=3)(batch)

    assert td["atomics"].shape == (1, 2, 3)
    assert td["atomics"][0, 0].tolist() == [0.0, 0.0, 1.0]
    assert td["padding_mask"].tolist() == [[False, True]]


def test_tabasco_tensordict_to_pointcloud_restores_masks_and_counts():
    from tensordict import TensorDict

    from MolecularDiffusion.modules.tasks.diffusion_tabasco import (
        TensorDictToPointCloudAdapter,
    )

    td = TensorDict(
        {
            "coords": torch.ones(2, 3, 3),
            "atomics": torch.tensor(
                [
                    [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
                ]
            ),
            "padding_mask": torch.tensor(
                [[False, False, True], [False, True, True]]
            ),
        },
        batch_size=[2],
    )

    out = TensorDictToPointCloudAdapter()(td)

    assert out["charges"].tolist() == [[1, 0, 1], [0, 1, 0]]
    assert out["node_mask"].tolist() == [[1, 1, 0], [1, 0, 0]]
    assert out["natoms"].tolist() == [2, 1]


def test_tabasco_factory_computes_missing_dataset_stats_from_cached_lists():
    from MolecularDiffusion.modules.tasks.diffusion_tabasco import ModelTaskFactory

    class CachedDataset:
        smiles_list = ["C", "O", "N"]
        n_atoms = [2, 3, 3]

        def __len__(self):
            return 3

    factory = ModelTaskFactory(
        task_type="diffusion_tabasco",
        transformer_config={},
        coords_interpolant_config={},
        atomics_interpolant_config={},
        flow_matching_config={},
        num_atom_types=4,
        dataset_stats={},
    )

    factory.compute_dataset_stats(CachedDataset())

    assert factory.dataset_stats["atom_count_histogram"] == {2: 1, 3: 2}
    assert factory.dataset_stats["all_smiles"] == ["C", "O", "N"]
    assert factory.dataset_stats["max_atoms"] == 3
