"""Model-specific compatibility tests for diffusion tasks."""

from __future__ import annotations

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.diffusion


class DummyDiffusionModel(nn.Module):
    in_node_nf = 5
    extra_norm_values = [0.5]
    include_charges = True
    norm_values = (2.0, 4.0, 10.0)


def test_diffusion_task_derives_atom_type_count_from_model_features():
    from MolecularDiffusion.modules.tasks.diffusion import GeomMolecularGenerative

    task = GeomMolecularGenerative(
        DummyDiffusionModel(),
        condition=["energy"],
        normalize_condition="mad",
    )

    assert task.condition == ["energy"]
    assert task.n_atom_types == 3
    assert task.n_dim_data == 5
    assert task.normalize_condition == "mad"


def test_diffusion_task_rejects_unknown_reference_freeze_mode():
    from MolecularDiffusion.modules.tasks.diffusion import GeomMolecularGenerative

    with pytest.raises(ValueError, match="reference_freeze_mode"):
        GeomMolecularGenerative(
            DummyDiffusionModel(),
            reference_freeze_mode="coordinates_only",
        )


def test_diffusion_preprocess_without_train_set_preserves_inference_defaults():
    from MolecularDiffusion.modules.tasks.diffusion import GeomMolecularGenerative

    task = GeomMolecularGenerative(DummyDiffusionModel(), n_node_dist={3: 1})

    task.preprocess(None)

    assert task.atomic_numbers == []
    assert task.atom_decoder == []
    assert task.atom_encoder == {}
    assert task.dataset_smiles_list == []
    assert task.max_n_nodes == 0
    assert task.n_node_dist == {}


def test_diffusion_reference_feature_stats_uses_modal_features_and_charges():
    from MolecularDiffusion.modules.tasks.diffusion import GeomMolecularGenerative

    task = GeomMolecularGenerative(
        DummyDiffusionModel(),
        reference_indices=[0, 1],
        reference_freeze_mode="features_only",
    )
    train_set = [
        {
            "node_feature": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
            "charges": torch.tensor([6.0, 1.0, 1.0]),
        },
        {
            "node_feature": torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
            "charges": torch.tensor([6.0, 1.0, 8.0]),
        },
    ]

    stats = task._compute_reference_feature_stats(train_set)

    assert stats["reference_indices"] == [0, 1]
    assert stats["node_feature"].shape == (1, 2, 2)
    assert stats["atomic_numbers"].shape == (1, 2, 1)
    assert torch.allclose(
        stats["node_feature"][0],
        torch.tensor([[0.25, 0.0], [0.0, 0.25]]),
    )
    assert torch.allclose(
        stats["atomic_numbers"][0, :, 0],
        torch.tensor([0.6, 0.1]),
    )


def test_diffusion_geometric_constraint_keys_are_filtered_from_condition_config():
    from MolecularDiffusion.modules.tasks.diffusion import _without_geometric_constraint_cfgs

    cfg = {
        "cfg_scale": 2.0,
        "connector_dicts": [{"a": 1}],
        "constraint_strength": 3.0,
        "scale_factor": 1.5,
    }

    assert _without_geometric_constraint_cfgs(cfg) == {"cfg_scale": 2.0}
