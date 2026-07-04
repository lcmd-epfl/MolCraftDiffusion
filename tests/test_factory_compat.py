"""Compatibility tests for model/task factory interfaces."""

from __future__ import annotations

import pytest


def test_egcl_factory_validates_adapter_condition_names():
    from MolecularDiffusion.runmodes.train.tasks_egcl import ModelTaskFactory

    with pytest.raises(ValueError, match="adapter_conditions entry"):
        ModelTaskFactory(
            task_type="diffusion",
            atom_vocab=["H", "C"],
            condition_names=["energy"],
            adapter_conditions=["gap"],
        )


def test_egcl_regression_factory_builds_property_task():
    from MolecularDiffusion.modules.tasks import ProperyPrediction
    from MolecularDiffusion.runmodes.train.tasks_egcl import ModelTaskFactory

    factory = ModelTaskFactory(
        task_type="regression",
        atom_vocab=["H", "C"],
        condition_names=[],
        hidden_size=8,
        num_layers=1,
        num_sublayers=1,
        task_learn=["energy"],
        num_mlp_layer=1,
        mlp_hidden_dim=8,
        prediction_mlp_type="pernode",
    )

    task = factory.build()

    assert isinstance(task, ProperyPrediction)
    assert list(task.task) == ["energy"]
    assert task.model.hidden_nf == 8
    assert task.mlp is not None


def test_egcl_diffusion_factory_preserves_condition_and_dimensions():
    from MolecularDiffusion.modules.tasks import GeomMolecularGenerative
    from MolecularDiffusion.runmodes.train.tasks_egcl import ModelTaskFactory

    factory = ModelTaskFactory(
        task_type="diffusion",
        atom_vocab=["H", "C"],
        condition_names=["energy"],
        hidden_size=8,
        num_layers=1,
        num_sublayers=1,
        diffusion_steps=2,
        diffusion_noise_schedule="polynomial_2",
        diffusion_loss_type="l2",
        normalize_factors=[1, 1, 1],
        extra_norm_values=[],
    )

    task = factory.build()

    assert isinstance(task, GeomMolecularGenerative)
    assert task.condition == ["energy"]
    assert task.model.in_node_nf == 3
    assert task.model.T == 2


def test_egt_factory_requires_context_for_adapter_module():
    from MolecularDiffusion.runmodes.train.tasks_egt import ModelTaskFactory

    with pytest.raises(ValueError, match="Must specify the contexts"):
        ModelTaskFactory(
            task_type="diffusion",
            train_set=[],
            atom_vocab=["H"],
            task_names=[],
            use_adapter_module=True,
        )
