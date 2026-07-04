"""Model-specific compatibility tests for regression tasks."""

from __future__ import annotations

import pytest
import torch

from tests.conftest import TinyBackbone, TinyRegressionDataset

pytestmark = pytest.mark.regression


def test_regression_task_standardizes_option_members():
    from MolecularDiffusion.modules.tasks.regression import ProperyPrediction

    task = ProperyPrediction(
        TinyBackbone(),
        task="energy",
        criterion="mse",
        metric=["mae", "rmse"],
        num_mlp_layer=1,
        mlp_hidden_dim=8,
    )

    assert task.task == {"energy": 1}
    assert task.criterion == {"mse": 1}
    assert task.metric == {"mae": 1, "rmse": 1}


def test_regression_preprocess_preserves_task_weights_for_multiple_targets():
    from MolecularDiffusion.modules.tasks.regression import ProperyPrediction

    task = ProperyPrediction(
        TinyBackbone(),
        task={"energy": 2.0, "gap": 0.5},
        criterion="mse",
        metric="mae",
        num_mlp_layer=1,
        mlp_hidden_dim=8,
    )
    train_set = TinyRegressionDataset(
        [
            {"energy": 1.0, "gap": 10.0, "labeled": True},
            {"energy": 3.0, "gap": 14.0, "labeled": True},
            {"energy": 99.0, "gap": 99.0, "labeled": False},
        ],
        targets={"energy": [1.0, 3.0], "gap": [10.0, 14.0]},
    )

    task.preprocess(train_set)

    assert torch.allclose(task.mean, torch.tensor([2.0, 12.0]))
    assert torch.allclose(task.weight, torch.tensor([2.0, 0.5]))
    assert task.num_class == [1, 1]


def test_regression_forward_with_unlabeled_batch_returns_zero_training_loss(
    tiny_regression_dataset, tiny_pyg_graphs, pyg_batch_cls
):
    from MolecularDiffusion.modules.tasks.regression import ProperyPrediction

    task = ProperyPrediction(
        TinyBackbone(),
        task="energy",
        criterion="mse",
        metric="mae",
        num_mlp_layer=1,
        mlp_hidden_dim=8,
    )
    task.device = torch.device("cpu")
    task.preprocess(tiny_regression_dataset)
    batch = {"graph": pyg_batch_cls.from_data_list(tiny_pyg_graphs)}

    loss, metrics = task(batch)

    assert torch.equal(loss, torch.tensor(0.0))
    assert metrics == {}


def test_regression_readout_rejects_unknown_method():
    from MolecularDiffusion.modules.tasks.regression import ProperyPrediction

    task = ProperyPrediction(TinyBackbone(), task="energy", readout="median")

    with pytest.raises(ValueError, match="Unsupported method"):
        task.readout_f(torch.zeros(2, 3, 4))
