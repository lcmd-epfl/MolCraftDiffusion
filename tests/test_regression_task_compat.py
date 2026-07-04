"""Compatibility tests for property-regression task behavior."""

from __future__ import annotations

import pytest
import torch

from tests.conftest import TinyBackbone, TinyRegressionDataset


def _make_regression_task(**kwargs):
    from MolecularDiffusion.modules.tasks.regression import ProperyPrediction

    task = ProperyPrediction(
        TinyBackbone(),
        task=("energy",),
        criterion="mse",
        metric=("mae", "rmse"),
        num_mlp_layer=1,
        mlp_hidden_dim=8,
        normalization=True,
        **kwargs,
    )
    task.device = torch.device("cpu")
    return task


def test_preprocess_uses_only_labeled_finite_targets(tiny_regression_dataset):
    task = _make_regression_task()

    task.preprocess(tiny_regression_dataset)

    assert torch.allclose(task.mean, torch.tensor([2.0]))
    assert torch.allclose(task.std, torch.tensor([2.0**0.5]), atol=1e-6)
    assert torch.equal(task.weight, torch.tensor([1.0]))
    assert task.num_class == [1]
    assert task.mlp is not None


def test_preprocess_rejects_missing_task_field():
    task = _make_regression_task()
    dataset = TinyRegressionDataset([{"energy": 1.0}], targets={"other": [1.0]})

    with pytest.raises(ValueError, match="Task energy not found"):
        task.preprocess(dataset)


def test_unsupported_prediction_mlp_type_raises():
    with pytest.raises(ValueError, match="Unsupported MLP type"):
        _make_regression_task(prediction_mlp_type="unknown")


def test_target_marks_unlabeled_rows_as_nan():
    task = _make_regression_task()
    batch = {
        "energy": torch.tensor([1.0, 9.0]),
        "labeled": torch.tensor([True, False]),
    }

    target = task.target(batch)

    assert target.shape == (2, 1)
    assert target[0, 0] == 1.0
    assert torch.isnan(target[1, 0])


def test_forward_returns_finite_loss_and_named_metrics(
    tiny_regression_dataset, tiny_pyg_graphs, pyg_batch_cls
):
    task = _make_regression_task()
    task.preprocess(tiny_regression_dataset)
    graph_batch = pyg_batch_cls.from_data_list(tiny_pyg_graphs)
    batch = {
        "graph": graph_batch,
        "energy": torch.tensor([1.0, 3.0]),
        "labeled": torch.tensor([True, True]),
    }

    loss, metrics = task(batch)

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["mean_squared_error"])


def test_evaluate_reports_per_task_metrics():
    task = _make_regression_task()
    pred = torch.tensor([[1.0], [3.0]])
    target = torch.tensor([[2.0], [3.0]])

    metrics = task.evaluate(pred, target)

    assert torch.allclose(metrics["mean_absolute_error [energy]"], torch.tensor(0.5))
    assert torch.allclose(
        metrics["root_mean_squared_error [energy]"], torch.tensor(2**-0.5)
    )


def test_mlp_regressor_readout_modes_are_shape_stable():
    from MolecularDiffusion.modules.tasks.regression import MLPRegressor_pernode

    latent = torch.ones(5, 4)
    batch_indices = torch.tensor([0, 0, 1, 1, 1])
    mean_head = MLPRegressor_pernode(
        4, 1, hidden_dim=4, num_layers=1, readout_method="mean"
    )
    sum_head = MLPRegressor_pernode(
        4, 1, hidden_dim=4, num_layers=1, readout_method="sum"
    )

    assert mean_head(latent, batch_indices).shape == (2, 1)
    assert sum_head(latent, batch_indices).shape == (2, 1)
