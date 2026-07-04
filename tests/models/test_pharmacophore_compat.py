"""Model-specific compatibility tests for pharmacophore diffusion tasks."""

from __future__ import annotations

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.pharmacophore


class DummyPharmacophoreModel(nn.Module):
    def forward(self, batch):  # noqa: ARG002
        input_dict = {}
        for modality in ("x1", "x2", "x3", "x4"):
            input_dict[modality] = {
                "decoder": {
                    "pos": torch.zeros(2, 3),
                    "pos_noise": torch.zeros(2, 3),
                    "x_noise": torch.zeros(2),
                    "direction_noise": torch.zeros(2, 3),
                    "virtual_node_mask": torch.tensor([False, True]),
                }
            }
        return input_dict, {}


def test_pharmacophore_task_uses_supplied_model_and_default_weights():
    from MolecularDiffusion.modules.tasks.pharmacophore import PharmacophoreGenerative

    model = DummyPharmacophoreModel()
    task = PharmacophoreGenerative(model=model)

    assert task.model is model
    assert task.task is task
    assert task.task_type == "diffusion_pharmacophore"
    assert task.modality_weights == {"x1": 1.0, "x2": 1.0, "x3": 1.0, "x4": 1.0}


def test_pharmacophore_forward_mock_mode_returns_weighted_loss_metrics():
    from MolecularDiffusion.modules.tasks.pharmacophore import PharmacophoreGenerative

    task = PharmacophoreGenerative(
        model=DummyPharmacophoreModel(),
        modality_weights={"x1": 2.0, "x2": 1.0, "x3": 0.5, "x4": 0.0},
    )

    loss, metrics = task(batch={})

    assert torch.allclose(loss, torch.tensor(0.35))
    assert set(metrics) == {"loss_x1", "loss_x2", "loss_x3", "loss_x4", "loss_total"}
    assert torch.allclose(metrics["loss_total"], torch.tensor(0.35))


def test_pharmacophore_preprocess_rejects_empty_training_set():
    from MolecularDiffusion.modules.tasks.pharmacophore import PharmacophoreGenerative

    task = PharmacophoreGenerative(model=DummyPharmacophoreModel())

    with pytest.raises(ValueError, match="Training set is empty"):
        task.preprocess([])


def test_pharmacophore_x4_loss_is_zero_when_all_nodes_are_virtual():
    from MolecularDiffusion.modules.tasks.pharmacophore import PharmacophoreGenerative

    task = PharmacophoreGenerative(model=DummyPharmacophoreModel())
    input_dict = {
        "x4": {
            "decoder": {
                "pos": torch.zeros(2, 3),
                "pos_noise": torch.zeros(2, 3),
                "x_noise": torch.zeros(2),
                "direction_noise": torch.zeros(2, 3),
                "virtual_node_mask": torch.tensor([True, True]),
            }
        }
    }
    output_dict = {
        "x4": {
            "decoder": {
                "denoiser": {
                    "pos_out": torch.ones(2, 3),
                    "x_out": torch.ones(2),
                    "direction_out": torch.ones(2, 3),
                }
            }
        }
    }

    loss = task._compute_x4_loss(input_dict, output_dict)

    assert torch.equal(loss.detach(), torch.tensor(0.0))
    assert loss.requires_grad


def test_pharmacophore_predict_target_evaluate_are_empty_generative_contracts():
    from MolecularDiffusion.modules.tasks.pharmacophore import PharmacophoreGenerative

    task = PharmacophoreGenerative(model=DummyPharmacophoreModel())

    assert task.predict({}).numel() == 0
    assert task.target({}).numel() == 0
    assert task.evaluate(torch.tensor([]), torch.tensor([])) == {}
