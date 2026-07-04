"""Compatibility tests for generation and checkpoint helper behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf


def test_select_ckpt_file_prefers_highest_metric(tmp_path):
    from MolecularDiffusion.cli.generate import _select_ckpt_file

    low = tmp_path / "epoch=1-metric=0.20.ckpt"
    high = tmp_path / "epoch=2-metric=0.90.ckpt"
    last = tmp_path / "last.ckpt"
    low.touch()
    high.touch()
    last.touch()

    assert _select_ckpt_file(str(tmp_path)) == str(high)


def test_select_ckpt_file_falls_back_to_last_checkpoint(tmp_path):
    from MolecularDiffusion.cli.generate import _select_ckpt_file

    first = tmp_path / "plain.ckpt"
    last = tmp_path / "last.ckpt"
    first.touch()
    last.touch()

    assert _select_ckpt_file(str(tmp_path)) == str(last)


def test_checkpoint_metadata_is_read_from_multiple_formats():
    from MolecularDiffusion.cli.generate import _get_ckpt_meta

    assert _get_ckpt_meta(
        {"hyperparameters": {"task_type": "diffusion", "condition_names": ["gap"]}}
    ) == ("diffusion", ["gap"])
    assert _get_ckpt_meta(
        {"hyper_parameters": {"task_type": "regression"}, "condition_names": ["e"]}
    ) == ("regression", ["e"])
    assert _get_ckpt_meta({"task_type": "guidance"}) == ("guidance", None)


def test_task_type_validation_rejects_mismatched_checkpoint():
    from MolecularDiffusion.cli.generate import _validate_task_type

    with pytest.raises(ValueError, match="Task type mismatch"):
        _validate_task_type({"task_type": "diffusion"}, "regression")


def test_extract_clean_state_dict_prefers_new_format_ema():
    from MolecularDiffusion.cli.generate import _extract_clean_state_dict

    ckpt = {
        "ema_model_state_dict": {"task.weight": torch.tensor([2.0])},
        "state_dict": {"task.weight": torch.tensor([1.0])},
    }

    state = _extract_clean_state_dict(ckpt, prefer_ema=True)

    assert list(state) == ["weight"]
    assert torch.equal(state["weight"], torch.tensor([2.0]))


def test_extract_clean_state_dict_supports_old_embedded_ema_keys():
    from MolecularDiffusion.cli.generate import _extract_clean_state_dict

    ckpt = {
        "state_dict": {
            "task.weight": torch.tensor([1.0]),
            "task.ema_model.weight": torch.tensor([3.0]),
        }
    }

    state = _extract_clean_state_dict(ckpt, prefer_ema=True)

    assert list(state) == ["weight"]
    assert torch.equal(state["weight"], torch.tensor([3.0]))


def test_extract_clean_state_dict_raw_path_drops_embedded_ema_keys():
    from MolecularDiffusion.cli.generate import _extract_clean_state_dict

    ckpt = {
        "state_dict": {
            "task.weight": torch.tensor([1.0]),
            "task.ema_model.weight": torch.tensor([3.0]),
        }
    }

    state = _extract_clean_state_dict(ckpt, prefer_ema=False)

    assert list(state) == ["weight"]
    assert torch.equal(state["weight"], torch.tensor([1.0]))


def test_total_step_override_updates_supported_task_shapes(namespace_factory):
    from MolecularDiffusion.cli.generate import _apply_total_step_override

    diffusion_task = namespace_factory(model=namespace_factory(T=900))
    fm_task = namespace_factory(model=namespace_factory(fm_num_timesteps=300))
    root_task = namespace_factory(T=100)
    ldm_task = namespace_factory(
        interpolant=namespace_factory(num_timesteps=100),
        model=namespace_factory(),
    )

    _apply_total_step_override(diffusion_task, 12)
    _apply_total_step_override(fm_task, 13)
    _apply_total_step_override(root_task, 14)
    _apply_total_step_override(ldm_task, 15)

    assert diffusion_task.model.T == 12
    assert fm_task.model.fm_num_timesteps == 13
    assert root_task.T == 14
    assert ldm_task.interpolant.num_timesteps == 15


def test_condition_names_are_stamped_from_checkpoint(namespace_factory):
    from MolecularDiffusion.cli.generate import _stamp_condition_names

    task = namespace_factory(condition=[])

    _stamp_condition_names(task, {"condition_names": ["energy", "gap"]})

    assert task.condition == ["energy", "gap"]


def test_generate_config_guard_rejects_training_or_incomplete_configs():
    from MolecularDiffusion.cli.generate import _assert_generate_config

    with pytest.raises(ValueError, match="training config"):
        _assert_generate_config(OmegaConf.create({"trainer": {}, "interference": {}}))

    with pytest.raises(ValueError, match="missing required"):
        _assert_generate_config(OmegaConf.create({"chkpt_directory": "run"}))

    _assert_generate_config(
        OmegaConf.create({"chkpt_directory": "run", "interference": {}})
    )


def test_generative_factory_clamps_molecule_size_to_training_distribution(tmp_path):
    from MolecularDiffusion.runmodes.generate.tasks_generate import GenerativeFactory

    task = SimpleNamespace(
        n_node_dist={3: 1, 5: 1},
        node_dist_model=None,
        prop_dist_model=None,
        condition=[],
        model=SimpleNamespace(T=10),
    )

    factory = GenerativeFactory(task=task, mol_size=[2, 50], output_path=str(tmp_path))

    assert factory.max_atom == 5
    assert factory.mol_size == [2, 5]


def test_generative_factory_validates_condition_property_count(tmp_path):
    from MolecularDiffusion.runmodes.generate.tasks_generate import GenerativeFactory

    task = SimpleNamespace(
        n_node_dist={3: 1},
        node_dist_model=None,
        prop_dist_model=None,
        condition=["energy", "gap"],
        model=SimpleNamespace(T=10),
    )

    with pytest.raises(ValueError, match="Property count mismatch"):
        GenerativeFactory(
            task=task,
            target_values=[1.0],
            property_names=["energy"],
            output_path=str(tmp_path),
        )


def test_train_load_weights_rejects_checkpoint_task_type_mismatch(tmp_path):
    from torch import nn

    from MolecularDiffusion.cli.train import load_weights

    ckpt_path = tmp_path / "model.ckpt"
    torch.save({"hyperparameters": {"task_type": "diffusion"}, "model": {}}, ckpt_path)

    with pytest.raises(ValueError, match="Task type mismatch"):
        load_weights(
            nn.Linear(1, 1),
            str(ckpt_path),
            task_module=SimpleNamespace(task_type="regression"),
        )
