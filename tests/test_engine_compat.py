"""Compatibility tests for original and Lightning engine wrappers."""

from __future__ import annotations

import torch

from tests.conftest import TinyGenerationTask, TinyTask


def test_engine_initializes_on_cpu_and_uses_default_graph_collate(monkeypatch):
    from MolecularDiffusion.core.engine import Engine
    from MolecularDiffusion.data.dataloader import graph_collate
    from MolecularDiffusion.utils import comm

    monkeypatch.delenv("SLURM_PROCID", raising=False)
    monkeypatch.delenv("SLURM_GPUS_ON_NODE", raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(comm, "get_world_size", lambda: 1)
    monkeypatch.setattr(comm, "get_rank", lambda: 0)

    task = TinyTask()
    engine = Engine(
        task=task,
        train_set=["train"],
        valid_set=["valid"],
        test_set=["test"],
        optimizer=None,
        logger="logging",
        pin_memory=False,
    )

    assert engine.device.type == "cpu"
    assert engine.collate_fn is graph_collate
    assert engine.train_set == ["train"]
    assert engine.valid_set == ["valid-replaced"]
    assert engine.test_set == ["test-replaced"]
    assert task.preprocess_calls == 1
    assert task.device == torch.device("cpu")


def test_engine_inference_mode_forces_logging_logger(monkeypatch):
    from MolecularDiffusion.core import LoggingLogger
    from MolecularDiffusion.core.engine import Engine
    from MolecularDiffusion.utils import comm

    monkeypatch.delenv("SLURM_PROCID", raising=False)
    monkeypatch.delenv("SLURM_GPUS_ON_NODE", raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(comm, "get_world_size", lambda: 1)
    monkeypatch.setattr(comm, "get_rank", lambda: 0)

    engine = Engine(
        task=None,
        train_set=None,
        valid_set=None,
        test_set=None,
        optimizer=None,
        logger="wandb",
        pin_memory=False,
    )

    assert isinstance(engine.logger, LoggingLogger)
    assert engine.model is None


def test_engine_lightning_saves_generation_metadata():
    from MolecularDiffusion.core.engine_lightning import EngineLightning

    task = TinyGenerationTask()
    task.node_dist_model = object()
    task.prop_dist_model = object()
    wrapper = EngineLightning(
        optimizer_config={"optimizer_choice": "adam", "lr": 1e-3},
        task=task,
    )
    checkpoint = {}

    wrapper.on_save_checkpoint(checkpoint)

    assert checkpoint["task_type"] == "diffusion"
    assert checkpoint["condition_names"] == ["energy"]
    assert checkpoint["node_dist_model"] is task.node_dist_model
    assert checkpoint["prop_dist_model"] is task.prop_dist_model
    assert checkpoint["n_node_dist"] == {3: 2, 5: 1}
    assert checkpoint["reference_indices"] == [0, 2]
    assert checkpoint["reference_freeze_mode"] == "features_only"
    assert torch.equal(checkpoint["reference_scaffold"], task.reference_scaffold)


def test_engine_lightning_restores_checkpoint_metadata_when_fresh_state_absent():
    from MolecularDiffusion.core.engine_lightning import EngineLightning

    task = TinyGenerationTask()
    task.node_dist_model = None
    task.prop_dist_model = None
    task.n_node_dist = {}
    task.reference_indices = None
    task.reference_feature_stats = None
    task.reference_scaffold = None
    wrapper = EngineLightning(
        optimizer_config={"optimizer_choice": "adam", "lr": 1e-3},
        task=task,
    )
    checkpoint = {
        "node_dist_model": "node-dist",
        "n_node_dist": {7: 4},
        "prop_dist_model": "prop-dist",
        "reference_indices": [1],
        "reference_freeze_mode": "all",
        "reference_feature_stats": {"node_feature": torch.ones(1, 1, 1)},
        "reference_scaffold": torch.ones(1, 1, 4),
    }

    wrapper.on_load_checkpoint(checkpoint)

    assert task.node_dist_model == "node-dist"
    assert task.n_node_dist == {7: 4}
    assert task.prop_dist_model == "prop-dist"
    assert task.reference_indices == [1]
    assert task.reference_freeze_mode == "all"
    assert torch.equal(task.reference_scaffold, torch.ones(1, 1, 4))


def test_engine_lightning_keeps_fresh_distribution_models_on_load():
    from MolecularDiffusion.core.engine_lightning import EngineLightning

    task = TinyGenerationTask()
    fresh_node = object()
    fresh_prop = object()
    task.node_dist_model = fresh_node
    task.prop_dist_model = fresh_prop
    wrapper = EngineLightning(
        optimizer_config={"optimizer_choice": "adam", "lr": 1e-3},
        task=task,
    )

    wrapper.on_load_checkpoint(
        {"node_dist_model": "old-node", "n_node_dist": {99: 1}, "prop_dist_model": "old-prop"}
    )

    assert task.node_dist_model is fresh_node
    assert task.prop_dist_model is fresh_prop
    assert task.n_node_dist == {3: 2, 5: 1}


def test_engine_lightning_extracts_old_ema_state_and_removes_legacy_keys():
    from MolecularDiffusion.core.engine_lightning import EngineLightning

    wrapper = EngineLightning(
        optimizer_config={"optimizer_choice": "adam", "lr": 1e-3},
        task=TinyGenerationTask(),
        ema_decay=0.99,
    )
    checkpoint = {
        "state_dict": {
            "task.weight": torch.tensor(1.0),
            "ema_model.weight": torch.tensor(2.0),
        }
    }

    wrapper.on_load_checkpoint(checkpoint)

    assert torch.equal(wrapper._pending_ema_state["weight"], torch.tensor(2.0))
    assert "ema_model.weight" not in checkpoint["state_dict"]


def test_saved_hyperparameters_exclude_datasets(monkeypatch):
    """Checkpoints must carry weights + inference config, never the dataset."""
    from MolecularDiffusion.core.engine import Engine
    from MolecularDiffusion.utils import comm

    monkeypatch.delenv("SLURM_PROCID", raising=False)
    monkeypatch.delenv("SLURM_GPUS_ON_NODE", raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(comm, "get_world_size", lambda: 1)
    monkeypatch.setattr(comm, "get_rank", lambda: 0)

    engine = Engine(
        task=TinyTask(),
        train_set=["train"],
        valid_set=["valid"],
        test_set=["test"],
        optimizer=None,
        logger="logging",
        pin_memory=False,
        batch_size=8,
    )

    cfg = engine.sanitized_config_dict()

    for key in ("train_set", "valid_set", "test_set", "optimizer", "scheduler", "collate_fn"):
        assert key not in cfg, f"{key} must not be pickled into the checkpoint"
    assert cfg["batch_size"] == 8
    assert "task" in cfg
