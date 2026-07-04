"""Compatibility tests for public config, registry, and import surfaces."""

from __future__ import annotations

import builtins

import pytest


def test_bundled_config_path_contains_core_groups():
    from MolecularDiffusion.cli._hydra import get_package_config_path

    config_path = get_package_config_path()

    assert (config_path / "tasks" / "diffusion.yaml").is_file()
    assert (config_path / "tasks" / "regression.yaml").is_file()
    assert (config_path / "data" / "mol_dataset.yaml").is_file()
    assert (config_path / "interference" / "gen_unconditional.yaml").is_file()


def test_hydra_composes_train_and_generation_configs():
    from MolecularDiffusion.cli._hydra import setup_hydra_config

    train_cfg = setup_hydra_config(
        "train.yaml", config_dir="configs", overrides=["seed=123"]
    )
    gen_cfg = setup_hydra_config(
        "generate.yaml",
        config_dir="configs",
        overrides=["interference.num_generate=2"],
    )

    assert train_cfg.seed == 123
    assert train_cfg.tasks.task_type == "diffusion"
    assert train_cfg.data._target_ == "MolecularDiffusion.runmodes.train.DataModule"
    assert gen_cfg.interference.num_generate == 2
    assert gen_cfg.interference.task_type == "unconditional"
    assert "trainer" not in gen_cfg


def test_registry_resolves_legacy_public_task_names():
    from MolecularDiffusion import core
    from MolecularDiffusion.modules.tasks import GeomMolecularGenerative, ProperyPrediction

    assert core.Registry.search("GeomMolecularGenerative") is GeomMolecularGenerative
    assert core.Registry.search("ProperyPrediction") is ProperyPrediction


def test_public_package_lazy_imports_are_cached():
    import MolecularDiffusion

    utils_first = MolecularDiffusion.utils
    utils_second = MolecularDiffusion.utils

    assert utils_first is utils_second
    with pytest.raises(AttributeError):
        _ = MolecularDiffusion.not_a_real_submodule


def test_xyzrender_optional_boundary_degrades_without_dependency(monkeypatch):
    import MolecularDiffusion.runmodes.generate.tasks_generate as generate_tasks

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "xyzrender":
            raise ImportError("xyzrender intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(generate_tasks, "_XYZRENDER_API", None)
    monkeypatch.setattr(generate_tasks, "_XYZRENDER_UNAVAILABLE", False)
    monkeypatch.setattr(generate_tasks, "_XYZRENDER_UNAVAILABLE_WARNED", False)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert generate_tasks._get_xyzrender_api() is None
    assert generate_tasks._XYZRENDER_UNAVAILABLE is True


def test_xyzrender_sanitizer_replaces_unknown_elements_for_render_only():
    from MolecularDiffusion.runmodes.generate.tasks_generate import (
        _sanitize_xyz_text_for_render,
    )

    xyz = "2\ncomment\nXx 0 0 0\nC 1 0 0\n"

    assert _sanitize_xyz_text_for_render(xyz) == "2\ncomment\nC 0 0 0\nC 1 0 0\n"
