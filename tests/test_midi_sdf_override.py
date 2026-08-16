"""The MiDi bond sidecar must honour the *generate* config's sdf_output_path.

A checkpoint this platform trained stores the training-time task config in
``hyper_parameters.model_config``, and the task is rebuilt from that -- so
without this recovery the training-time ``sdf_output_path: null`` wins and no
``.sdf`` is written, silently.
"""

import sys

from MolecularDiffusion.modules.tasks.diffusion_midi import (
    _generate_config_override,
)


def _write(tmp_path, body):
    path = tmp_path / "generate.yaml"
    path.write_text(body)
    return str(path)


def test_reads_tasks_block_from_the_invoked_yaml(tmp_path, monkeypatch):
    cfg = _write(tmp_path, "tasks:\n  sdf_output_path: out/mols.sdf\n")
    monkeypatch.setattr(sys, "argv", ["MolCraftDiff", "generate", cfg])
    assert (
        _generate_config_override("sdf_output_path", default="stale.sdf")
        == "out/mols.sdf"
    )


def test_cli_override_wins(tmp_path, monkeypatch):
    cfg = _write(tmp_path, "tasks:\n  sdf_output_path: out/mols.sdf\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["MolCraftDiff", "generate", cfg, "tasks.sdf_output_path=cli.sdf"],
    )
    assert _generate_config_override("sdf_output_path") == "cli.sdf"


def test_explicit_null_disables(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["MolCraftDiff", "generate", "tasks.sdf_output_path=null"]
    )
    assert _generate_config_override("sdf_output_path", default="on.sdf") is None


def test_absent_key_keeps_the_task_config_value(tmp_path, monkeypatch):
    cfg = _write(tmp_path, "tasks:\n  n_layers: 2\n")
    monkeypatch.setattr(sys, "argv", ["MolCraftDiff", "generate", cfg])
    assert (
        _generate_config_override("sdf_output_path", default="kept.sdf")
        == "kept.sdf"
    )


def test_no_cli_config_keeps_the_task_config_value(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pytest"])
    assert (
        _generate_config_override("sdf_output_path", default="kept.sdf")
        == "kept.sdf"
    )
