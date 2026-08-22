"""The `.sdf` bond sidecar must survive the checkpoint->generate config hop.

`cli/generate.py` rebuilds a task from the checkpoint's TRAINING-time config,
where `sdf_output_path` is `null`. Only keys a factory declares in
`generation_time_keys` are taken from the generate config. If a bond model
stops declaring it, generation silently emits `.xyz` only and the whole 2D
half of the model is lost with no error -- which is exactly what happened
when a bundled example config was run from outside the repo.
"""

import pytest
from omegaconf import OmegaConf

from MolecularDiffusion.cli.generate import _caller_overrides

BOND_FACTORIES = [
    "MolecularDiffusion.modules.tasks.diffusion_midi.ModelTaskFactory",
    "MolecularDiffusion.modules.tasks.diffusion_jodo.ModelTaskFactory",
    "MolecularDiffusion.modules.tasks.diffusion_flowmol_graph3d."
    "FlowMolGraph3DTaskFactory",
]


@pytest.mark.parametrize("target", BOND_FACTORIES)
def test_generate_config_sdf_path_beats_checkpoint(target):
    pytest.importorskip("torch")
    ckpt_config = OmegaConf.create({"_target_": target, "sdf_output_path": None})
    generate_config = OmegaConf.create(
        {"_target_": target, "sdf_output_path": "out/mols.sdf"}
    )
    overrides = _caller_overrides(ckpt_config, generate_config)
    assert overrides.get("sdf_output_path") == "out/mols.sdf", (
        f"{target} no longer routes sdf_output_path through "
        "`generation_time_keys`; the bond sidecar will be silently dropped."
    )


@pytest.mark.parametrize("target", BOND_FACTORIES)
def test_sdf_path_is_a_plain_writable_attribute(target):
    """No argv scanning: the value must be whatever the factory was given."""
    pytest.importorskip("torch")
    import hydra.utils

    factory_cls = hydra.utils.get_class(target)
    assert "sdf_output_path" in getattr(factory_cls, "generation_time_keys", ())
    assert not isinstance(
        getattr(factory_cls, "sdf_output_path", None), property
    )


@pytest.mark.parametrize("target", BOND_FACTORIES)
def test_null_in_generate_config_leaves_the_checkpoint_value(target):
    """`null` is not an override -- the platform-wide rule for every key.

    Harmless for the sidecar: a checkpoint's training-time value is `null`
    too, so `tasks.sdf_output_path=null` still means "no sidecar".
    """
    pytest.importorskip("torch")
    ckpt_config = OmegaConf.create({"_target_": target, "sdf_output_path": None})
    generate_config = OmegaConf.create(
        {"_target_": target, "sdf_output_path": None}
    )
    assert "sdf_output_path" not in _caller_overrides(ckpt_config, generate_config)
