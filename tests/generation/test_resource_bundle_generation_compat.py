"""Generation compatibility tests for published resource-bundle examples."""

from __future__ import annotations

import os
import pickle
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

pytestmark = pytest.mark.generation


RESOURCE_BUNDLE_GENERATION_CONFIGS = {
    "gen_uncond": """
defaults:
  - tasks: diffusion
  - interference: gen_unconditional
  - _self_
chkpt_directory: training_outputs/benchmark/EDM_geom/
atom_vocab: [H,B,C,N,O,F,Al,Si,P,S,Cl,As,Se,Br,I,Hg,Bi]
diffusion_steps: 900
interference:
  _target_: MolecularDiffusion.runmodes.generate.GenerativeFactory
  task_type: unconditional
  sampling_mode: ddpm
  num_generate: 100
  batch_size: 12
  max_mol_size: 100
  mol_size: [0,0]
  target_values: []
  property_names: []
  seed: 86
  output_path: ${chkpt_directory}/output
""",
    "gen_cfg": """
defaults:
  - tasks: diffusion
  - interference: gen_cfg
  - _self_
chkpt_directory: training_outputs/benchmark/EDM_tlgeom_formed_energyscore/
atom_vocab: [H,B,C,N,O,F,Al,Si,P,S,Cl,As,Se,Br,I,Hg,Bi]
diffusion_steps: 900
interference:
  _target_: MolecularDiffusion.runmodes.generate.GenerativeFactory
  task_type: cfg
  sampling_mode: ddpm
  num_generate: 100
  batch_size: 12
  max_mol_size: 100
  mol_size: [0,0]
  target_values: []
  condition_configs:
    cfg_scale: 1.0
  property_names: []
  seed: 86
  output_path: ${chkpt_directory}/output
""",
    "gen_gg": """
defaults:
  - tasks: diffusion
  - interference: gen_gg
  - _self_
chkpt_directory: training_outputs/benchmark/EDM_geom/
atom_vocab: [H,B,C,N,O,F,Al,Si,P,S,Cl,As,Se,Br,I,Hg,Bi]
diffusion_steps: 900
interference:
  _target_: MolecularDiffusion.runmodes.generate.GenerativeFactory
  task_type: gg
  sampling_mode: ddpm
  num_generate: 100
  batch_size: 12
  max_mol_size: 100
  mol_size: [0,0]
  target_values: []
  condition_configs:
    gg_scale: 1e-3
    max_norm: 0.1
    guidance_at: 1
    guidance_stop: 0
    n_backwards: 2
    target_function:
      _target_: scripts.gradient_guidance.sf_energy_score.SFEnergyScore
      _partial_: true
      chkpt_directory: training_outputs/egcl_guidance/last.ckpt
  property_names: []
  seed: 86
  output_path: ${chkpt_directory}/output
""",
    "gen_cfggg": """
defaults:
  - tasks: diffusion
  - interference: gen_cfggg
  - _self_
chkpt_directory: training_outputs/benchmark/EDM_tlgeom_formed_energyscore/
atom_vocab: [H,B,C,N,O,F,Al,Si,P,S,Cl,As,Se,Br,I,Hg,Bi]
diffusion_steps: 900
interference:
  _target_: MolecularDiffusion.runmodes.generate.GenerativeFactory
  task_type: cfggg
  sampling_mode: ddpm
  num_generate: 100
  batch_size: 12
  max_mol_size: 100
  mol_size: [0,0]
  target_values: []
  condition_configs:
    cfg_scale: 1.0
    gg_scale: 1e-3
    max_norm: 0.1
    guidance_at: 1
    guidance_stop: 0
    n_backwards: 2
    target_function:
      _target_: scripts.gradient_guidance.sf_energy_score.SFEnergyScore
      _partial_: true
      chkpt_directory: training_outputs/egcl_guidance/last.ckpt
  property_names: []
  seed: 86
  output_path: ${chkpt_directory}/output
""",
    "gen_ip_cp": """
defaults:
  - tasks: diffusion
  - interference: gen_inpaint
  - _self_
chkpt_directory: training_outputs/benchmark/EDM_geom/
atom_vocab: [H,B,C,N,O,F,Al,Si,P,S,Cl,As,Se,Br,I,Hg,Bi]
diffusion_steps: 900
interference:
  _target_: MolecularDiffusion.runmodes.generate.GenerativeFactory
  task_type: inpaint
  sampling_mode: ddpm
  num_generate: 100
  batch_size: 1
  max_mol_size: 100
  mol_size: [0,0]
  target_values: []
  property_names: []
  seed: 86
  output_path: ${chkpt_directory}/output_cp_inpaint
  condition_configs:
    reference_structure_path: docs/data/CpHHH.xyz
    inpaint_cfgs:
      denoising_strength: 0.5
      mask_node_index: [5, 30, 31]
      noise_initial_mask: true
""",
    "gen_op_cp": """
defaults:
  - tasks: diffusion
  - interference: gen_outpaint
  - _self_
chkpt_directory: training_outputs/benchmark/EDM_geom/
atom_vocab: [H,B,C,N,O,F,Al,Si,P,S,Cl,As,Se,Br,I,Hg,Bi]
diffusion_steps: 900
interference:
  _target_: MolecularDiffusion.runmodes.generate.GenerativeFactory
  task_type: outpaint
  sampling_mode: ddpm
  num_generate: 100
  batch_size: 12
  max_mol_size: 100
  mol_size: [0,0]
  target_values: []
  property_names: []
  seed: 86
  output_path: ${chkpt_directory}/output_cp_inpaint
  condition_configs:
    reference_structure_path: docs/data/Cp_op.xyz
    outpaint_cfgs:
      t_start: 0.75
      t_critical: 0.0
      connector_dicts:
        1: [2]
      spread: 1.25
      seed_dist: 1.25
      min_dist: 2.5
""",
    "gen_op_iflp": """
defaults:
  - tasks: diffusion
  - interference: gen_outpaint
  - _self_
chkpt_directory: training_outputs/benchmark/EDM_tlgeom_iflp_outpaint/
atom_vocab: [H,B,C,N,O,F,Al,Si,P,S,Cl,As,Se,Br,I,Hg,Bi]
diffusion_steps: 900
interference:
  _target_: MolecularDiffusion.runmodes.generate.GenerativeFactory
  task_type: outpaintft
  sampling_mode: ddpm
  num_generate: 100
  batch_size: 12
  max_mol_size: 100
  mol_size: [0,0]
  target_values: []
  property_names: []
  seed: 86
  output_path: ${chkpt_directory}/output_iflp_inpaint
  condition_configs:
    reference_structure_path: docs/data/INT2_0.xyz
    outpaint_cfgs:
      t_start: 0.9
      t_critical: 0.0
      connector_dicts:
        0: [4]
      spread: 1.25
      seed_dist: 1.25
      min_dist: 2.5
""",
    "gen_pharmacophore_uncondx1x3": """
defaults:
  - tasks: pharmacophore
  - interference: pharm_unconditional
  - _self_
name: pharm_unconditional
chkpt_directory: trained_models/shephard_models/x1x3_diffusion_gdb17_20240824
seed: 42
atom_vocab: null
diffusion_steps: 0
interference:
  compute_x1: true
  compute_x2: false
  compute_x3: true
  compute_x4: false
  num_generate: 1012
  batch_size: 4
  N_x1: [11, 60]
  N_x1_sampling: uniform
  N_x4: 0
  distributions_path: data/shepherd_data/atom_pharm_count.npz
  distributions_key: gdb
  num_steps: 100
  output_path: ${chkpt_directory}/gen_unconditional
""",
}

EXPECTED_GENERATION_MODES = {
    "gen_uncond": ("unconditional", "GenerativeFactory"),
    "gen_cfg": ("cfg", "GenerativeFactory"),
    "gen_gg": ("gg", "GenerativeFactory"),
    "gen_cfggg": ("cfggg", "GenerativeFactory"),
    "gen_ip_cp": ("inpaint", "GenerativeFactory"),
    "gen_op_cp": ("outpaint", "GenerativeFactory"),
    "gen_op_iflp": ("outpaintft", "GenerativeFactory"),
    "gen_pharmacophore_uncondx1x3": ("unconditional", "PharmacophoreConditionGenerator"),
}


@pytest.mark.parametrize("name", sorted(RESOURCE_BUNDLE_GENERATION_CONFIGS))
def test_resource_bundle_generation_examples_compose_with_bundled_defaults(
    tmp_path, name
):
    from MolecularDiffusion.cli._hydra import setup_hydra_config
    from MolecularDiffusion.cli.generate import _assert_generate_config

    cfg_path = tmp_path / f"{name}.yaml"
    cfg_path.write_text(RESOURCE_BUNDLE_GENERATION_CONFIGS[name], encoding="utf-8")

    cfg = setup_hydra_config(cfg_path.name, config_dir=os.fspath(tmp_path))
    expected_mode, expected_target_suffix = EXPECTED_GENERATION_MODES[name]

    _assert_generate_config(cfg)
    assert cfg.interference.task_type == expected_mode
    assert cfg.interference._target_.endswith(expected_target_suffix)
    assert cfg.interference.num_generate > 0
    assert cfg.interference.batch_size > 0
    assert cfg.chkpt_directory


@pytest.mark.parametrize(
    ("task_type", "expected_method"),
    [
        ("unconditional", "unconditional_generation"),
        ("conditional", "conditional_generation"),
        ("cfg", "conditional_generation"),
        ("gradient_guidance", "property_guidance"),
        ("gg", "property_guidance"),
        ("cfggg", "property_guidance"),
        ("inpaint", "structural_guidance"),
        ("outpaint", "structural_guidance"),
        ("outpaintft", "structural_guidance"),
        ("inpaint_cfg", "hybrid_guidance"),
        ("outpaint_cfggg", "hybrid_guidance"),
    ],
)
def test_generative_factory_routes_published_modes_and_aliases(
    monkeypatch, tmp_path, task_type, expected_method
):
    from MolecularDiffusion.runmodes.generate.tasks_generate import GenerativeFactory

    task = SimpleNamespace(
        n_node_dist={3: 1},
        node_dist_model=None,
        prop_dist_model=None,
        condition=[],
        model=SimpleNamespace(T=10),
    )
    factory = GenerativeFactory(
        task=task,
        task_type=task_type,
        mol_size=[3],
        output_path=os.fspath(tmp_path),
    )
    called = []

    for method_name in (
        "unconditional_generation",
        "conditional_generation",
        "property_guidance",
        "structural_guidance",
        "hybrid_guidance",
    ):
        monkeypatch.setattr(
            factory,
            method_name,
            lambda method_name=method_name: called.append(method_name),
        )

    factory.run()

    assert called == [expected_method]


def test_load_model_supports_original_engine_edm_resource_bundle_layout(
    monkeypatch, tmp_path
):
    import MolecularDiffusion.cli.generate as generate_cli

    class FakeTask:
        def __init__(self):
            self.node_dist_model = None
            self.prop_dist_model = None
            self.condition = []

        def eval(self):
            self.evaluated = True
            return self

    fake_task = FakeTask()

    class FakeEngine:
        def __init__(self, *args, **kwargs):
            pass

        def load_from_checkpoint(self, checkpoint_path, interference_mode=False):
            assert checkpoint_path == os.fspath(tmp_path / "edm_chem.pkl")
            assert interference_mode is True
            return SimpleNamespace(model=fake_task)

    monkeypatch.setattr(generate_cli, "Engine", FakeEngine)
    torch.save(
        {
            "task_type": "diffusion",
            "condition_names": ["energy_score"],
            "reference_indices": [0, 1],
        },
        tmp_path / "edm_chem.pkl",
    )
    with open(tmp_path / "edm_stat.pkl", "wb") as f:
        pickle.dump(
            {
                "node": "node-dist",
                "prop": "prop-dist",
                "reference_freeze_mode": "features_only",
                "reference_feature_stats": {"node_feature": torch.ones(1, 1, 1)},
            },
            f,
        )

    task = generate_cli.load_model(
        os.fspath(tmp_path),
        task_config=OmegaConf.create({"task_type": "diffusion"}),
    )

    assert task is fake_task
    assert task.evaluated is True
    assert task.condition == ["energy_score"]
    assert task.reference_indices == [0, 1]
    assert task.node_dist_model == "node-dist"
    assert task.prop_dist_model == "prop-dist"
    assert task.reference_freeze_mode == "features_only"


def test_load_model_supports_generic_pkl_resource_bundle_layout(monkeypatch, tmp_path):
    import MolecularDiffusion.cli.generate as generate_cli

    class FakeTask:
        def eval(self):
            self.evaluated = True
            return self

    fake_task = FakeTask()

    class FakeEngine:
        def __init__(self, *args, **kwargs):
            pass

        def load_from_checkpoint(self, checkpoint_path, interference_mode=False):
            assert checkpoint_path == os.fspath(tmp_path / "model.pkl")
            assert interference_mode is True
            return SimpleNamespace(model=fake_task)

    monkeypatch.setattr(generate_cli, "Engine", FakeEngine)
    torch.save({"task_type": "diffusion_tabasco"}, tmp_path / "model.pkl")

    task = generate_cli.load_model(
        os.fspath(tmp_path),
        task_config=OmegaConf.create({"task_type": "diffusion_tabasco"}),
    )

    assert task is fake_task
    assert task.evaluated is True
