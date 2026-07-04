"""Deeper offline tests for generation execution control flow."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.generation


class FixedNodeDist:
    def __init__(self, sizes):
        self.sizes = list(sizes)
        self.calls = []

    def sample(self, n_samples):
        self.calls.append(n_samples)
        return torch.tensor(self.sizes[:n_samples], dtype=torch.long)


class FakeTargetFactory:
    def __init__(self):
        self.atom_vocab = None
        self.norm_factor = None
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self


class FakeSchedulerFactory:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self


class FakeGenerationTask:
    atom_vocab = ["H", "C"]
    prop_dist_model = None
    normalize_condition = None
    predictive_model = None
    device = torch.device("cpu")

    def __init__(self):
        self.model = SimpleNamespace(T=10, norm_values=[1, 1, 1])
        self.n_node_dist = {3: 1, 4: 1, 8: 1}
        self.node_dist_model = FixedNodeDist([8, 4, 3])
        self.calls = []

    def _payload(self, nodesxsample, n_frames=0):
        batch_size = int(len(nodesxsample))
        n_nodes = int(torch.as_tensor(nodesxsample).max().item())
        n_atom_types = len(self.atom_vocab)
        if n_frames:
            one_hot = torch.zeros(n_frames, batch_size, n_nodes, n_atom_types)
            one_hot[..., 0] = 1
            charges = torch.ones(n_frames, batch_size, n_nodes, 1)
            coords = torch.zeros(n_frames, batch_size, n_nodes, 3)
            node_mask = torch.ones(batch_size, n_nodes)
        else:
            one_hot = torch.zeros(batch_size, n_nodes, n_atom_types)
            one_hot[..., 0] = 1
            charges = torch.ones(batch_size, n_nodes, 1)
            coords = torch.zeros(batch_size, n_nodes, 3)
            node_mask = torch.ones(batch_size, n_nodes)
        return one_hot, charges, coords, node_mask

    def sample(self, nodesxsample, **kwargs):
        self.calls.append(("sample", nodesxsample.clone(), kwargs))
        return self._payload(nodesxsample, n_frames=kwargs.get("n_frames", 0))

    def sample_conditonal(self, nodesxsample, target_value, **kwargs):
        self.calls.append(
            ("sample_conditonal", nodesxsample.clone(), target_value, kwargs)
        )
        return self._payload(nodesxsample, n_frames=kwargs.get("n_frames", 0))

    def sample_guidance_conitional(self, target_value, nodesxsample, **kwargs):
        self.calls.append(
            (
                "sample_guidance_conitional",
                nodesxsample.clone(),
                target_value,
                kwargs,
            )
        )
        return self._payload(nodesxsample, n_frames=kwargs.get("n_frames", 0))

    def sample_guidance(self, target_function, nodesxsample, **kwargs):
        self.calls.append(("sample_guidance", nodesxsample.clone(), target_function, kwargs))
        return self._payload(nodesxsample)

    def sample_hybrid_guidance(self, target_function, target_value, nodesxsample, **kwargs):
        self.calls.append(
            (
                "sample_hybrid_guidance",
                nodesxsample.clone(),
                target_function,
                target_value,
                kwargs,
            )
        )
        return self._payload(nodesxsample, n_frames=kwargs.get("n_frames", 0))


@pytest.fixture
def fake_xyz_io(monkeypatch):
    import MolecularDiffusion.runmodes.generate.tasks_generate as gen_tasks

    writes = []
    moves = []

    def write_xyz(output_path, one_hot, coords, atom_decoder=None, **kwargs):
        del one_hot, coords, atom_decoder, kwargs
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "molecule_000.xyz"), "w", encoding="utf-8") as f:
            f.write("1\nfake\nH 0 0 0\n")
        writes.append(("one_hot", output_path))

    def write_xyz_atomic(output_path, coords, charges, **kwargs):
        del coords, charges, kwargs
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "molecule_000.xyz"), "w", encoding="utf-8") as f:
            f.write("1\nfake\nH 0 0 0\n")
        writes.append(("atomic", output_path))

    def move_xyz(self, src_path, mol_idx, trajectory_dir=None):
        dest = os.path.join(self.output_path, f"molecule_{mol_idx:04d}.xyz")
        os.makedirs(self.output_path, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(f"1\nfinal-{mol_idx}\nH {mol_idx}.0 0.0 0.0\n")
        if os.path.exists(src_path):
            os.unlink(src_path)
        moves.append((mol_idx, trajectory_dir))
        return dest

    monkeypatch.setattr(gen_tasks, "save_xyz_file", write_xyz)
    monkeypatch.setattr(gen_tasks, "save_xyz_file_atomic_numbers", write_xyz_atomic)
    monkeypatch.setattr(gen_tasks.GenerativeFactory, "_move_xyz", move_xyz)
    return SimpleNamespace(writes=writes, moves=moves)


def test_unconditional_generation_splits_batches_clamps_node_counts_and_writes_files(
    tmp_path, fake_xyz_io
):
    from MolecularDiffusion.runmodes.generate.tasks_generate import GenerativeFactory

    task = FakeGenerationTask()
    factory = GenerativeFactory(
        task=task,
        task_type="unconditional",
        num_generate=3,
        batch_size=2,
        mol_size=[0, 0],
        max_mol_size=5,
        output_path=os.fspath(tmp_path),
    )

    factory.unconditional_generation()

    sample_calls = [call for call in task.calls if call[0] == "sample"]
    assert [call[1].tolist() for call in sample_calls] == [[5, 4], [5]]
    assert task.node_dist_model.calls == [2, 1]
    assert [move[0] for move in fake_xyz_io.moves] == [0, 1, 2]
    final_files = sorted(tmp_path.glob("molecule_*.xyz"))
    assert [path.name for path in final_files] == [
        "molecule_0000.xyz",
        "molecule_0001.xyz",
        "molecule_0002.xyz",
    ]
    for idx, path in enumerate(final_files):
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "1"
        assert lines[1] == f"final-{idx}"
        assert lines[2] == f"H {idx}.0 0.0 0.0"


def test_conditional_generation_passes_target_values_and_cfg_parameters(
    tmp_path, fake_xyz_io
):
    from MolecularDiffusion.runmodes.generate.tasks_generate import GenerativeFactory

    task = FakeGenerationTask()
    factory = GenerativeFactory(
        task=task,
        task_type="cfg",
        num_generate=2,
        batch_size=2,
        mol_size=[4],
        target_values=[3.0],
        property_names=["energy"],
        negative_target_values=[-1.0],
        condition_configs={"cfg_scale": 2.5, "cfg_scale_schedule": "linear"},
        output_path=os.fspath(tmp_path),
    )

    factory.conditional_generation()

    method, nodesxsample, target_value, kwargs = task.calls[0]
    assert method == "sample_guidance_conitional"
    assert nodesxsample.tolist() == [4, 4]
    assert target_value == [3.0]
    assert kwargs["negative_target_value"] == [-1.0]
    assert kwargs["cfg_scale"] == 2.5
    assert kwargs["cfg_scale_schedule"] == "linear"
    assert kwargs["guidance_ver"] == "cfg"


def test_property_guidance_builds_target_and_scheduler_then_calls_sampler(
    tmp_path, fake_xyz_io
):
    from MolecularDiffusion.runmodes.generate.tasks_generate import GenerativeFactory

    task = FakeGenerationTask()
    target_factory = FakeTargetFactory()
    scheduler_factory = FakeSchedulerFactory()
    factory = GenerativeFactory(
        task=task,
        task_type="gg",
        num_generate=1,
        batch_size=1,
        mol_size=[3],
        condition_configs={
            "target_function": target_factory,
            "scheduler": scheduler_factory,
            "gg_scale": 0.25,
            "max_norm": 0.5,
            "guidance_ver": 2,
            "guidance_at": 0.8,
            "guidance_stop": 0.1,
            "n_backwards": 4,
        },
        output_path=os.fspath(tmp_path),
    )

    factory.property_guidance()

    method, nodesxsample, target_function, kwargs = task.calls[0]
    assert method == "sample_guidance"
    assert nodesxsample.tolist() == [3]
    assert target_function is target_factory
    assert target_factory.calls == 1
    assert target_factory.atom_vocab == task.atom_vocab
    assert target_factory.norm_factor == task.model.norm_values
    assert scheduler_factory.calls == 1
    assert kwargs["scale"] == 0.25
    assert kwargs["max_norm"] == 0.5
    assert kwargs["guidance_ver"] == 2
    assert kwargs["n_backwards"] == 4


def test_structural_guidance_clamps_to_reference_size_and_passes_outpaint_config(
    monkeypatch, tmp_path, fake_xyz_io
):
    from MolecularDiffusion.runmodes.generate.tasks_generate import GenerativeFactory

    task = FakeGenerationTask()
    factory = GenerativeFactory(
        task=task,
        task_type="outpaint",
        num_generate=1,
        batch_size=1,
        mol_size=[2],
        condition_configs={
            "n_retrys": 0,
            "condition_component": "xh",
            "outpaint_cfgs": {"t_start": 0.75, "connector_dicts": {0: [1]}},
        },
        output_path=os.fspath(tmp_path),
    )
    monkeypatch.setattr(
        factory,
        "preprocess_ref_structure",
        lambda device: torch.zeros(1, 4, 6, device=device),
    )

    factory.structural_guidance()

    method, nodesxsample, kwargs = task.calls[0]
    assert method == "sample"
    assert nodesxsample.tolist() == [4]
    assert kwargs["condition_mode"] == "outpaint_xh"
    assert kwargs["condition_tensor"].shape == (1, 4, 6)
    assert kwargs["outpaint_cfgs"] == {"t_start": 0.75, "connector_dicts": {0: [1]}}


def test_hybrid_cfg_filters_geometric_constraint_keys_before_sampling(
    monkeypatch, tmp_path, fake_xyz_io
):
    from MolecularDiffusion.runmodes.generate.tasks_generate import GenerativeFactory

    task = FakeGenerationTask()
    factory = GenerativeFactory(
        task=task,
        task_type="outpaint_cfg",
        num_generate=1,
        batch_size=1,
        mol_size=[5],
        target_values=[1.0],
        property_names=["energy"],
        condition_configs={
            "cfg_scale": 3.0,
            "condition_component": "xh",
            "outpaint_cfgs": {
                "t_start": 0.4,
                "connector_dicts": {0: [1]},
                "constraint_strength": 2.0,
                "scale_factor": 1.2,
            },
        },
        output_path=os.fspath(tmp_path),
    )
    monkeypatch.setattr(
        factory,
        "preprocess_ref_structure",
        lambda device: torch.zeros(1, 3, 6, device=device),
    )

    factory.hybrid_guidance()

    method, nodesxsample, target_function, target_value, kwargs = task.calls[0]
    assert method == "sample_hybrid_guidance"
    assert nodesxsample.tolist() == [5]
    assert target_function is None
    assert target_value == [1.0]
    assert kwargs["guidance_ver"] == "cfg"
    assert kwargs["cfg_scale"] == 3.0
    assert kwargs["condition_mode"] == "outpaint_xh"
    assert kwargs["outpaint_cfgs"] == {"t_start": 0.4}


def test_trajectory_generation_computes_frame_schedule_and_moves_each_frame_batch(
    tmp_path, fake_xyz_io
):
    from MolecularDiffusion.runmodes.generate.tasks_generate import GenerativeFactory

    task = FakeGenerationTask()
    task.model.T = 12
    factory = GenerativeFactory(
        task=task,
        task_type="unconditional",
        num_generate=2,
        batch_size=2,
        mol_size=[3],
        n_frames=4,
        condition_configs={"denoising_strength": 0.5},
        output_path=os.fspath(tmp_path),
    )

    factory.unconditional_generation()

    assert factory.visualize_trajectory is True
    assert factory.s_saves.tolist() == [6, 4, 2, 0]
    assert [move[0] for move in fake_xyz_io.moves] == [0, 1]
    assert all(move[1] is not None for move in fake_xyz_io.moves)
