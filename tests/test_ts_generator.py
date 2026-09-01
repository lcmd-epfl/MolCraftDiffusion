"""The shared transition-state seam, and what it must keep doing.

Two things are pinned here:

* **The seam is general.** A connectivity-only reaction -- no reactant or
  product coordinates at all, which is what GoFlow and RitS carry -- drives
  the real ``PocketGenerator.run`` loop end to end through
  :class:`TSGenerator` and writes no ``reactant.xyz`` it cannot back up.
  That is the claim the seam makes; if it breaks, the next TS integration
  quietly writes its own generator again.
* **OA-ReactDiff's Hydra key surface is unchanged.** Moving the plumbing into
  a base class must not rename, reorder or re-default a single interference
  key -- three shipped configs and a zoo entry set them by name.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import torch

# `component.dataset -> core -> runmodes.train.data -> data.dataset` is a
# pre-existing circular import; touching `data.dataset` first breaks it.
import MolecularDiffusion.data.dataset  # noqa: F401
from MolecularDiffusion.data.component.reaction_data import (
    Reaction,
    ReactionSide,
)
from MolecularDiffusion.modules.tasks.diffusion_oareactdiff import (
    OAReactDiffTSGenerator,
)
from MolecularDiffusion.modules.tasks.pocket_generator import PocketGenerator
from MolecularDiffusion.modules.tasks.ts_generator import (
    _TASK_TO_TS_GENERATOR,
    TSGenerator,
)

N_ATOMS = 3
Z = torch.tensor([6, 1, 8])


class _StubTask:
    """The narrowest thing ``PocketGenerator.run`` will drive."""

    atom_vocab = ["H", "C", "N", "O", "F"]

    def __init__(self) -> None:
        self.seen: List[Dict[str, Any]] = []

    def to(self, device: Any) -> "_StubTask":
        return self

    def eval(self) -> "_StubTask":
        return self

    def sample(
        self,
        batch: Any = None,
        nodesxsample: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        **kwargs: Any,
    ):
        self.seen.append(
            {
                "batch": batch,
                "sizes": nodesxsample,
                "steps": num_steps,
                **kwargs,
            }
        )
        n = int(nodesxsample.numel())
        natoms = int(nodesxsample[0])
        one_hot = torch.zeros(n, natoms, 5)
        one_hot[..., 1] = 1.0
        # Non-zero: save_xyz_file drops atoms sitting on the origin.
        coords = torch.arange(1.0, n * natoms * 3 + 1).view(n, natoms, 3)
        return one_hot, torch.zeros(n, natoms), coords, torch.ones(n, natoms)


def _reaction(*, geometry: bool, ts: bool = True) -> Reaction:
    """One reaction, carried either as 3D endpoints or as bonds only."""
    if geometry:
        sides = (
            ReactionSide(pos=torch.zeros(N_ATOMS, 3) + 1.0),
            ReactionSide(pos=torch.zeros(N_ATOMS, 3) + 2.0),
        )
    else:
        bonds = torch.tensor([[0, 1], [1, 0]])
        sides = (
            ReactionSide(bond_index=bonds, bond_type=torch.tensor([1, 1])),
            ReactionSide(bond_index=bonds, bond_type=torch.tensor([2, 2])),
        )
    return Reaction(
        z=Z,
        reactant=sides[0],
        product=sides[1],
        ts_pos=torch.zeros(N_ATOMS, 3) + 3.0 if ts else None,
        meta={"rxn": "rxn0001"},
    )


class _FakeTSGenerator(TSGenerator):
    """What a future TS integration has to write. All of it."""

    tag = "fake"

    def __init__(self, task: Any, reaction: Reaction, **kwargs: Any) -> None:
        super().__init__(task, reaction_source="memory", **kwargs)
        self._source = [reaction]

    def _reactions(self) -> Any:
        return self._source

    def _collate(self, reaction: Reaction, n: int) -> Dict[str, Any]:
        return {"z": reaction.z.repeat(n)}


def _run(tmp_path: Path, reaction: Reaction, **kwargs: Any) -> _StubTask:
    task = _StubTask()
    gen = _FakeTSGenerator(
        task,
        reaction,
        num_generate=2,
        batch_size=2,
        num_steps=7,
        output_path=str(tmp_path),
        **kwargs,
    )
    gen.run()
    return task


def test_connectivity_only_reaction_drives_the_shared_loop(tmp_path) -> None:
    """No endpoint geometry anywhere -- the loop must still complete."""
    task = _run(tmp_path, _reaction(geometry=False))

    written = sorted(p.name for p in tmp_path.glob("*.xyz"))
    assert written == [
        "molecule_000.xyz",
        "molecule_001.xyz",
        "reference_ts.xyz",
    ]
    # The tiled context reached sample(), and the size came off the reaction.
    assert torch.equal(task.seen[0]["batch"]["z"], Z.repeat(2))
    assert torch.equal(task.seen[0]["sizes"], torch.tensor([N_ATOMS] * 2))
    assert task.seen[0]["steps"] == 7


def test_geometry_reaction_writes_all_three_references(tmp_path) -> None:
    _run(tmp_path, _reaction(geometry=True))

    assert (tmp_path / "reactant.xyz").exists()
    assert (tmp_path / "product.xyz").exists()
    lines = (tmp_path / "reactant.xyz").read_text().splitlines()
    assert lines[0] == str(N_ATOMS)
    # Symbols come from the reaction's own atomic numbers, not a task vocab.
    assert [ln.split()[0] for ln in lines[2:]] == ["C", "H", "O"]


def test_no_reference_ts_when_the_corpus_has_none(tmp_path) -> None:
    """An inference-time reaction has no answer to write; not an error."""
    _run(tmp_path, _reaction(geometry=False, ts=False))

    assert not (tmp_path / "reference_ts.xyz").exists()
    assert (tmp_path / "molecule_000.xyz").exists()


def test_reaction_index_out_of_range_is_explicit(tmp_path) -> None:
    gen = _FakeTSGenerator(
        _StubTask(),
        _reaction(geometry=False),
        reaction_index=5,
        output_path=str(tmp_path),
    )
    with pytest.raises(IndexError, match="reaction_index 5 is out of range"):
        gen.run()


def test_unknown_interference_key_is_still_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="does not accept interference key"):
        _FakeTSGenerator(
            _StubTask(),
            _reaction(geometry=False),
            output_path=str(tmp_path),
            guide_mdoe="typo",
        )


def test_mol_size_is_refused_rather_than_ignored(tmp_path) -> None:
    """The base declares it; a TS run would swallow it and do nothing."""
    with pytest.raises(ValueError, match="mol_size does not apply"):
        _FakeTSGenerator(
            _StubTask(),
            _reaction(geometry=False),
            output_path=str(tmp_path),
            mol_size=[7, 9],
        )


def test_oareactdiff_rides_the_seam() -> None:
    assert issubclass(OAReactDiffTSGenerator, TSGenerator)
    assert issubclass(TSGenerator, PocketGenerator)
    # The loop is inherited, not re-implemented -- in both classes.
    assert "run" not in vars(TSGenerator)
    assert "run" not in vars(OAReactDiffTSGenerator)


def test_oareactdiff_hydra_key_surface_unchanged() -> None:
    """Three shipped configs and a zoo entry set these by name."""
    params = inspect.signature(OAReactDiffTSGenerator.__init__).parameters
    # reaction_pkl is a `source_key`, not a parameter -- see the aliasing
    # tests below. It is still a valid interference key.
    assert OAReactDiffTSGenerator.source_key == "reaction_pkl"
    assert [p for p in params if p not in ("self", "kwargs")] == [
        "task",
        "reaction_index",
        "num_generate",
        "batch_size",
        "num_steps",
        "resamplings",
        "jump_length",
        "noise_schedule",
        "output_path",
        "seed",
        "device",
    ]
    defaults = {
        k: v.default
        for k, v in params.items()
        if v.default is not inspect.Parameter.empty
    }
    assert defaults == {
        "reaction_index": 0,
        "num_generate": 20,
        "batch_size": 4,
        "num_steps": 250,
        "resamplings": 5,
        "jump_length": 5,
        "noise_schedule": "polynomial_2",
        "output_path": "generated_oareactdiff",
        "seed": 42,
        "device": None,
    }


class OAReactDiffTask:  # noqa: N801 - the registry keys on this name
    """Stands in for the real task: dispatch keys on the class NAME only."""

    atom_vocab = ["H", "C", "N", "O", "F"]


def _oareactdiff(**kwargs: Any) -> Any:
    """Build through the SHARED config's path: TSGenerator + a task."""
    return TSGenerator(OAReactDiffTask(), **kwargs)


def test_shared_config_dispatches_on_the_task() -> None:
    """`gen_ts.yaml` names TSGenerator; the subclass comes off the task."""
    gen = _oareactdiff(reaction_source="corpus.pkl")
    assert type(gen) is OAReactDiffTSGenerator
    assert gen.pocket_db == "corpus.pkl"
    # and the subclass's own __init__ ran, with its own defaults
    assert gen.num_steps == 250
    assert gen.resamplings == 5


def test_dispatch_registry_lists_every_ts_generator() -> None:
    assert _TASK_TO_TS_GENERATOR == {
        "OAReactDiffTask": ("diffusion_oareactdiff", "OAReactDiffTSGenerator"),
        "ReactOTTask": ("diffusion_reactot", "ReactOTTSGenerator"),
        "GoFlowTask": ("diffusion_goflow", "GoFlowTSGenerator"),
    }


def test_shared_config_on_a_non_ts_task_says_so() -> None:
    class DiffSBDDTask:
        pass

    with pytest.raises(ValueError, match="not a transition-state task"):
        TSGenerator(DiffSBDDTask(), reaction_source="corpus.pkl")


def test_per_model_key_still_works() -> None:
    """All three shipped configs say reaction_pkl."""
    gen = _oareactdiff(reaction_pkl="corpus.pkl")
    assert gen.pocket_db == "corpus.pkl"
    # ... and directly on the concrete class, which is what those configs do
    direct = OAReactDiffTSGenerator(
        OAReactDiffTask(), reaction_pkl="corpus.pkl"
    )
    assert direct.pocket_db == "corpus.pkl"


def test_shared_key_works_on_the_per_model_class() -> None:
    direct = OAReactDiffTSGenerator(
        OAReactDiffTask(), reaction_source="corpus.pkl"
    )
    assert direct.pocket_db == "corpus.pkl"


def test_both_names_at_once_is_refused() -> None:
    with pytest.raises(ValueError, match="not both"):
        _oareactdiff(reaction_source="a.pkl", reaction_pkl="b.pkl")


def test_a_typo_near_the_alias_still_raises() -> None:
    with pytest.raises(ValueError, match="does not accept interference key"):
        _oareactdiff(reaction_pkll="corpus.pkl")


def test_missing_corpus_names_both_keys() -> None:
    with pytest.raises(ValueError, match="reaction_pkl.*reaction_source"):
        _oareactdiff()
