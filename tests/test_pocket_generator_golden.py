"""Golden snapshots of the seven pocket-conditioned generators.

The fixture ``tests/data/pocket_generator_golden.pt`` was generated against
git HEAD ``701ed38``, BEFORE the generator refactor. It pins, per model:

* the tiled pocket batch produced by ``_repeat`` / ``_pocket_batch``
  (values AND dtypes -- the scatter-index tensors are ``int64`` and a silent
  dtype change is exactly the regression this file exists to catch);
* the constructor's ordered parameter names and defaults, i.e. the Hydra key
  surface an ``interference/gen_*_pocket.yaml`` may set;
* the explicit-``mol_size`` size-selection branch under ``random.seed(0)``;
* DiffSBDD's ``_fixed_ligand`` inpainting builder on the ``add_n_nodes`` path.

**Regenerating the fixture defeats the purpose of this test.** It must only
be regenerated when the OLD behaviour is deliberately changed, via
``python tests/test_pocket_generator_golden.py``. A plain pytest run never
regenerates it.
"""

from __future__ import annotations

import inspect
import random
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from MolecularDiffusion.modules.tasks.diffusion_diffsbdd import (
    DiffSBDDPocketGenerator,
)
from MolecularDiffusion.modules.tasks.diffusion_diffint import (
    DiffIntPocketGenerator,
)
from MolecularDiffusion.modules.tasks.diffusion_kgdiff import (
    KGDiffPocketGenerator,
)
from MolecularDiffusion.modules.tasks.diffusion_ipdiff import (
    IPDiffPocketGenerator,
)
from MolecularDiffusion.modules.tasks.diffusion_pmdm import PMDMPocketGenerator
from MolecularDiffusion.modules.tasks.diffusion_apo2mol import (
    Apo2MolPocketGenerator,
)
from MolecularDiffusion.modules.tasks.diffusion_diffpharma import (
    DiffPharmaPocketGenerator,
)

FIXTURE = Path(__file__).parent / "data" / "pocket_generator_golden.pt"

#: 5 pocket atoms / 3 residues, 4 reference-ligand atoms. Tiled with n=3.
N_POCKET = 5
N_RES = 3
N_LIG = 4
N_TILE = 3
MOL_SIZE = [7, 9, 11]
N_SIZES = 4

CLASSES = {
    "diffsbdd": DiffSBDDPocketGenerator,
    "diffint": DiffIntPocketGenerator,
    "kgdiff": KGDiffPocketGenerator,
    "ipdiff": IPDiffPocketGenerator,
    "pmdm": PMDMPocketGenerator,
    "apo2mol": Apo2MolPocketGenerator,
    "diffpharma": DiffPharmaPocketGenerator,
}


def _f(*shape: int) -> torch.Tensor:
    """Deterministic float32 filler, distinct per element."""
    n = 1
    for s in shape:
        n *= s
    return (torch.arange(n, dtype=torch.float32) * 0.5 - 3.0).reshape(*shape)


def _onehot(rows: int, cols: int) -> torch.Tensor:
    idx = torch.arange(rows) % cols
    return torch.nn.functional.one_hot(idx, cols).float()


def _items() -> Dict[str, Dict[str, Any]]:
    """One synthetic Dataset row per model, keys/dtypes as the real ones."""
    diffsbdd_item = {
        "name": "pocket_synthetic",
        "lig_coords": _f(N_LIG, 3),
        "lig_one_hot": _onehot(N_LIG, 10),
        "pocket_coords": _f(N_POCKET, 3),
        "pocket_one_hot": _onehot(N_POCKET, 20),
    }
    diffint_item = {
        "name": "pocket_synthetic",
        "lig_coords": _f(N_LIG, 3),
        "lig_one_hot": _onehot(N_LIG, 10),
        "pocket_coords": _f(N_POCKET, 3),
        "pocket_one_hot": _onehot(N_POCKET, 22),
        "n_residues": N_RES,
    }
    kgdiff_item = {
        "name": "pocket_synthetic",
        "ligand_pos": _f(N_LIG, 3),
        "ligand_v": torch.arange(N_LIG, dtype=torch.int64),
        "protein_pos": _f(N_POCKET, 3),
        "protein_v": _onehot(N_POCKET, 27),
        "affinity": torch.tensor(0.75, dtype=torch.float32),
    }
    pmdm_item = {
        "name": "pocket_synthetic",
        "ligand_pos": _f(N_LIG, 3),
        "ligand_atom_feature": _onehot(N_LIG, 10),
        "ligand_atom_feature_full": _onehot(N_LIG, 16),
        "protein_pos": _f(N_POCKET, 3),
        "protein_atom_feature": _onehot(N_POCKET, 6),
        "protein_atom_feature_full": _onehot(N_POCKET, 31),
        "protein_is_backbone": torch.zeros(N_POCKET, dtype=torch.bool),
    }
    apo2mol_item = {
        "name": "pocket_synthetic",
        "ligand_pos": _f(N_LIG, 3),
        "ligand_v": torch.arange(N_LIG, dtype=torch.int64),
        "protein_pos": _f(N_POCKET, 3),
        "protein_pos_holo": _f(N_POCKET, 3) + 1.25,
        "protein_v": _onehot(N_POCKET, 27),
        "protein_atom_name": ["N", "CA", "C", "O", "CB"],
        "protein_atom_to_aa_name": ["ALA", "ALA", "ALA", "GLY", "SER"],
        "protein_atom_to_aa_group": torch.tensor(
            [0, 0, 0, 1, 2], dtype=torch.int64
        ),
        "protein_rotations": _f(N_RES, 4),
        "protein_translations": _f(N_RES, 3),
        "protein_chi_apo": _f(N_RES, 5),
        "protein_chi_holo": _f(N_RES, 5) + 0.25,
        "protein_chi_mask": _onehot(N_RES, 5),
    }
    diffpharma_item = {
        "lig_coords": _f(N_LIG, 3),
        "lig_one_hot": _onehot(N_LIG, 11),
        "pocket_coords": _f(N_POCKET, 3),
        "pocket_one_hot": _onehot(N_POCKET, 11),
        "interh_coords": _f(2, 3),
        "interh_one_hot": _onehot(2, 3),
        "interhp_coords": _f(3, 3),
        "interhp_one_hot": _onehot(3, 6),
    }
    return {
        "diffsbdd": diffsbdd_item,
        "diffint": diffint_item,
        "kgdiff": kgdiff_item,
        "ipdiff": kgdiff_item,
        "pmdm": pmdm_item,
        "apo2mol": apo2mol_item,
        "diffpharma": diffpharma_item,
    }


def _generators() -> Dict[str, Any]:
    """One instance per model; all seven constructors only store attributes."""
    return {
        "diffsbdd": DiffSBDDPocketGenerator(
            task=None,
            pocket_db="synthetic.db",
            mol_size=MOL_SIZE,
            fixed_atom_indices=[0, 2],
            add_n_nodes=3,
            device="cpu",
        ),
        "diffint": DiffIntPocketGenerator(
            task=None,
            pocket_db="synthetic.db",
            mol_size=MOL_SIZE,
            device="cpu",
        ),
        "kgdiff": KGDiffPocketGenerator(
            task=None,
            pocket_db="synthetic.db",
            mol_size=MOL_SIZE,
            device="cpu",
        ),
        "ipdiff": IPDiffPocketGenerator(
            task=None,
            pocket_db="synthetic.db",
            mol_size=MOL_SIZE,
            device="cpu",
        ),
        "pmdm": PMDMPocketGenerator(
            task=None,
            pocket_db="synthetic.db",
            mol_size=MOL_SIZE,
            device="cpu",
        ),
        "apo2mol": Apo2MolPocketGenerator(
            task=None,
            pocket_db="synthetic.db",
            mol_size=MOL_SIZE,
            device="cpu",
        ),
        "diffpharma": DiffPharmaPocketGenerator(
            task=None,
            pocket_db="synthetic.db",
            mol_size=MOL_SIZE,
            device="cpu",
        ),
    }


def _tiled(name: str, gen: Any, item: Dict[str, Any]) -> Dict[str, Any]:
    if name in ("diffsbdd", "diffint"):
        return gen._pocket_batch(item, N_TILE)  # noqa: SLF001
    return gen._repeat(item, N_TILE)  # noqa: SLF001


def _sizes(name: str, gen: Any, tiled: Dict[str, Any]) -> torch.Tensor:
    random.seed(0)
    if name in ("diffsbdd", "diffint"):
        return gen._explicit_sizes(N_SIZES)  # noqa: SLF001
    if name == "diffpharma":
        return gen._sizes(tiled, N_SIZES)  # noqa: SLF001
    return gen._sizes(N_SIZES)  # noqa: SLF001


def _signature(cls: type) -> list:
    return [
        (p.name, None if p.default is inspect.Parameter.empty else p.default)
        for p in inspect.signature(cls.__init__).parameters.values()
        if p.name != "self"
    ]


def build_snapshot() -> Dict[str, Any]:
    """Everything this test pins, per model."""
    items = _items()
    gens = _generators()
    snap: Dict[str, Any] = {}
    for name, gen in gens.items():
        tiled = _tiled(name, gen, items[name])
        snap[name] = {
            "tiled": tiled,
            "sizes": _sizes(name, gen, tiled),
            "signature": _signature(CLASSES[name]),
        }
    # DiffSBDD inpainting, add_n_nodes path only.
    ligand, lig_fixed, n_fixed = gens["diffsbdd"]._fixed_ligand(  # noqa: SLF001
        items["diffsbdd"], N_TILE
    )
    snap["diffsbdd"]["fixed_ligand"] = {
        "x": ligand["x"],
        "one_hot": ligand["one_hot"],
        "size": ligand["size"],
        "mask": ligand["mask"],
        "lig_fixed": lig_fixed,
        "n_fixed": n_fixed,
    }
    return snap


def _assert_same(got: Any, want: Any, where: str) -> None:
    if isinstance(want, torch.Tensor):
        assert isinstance(got, torch.Tensor), f"{where}: not a tensor"
        assert got.dtype == want.dtype, (
            f"{where}: dtype {got.dtype} != {want.dtype}"
        )
        assert got.shape == want.shape, (
            f"{where}: shape {tuple(got.shape)} != {tuple(want.shape)}"
        )
        assert torch.equal(got, want), f"{where}: values differ"
    elif isinstance(want, dict):
        assert set(got) == set(want), (
            f"{where}: keys {sorted(got)} != {sorted(want)}"
        )
        for k in want:
            _assert_same(got[k], want[k], f"{where}.{k}")
    elif isinstance(want, (list, tuple)):
        assert type(got) is type(want), f"{where}: type {type(got)}"
        assert len(got) == len(want), f"{where}: length"
        for i, (g, w) in enumerate(zip(got, want)):
            _assert_same(g, w, f"{where}[{i}]")
    else:
        assert type(got) is type(want), (
            f"{where}: type {type(got)} != {type(want)}"
        )
        assert got == want, f"{where}: {got!r} != {want!r}"


@pytest.fixture(scope="module")
def golden() -> Dict[str, Any]:
    assert FIXTURE.exists(), (
        f"missing golden fixture {FIXTURE}; regenerate deliberately with "
        f"`python {Path(__file__).name}`"
    )
    return torch.load(FIXTURE, weights_only=False)


@pytest.fixture(scope="module")
def current() -> Dict[str, Any]:
    return build_snapshot()


@pytest.mark.parametrize("model", list(CLASSES))
def test_tiled_pocket_batch_unchanged(model, golden, current) -> None:
    _assert_same(
        current[model]["tiled"], golden[model]["tiled"], f"{model}.tiled"
    )


@pytest.mark.parametrize("model", list(CLASSES))
def test_explicit_size_selection_unchanged(model, golden, current) -> None:
    _assert_same(
        current[model]["sizes"], golden[model]["sizes"], f"{model}.sizes"
    )


@pytest.mark.parametrize("model", list(CLASSES))
def test_constructor_signature_unchanged(model, golden, current) -> None:
    _assert_same(
        current[model]["signature"],
        golden[model]["signature"],
        f"{model}.signature",
    )


def test_diffsbdd_fixed_ligand_unchanged(golden, current) -> None:
    _assert_same(
        current["diffsbdd"]["fixed_ligand"],
        golden["diffsbdd"]["fixed_ligand"],
        "diffsbdd.fixed_ligand",
    )


# --- validation errors the refactor must preserve -------------------------


@pytest.mark.parametrize(
    "cls",
    [
        DiffSBDDPocketGenerator,
        KGDiffPocketGenerator,
        IPDiffPocketGenerator,
        PMDMPocketGenerator,
        Apo2MolPocketGenerator,
    ],
)
def test_pocket_db_is_required(cls) -> None:
    with pytest.raises(ValueError, match="pocket_db"):
        cls(task=None)


@pytest.mark.parametrize(
    "cls", [DiffIntPocketGenerator, DiffPharmaPocketGenerator]
)
def test_exactly_one_pocket_source(cls) -> None:
    with pytest.raises(ValueError, match="exactly one pocket source"):
        cls(task=None)
    with pytest.raises(ValueError, match="exactly one pocket source"):
        cls(task=None, pocket_db="a.db", pocket_pdb="a.pdb")


@pytest.mark.parametrize(
    "cls", [DiffIntPocketGenerator, DiffPharmaPocketGenerator]
)
def test_ref_sdf_required_with_pocket_pdb(cls) -> None:
    with pytest.raises(ValueError, match="ref_sdf"):
        cls(task=None, pocket_pdb="a.pdb")


def test_diffint_rejects_inpainting() -> None:
    with pytest.raises(NotImplementedError, match="inpainting"):
        DiffIntPocketGenerator(
            task=None, pocket_db="a.db", fixed_atom_indices=[0, 1]
        )


if __name__ == "__main__":
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_snapshot(), FIXTURE)
    print(f"regenerated {FIXTURE}")
