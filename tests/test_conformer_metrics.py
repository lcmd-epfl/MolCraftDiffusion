"""The ``conformer`` metrics group: stereo scoring, layouts, aggregation.

Chemistry fixtures are built with RDKit only -- no checkpoint, no xtb.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from MolecularDiffusion.runmodes.analyze import conformer_metrics as cm


def _embed(smiles: str):
    """3D-embedded, H-explicit molecule with CIP tags from its geometry."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=0xF00D) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    Chem.AssignStereochemistryFrom3D(mol)
    return mol


# L-alanine-like: two centres so a single inversion is a diastereomer, not
# the enantiomer.
CHIRAL = "C[C@H](N)[C@@H](O)C(=O)O"
ONE_INVERTED = "C[C@@H](N)[C@@H](O)C(=O)O"
ENANTIOMER = "C[C@@H](N)[C@H](O)C(=O)O"
ACHIRAL = "CCO"


def test_get_xtb_energy_still_importable_for_generation():
    """``tasks_conformer`` lazily imports this; losing it breaks GENERATION."""
    from MolecularDiffusion.runmodes.analyze.compare_to_optimized import (
        get_xtb_energy,
    )

    assert callable(get_xtb_energy)


def test_identical_molecule_scores_one():
    mol = _embed(CHIRAL)
    scores = cm.stereo_scores([mol], [mol])
    assert scores["rs_score"] == 1.0
    assert scores["n_rs_scored"] == 1


def test_partially_inverted_stereochemistry_scores_below_one():
    """One centre flipped is a diastereomer -- must NOT count as correct."""
    scores = cm.stereo_scores([_embed(ONE_INVERTED)], [_embed(CHIRAL)])
    assert scores["rs_score"] < 1.0


def test_full_enantiomer_scores_one():
    """Upstream semantics, kept deliberately for comparability with LoQI."""
    scores = cm.stereo_scores([_embed(ENANTIOMER)], [_embed(CHIRAL)])
    assert scores["rs_score"] == 1.0


def test_no_stereocentres_gives_none_not_zero():
    """An empty denominator must be distinguishable from 'everything wrong'."""
    mol = _embed(ACHIRAL)
    scores = cm.stereo_scores([mol], [mol])
    assert scores["rs_score"] is None
    assert scores["ez_score"] is None
    assert scores["n_rs_scored"] == 0


def test_mismatched_lengths_raise_not_assert():
    mol = _embed(ACHIRAL)
    with pytest.raises(ValueError, match="same length"):
        cm.stereo_scores([mol, mol], [mol])


def test_ez_double_bond_geometry():
    e_mol, z_mol = _embed("C/C=C/C"), _embed(r"C/C=C\C")
    assert cm.stereo_scores([e_mol], [e_mol])["ez_score"] == 1.0
    assert cm.stereo_scores([z_mol], [e_mol])["ez_score"] == 0.0


# ---------------------------------------------------------------------------
# Layout detection and end-to-end aggregation
# ---------------------------------------------------------------------------


def _write_layout1(root: Path, n_conformers: int = 3, sdf_records=None):
    """Minimal ConformerFactory-shaped output directory."""
    ref = _embed(CHIRAL)
    rows = []
    mol_dir = root / "mol_0000"
    mol_dir.mkdir(parents=True)
    with Chem.SDWriter(str(mol_dir / "reference.sdf")) as w:
        w.write(ref)
    records = sdf_records if sdf_records is not None else [ref] * n_conformers
    with Chem.SDWriter(str(mol_dir / "conformers.sdf")) as w:
        for m in records:
            w.write(m)
    for i in range(n_conformers):
        rows.append(
            {
                "mol_index": 0,
                "conformer_index": i,
                "smiles": Chem.MolToSmiles(ref),
                "n_atoms": ref.GetNumAtoms(),
                "xyz": f"mol_0000/conformer_{i:03d}.xyz",
                "rmsd": 0.1 * (i + 1),
                "energy_hartree": None,
            }
        )
    pd.DataFrame(rows).to_csv(root / "conformers.csv", index=False)
    return root


def test_layout_detection_sdf_pairs(tmp_path):
    assert cm.detect_layout(_write_layout1(tmp_path)) == "sdf_pairs"


def test_bare_xyz_directory_is_a_clear_error(tmp_path):
    (tmp_path / "molecule_0.xyz").write_text("1\n\nC 0.0 0.0 0.0\n")
    with pytest.raises(ValueError, match="paired structures"):
        cm.detect_layout(tmp_path)


def test_layout_detection_xyz_optimized(tmp_path):
    (tmp_path / "a.xyz").write_text("1\n\nC 0.0 0.0 0.0\n")
    opt = tmp_path / "optimized_xyz"
    opt.mkdir()
    (opt / "a_opt.xyz").write_text("1\n\nC 0.0 0.0 0.0\n")
    assert cm.detect_layout(tmp_path) == "xyz_optimized"


def test_end_to_end_aggregation(tmp_path):
    df, summary = cm.compute_conformer_metrics(
        _write_layout1(tmp_path), rmsd_threshold=0.25
    )
    assert len(df) == 3
    assert list(df["conformer_index"]) == [0, 1, 2]
    # rmsd is READ from the csv, never recomputed
    assert list(df["rmsd"]) == [0.1, 0.2, 0.3]
    assert summary["rmsd_median"] == 0.2
    assert summary["coverage_at_threshold"] == pytest.approx(2 / 3)
    assert summary["rmsd_best_per_mol_mean"] == 0.1
    # generated == reference here, so every paired geometry diff is zero
    assert summary["bond_length_mean_mean"] == pytest.approx(0.0)
    assert summary["rs_score"] == 1.0
    assert summary["n_skipped"] == 0
    # xtb is optional: never a fabricated number
    if not cm.xtb_available():
        assert df["xtb_strain_kcal"].isna().all()


def test_sdf_shorter_than_csv_does_not_zip_blindly(tmp_path):
    """A failed rebuild means fewer SDF records than csv rows."""
    root = _write_layout1(tmp_path, n_conformers=3)
    # drop the middle record and blank its rmsd, as ConformerFactory would
    csv = pd.read_csv(root / "conformers.csv")
    csv.loc[1, "rmsd"] = None
    csv.to_csv(root / "conformers.csv", index=False)
    ref = _embed(CHIRAL)
    with Chem.SDWriter(str(root / "mol_0000" / "conformers.sdf")) as w:
        w.write(ref)
        w.write(ref)

    df, _ = cm.compute_conformer_metrics(root)
    assert list(df["conformer_index"]) == [0, 2]
    assert list(df["rmsd"]) == [0.1, 0.3]


def test_runner_writes_csv_and_summary(tmp_path):
    """The compute_metrics block: output paths, result table, summary json."""
    from MolecularDiffusion.runmodes.analyze.compute_metrics import runner

    root = _write_layout1(tmp_path)
    out = tmp_path / "out.csv"
    runner(
        SimpleNamespace(
            input=str(root), output=str(out), filter=None,
            filtered_output=None, metrics="conformer", recheck_topo=False,
            check_neutrality=False, portion=1.0, mol_converter="xyz2mol",
            skip_atoms=None, split=1, timeout=10, reference_mol=None,
            mol_idx=0, train_smiles=None, receptor=None, ref_ligand=None,
            dock_mode="dock", exhaustiveness=8, rdkit_rmsd=False,
            rmsd_n_conf=20, rmsd_threshold=0.5, charge=0, level="gfn2",
            xtb_timeout=120,
        )
    )
    csv_path = tmp_path / "out_conformer.csv"
    assert csv_path.exists()
    summary = json.loads((tmp_path / "out_conformer_summary.json").read_text())
    assert summary["metrics"] == "conformer"
    assert summary["layout"] == "sdf_pairs"
    assert summary["rs_score"] == 1.0


def test_conformer_is_not_part_of_all():
    """`all` must never try to run the paired group on unpaired input."""
    import inspect

    from MolecularDiffusion.runmodes.analyze import compute_metrics

    source = inspect.getsource(compute_metrics.runner)
    assert 'if args.metrics == "conformer":' in source
    assert '"all", "conformer"' not in source
