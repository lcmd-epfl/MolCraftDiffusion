"""Unit tests for the `analyze metrics --metrics sbdd` building blocks."""

import numpy as np
import pytest

vina = pytest.importorskip("vina", reason="needs the [sbdd] extra")
meeko = pytest.importorskip("meeko", reason="needs the [sbdd] extra")

METHANE_XYZ = (
    "5\n\n"
    "C 0.000 0.000 0.000\n"
    "H 0.629 0.629 0.629\n"
    "H -0.629 -0.629 0.629\n"
    "H -0.629 0.629 -0.629\n"
    "H 0.629 -0.629 -0.629\n"
)


class TestLoadPose:
    def test_keeps_the_conformer(self, tmp_path):
        """Docking needs the generated coordinates, not just the topology."""
        from MolecularDiffusion.runmodes.analyze.docking import load_pose

        xyz = tmp_path / "methane.xyz"
        xyz.write_text(METHANE_XYZ)
        smiles, mol = load_pose(str(xyz))
        assert mol is not None
        assert mol.GetNumConformers() == 1
        assert smiles

    def test_unreadable_file_returns_none(self, tmp_path):
        from MolecularDiffusion.runmodes.analyze.docking import load_pose

        bad = tmp_path / "corrupt.xyz"
        bad.write_text("not\nan xyz file\n")
        assert load_pose(str(bad)) == (None, None)

    def test_bad_valence_returns_none_without_raising(self, tmp_path):
        """A structure RDKit cannot sanitize is data, not an exception."""
        from MolecularDiffusion.runmodes.analyze.docking import load_pose

        # five atoms crowded around one N -> hypervalent after perception
        xyz = tmp_path / "bad.xyz"
        xyz.write_text(
            "6\n\n"
            "N 0.000 0.000 0.000\n"
            "C 1.470 0.000 0.000\n"
            "C -1.470 0.000 0.000\n"
            "C 0.000 1.470 0.000\n"
            "C 0.000 -1.470 0.000\n"
            "C 0.000 0.000 1.470\n"
        )
        smiles, mol = load_pose(str(xyz))
        assert mol is None


class TestBox:
    def test_box_is_extent_plus_buffer(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from MolecularDiffusion.runmodes.analyze.docking import box_for

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        center, size = box_for(mol)

        pos = mol.GetConformer(0).GetPositions()
        assert center == pytest.approx(((pos.max(0) + pos.min(0)) / 2).tolist())
        assert size == pytest.approx(((pos.max(0) - pos.min(0)) + 5.0).tolist())

    def test_buffer_is_configurable(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from MolecularDiffusion.runmodes.analyze.docking import box_for

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        _, small = box_for(mol, buffer=0.0)
        _, big = box_for(mol, buffer=10.0)
        assert np.allclose(np.array(big) - np.array(small), 10.0)


class TestLigandPdbqt:
    def test_writes_pdbqt_for_a_sane_molecule(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from MolecularDiffusion.runmodes.analyze.docking import ligand_pdbqt

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        text = ligand_pdbqt(mol)
        assert "ATOM" in text or "ROOT" in text


class TestScoreMode:
    def test_rejects_unknown_mode(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from MolecularDiffusion.runmodes.analyze.docking import score_pose

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        with pytest.raises(ValueError, match="mode must be one of"):
            score_pose(mol, "unused.pdbqt", mode="nope")


class TestPrepareReceptor:
    def test_pdbqt_is_passed_through_untouched(self, tmp_path):
        """An already-prepared receptor must be reused, not re-derived."""
        from MolecularDiffusion.runmodes.analyze.docking import prepare_receptor

        rec = tmp_path / "rec.pdbqt"
        rec.write_text("ATOM\n")
        assert prepare_receptor(str(rec)) == str(rec)
