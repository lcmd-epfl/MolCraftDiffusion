"""Unit tests for the `analyze metrics --metrics core` building blocks."""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# shape_measure
# ---------------------------------------------------------------------------


class TestShapeMeasure:
    """Continuous shape measures against textbook reference values."""

    @staticmethod
    def _with_centre(vertices):
        return np.vstack([np.zeros(3), np.asarray(vertices, dtype=float)])

    @property
    def tetrahedron(self):
        v = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ]
        )
        return self._with_centre(v / np.sqrt(3.0))

    @property
    def square(self):
        v = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
        return self._with_centre(v)

    @property
    def octahedron(self):
        v = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
        return self._with_centre(v)

    def test_perfect_match_is_zero(self):
        from MolecularDiffusion.utils.shape_measure import shape_measure

        assert shape_measure(self.tetrahedron, "T-4") == pytest.approx(
            0.0, abs=1e-9
        )
        assert shape_measure(self.octahedron, "OC-6") == pytest.approx(
            0.0, abs=1e-9
        )
        assert shape_measure(self.square, "SP-4") == pytest.approx(
            0.0, abs=1e-9
        )

    def test_tetrahedron_versus_square_planar(self):
        """The textbook tetrahedron/square-planar shape measure is 33.33."""
        from MolecularDiffusion.utils.shape_measure import shape_measure

        assert shape_measure(self.tetrahedron, "SP-4") == pytest.approx(
            33.3333, abs=1e-3
        )
        assert shape_measure(self.square, "T-4") == pytest.approx(
            33.3333, abs=1e-3
        )

    def test_trigonal_prism_versus_octahedron(self):
        """The published prism/octahedron shape measure is 16.737."""
        from MolecularDiffusion.utils.shape_measure import (
            _equal_edge_prism,
            shape_measure,
        )

        prism = self._with_centre(_equal_edge_prism())
        assert shape_measure(prism, "OC-6") == pytest.approx(16.737, abs=1e-2)

    def test_invariant_to_rotation_translation_and_scale(self):
        from MolecularDiffusion.utils.shape_measure import shape_measure

        rng = np.random.default_rng(0)
        rot, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(rot) < 0:
            rot[:, 0] *= -1
        moved = self.tetrahedron @ rot.T * 2.7 + np.array([3.0, -1.0, 0.5])
        assert shape_measure(moved, "SP-4") == pytest.approx(33.3333, abs=1e-3)

    def test_central_atom_can_be_any_row(self):
        from MolecularDiffusion.utils.shape_measure import shape_measure

        shuffled = np.vstack([self.tetrahedron[1:], self.tetrahedron[0]])
        assert shape_measure(shuffled, "T-4", central_atom=5) == pytest.approx(
            0.0, abs=1e-9
        )

    def test_wrong_vertex_count_is_rejected(self):
        from MolecularDiffusion.utils.shape_measure import shape_measure

        with pytest.raises(ValueError, match="expects"):
            shape_measure(self.tetrahedron, "OC-6")

    def test_unknown_shape_is_rejected(self):
        from MolecularDiffusion.utils.shape_measure import shape_measure

        with pytest.raises(KeyError, match="Unknown shape"):
            shape_measure(self.tetrahedron, "NOPE-4")


# ---------------------------------------------------------------------------
# check_validity_v1
# ---------------------------------------------------------------------------


class TestSkipIndices:
    def test_skipped_atoms_leave_the_denominator(self):
        """--skip-atoms must not score the skipped atoms as invalid."""
        import torch

        from MolecularDiffusion.utils.geom_metrics import check_validity_v1
        from MolecularDiffusion.utils.geom_utils import (
            correct_edges,
            create_pyg_graph,
        )

        # methane: a clean tetrahedral carbon
        pos = torch.tensor(
            [
                [0.000, 0.000, 0.000],
                [0.629, 0.629, 0.629],
                [-0.629, -0.629, 0.629],
                [-0.629, 0.629, -0.629],
                [0.629, -0.629, -0.629],
            ]
        )
        z = torch.tensor([6, 1, 1, 1, 1])
        data = correct_edges(
            create_pyg_graph(pos, z, xyz_filename="x", r=4), scale_factor=1.2
        )

        _, pct_all, _, _, _ = check_validity_v1(data, skip_indices=[])
        _, pct_skip, _, _, _ = check_validity_v1(data, skip_indices=[1, 2])

        assert pct_all == pytest.approx(1.0)
        # two atoms skipped -> still 100%, not 3/5
        assert pct_skip == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# core-metric helpers
# ---------------------------------------------------------------------------


class TestRdkitValid:
    def test_none_is_invalid(self):
        from MolecularDiffusion.runmodes.analyze.compute_metrics import (
            _rdkit_valid,
        )

        assert _rdkit_valid(None) is False

    def test_sane_molecule_is_valid(self):
        from rdkit import Chem

        from MolecularDiffusion.runmodes.analyze.compute_metrics import (
            _rdkit_valid,
        )

        assert _rdkit_valid(Chem.MolFromSmiles("CCO")) is True

    def test_hypervalent_carbon_is_invalid(self):
        from rdkit import Chem

        from MolecularDiffusion.runmodes.analyze.compute_metrics import (
            _rdkit_valid,
        )

        # five bonds on carbon: builds unsanitized, fails sanitization
        mol = Chem.MolFromSmiles("C(C)(C)(C)(C)C", sanitize=False)
        assert _rdkit_valid(mol) is False


class TestSetLevelMetrics:
    def test_empty_input(self):
        from MolecularDiffusion.runmodes.analyze.compute_metrics import (
            _set_level_metrics,
        )

        out = _set_level_metrics([])
        assert out["n_valid_smiles"] == 0
        assert out["uniqueness"] is None
        assert out["diversity"] is None

    def test_uniqueness_counts_duplicates(self):
        from MolecularDiffusion.runmodes.analyze.compute_metrics import (
            _set_level_metrics,
        )

        out = _set_level_metrics(["CCO", "CCO", "c1ccccc1"])
        assert out["uniqueness"] == pytest.approx(2 / 3)

    def test_novelty_against_reference(self):
        from MolecularDiffusion.runmodes.analyze.compute_metrics import (
            _set_level_metrics,
        )

        out = _set_level_metrics(["CCO", "c1ccccc1"], train_smiles={"CCO"})
        assert out["novelty"] == pytest.approx(0.5)

    def test_diversity_of_identical_molecules_is_zero(self):
        from MolecularDiffusion.runmodes.analyze.compute_metrics import (
            _set_level_metrics,
        )

        # duplicates collapse to one unique molecule -> no pairs -> None
        assert _set_level_metrics(["CCO", "CCO"])["diversity"] is None
        # two very different molecules -> high diversity
        assert _set_level_metrics(["CCO", "c1ccccc1"])["diversity"] > 0.5

    def test_nulls_are_ignored(self):
        from MolecularDiffusion.runmodes.analyze.compute_metrics import (
            _set_level_metrics,
        )

        assert _set_level_metrics(["CCO", None, ""])["n_valid_smiles"] == 1


class TestPerceiveMol:
    def test_unreadable_file_returns_none_without_raising(self, tmp_path):
        """A failed conversion yields None, not the previous result."""
        from MolecularDiffusion.runmodes.analyze.compute_metrics import (
            _perceive_mol,
        )

        bad = tmp_path / "corrupt.xyz"
        bad.write_text("not\nan xyz file at all\n")
        assert _perceive_mol(str(bad), "xyz2mol", timeout=5) == (None, None)

    def test_reads_a_real_xyz(self, tmp_path):
        from MolecularDiffusion.runmodes.analyze.compute_metrics import (
            _perceive_mol,
        )

        xyz = tmp_path / "methane.xyz"
        xyz.write_text(
            "5\n\n"
            "C 0.000 0.000 0.000\n"
            "H 0.629 0.629 0.629\n"
            "H -0.629 -0.629 0.629\n"
            "H -0.629 0.629 -0.629\n"
            "H 0.629 -0.629 -0.629\n"
        )
        smiles, mol = _perceive_mol(str(xyz), "xyz2mol", timeout=30)
        assert mol is not None
        assert smiles is not None


class TestLoadTrainSmiles:
    def test_none_passthrough(self):
        from MolecularDiffusion.runmodes.analyze.compute_metrics import (
            _load_train_smiles,
        )

        assert _load_train_smiles(None) is None

    def test_txt_one_per_line(self, tmp_path):
        from MolecularDiffusion.runmodes.analyze.compute_metrics import (
            _load_train_smiles,
        )

        p = tmp_path / "train.txt"
        p.write_text("CCO\nc1ccccc1\n\n")
        assert _load_train_smiles(str(p)) == {"CCO", "c1ccccc1"}

    def test_csv_smiles_column(self, tmp_path):
        from MolecularDiffusion.runmodes.analyze.compute_metrics import (
            _load_train_smiles,
        )

        p = tmp_path / "train.csv"
        p.write_text("smiles,x\nCCO,1\nc1ccccc1,2\n")
        assert _load_train_smiles(str(p)) == {"CCO", "c1ccccc1"}
