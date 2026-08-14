"""Unit tests for the `analyze metrics --metrics druglike` descriptors."""

import pytest
from rdkit import Chem


def _mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return mol


class TestRingStats:
    def test_benzene_is_one_aromatic_ring(self):
        from MolecularDiffusion.runmodes.analyze.druglike import _ring_stats

        stats = _ring_stats(_mol("c1ccccc1"))
        assert stats["n_rings"] == 1
        assert stats["n_aromatic_rings"] == 1
        assert stats["n_aliphatic_rings"] == 0
        assert stats["ring_size_6"] is True
        assert stats["ring_size_5"] is False

    def test_cyclohexane_is_one_aliphatic_ring(self):
        from MolecularDiffusion.runmodes.analyze.druglike import _ring_stats

        stats = _ring_stats(_mol("C1CCCCC1"))
        assert stats["n_aromatic_rings"] == 0
        assert stats["n_aliphatic_rings"] == 1
        assert stats["ring_size_6"] is True

    def test_cyclopropane_is_size_three(self):
        from MolecularDiffusion.runmodes.analyze.druglike import _ring_stats

        stats = _ring_stats(_mol("C1CC1"))
        assert stats["ring_size_3"] is True
        assert stats["ring_size_6"] is False

    def test_acyclic_molecule_has_no_rings(self):
        from MolecularDiffusion.runmodes.analyze.druglike import _ring_stats

        stats = _ring_stats(_mol("CCO"))
        assert stats["n_rings"] == 0
        assert not any(stats[f"ring_size_{n}"] for n in range(3, 10))


class TestLipinski:
    def test_aspirin_obeys_all_five(self):
        from MolecularDiffusion.runmodes.analyze.druglike import _lipinski

        assert _lipinski(_mol("CC(=O)Oc1ccccc1C(=O)O")) == 5

    def test_a_greasy_giant_breaks_rules(self):
        from MolecularDiffusion.runmodes.analyze.druglike import _lipinski

        # long alkane: too heavy, too lipophilic, too many rotatable bonds
        assert _lipinski(_mol("C" * 40)) < 5


class TestPains:
    def test_benign_molecule_passes(self):
        from MolecularDiffusion.runmodes.analyze.druglike import _pains_pass

        assert _pains_pass(_mol("CC(=O)Oc1ccccc1C(=O)O")) is True

    def test_quinone_is_flagged(self):
        from MolecularDiffusion.runmodes.analyze.druglike import _pains_pass

        # p-benzoquinone matches the quinone_A alert in the PAINS_A catalog
        assert _pains_pass(_mol("O=C1C=CC(=O)C=C1")) is False

    def test_azobenzene_is_flagged(self):
        from MolecularDiffusion.runmodes.analyze.druglike import _pains_pass

        assert _pains_pass(_mol("c1ccc(/N=N/c2ccccc2)cc1")) is False


class TestRingFilter:
    def test_six_membered_ring_passes(self):
        from MolecularDiffusion.runmodes.analyze.druglike import _ring_filter_pass

        assert _ring_filter_pass(_mol("C1CCCCC1")) is True

    def test_large_aliphatic_ring_fails(self):
        from MolecularDiffusion.runmodes.analyze.druglike import _ring_filter_pass

        assert _ring_filter_pass(_mol("C1CCCCCCC1")) is False

    def test_large_aromatic_ring_is_allowed(self):
        from MolecularDiffusion.runmodes.analyze.druglike import _ring_filter_pass

        # azulene: a fused 7-membered aromatic ring must not be rejected
        assert _ring_filter_pass(_mol("c1ccc2cccc2cc1")) is True


class TestRdkitRmsd:
    def test_returns_ordered_statistics(self):
        from MolecularDiffusion.runmodes.analyze.druglike import rdkit_rmsd

        mol = Chem.AddHs(_mol("CCO"))
        from rdkit.Chem import AllChem

        AllChem.EmbedMolecule(mol, randomSeed=42)
        low, mid, high = rdkit_rmsd(mol, n_conf=3)
        assert low is not None
        assert low <= mid <= high

    def test_failure_returns_none_triple(self):
        from MolecularDiffusion.runmodes.analyze.druglike import rdkit_rmsd

        assert rdkit_rmsd(None) == (None, None, None)


class TestCompute:
    def test_row_has_every_column(self):
        from MolecularDiffusion.runmodes.analyze.druglike import RING_SIZES, compute

        row = compute(_mol("CC(=O)Oc1ccccc1C(=O)O"))
        for key in ("QED", "SA_score", "LogP", "lipinski", "pains_pass",
                    "ring_filter_pass", "n_rings", "n_aromatic_rings",
                    "n_aliphatic_rings"):
            assert key in row, key
        for size in RING_SIZES:
            assert f"ring_size_{size}" in row
        # opt-in only
        assert "rdkit_rmsd_min" not in row

    def test_rdkit_rmsd_is_opt_in(self):
        from MolecularDiffusion.runmodes.analyze.druglike import compute

        row = compute(_mol("CCO"), with_rdkit_rmsd=True, n_conf=2)
        assert "rdkit_rmsd_min" in row
        assert "rdkit_rmsd_median" in row
        assert "rdkit_rmsd_max" in row


class TestSimilarityReferenceLoading:
    def test_unsupported_format_is_rejected(self, tmp_path):
        from MolecularDiffusion.runmodes.analyze.similarity3d import load_reference_source

        bad = tmp_path / "ref.txt"
        bad.write_text("nope")
        with pytest.raises(ValueError, match="Unsupported reference format"):
            load_reference_source(str(bad))

    def test_compare_without_reference_returns_zeros(self):
        from MolecularDiffusion.runmodes.analyze.similarity3d import compare

        scores = compare(_mol("CCO"), None)
        assert scores == {"shape_sim": 0.0, "pharm_sim": 0.0, "esp_sim": 0.0}
