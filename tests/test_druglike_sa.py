"""SA_score must be computed on the heavy-atom graph, not on explicit H atoms."""

import math

import pytest

pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402

from MolecularDiffusion.utils.geom_metrics import compute_drug_likeness  # noqa: E402
from MolecularDiffusion.utils.sascore import calculateScore  # noqa: E402

SMILES = "COc1ccc(-c2cc(C(N)=O)c3ccccc3n2)cc1"


def test_sa_score_ignores_explicit_hydrogens():
    heavy = Chem.MolFromSmiles(SMILES)
    with_h = Chem.AddHs(heavy)
    sa = compute_drug_likeness(with_h)["SA_score"]
    assert sa == pytest.approx(calculateScore(heavy))
    assert 1.0 <= sa < 4.0  # with explicit H this molecule scored ~7


def test_scscore_is_nan_without_model(monkeypatch):
    from MolecularDiffusion.runmodes.analyze import druglike
    from MolecularDiffusion.runmodes.data import preparation

    monkeypatch.setattr(preparation, "get_scscore_model", lambda: None)
    assert math.isnan(druglike._scscore(Chem.MolFromSmiles("CCO")))


def test_scscore_reuses_given_mol_and_keeps_explicit_hydrogens(monkeypatch):
    # Unlike SA_score, the established procedure for this project scores
    # SCScore with explicit H atoms kept (removeHs=False) -- see
    # synthesis_and_novelty_metrics_procedure.md.
    from MolecularDiffusion.runmodes.analyze import druglike
    from MolecularDiffusion.runmodes.data import preparation

    class Fake:
        def get_score_from_smi(self, smi):
            assert "[H]" in smi
            return smi, 2.5

    monkeypatch.setattr(preparation, "get_scscore_model", lambda: Fake())
    assert druglike._scscore(Chem.AddHs(Chem.MolFromSmiles("CCO"))) == 2.5
