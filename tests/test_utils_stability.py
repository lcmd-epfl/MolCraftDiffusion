"""Unit tests for MolecularDiffusion.utils.geom_stability."""

import pytest
from rdkit import Chem


# ---------------------------------------------------------------------------
# load_valency_table
# ---------------------------------------------------------------------------

class TestLoadValencyTable:
    def test_load_tuple_table(self):
        from MolecularDiffusion.utils.geom_stability import load_valency_table

        table = load_valency_table("tuple")
        assert isinstance(table, dict)
        assert len(table) > 0

    def test_load_legacy_table(self):
        from MolecularDiffusion.utils.geom_stability import load_valency_table

        table = load_valency_table("legacy")
        assert isinstance(table, dict)
        assert len(table) > 0

    def test_table_contains_carbon(self):
        from MolecularDiffusion.utils.geom_stability import load_valency_table

        table = load_valency_table("tuple")
        assert "C" in table

    def test_charge_keys_are_integers(self):
        from MolecularDiffusion.utils.geom_stability import load_valency_table

        table = load_valency_table("tuple")
        for element, charge_dict in table.items():
            for charge in charge_dict:
                assert isinstance(charge, int), f"charge key for {element} is not int"

    def test_tuple_entries_are_tuples_or_ints(self):
        from MolecularDiffusion.utils.geom_stability import load_valency_table

        table = load_valency_table("tuple")
        for element, charge_dict in table.items():
            for charge, valencies in charge_dict.items():
                if valencies:
                    first = valencies[0]
                    assert isinstance(first, (tuple, int, list))


# ---------------------------------------------------------------------------
# get_default_valencies / get_simple_valencies
# ---------------------------------------------------------------------------

class TestCachedValencies:
    def test_get_default_returns_same_object_twice(self):
        from MolecularDiffusion.utils.geom_stability import get_default_valencies

        t1 = get_default_valencies()
        t2 = get_default_valencies()
        assert t1 is t2  # cached

    def test_get_simple_returns_same_object_twice(self):
        from MolecularDiffusion.utils.geom_stability import get_simple_valencies

        t1 = get_simple_valencies()
        t2 = get_simple_valencies()
        assert t1 is t2


# ---------------------------------------------------------------------------
# is_valid
# ---------------------------------------------------------------------------

class TestIsValid:
    def _mol(self, smiles):
        return Chem.MolFromSmiles(smiles)

    def test_valid_simple_molecule(self):
        from MolecularDiffusion.utils.geom_stability import is_valid

        assert is_valid(self._mol("CCO")) is True

    def test_valid_aromatic(self):
        from MolecularDiffusion.utils.geom_stability import is_valid

        assert is_valid(self._mol("c1ccccc1")) is True

    def test_none_input_is_invalid(self):
        from MolecularDiffusion.utils.geom_stability import is_valid

        assert is_valid(None) is False

    def test_multifragment_is_invalid(self):
        from MolecularDiffusion.utils.geom_stability import is_valid

        # Two disconnected fragments → should fail validity
        mol = Chem.MolFromSmiles("CC.OO")
        assert is_valid(mol) is False

    def test_valid_heteroaromatic(self):
        from MolecularDiffusion.utils.geom_stability import is_valid

        assert is_valid(self._mol("c1ccncc1")) is True

    def test_valid_charged_molecule(self):
        from MolecularDiffusion.utils.geom_stability import is_valid

        assert is_valid(self._mol("[NH4+]")) is True
