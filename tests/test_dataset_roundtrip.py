"""Checks for the two silent-corruption paths in data/component/dataset.py."""

import torch

# Importing component.dataset first hits a pre-existing circular import
# (component.dataset -> core -> runmodes.train.data -> data.dataset). Importing
# data.dataset first resolves the cycle.
import MolecularDiffusion.data.dataset  # noqa: F401
from MolecularDiffusion.data.component.dataset import GraphDataset, _check_any_loaded


def _compacted_dataset(tmp_path):
    """A 2-molecule pyG dataset stored with pack_ohe + int8_z compaction."""
    ds = GraphDataset()
    ds.load_npy(
        coords=torch.tensor(
            [
                [0, 6, 0.0, 0.0, 0.0],
                [0, 8, 1.2, 0.0, 0.0],
                [1, 6, 0.0, 0.0, 0.0],
                [1, 7, 1.4, 0.0, 0.0],
            ]
        ),
        natoms=torch.tensor([2, 2]),
        smiles_list=["CO", "CN"],
        targets={"y": [1.0, 2.0]},
        atom_vocab=["C", "N", "O"],
        edge_type="fully_connected",
        compact={"pack_ohe": True, "int8_z": True},
    )
    return ds


def test_compaction_survives_pickle_roundtrip(tmp_path):
    ds = _compacted_dataset(tmp_path)
    before = ds[0]["graph"]
    assert before.x.shape == (2, 3), "expected expanded OHE from the live dataset"

    pkl = str(tmp_path / "ds.pt")
    ds.save_pickle(pkl)

    reloaded = GraphDataset()
    reloaded.load_pickle(pkl)
    after = reloaded[0]["graph"]

    # Without the compact metadata in the pickle this comes back as (2,) int8
    # argmax indices and the model silently trains on garbage.
    assert after.x.shape == before.x.shape
    assert torch.equal(after.x, before.x)
    assert after.atomic_numbers.dtype == torch.long


def test_check_any_loaded_raises_when_everything_was_discarded():
    try:
        _check_any_loaded("some source", 10, 0, None)
    except ValueError as exc:
        assert "some source" in str(exc)
    else:
        raise AssertionError("expected ValueError on a fully-discarded source")

    _check_any_loaded("some source", 10, 1, None)  # partial load is fine
    _check_any_loaded("some source", 0, 0, None)  # empty source is fine


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        test_compaction_survives_pickle_roundtrip(Path(tmp))
    test_check_any_loaded_raises_when_everything_was_discarded()
    print("ok")
