"""Tests for the graph3d (3D molecular graph with explicit bonds) data layer.

The load-bearing check is (c): a molecule that survives
converter -> ASE db -> dataset -> RDKit rebuild with its canonical SMILES
intact proves the bond pipeline is lossless. Everything else guards an
invariant that, if broken, would corrupt training silently rather than crash.
"""

import os

import numpy as np
import pytest
import torch

# component.dataset first would hit a pre-existing circular import
# (component.dataset -> core -> runmodes.train.data -> data.dataset).
import MolecularDiffusion.data.dataset  # noqa: F401

from MolecularDiffusion.data.component.graph3d_dataset import (
    BOND_ORDER_TO_CLASS,
    N_BOND_CLASSES,
    Graph3DDataset,
    graph3d_dense_collate,
    build_rdkit_mol,
    remap_bonds_after_atom_removal,
)

Chem = pytest.importorskip("rdkit.Chem")


# --- fixtures ---------------------------------------------------------------

#: benzene exercises aromatic bonds; the ammonium exercises a formal charge;
#: ethanol exercises an H on a heteroatom (the O-H bond must vanish under
#: with_hydrogen=False without orphaning the O).
SMILES = ["c1ccccc1", "[NH4+]", "CCO", "CC(=O)Nc1ccccc1"]


def _embed(smiles):
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE) == 0
    Chem.SanitizeMol(mol)
    return mol


def _write_db(tmp_path, smiles_list=SMILES, split_cycle=("train", "val", "test")):
    from ase.db import connect

    from MolecularDiffusion.runmodes.data.graph3d_import import mol_to_row

    path = str(tmp_path / "graph3d.db")
    db = connect(path)
    with db:
        for i, smi in enumerate(smiles_list):
            atoms, data = mol_to_row(_embed(smi))
            data["source"] = "test"
            data["split"] = split_cycle[i % len(split_cycle)]
            db.write(atoms, data=data)
    return path


VOCAB = ["H", "C", "N", "O", "F"]


# --- (a) storage invariants -------------------------------------------------


def test_stored_bonds_are_upper_triangular_and_in_vocabulary(tmp_path):
    ds = Graph3DDataset.__new__(Graph3DDataset)
    ds.load_db(_write_db(tmp_path), atom_vocab=VOCAB, max_atom=64)

    assert len(ds) == len(SMILES)
    for i in range(len(ds)):
        g = ds[i]["graph"]
        bi, bt = g.bond_index, g.bond_type
        assert bi.shape[0] == 2
        assert (bi[0] < bi[1]).all(), "bond_index must be upper-triangular"
        assert bt.min() >= 1 and bt.max() < N_BOND_CLASSES
        assert bi.max() < g.n_nodes


def test_aromatic_class_is_populated(tmp_path):
    """Guards the sanitize decision: benzene must land as class 4, not Kekule."""
    ds = Graph3DDataset.__new__(Graph3DDataset)
    ds.load_db(_write_db(tmp_path, ["c1ccccc1"]), atom_vocab=VOCAB, max_atom=64)
    assert (ds[0]["graph"].bond_type == 4).sum() == 6


def test_kekulize_replaces_aromatic_with_alternating_orders(tmp_path):
    ds = Graph3DDataset.__new__(Graph3DDataset)
    ds.load_db(
        _write_db(tmp_path, ["c1ccccc1"]), atom_vocab=VOCAB, max_atom=64, kekulize=True
    )
    bt = ds[0]["graph"].bond_type
    assert (bt == 4).sum() == 0, "kekulize=True must leave no aromatic bonds"
    # 12 bonds total: 6 aromatic ring bonds -> 3 single + 3 double, plus the
    # 6 C-H bonds that were already single.
    assert (bt == 2).sum() == 3, "a kekulized ring alternates three doubles"
    assert (bt == 1).sum() == 9


def test_formal_charge_is_raw_and_signed(tmp_path):
    ds = Graph3DDataset.__new__(Graph3DDataset)
    ds.load_db(_write_db(tmp_path, ["[NH4+]"]), atom_vocab=VOCAB, max_atom=64)
    assert int(ds[0]["graph"].fc.sum()) == 1


# --- (b) hydrogen removal ---------------------------------------------------


def test_remap_bonds_after_atom_removal_is_monotone():
    # atoms 0..4, drop 1 and 3; bond (0,2) survives, (1,4) does not.
    keep = np.array([True, False, True, False, True])
    bi = np.array([[0, 1, 2], [2, 4, 4]])
    bt = np.array([1, 2, 3])
    new_bi, new_bt = remap_bonds_after_atom_removal(bi, bt, keep)
    assert new_bi.tolist() == [[0, 1], [1, 2]]
    assert new_bt.tolist() == [1, 3]
    assert (new_bi[0] < new_bi[1]).all()


def test_with_hydrogen_false_drops_h_and_keeps_bonds_valid(tmp_path):
    path = _write_db(tmp_path, ["CCO"])
    heavy = Graph3DDataset.__new__(Graph3DDataset)
    heavy.load_db(path, atom_vocab=["C", "N", "O", "F"], max_atom=64, with_hydrogen=False)

    g = heavy[0]["graph"]
    assert int(g.n_nodes) == 3, "CCO has three heavy atoms"
    assert g.bond_index.shape[1] == 2, "only C-C and C-O survive"
    assert g.bond_index.max() < g.n_nodes
    assert (g.bond_index[0] < g.bond_index[1]).all()
    assert g.fc.shape[0] == 3


# --- (c) the round-trip that actually proves correctness --------------------


def _canonical(mol_or_smiles):
    """Normalize both sides of a comparison through the same path.

    The stored SMILES comes from an explicit-H molecule and a rebuilt mol
    re-emits explicit H atoms, so the two differ in H *notation* while being
    the same molecule. Round-tripping both through ``CanonSmiles`` removes
    exactly that difference and nothing else.
    """
    smiles = (
        mol_or_smiles
        if isinstance(mol_or_smiles, str)
        else Chem.MolToSmiles(mol_or_smiles)
    )
    return Chem.CanonSmiles(smiles)


def _rebuild(graph):
    return build_rdkit_mol(
        graph.z.numpy(),
        graph.bond_index.numpy(),
        graph.bond_type.numpy(),
        graph.fc.numpy(),
        coords=graph.pos.numpy(),
    )


def test_smiles_survives_db_dataset_rebuild_roundtrip(tmp_path):
    ds = Graph3DDataset.__new__(Graph3DDataset)
    ds.load_db(_write_db(tmp_path), atom_vocab=VOCAB, max_atom=64)

    for i in range(len(ds)):
        g = ds[i]["graph"]
        rebuilt = _rebuild(g)
        assert _canonical(rebuilt) == _canonical(g.smiles), f"lossy for {g.smiles}"


def test_charged_atom_keeps_its_hydrogen_count(tmp_path):
    """Regression: building with SetNoImplicit(True) silently turned [NH3+]
    into [N+], which cost ~9% of QM9 in the round-trip."""
    ds = Graph3DDataset.__new__(Graph3DDataset)
    ds.load_db(
        _write_db(tmp_path, ["[NH4+]", "C[NH+]1CC1"]), atom_vocab=VOCAB, max_atom=64
    )
    for i in range(len(ds)):
        g = ds[i]["graph"]
        got = _canonical(_rebuild(g))
        assert "H" in got, f"hydrogen count lost on a charged atom: {got}"
        assert got == _canonical(g.smiles)


def test_stereocentre_is_recovered_from_coordinates(tmp_path):
    """Stereochemistry lives in the geometry, not the bond table, so it comes
    back only when coords are handed to the rebuild."""
    ds = Graph3DDataset.__new__(Graph3DDataset)
    ds.load_db(_write_db(tmp_path, ["C[C@H]1CO1"]), atom_vocab=VOCAB, max_atom=64)
    g = ds[0]["graph"]

    assert "@" in _canonical(_rebuild(g))
    without_coords = build_rdkit_mol(
        g.z.numpy(), g.bond_index.numpy(), g.bond_type.numpy(), g.fc.numpy()
    )
    assert "@" not in _canonical(without_coords)


# --- (d) collate ------------------------------------------------------------


def test_dense_collate_is_symmetric_with_zero_diagonal(tmp_path):
    ds = Graph3DDataset.__new__(Graph3DDataset)
    ds.load_db(_write_db(tmp_path), atom_vocab=VOCAB, max_atom=64)

    batch = graph3d_dense_collate([ds[i] for i in range(len(ds))])
    E = batch["bond_type"]
    assert E.shape[0] == len(ds) and E.shape[1] == E.shape[2]
    assert torch.equal(E, E.transpose(1, 2)), "MiDi asserts this itself"
    assert torch.equal(E.diagonal(dim1=1, dim2=2), torch.zeros_like(E[:, :, 0]))
    assert batch["node_mask"].sum(1).tolist() == batch["natoms"].tolist()

    # every real bond appears twice (once per triangle)
    for b in range(len(ds)):
        assert int((E[b] > 0).sum()) == 2 * ds[b]["graph"].bond_type.numel()


def test_raw_collate_offsets_bond_index_across_the_batch(tmp_path):
    """PyG only auto-increments keys containing 'index'. If this breaks, batched
    graphs are silently wrong rather than raising."""
    from MolecularDiffusion.data.dataloader import graph_collate

    ds = Graph3DDataset.__new__(Graph3DDataset)
    ds.load_db(_write_db(tmp_path, ["CCO", "CCO"]), atom_vocab=VOCAB, max_atom=64)

    batch = graph_collate([ds[0], ds[1]])["graph"]
    n0 = int(ds[0]["graph"].n_nodes)
    per_mol = ds[0]["graph"].bond_index.shape[1]
    assert torch.equal(batch.bond_index[:, per_mol:], ds[1]["graph"].bond_index + n0)


# --- (e) statistics ---------------------------------------------------------


def test_statistics_derive_the_no_bond_class_from_pair_counts(tmp_path):
    ds = Graph3DDataset.__new__(Graph3DDataset)
    ds.load_db(_write_db(tmp_path), atom_vocab=VOCAB, max_atom=64)

    stats = ds.graph3d_stats
    assert stats.n_molecules == len(SMILES)

    expected_none = sum(
        n * (n - 1) // 2 - ds[i]["graph"].bond_type.numel()
        for i, n in enumerate(int(ds[j]["graph"].n_nodes) for j in range(len(ds)))
    )
    assert int(stats.bond_type_counts[0]) == expected_none

    marginal = stats.bond_type_marginal()
    assert marginal.sum() == pytest.approx(1.0)
    assert marginal.argmax() == 0, "no-bond must dominate a sparse molecule"
    # directed is MiDi's convention; normalization makes the two identical
    assert stats.bond_type_marginal(directed=True) == pytest.approx(marginal)

    sizes, counts = stats.n_atoms_histogram()
    assert counts.sum() == len(SMILES)
    assert sizes.max() <= 64


# --- (f) persistence --------------------------------------------------------


def test_pickle_roundtrip_preserves_bonds_and_stats(tmp_path):
    ds = Graph3DDataset.__new__(Graph3DDataset)
    ds.load_db(_write_db(tmp_path), atom_vocab=VOCAB, max_atom=64)

    pkl = str(tmp_path / "cache.pt")
    ds.save_pickle(pkl)

    reloaded = Graph3DDataset.__new__(Graph3DDataset)
    reloaded.max_atom = 64
    reloaded.load_pickle(pkl)

    assert len(reloaded) == len(ds)
    assert reloaded.graph3d_stats.n_molecules == ds.graph3d_stats.n_molecules
    for i in range(len(ds)):
        assert torch.equal(reloaded[i]["graph"].bond_type, ds[i]["graph"].bond_type)
        assert torch.equal(reloaded[i]["graph"].bond_index, ds[i]["graph"].bond_index)
    assert reloaded.splits == ds.splits


def test_bond_order_map_covers_the_canonical_vocabulary():
    assert sorted(BOND_ORDER_TO_CLASS.values()) == [1, 2, 3, 4]


def test_chunked_and_unchunked_builds_agree(tmp_path):
    """Chunking must be a storage detail, never a data difference."""
    from MolecularDiffusion.data.component.dataset import LazyChunkedGraphDataset

    db = _write_db(tmp_path)
    chunk_dir = str(tmp_path / "chunks")

    plain = Graph3DDataset.__new__(Graph3DDataset)
    plain.load_db(db, atom_vocab=VOCAB, max_atom=64)

    chunked = Graph3DDataset.__new__(Graph3DDataset)
    chunked.load_db(db, atom_vocab=VOCAB, max_atom=64, chunk_size=2, chunk_dir=chunk_dir)

    meta = torch.load(os.path.join(chunk_dir, "meta.pt"), weights_only=False)
    assert meta["kind"] == "graph3d"
    assert meta["graph3d_stats"].n_molecules == plain.graph3d_stats.n_molecules

    lazy = LazyChunkedGraphDataset(chunk_dir)
    assert len(lazy) == len(plain)
    for i in range(len(plain)):
        a, b = plain[i]["graph"], lazy[i]["graph"]
        assert torch.equal(a.bond_index, b.bond_index)
        assert torch.equal(a.bond_type, b.bond_type)
        assert torch.equal(a.fc, b.fc)


# --- (g) DataModule integration --------------------------------------------


def _graph3d_module(tmp_path, db, name, **kwargs):
    from MolecularDiffusion.runmodes.train.data import DataModule

    return DataModule(
        root=str(tmp_path),
        filename=None,
        task_type="diffusion",
        atom_vocab=VOCAB,
        with_hydrogen=True,
        max_atom=64,
        ase_db_path=db,
        dataset_name=name,
        data_type="graph3d",
        batch_size=2,
        **kwargs,
    )


def test_graph3d_data_type_does_not_disturb_existing_cache_paths(tmp_path):
    """The two original data_types must keep their historic cache filenames, or
    every processed_data_*.pt already on disk is orphaned."""
    from MolecularDiffusion.runmodes.train.data import DataModule

    common = dict(
        root="data/",
        filename=None,
        task_type="diffusion",
        atom_vocab=VOCAB,
        with_hydrogen=True,
        max_atom=29,
    )
    for data_type in ("pointcloud", "pyg"):
        module = DataModule(dataset_name="qm9", data_type=data_type, **common)
        assert module.root_path == "data/processed_data_qm9.pt"
        assert module.chunk_dir == "data/chunks_qm9"

    graph3d = DataModule(dataset_name="qm9", data_type="graph3d", **common)
    assert graph3d.root_path == "data/processed_data_qm9_graph3d.pt"
    assert graph3d.chunk_dir == "data/chunks_qm9_graph3d"


def test_stored_split_is_honoured_and_falls_back_when_absent(tmp_path):
    os.makedirs(tmp_path / "a", exist_ok=True)
    os.makedirs(tmp_path / "b", exist_ok=True)
    labelled = _write_db(tmp_path / "a", split_cycle=("train", "train", "val", "test"))
    unlabelled = _write_db(tmp_path / "b", split_cycle=("",))

    m = _graph3d_module(tmp_path, labelled, "lab", use_stored_split=True)
    m.load()
    assert (len(m.train_set), len(m.valid_set), len(m.test_set)) == (2, 1, 1)
    assert {m.train_set.dataset.splits[i] for i in m.train_set.indices} == {"train"}

    # No labels -> must fall back rather than produce an empty train set.
    m2 = _graph3d_module(tmp_path, unlabelled, "unlab", use_stored_split=True)
    m2.load()
    assert len(m2.train_set) > 0


def test_datamodule_selects_the_requested_collate(tmp_path):
    from MolecularDiffusion.data.dataloader import graph_collate

    db = _write_db(tmp_path)
    assert _graph3d_module(tmp_path, db, "raw").collate_fn is graph_collate
    assert (
        _graph3d_module(tmp_path, db, "dense", bond_collate="dense").collate_fn
        is graph3d_dense_collate
    )
