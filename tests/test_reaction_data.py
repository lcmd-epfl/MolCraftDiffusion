"""Tests for the shared reaction container and OA-ReactDiff's data layer.

The load-bearing checks are the ones that would corrupt training *silently*
rather than crash: a side whose tensors disagree with the reaction's atom
count (a mis-sized ``pos`` broadcasts; an out-of-range ``bond_index`` indexes
the wrong atoms), and the collate's object ordering, which binds each of
reactant / TS / product to its own encoder-decoder pair in the state dict --
swap two and the weights load fine and mean the wrong thing.

``bond_index`` being **directed and per-side** is asserted here on purpose:
it is the one place this container deliberately departs from the ``graph3d``
upper-triangular storage rule, and "fixing" it would racemize a future
stereochemistry-aware model's whole corpus.
"""

import os

import pytest
import torch

# component.dataset first would hit a pre-existing circular import
# (component.dataset -> core -> runmodes.train.data -> data.dataset).
import MolecularDiffusion.data.dataset  # noqa: F401

from MolecularDiffusion.data.component.oareactdiff_data import (
    FRAGMENT_ORDER,
    N_ELEMENT,
    TS_INDEX,
    OAReactDiffTS1xDataset,
    oareactdiff_collate,
)
from MolecularDiffusion.data.component.reaction_data import (
    N_BOND_CLASSES,
    Reaction,
    ReactionSide,
)

PKL = "docs/model_integrations/oareactdiff/data/valid_addprop.pkl"


def _reaction(n=4, ts=True):
    pos = torch.randn(n, 3)
    return Reaction(
        z=torch.tensor([6] * (n - 1) + [1]),
        reactant=ReactionSide(pos=pos),
        product=ReactionSide(pos=pos + 0.1),
        ts_pos=pos + 0.05 if ts else None,
        meta={"rxn": "rxn0", "smi_reactant": ["C"], "smi_product": ["C"]},
    )


def test_representation_is_derived_not_declared():
    """has_geometry / has_connectivity read the tensors, not a stored enum."""
    geometry_only = ReactionSide(pos=torch.randn(3, 3))
    assert geometry_only.has_geometry
    assert not geometry_only.has_connectivity

    topology_only = ReactionSide(
        bond_index=torch.tensor([[0, 1], [1, 2]]),
        bond_type=torch.tensor([1, 2]),
    )
    assert not topology_only.has_geometry
    assert topology_only.has_connectivity


def test_bond_index_stays_directed():
    """A directed edge and its reverse are both storable and both kept.

    Not a style point: a directed 3-cycle is how a stereochemistry-aware
    model encodes tetrahedral chirality, so canonicalising to i < j would
    flip R and S without erroring.
    """
    side = ReactionSide(
        bond_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),  # 0->1->2->0
        bond_type=torch.tensor([1, 1, 1]),
    )
    side.validate(3)
    assert side.bond_index.tolist() == [[0, 1, 2], [1, 2, 0]]


@pytest.mark.parametrize(
    ("side", "match"),
    [
        (ReactionSide(pos=torch.randn(9, 3)), "expected"),
        (ReactionSide(feat=torch.randn(9, 2)), "rows"),
        (ReactionSide(bond_index=torch.tensor([[0], [1]])), "both"),
        (
            ReactionSide(
                bond_index=torch.tensor([[0], [1]]),
                bond_type=torch.tensor([1, 2]),
            ),
            "edges",
        ),
        (
            ReactionSide(
                bond_index=torch.tensor([[0], [99]]),
                bond_type=torch.tensor([1]),
            ),
            "only",
        ),
        (
            ReactionSide(
                bond_index=torch.tensor([[0], [1]]),
                bond_type=torch.tensor([N_BOND_CLASSES]),
            ),
            "BOND_VOCAB",
        ),
    ],
)
def test_bad_side_is_rejected(side, match):
    with pytest.raises(ValueError, match=match):
        side.validate(4)


def test_ts_pos_is_validated_against_z():
    with pytest.raises(ValueError, match="ts_pos"):
        Reaction(
            z=torch.tensor([6, 1]),
            reactant=ReactionSide(pos=torch.randn(2, 3)),
            product=ReactionSide(pos=torch.randn(2, 3)),
            ts_pos=torch.randn(5, 3),
        )


def test_collate_layout_and_object_order():
    batch = [_reaction(4), _reaction(6)]
    representations, conditions = oareactdiff_collate(batch)

    assert len(representations) == len(FRAGMENT_ORDER) == 3
    assert FRAGMENT_ORDER[TS_INDEX] == "transition_state"
    assert conditions.shape == (2, 1)
    assert not conditions.any(), "the condition channel is inert (zeros)"

    for rep in representations:
        assert rep["size"].tolist() == [4, 6]
        assert rep["pos"].shape == (10, 3)
        assert rep["one_hot"].shape == (10, N_ELEMENT)
        assert rep["charge"].shape == (10, 1)
        # torch_scatter needs int64 segment ids restarting at 0 per batch.
        assert rep["mask"].dtype == torch.int64
        assert rep["mask"].tolist() == [0] * 4 + [1] * 6

    # Object 1 must be the TS, object 0 the reactant: the order binds each
    # object to its own encoder/decoder pair in the checkpoint.
    assert torch.equal(representations[0]["pos"][:4], batch[0].reactant.pos)
    assert torch.equal(representations[TS_INDEX]["pos"][:4], batch[0].ts_pos)
    assert torch.equal(representations[2]["pos"][:4], batch[0].product.pos)

    # Every object gets its own tensor: the normalizer mutates in place.
    assert (
        representations[0]["one_hot"].data_ptr()
        != representations[1]["one_hot"].data_ptr()
    )


def test_collate_refuses_a_missing_transition_state():
    with pytest.raises(ValueError, match="ts_pos"):
        oareactdiff_collate([_reaction(4, ts=False)])


@pytest.mark.skipif(
    not os.path.exists(PKL), reason="run scripts/convert_dataset.py first"
)
def test_published_split_and_swapping():
    """The split lives in `use_ind`, and the swapped half is the reverse."""
    dataset = OAReactDiffTS1xDataset(PKL, limit=8)
    assert len(dataset) == 16, "swapping_react_prod doubles the set"

    forward, backward = dataset[0], dataset[8]
    assert torch.equal(forward.reactant.pos, backward.product.pos)
    assert torch.equal(forward.product.pos, backward.reactant.pos)
    assert torch.equal(forward.ts_pos, backward.ts_pos)
    assert forward.meta["rxn"] == backward.meta["rxn"]

    # Per-object CoM centring, and the shared atom ordering the container
    # is built on.
    assert forward.reactant.pos.mean(0).abs().max() < 1e-5
    assert len(forward.reactant.pos) == len(forward.z) == len(forward.ts_pos)

    # `use_by_ind=False` widens the set: proof the split is the use_ind list,
    # not the file.
    everything = OAReactDiffTS1xDataset(PKL, use_by_ind=False)
    honoured = OAReactDiffTS1xDataset(PKL, use_by_ind=True)
    assert len(everything) > len(honoured)
