"""Upstream's RDKit node/edge featurization (``mol_utils/featurization.py``).

``featurize_mol`` produces the exact ``x`` layout DMT was trained on. Getting a
single column wrong is silent: the model still runs and still emits plausible
coordinates.

Layout of ``x`` (QM9: 5 + 39 = 44 columns; GEOM-Drugs: 35 + 39 = 74):

===========  =====  ===========================================================
columns      width  content
===========  =====  ===========================================================
0..T         |T|    atom-symbol one-hot over ``qm9_types`` / ``drugs_types``
+0            1     atomic number, as a RAW INTEGER (not one-hot, not scaled)
+1            1     is-aromatic flag
+2..+9        8     degree, ``one_k_encoding(.., [0..6])`` -> 7 + 1 catch-all
+10..+15      6     hybridization over SP/SP2/SP3/SP3D/SP3D2 -> 5 + 1
+16..+23      8     implicit valence over [0..6] -> 7 + 1
+24..+27      4     FORMAL CHARGE over [-1, 0, +1] -> 3 + 1 catch-all
+28..+33      6     in-ring-of-size 3,4,5,6,7,8 flags
+34..+38      5     number of rings the atom is in, over [0,1,2,3] -> 4 + 1
===========  =====  ===========================================================

``one_k_encoding`` puts anything outside ``choices`` in the LAST slot, which is
how a formal charge of, say, +2 is represented -- there is no separate charge
head, charges are input-only conditioning.

Bond classes are upstream's four: ``SINGLE=0, DOUBLE=1, TRIPLE=2, AROMATIC=3``.
"No bond" is NOT class 0 -- it is the all-zero 4-vector that
``to_dense_adj`` leaves on every non-bonded node pair. The platform's canonical
vocabulary maps onto it as ``nextmol_col = canonical_class - 1``, a pass-through
of the "class 0 is never materialized" storage rule.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT

__all__ = [
    "BOND_CLASSES",
    "atom_types_for",
    "drugs_types",
    "featurize_mol",
    "qm9_types",
]

#: upstream ``featurization.py:19``. Four classes; index 0 is SINGLE, not "none".
BOND_CLASSES = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

qm9_types = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}

drugs_types = {
    "H": 0, "Li": 1, "B": 2, "C": 3, "N": 4, "O": 5, "F": 6, "Na": 7, "Mg": 8,
    "Al": 9, "Si": 10, "P": 11, "S": 12, "Cl": 13, "K": 14, "Ca": 15, "V": 16,
    "Cr": 17, "Mn": 18, "Cu": 19, "Zn": 20, "Ga": 21, "Ge": 22, "As": 23,
    "Se": 24, "Br": 25, "Ag": 26, "In": 27, "Sb": 28, "I": 29, "Gd": 30,
    "Pt": 31, "Au": 32, "Hg": 33, "Bi": 34,
}  # fmt: skip

_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]


def atom_types_for(dataset: str) -> dict:
    """``'qm9' | 'drugs'`` -> the symbol->index map DMT was trained with."""
    if dataset in ("qm9", "QM9-df", "QM9-jodo"):
        return qm9_types
    if dataset in ("drugs", "geom", "Geom-drugs-df", "Geom-drugs-jodo"):
        return drugs_types
    msg = f"Unknown dataset {dataset!r}; expected 'qm9' or 'drugs'."
    raise ValueError(msg)


def one_k_encoding(value, choices: list) -> list[int]:
    """One-hot with a trailing catch-all slot for out-of-vocabulary values."""
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def featurize_mol(mol, types=drugs_types):
    """RDKit mol -> ``(x, z, edge_index, edge_attr)``.

    ``edge_index`` is **directed, both directions** ``(2, 2E)`` over real bonds
    only, with the same ``edge_attr`` on both -- which is how upstream enforces
    bond symmetry. There is no symmetry assertion anywhere downstream, so an
    unmirrored edge list would silently produce an asymmetric dense adjacency.
    """
    if isinstance(types, str):
        types = atom_types_for(types)

    n = mol.GetNumAtoms()
    atom_type_idx, atomic_number, atom_features = [], [], []
    ring = mol.GetRingInfo()
    for i, atom in enumerate(mol.GetAtoms()):
        atom_type_idx.append(types[atom.GetSymbol()])
        atomic_number.append(atom.GetAtomicNum())
        atom_features.extend(
            [atom.GetAtomicNum(), 1 if atom.GetIsAromatic() else 0]
        )
        atom_features.extend(one_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))
        atom_features.extend(one_k_encoding(atom.GetHybridization(), _HYBRIDIZATIONS))
        atom_features.extend(
            one_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])
        )
        atom_features.extend(one_k_encoding(atom.GetFormalCharge(), [-1, 0, 1]))
        atom_features.extend(
            [int(ring.IsAtomInRingOfSize(i, s)) for s in (3, 4, 5, 6, 7, 8)]
        )
        atom_features.extend(one_k_encoding(int(ring.NumAtomRings(i)), [0, 1, 2, 3]))

    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_CLASSES[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long).reshape(2, -1)
    edge_attr = F.one_hot(
        torch.tensor(edge_type, dtype=torch.long), num_classes=len(BOND_CLASSES)
    ).to(torch.float)

    x1 = F.one_hot(torch.tensor(atom_type_idx), num_classes=len(types))
    x2 = torch.tensor(atom_features).view(n, -1)
    x = torch.cat([x1.to(torch.float), x2.to(torch.float)], dim=-1)
    return x, z, edge_index, edge_attr
