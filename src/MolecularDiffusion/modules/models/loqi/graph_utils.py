"""LoQI edge-graph construction: stereo augmentation + fully-connected edges.

Two functions, both ported verbatim (NVIDIA, Apache-2.0):

* :func:`derive_stereo_edges` -- ``data_processing/utils_data.py::add_stereo_bonds``.
  Emits LoQI's auxiliary edge classes 5-8, which encode E/Z and R/S. These are
  **not bonds**: they are directed (class 8 deliberately asymmetric -- the
  direction *is* the R/S signal), so the platform's symmetric,
  upper-triangular ``bond_index``/``bond_type`` storage cannot hold them. The
  approved plan therefore re-derives them per batch from the geometry, using the
  same ``AssignStereochemistryFrom3D`` that produced them upstream.

* :func:`make_graph_fully_connected` --
  ``src/megalodon/data/batch_preprocessor_conf.py:8-40``. Merges the directed
  real-bond + stereo edge list into a fully-connected directed edge list,
  materializing class 0 ("no bond") for every remaining pair. Where a stereo
  edge lands on a real bond, ``coalesce(reduce="min")`` on the bond list first,
  then ``reduce="max"`` against the zero-filled complete graph, means the bond
  order wins -- upstream's behaviour, preserved.

Canonical edge vocabulary here (9 classes):
``0=none 1=SINGLE 2=DOUBLE 3=TRIPLE 4=AROMATIC 5=E 6=Z 7=chirality(sym)
8=chirality(directed)``. Class 4 never occurs for LoQI: the pipeline kekulizes.
"""

from __future__ import annotations

import torch
from torch_geometric.utils import coalesce, dense_to_sparse, sort_edge_index

#: LoQI's 9-class edge vocabulary width.
N_EDGE_CLASSES = 9

#: ``(symmetric, directed)`` chirality classes, ``process_chembl3d.py:219-222``.
CHI_BONDS = (7, 8)


def _ez_bonds() -> dict:
    from rdkit import Chem

    return {Chem.BondStereo.STEREOE: 5, Chem.BondStereo.STEREOZ: 6}


def derive_stereo_edges(  # noqa: C901, PLR0912
    mol,
    chi_bonds: tuple[int, int] = CHI_BONDS,
    ez_bonds: dict | None = None,
    from_3D: bool = True,
) -> list[tuple[int, int, int]]:
    """RDKit mol (with a conformer) -> list of ``(i, j, class)`` stereo edges.

    Returns an empty list for an achiral, non-E/Z molecule. Duplicated entries
    are possible and intentional -- upstream re-emits a stereocentre's block
    once per incident bond, and the later ``coalesce`` deduplicates.
    """
    from rdkit import Chem

    ez_bonds = ez_bonds if ez_bonds is not None else _ez_bonds()
    result: list[tuple[int, int, int]] = []

    if from_3D and mol.GetNumConformers() > 0:
        Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
    else:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if bond.GetBondType() == Chem.BondType.DOUBLE and stereo in ez_bonds:
            idx_3, idx_4 = bond.GetStereoAtoms()
            atom_1, atom_2 = bond.GetBeginAtom(), bond.GetEndAtom()
            idx_1, idx_2 = atom_1.GetIdx(), atom_2.GetIdx()

            idx_5 = [
                n.GetIdx()
                for n in atom_1.GetNeighbors()
                if n.GetIdx() not in {idx_2, idx_3}
            ]
            idx_6 = [
                n.GetIdx()
                for n in atom_2.GetNeighbors()
                if n.GetIdx() not in {idx_1, idx_4}
            ]

            inv = (
                Chem.BondStereo.STEREOE
                if stereo == Chem.BondStereo.STEREOZ
                else Chem.BondStereo.STEREOZ
            )
            result.extend(
                [(idx_3, idx_4, ez_bonds[stereo]), (idx_4, idx_3, ez_bonds[stereo])]
            )
            if idx_5:
                result.extend(
                    [
                        (idx_5[0], idx_4, ez_bonds[inv]),
                        (idx_4, idx_5[0], ez_bonds[inv]),
                    ]
                )
            if idx_6:
                result.extend(
                    [
                        (idx_3, idx_6[0], ez_bonds[inv]),
                        (idx_6[0], idx_3, ez_bonds[inv]),
                    ]
                )
            if idx_5 and idx_6:
                result.extend(
                    [
                        (idx_5[0], idx_6[0], ez_bonds[stereo]),
                        (idx_6[0], idx_5[0], ez_bonds[stereo]),
                    ]
                )

        begin = bond.GetBeginAtom()
        if begin.HasProp("_CIPCode"):
            chirality = begin.GetProp("_CIPCode")
            neighbors = begin.GetNeighbors()
            if all(n.HasProp("_CIPRank") for n in neighbors):
                ranked = sorted(
                    neighbors, key=lambda a: int(a.GetProp("_CIPRank")), reverse=True
                )
                ranked = [a.GetIdx() for a in ranked]
                a, b, c = ranked[:3] if chirality == "R" else ranked[:3][::-1]
                d = ranked[-1]
                result.extend(
                    [
                        (a, d, chi_bonds[0]),
                        (b, d, chi_bonds[0]),
                        (c, d, chi_bonds[0]),
                        (d, a, chi_bonds[0]),
                        (d, b, chi_bonds[0]),
                        (d, c, chi_bonds[0]),
                        # The three DIRECTED class-8 edges. The direction is the
                        # R/S signal; do not symmetrize them.
                        (b, a, chi_bonds[1]),
                        (c, b, chi_bonds[1]),
                        (a, c, chi_bonds[1]),
                    ]
                )

    return result


def make_graph_fully_connected(
    edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor
):
    """Directed real+stereo edges -> fully-connected directed edges + 9 classes."""
    bond_edge_index, bond_edge_attr = sort_edge_index(
        edge_index=edge_index, edge_attr=edge_attr, sort_by_row=False
    )
    bond_edge_index, bond_edge_attr = coalesce(
        bond_edge_index, bond_edge_attr, reduce="min"
    )

    edge_index_global = (
        torch.eq(batch.unsqueeze(0), batch.unsqueeze(-1)).int().fill_diagonal_(0)
    )
    edge_index_global, _ = dense_to_sparse(edge_index_global)
    edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
    edge_attr_tmp = torch.zeros(
        edge_index_global.size(-1), device=edge_index_global.device, dtype=torch.long
    )

    edge_index_global = torch.cat([edge_index_global, bond_edge_index], dim=-1)
    edge_attr_tmp = torch.cat([edge_attr_tmp, bond_edge_attr], dim=0)
    edge_index_global, edge_attr_global = coalesce(
        edge_index_global, edge_attr_tmp, reduce="max"
    )
    return sort_edge_index(
        edge_index=edge_index_global, edge_attr=edge_attr_global, sort_by_row=False
    )
