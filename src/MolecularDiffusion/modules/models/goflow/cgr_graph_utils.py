"""Condensed-graph-of-reaction edge extension, vendored UNMODIFIED.

Ported verbatim from ``gotennet/utils/cgr_graph_utils.py`` (commit
``3ec00a09``). This is architecture-load-bearing logic -- the decode of the
packed ``edge_type = r_native * 22 + p_native`` union back into per-side
bond types, plus the higher-order (multi-hop, up to ``edge_order``)
connectivity and the runtime ``radius_graph`` union -- not glue, and is not
touched beyond the import path. See ``INTEGRATION_PLAN.md``, Repo Inspection
and Bond Representation Mapping, for the full derivation.

``BOND_TYPES`` here is RDKit's native 22-class ``BondType`` enum position
(``UNSPECIFIED=0`` shared with "no bond", ..., ``AROMATIC=12``, ...) -- the
scale ``edge_type`` on the batch is packed in, decoded by ``// 22`` / ``%
22`` below. This is a DIFFERENT vocabulary from the platform's canonical
5-class ``BOND_VOCAB`` (``graph3d_dataset.py``); the round trip between them
happens once, in ``data/component/goflow_data.py::goflow_collate``, never
here.
"""

from __future__ import annotations

import torch
from rdkit.Chem.rdchem import BondType
from torch_geometric.nn import radius, radius_graph
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_sparse import coalesce

BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}


def _extend_condensed_graph_edge(pos, bond_index, bond_type, batch, cutoff=5.0, edge_order=4):
    N = pos.size(0)
    edge_index_global, edge_index_local, edge_type_r, edge_type_p = extend_ts_graph_order_radius(
        N, pos, bond_index, bond_type, batch, order=edge_order, cutoff=cutoff
    )

    edge_type_global = torch.zeros_like(edge_index_global[0]) - 1
    adj_global = to_dense_adj(
        edge_index_global, edge_attr=edge_type_global, max_num_nodes=N
    )
    adj_local_r = to_dense_adj(
        edge_index_local, edge_attr=edge_type_r, max_num_nodes=N
    )
    adj_local_p = to_dense_adj(
        edge_index_local, edge_attr=edge_type_p, max_num_nodes=N
    )
    adj_global_r = torch.where(adj_local_r != 0, adj_local_r, adj_global)
    adj_global_p = torch.where(adj_local_p != 0, adj_local_p, adj_global)
    edge_index_global_r, edge_type_global_r = dense_to_sparse(adj_global_r)
    edge_index_global_p, edge_type_global_p = dense_to_sparse(adj_global_p)
    edge_type_global_r[edge_type_global_r < 0] = 0
    edge_type_global_p[edge_type_global_p < 0] = 0
    edge_index_global = edge_index_global_r

    return edge_index_global, edge_index_local, edge_type_global_r, edge_type_global_p


def extend_ts_graph_order_radius(
    num_nodes, pos, edge_index, edge_type, batch, order=3, cutoff=10.0,
):
    edge_index_local, edge_type_r, edge_type_p = _extend_ts_graph_order(
        num_nodes, edge_index, edge_type, batch, order=order
    )
    edge_index_global, _ = _extend_to_radius_graph(
        pos, edge_index_local, edge_type_r, cutoff, batch
    )
    return edge_index_global, edge_index_local, edge_type_r, edge_type_p


def _extend_ts_graph_order(num_nodes, edge_index, edge_type, batch, order=3):  # noqa: ARG001
    def binarize(x):
        return (x > 0).float()

    def get_higher_order_adj_matrix(adj, order):
        adj_mats = [
            torch.eye(adj.size(0), dtype=torch.long, device=adj.device),
            binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device)),
        ]
        for i in range(2, order + 1):
            adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
        order_mat = torch.zeros_like(adj)
        for i in range(1, order + 1):
            order_mat += (adj_mats[i] - adj_mats[i - 1]) * i
        return order_mat

    num_types = len(BOND_TYPES)
    N = num_nodes

    bond_type_r = edge_type // num_types
    mask_r = bond_type_r != 0
    bond_index_r = edge_index[:, mask_r]
    bond_type_r = bond_type_r[mask_r]

    bond_type_p = edge_type % num_types
    mask_p = bond_type_p != 0
    bond_index_p = edge_index[:, mask_p]
    bond_type_p = bond_type_p[mask_p]

    adj_r = to_dense_adj(bond_index_r, max_num_nodes=N).squeeze(0)
    adj_order_r = get_higher_order_adj_matrix(adj_r, order)
    type_mat_r = to_dense_adj(bond_index_r, edge_attr=bond_type_r, max_num_nodes=N).squeeze(0)
    type_highorder_r = torch.where(
        adj_order_r > 1, num_types + adj_order_r - 1, torch.zeros_like(adj_order_r)
    )
    assert (type_mat_r * type_highorder_r == 0).all()
    type_new_r = type_mat_r + type_highorder_r
    type_mask_r = -(type_new_r != 0).to(torch.float)

    adj_p = to_dense_adj(bond_index_p, max_num_nodes=N).squeeze(0)
    adj_order_p = get_higher_order_adj_matrix(adj_p, order)
    type_mat_p = to_dense_adj(bond_index_p, edge_attr=bond_type_p, max_num_nodes=N).squeeze(0)
    type_highorder_p = torch.where(
        adj_order_p > 1, num_types + adj_order_p - 1, torch.zeros_like(adj_order_p)
    )
    assert (type_mat_p * type_highorder_p == 0).all()
    type_new_p = type_mat_p + type_highorder_p
    type_mask_p = -(type_new_p != 0).to(torch.float)

    type_r = torch.where(type_new_r != 0, type_new_r, type_mask_p).to(torch.long)
    type_p = torch.where(type_new_p != 0, type_new_p, type_mask_r).to(torch.long)

    edge_index_r, edge_type_r = dense_to_sparse(type_r)
    edge_index_p, edge_type_p = dense_to_sparse(type_p)
    edge_type_r[edge_type_r < 0] = 0
    edge_type_p[edge_type_p < 0] = 0

    assert (edge_index_r == edge_index_p).all()

    edge_index_local, edge_type_r = coalesce(edge_index_r, edge_type_r.long(), N, N)
    _, edge_type_p = coalesce(edge_index_p, edge_type_p.long(), N, N)

    return edge_index_local, edge_type_r, edge_type_p


def _extend_to_radius_graph(
    pos, edge_index, edge_type, cutoff, batch, unspecified_type_number=0, is_sidechain=None,
):
    assert edge_type.dim() == 1
    N = pos.size(0)

    bgraph_adj = torch.sparse_coo_tensor(
        edge_index, edge_type, torch.Size([N, N]), dtype=torch.long, device=pos.device
    )

    if is_sidechain is None:
        rgraph_edge_index = radius_graph(pos, r=cutoff, batch=batch)
    else:
        is_sidechain = is_sidechain.bool()
        dummy_index = torch.arange(pos.size(0), device=pos.device)
        sidechain_pos = pos[is_sidechain]
        sidechain_index = dummy_index[is_sidechain]
        sidechain_batch = batch[is_sidechain]

        assign_index = radius(
            x=pos, y=sidechain_pos, r=cutoff, batch_x=batch, batch_y=sidechain_batch
        )
        r_edge_index_x = assign_index[1]
        r_edge_index_y = sidechain_index[assign_index[0]]

        rgraph_edge_index1 = torch.stack((r_edge_index_x, r_edge_index_y))
        rgraph_edge_index2 = torch.stack((r_edge_index_y, r_edge_index_x))
        rgraph_edge_index = torch.cat((rgraph_edge_index1, rgraph_edge_index2), dim=-1)
        rgraph_edge_index = rgraph_edge_index[:, (rgraph_edge_index[0] != rgraph_edge_index[1])]

    rgraph_adj = torch.sparse_coo_tensor(
        indices=rgraph_edge_index,
        values=torch.ones(rgraph_edge_index.size(1), dtype=torch.long, device=pos.device)
        * unspecified_type_number,
        size=torch.Size([N, N]),
        dtype=torch.long,
        device=pos.device,
    )

    composed_adj = (bgraph_adj + rgraph_adj).coalesce()
    new_edge_index = composed_adj.indices()
    new_edge_type = composed_adj.values().long()

    return new_edge_index, new_edge_type
