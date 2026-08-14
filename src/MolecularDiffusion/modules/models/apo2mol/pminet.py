"""PMINet -- Apo2Mol's frozen protein-molecule interaction prior.

Ported from ``others/Apo2Mol/graphbap/bapnet.py``. Its per-node 128-d output
is folded into every token embedding (``score_model.py``, the ``hbap_*``
arguments) and re-computed from the current prediction at every reverse step.

**This is NOT IPDiff's IPNet, despite both being called ``BAPNet``.**
Verified before porting: the two checkpoints hold 55 vs 58 tensors over
disjoint key sets, and 0 of the 45 same-named tensors are byte-identical.
Structurally, Apo2Mol's complex/geometry/fusion stages are ``GATConv``
attention layers while IPDiff's are EGNN ``GCL`` stacks. Hence a separate
module rather than an import of ``modules/models/ipdiff/bapnet.py``.

The equivariant ligand/pocket blocks (``GCL`` / ``EquivariantUpdate`` /
``EquivariantBlock``) are the usual EGNN pieces and are ported here too, so
this package has no cross-model import.

Not ported: ``SinusoidsEmbeddingNew`` is instantiated only when
``sin_embedding=True``, which the release never sets (``edge_feat_nf`` is 2 in
the checkpoint, the ``sin_embedding=False`` branch); ``check_memory`` is a
debug print.
"""

from __future__ import annotations

import logging

import os
from typing import List, Optional

import torch
from torch import nn
from torch_geometric.nn import GATConv
from torch_scatter import scatter_mean

logger = logging.getLogger(__name__)

#: ``bapnet.py:8-11`` -- the embedding tables' vocabularies. The ligand table
#: is indexed by the same 13-class (element, aromatic) index the diffusion
#: model uses; the pocket tables by argmax of the 27-dim protein feature's
#: element block (first 6) and amino-acid block (next 20). All three get one
#: extra row for an unused padding index.
LIGAND_ATOM_ADD_AROMATIC_TYPES = [
    "H", "C1", "C2", "N1", "N2", "O1", "O2", "F", "P1", "P2", "S1", "S2", "Cl",
]
POCKET_ATOM_TYPES = ["H", "C", "N", "O", "S", "Se"]
RESIDUE_TYPES = [
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
    "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR",
]


def get_edges(mask, x=None, edge_cutoff=None):
    """Dense within-graph edge list. NB: fully connected per graph."""
    adj = mask[:, None] == mask[None, :]
    if edge_cutoff is not None:
        adj = adj & (torch.cdist(x, x) <= float(edge_cutoff))
    return torch.stack(torch.where(adj), dim=0)


def remove_mean_batch_ligand(x_lig, x_pocket, lig_indices, pocket_indices):
    """Centre ligand and pocket on their OWN per-graph centroids, separately."""
    lig_mean = scatter_mean(x_lig, lig_indices, dim=0)
    pocket_mean = scatter_mean(x_pocket, pocket_indices, dim=0)
    return x_lig - lig_mean[lig_indices], x_pocket - pocket_mean[pocket_indices]


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    return radial, coord_diff / (norm + norm_constant)


def unsorted_segment_sum(
    data, segment_ids, num_segments, normalization_factor, aggregation_method
):
    result = data.new_full((num_segments, data.size(1)), 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == "sum":
        result = result / normalization_factor
    if aggregation_method == "mean":
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


class GCL(nn.Module):
    """EGNN scalar message-passing layer (``bapnet.py:236``)."""

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        normalization_factor,
        aggregation_method,
        edges_in_d=0,
        nodes_att_dim=0,
        act_fn=None,
        attention=False,
    ) -> None:
        super().__init__()
        act_fn = act_fn if act_fn is not None else nn.SiLU()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )
        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)
        out = mij * self.att_mlp(mij) if self.attention else mij
        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, _col = edge_index
        agg = unsorted_segment_sum(
            edge_attr,
            row,
            num_segments=x.size(0),
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method,
        )
        agg = (
            torch.cat([x, agg, node_attr], dim=1)
            if node_attr is not None
            else torch.cat([x, agg], dim=1)
        )
        return x + self.node_mlp(agg), agg

    def forward(
        self, h, edge_index, edge_attr=None, node_attr=None,
        node_mask=None, edge_mask=None,
    ):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, _agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij


class EquivariantUpdate(nn.Module):
    """EGNN coordinate update (``bapnet.py:299``)."""

    def __init__(
        self,
        hidden_nf,
        normalization_factor,
        aggregation_method,
        edges_in_d=1,
        act_fn=None,
        tanh=False,
        coords_range=10.0,
    ) -> None:
        super().__init__()
        act_fn = act_fn if act_fn is not None else nn.SiLU()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer,
        )
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(
        self, h, coord, edge_index, coord_diff, edge_attr, edge_mask,
        update_coords_mask=None,
    ):
        row, _col = edge_index
        input_tensor = torch.cat([h[row], h[_col], edge_attr], dim=1)
        if self.tanh:
            trans = (
                coord_diff
                * torch.tanh(self.coord_mlp(input_tensor))
                * self.coords_range
            )
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(
            trans,
            row,
            num_segments=coord.size(0),
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method,
        )
        if update_coords_mask is not None:
            agg = update_coords_mask * agg
        return coord + agg

    def forward(
        self, h, coord, edge_index, coord_diff, edge_attr=None,
        node_mask=None, edge_mask=None, update_coords_mask=None,
    ):
        coord = self.coord_model(
            h, coord, edge_index, coord_diff, edge_attr, edge_mask,
            update_coords_mask=update_coords_mask,
        )
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):
    """``n_layers`` GCLs then one coordinate update (``bapnet.py:345``)."""

    def __init__(
        self,
        hidden_nf,
        edge_feat_nf=2,
        act_fn=None,
        n_layers=2,
        attention=True,
        norm_diff=True,
        tanh=False,
        coords_range=15,
        norm_constant=1,
        sin_embedding=None,
        normalization_factor=100,
        aggregation_method="sum",
    ) -> None:
        super().__init__()
        act_fn = act_fn if act_fn is not None else nn.SiLU()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        for i in range(n_layers):
            self.add_module(
                f"gcl_{i}",
                GCL(
                    hidden_nf, hidden_nf, hidden_nf,
                    edges_in_d=edge_feat_nf,
                    act_fn=act_fn,
                    attention=attention,
                    normalization_factor=normalization_factor,
                    aggregation_method=aggregation_method,
                ),
            )
        self.add_module(
            "gcl_equiv",
            EquivariantUpdate(
                hidden_nf,
                edges_in_d=edge_feat_nf,
                act_fn=nn.SiLU(),
                tanh=tanh,
                coords_range=self.coords_range_layer,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
            ),
        )

    def forward(
        self, h, x, edge_index, node_mask=None, edge_mask=None,
        edge_attr=None, update_coords_mask=None,
    ):
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        for i in range(self.n_layers):
            h, _ = self._modules[f"gcl_{i}"](
                h, edge_index, edge_attr=edge_attr,
                node_mask=node_mask, edge_mask=edge_mask,
            )
        x = self._modules["gcl_equiv"](
            h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask,
            update_coords_mask=update_coords_mask,
        )
        if node_mask is not None:
            h = h * node_mask
        return h, x


class BAPNet(nn.Module):
    """Apo2Mol's PMINet (``bapnet.py:44``).

    Unlike IPDiff's same-named class, ``ckpt_path`` is **optional** here: the
    prior's weights live under ``net_cond.*`` in Apo2Mol's own released
    checkpoint, so at generate time they arrive with the rest of the task and
    there is nothing to pre-load. Pass ``ckpt_path`` when training from
    scratch, where the prior is supposed to start from the released PMINet
    weights (upstream's ``train_pl.py:81`` always does).

    The prior is not registered with the optimiser upstream
    (``pl_model.py:configure_optimizers`` passes ``self.model`` only), so it
    is effectively frozen while still being checkpointed. The task mirrors
    that by calling :meth:`freeze`.
    """

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        hidden_nf: int = 128,
        act_fn=None,
        GAT_head: int = 2,  # noqa: N803 - upstream's argument name
        graph_layers: int = 1,
        attention: bool = False,
        norm_diff: bool = True,
        tanh: bool = False,
        coords_range: int = 15,
        norm_constant: int = 1,
        inv_sublayers: int = 1,
        sin_embedding: bool = False,
        normalization_factor: int = 100,
        aggregation_method: str = "sum",
        edge_cutoff=None,
        ignore_keys: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        act_fn = act_fn if act_fn is not None else nn.SiLU()
        graph_dim = hidden_nf
        self.graph_dim = graph_dim
        self.hidden_nf = hidden_nf
        self.graph_layers = graph_layers

        self.ligand_atom_type_embed = nn.Embedding(
            len(LIGAND_ATOM_ADD_AROMATIC_TYPES) + 1, graph_dim
        )
        self.pocket_atom_type_embed = nn.Embedding(
            len(POCKET_ATOM_TYPES) + 1, graph_dim
        )
        self.pocket_residue_type_embed = nn.Embedding(
            len(RESIDUE_TYPES) + 1, graph_dim
        )
        self.pocket_type_fusion = nn.Linear(graph_dim * 2, graph_dim)

        self.id_embed = nn.Embedding(2, 4)
        self.embed_fusion = nn.Linear(graph_dim + 4, graph_dim)

        self.edge_cutoff = edge_cutoff
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        if sin_embedding:
            raise NotImplementedError(
                "sin_embedding=True is not ported: the released PMINet was "
                "trained with edge_feat_nf=2, i.e. sin_embedding=False."
            )
        self.sin_embedding = None
        edge_feat_nf = 2

        def _block():
            return EquivariantBlock(
                hidden_nf,
                edge_feat_nf=edge_feat_nf,
                act_fn=act_fn,
                n_layers=inv_sublayers,
                attention=attention,
                norm_diff=norm_diff,
                tanh=tanh,
                coords_range=coords_range,
                norm_constant=norm_constant,
                sin_embedding=self.sin_embedding,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method,
            )

        self.ComplexesGraph = nn.ModuleList(
            [GATConv(graph_dim, graph_dim, GAT_head, concat=False)]
        )
        self.LigandGraph = nn.ModuleList([_block()])
        self.PocketGraph = nn.ModuleList([_block()])

        for _ in range(graph_layers - 1):
            self.ComplexesGraph.append(
                GATConv(graph_dim, graph_dim, GAT_head, concat=False)
            )
            self.LigandGraph.append(_block())
            self.PocketGraph.append(_block())

        self.GeoGraph = nn.ModuleList(
            [GATConv(graph_dim, graph_dim, GAT_head, concat=False)]
        )
        self.FusionGraph = nn.ModuleList(
            [GATConv(graph_dim * 2, graph_dim, GAT_head, concat=False)]
        )

        self.OutputLayer = nn.Sequential(
            nn.Linear(graph_dim, graph_dim),
            nn.Hardswish(),
            nn.Linear(graph_dim, graph_dim),
        )
        self.FinalOutput = nn.Linear(graph_dim, 1)

        if ckpt_path:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys or [])

    def freeze(self) -> None:
        """Upstream never adds these parameters to the optimiser."""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def init_from_ckpt(self, path: str, ignore_keys: List[str]) -> None:
        """Load a converted PMINet state dict.

        Expects the output of
        ``docs/model_integrations/apo2mol/scripts/convert_checkpoint.py``,
        NOT the raw ``others/Apo2Mol/pretrained_models/PMINet``: the raw file
        was written by an older ``torch_geometric`` whose ``GATConv`` split
        its input projection into ``lin_src`` / ``lin_dst`` (an alias of the
        same tensor). The converter proves they are byte-identical and folds
        them into today's single ``lin``.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"PMINet weights not found: {path}. Convert them first with "
                "docs/model_integrations/apo2mol/scripts/convert_checkpoint.py"
            )
        blob = torch.load(path, map_location="cpu", weights_only=False)
        sd = blob.get("state_dict", blob)
        for k in list(sd):
            if any(k.startswith(ik) for ik in ignore_keys):
                del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        logger.info(
            "PMINet restored from %s: %d missing, %d unexpected",
            path, len(missing), len(unexpected),
        )
        if missing or unexpected:
            logger.warning(
                "PMINet key mismatch -- the prior is Apo2Mol's whole "
                "conditioning signal, so a large count means it is running on "
                "random weights. missing=%s unexpected=%s",
                missing[:5], unexpected[:5],
            )

    def extract_features(
        self, lig_coords, pocket_coords, lig_a_hidx, pocket_a_hidx,
        pocket_r_hidx, lig_mask, pocket_mask,
    ):
        """Per-node 128-d interaction features for ligand and pocket.

        Returns ``(ligand_feats, pocket_feats)``, both detached: this is a
        frozen prior, gradients never flow back into it.

        # ponytail: `get_edges` builds a DENSE within-graph adjacency, so cost
        # is O((n_lig + n_pocket)^2) per complex, same as upstream. That is the
        # memory ceiling on batch_size; sparsify with `edge_cutoff` if it bites.
        """
        lig_coords, pocket_coords = remove_mean_batch_ligand(
            lig_coords, pocket_coords, lig_mask, pocket_mask
        )
        device = lig_coords.device
        num_lig = lig_coords.shape[0]

        complexes_mask = torch.cat([lig_mask, pocket_mask], dim=0)
        complexes_id = torch.cat(
            [
                torch.zeros(num_lig, dtype=torch.long, device=device),
                torch.ones(
                    pocket_coords.shape[0], dtype=torch.long, device=device
                ),
            ]
        )

        complexes_id_emb = self.id_embed(complexes_id)
        lig_atom_type_emb = self.ligand_atom_type_embed(lig_a_hidx)
        pocket_type_emb = self.pocket_type_fusion(
            torch.cat(
                [
                    self.pocket_atom_type_embed(pocket_a_hidx),
                    self.pocket_residue_type_embed(pocket_r_hidx),
                ],
                dim=1,
            )
        )
        complexes_type_emb = torch.cat([lig_atom_type_emb, pocket_type_emb], dim=0)
        complexes_emb = self.embed_fusion(
            torch.cat([complexes_type_emb, complexes_id_emb], dim=-1)
        )

        complexes_edge_index = get_edges(mask=complexes_mask).to(device)
        ligand_edge_index = get_edges(mask=lig_mask).to(device)
        pocket_edge_index = get_edges(mask=pocket_mask).to(device)

        lig_emb, pocket_emb = complexes_emb[:num_lig], complexes_emb[num_lig:]

        pocket_distances, _ = coord2diff(pocket_coords, pocket_edge_index)
        lig_distances, _ = coord2diff(lig_coords, ligand_edge_index)

        o_c, o_l, o_p = complexes_emb, lig_emb, pocket_emb
        for i in range(self.graph_layers):
            o_c = self.ComplexesGraph[i](o_c, complexes_edge_index)
            o_l, lig_coords = self.LigandGraph[i](
                o_l, lig_coords, ligand_edge_index,
                node_mask=None, edge_mask=None,
                edge_attr=lig_distances, update_coords_mask=None,
            )
            o_p, pocket_coords = self.PocketGraph[i](
                o_p, pocket_coords, pocket_edge_index,
                node_mask=None, edge_mask=None,
                edge_attr=pocket_distances, update_coords_mask=None,
            )

        o_lp = self.GeoGraph[0](
            torch.cat([o_l, o_p], dim=0), complexes_edge_index
        )
        o_c = self.FusionGraph[0](
            torch.cat([o_c, o_lp], dim=1), complexes_edge_index
        )

        return o_c[:num_lig].detach(), o_c[num_lig:].detach()
