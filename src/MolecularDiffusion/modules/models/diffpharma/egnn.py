"""Three-graph EGNN backbone for DiffPharma.

Faithful port of ``others/DiffPharma/equivariant_diffusion/egnn_new.py`` with
import rewrites and dead code (the ``GNN`` variant, ``EGNN.noh_skip``, the
commented-out debug scaffolding) removed. Parameter names and module
attribute names are unchanged so the released checkpoint maps 1:1.

Aggregation uses plain ``Tensor.scatter_add_`` -- no torch_scatter here.
"""

import math

import torch
from torch import nn


def unsorted_segment_sum(
    data, segment_ids, num_segments, normalization_factor, aggregation_method
):
    """TensorFlow-style ``unsorted_segment_sum`` ('sum' or 'mean')."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
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


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff / (norm + norm_constant)
    return radial, coord_diff


def coord2cross(x, edge_index, batch_mask, norm_constant=1):
    mean = unsorted_segment_sum(
        x,
        batch_mask,
        num_segments=batch_mask.max() + 1,
        normalization_factor=None,
        aggregation_method="mean",
    )
    row, col = edge_index
    cross = torch.cross(
        x[row] - mean[batch_mask[row]], x[col] - mean[batch_mask[col]], dim=1
    )
    norm = torch.linalg.norm(cross, dim=1, keepdim=True)
    return cross / (norm + norm_constant)


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15.0, min_res=15.0 / 2000.0, div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = (
            2 * math.pi * div_factor ** torch.arange(self.n_frequencies) / max_res
        )
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


class GCL(nn.Module):
    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        normalization_factor,
        aggregation_method,
        edges_in_d=0,
        nodes_att_dim=0,
        act_fn=nn.SiLU(),
        attention=False,
    ):
        super().__init__()
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
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        return x + self.node_mlp(agg), agg

    def forward(
        self,
        h,
        edge_index,
        edge_attr=None,
        node_attr=None,
        node_mask=None,
        edge_mask=None,
    ):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, _agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij


class EquivariantUpdate(nn.Module):
    def __init__(
        self,
        hidden_nf,
        normalization_factor,
        aggregation_method,
        edges_in_d=1,
        act_fn=nn.SiLU(),
        tanh=False,
        coords_range=10.0,
        reflection_equiv=True,
    ):
        super().__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        self.reflection_equiv = reflection_equiv
        input_edge = hidden_nf * 2 + edges_in_d
        # NOTE (upstream behaviour, deliberately preserved): the SAME `layer`
        # instance is the last block of both MLPs below, i.e. the final linear
        # is shared between the coord and cross-product heads. The released
        # checkpoint stores it under both key prefixes.
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer,
        )
        self.cross_product_mlp = (
            nn.Sequential(
                nn.Linear(input_edge, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn,
                layer,
            )
            if not self.reflection_equiv
            else None
        )
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(
        self,
        h,
        coord,
        edge_index,
        coord_diff,
        coord_cross,
        edge_attr,
        edge_mask,
        update_coords_mask=None,
    ):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = (
                coord_diff
                * torch.tanh(self.coord_mlp(input_tensor))
                * self.coords_range
            )
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)

        if not self.reflection_equiv:
            phi_cross = self.cross_product_mlp(input_tensor)
            if self.tanh:
                phi_cross = torch.tanh(phi_cross) * self.coords_range
            trans = trans + coord_cross * phi_cross

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
        self,
        h,
        coord,
        edge_index,
        coord_diff,
        coord_cross,
        edge_attr=None,
        node_mask=None,
        edge_mask=None,
        update_coords_mask=None,
    ):
        coord = self.coord_model(
            h,
            coord,
            edge_index,
            coord_diff,
            coord_cross,
            edge_attr,
            edge_mask,
            update_coords_mask=update_coords_mask,
        )
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):
    def __init__(
        self,
        hidden_nf,
        edge_feat_nf=2,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=2,
        attention=True,
        norm_diff=True,
        tanh=False,
        coords_range=15,
        norm_constant=1,
        sin_embedding=None,
        normalization_factor=100,
        aggregation_method="sum",
        reflection_equiv=True,
    ):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.reflection_equiv = reflection_equiv

        for i in range(n_layers):
            self.add_module(
                "gcl_%d" % i,
                GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=edge_feat_nf,
                    act_fn=act_fn,
                    attention=attention,
                    normalization_factor=self.normalization_factor,
                    aggregation_method=self.aggregation_method,
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
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method,
                reflection_equiv=self.reflection_equiv,
            ),
        )
        self.to(self.device)

    def forward(
        self,
        h,
        x,
        edge_index,
        node_mask=None,
        edge_mask=None,
        edge_attr=None,
        update_coords_mask=None,
        batch_mask=None,
    ):
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        coord_cross = (
            None
            if self.reflection_equiv
            else coord2cross(x, edge_index, batch_mask, self.norm_constant)
        )
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        for i in range(self.n_layers):
            h, _ = self._modules["gcl_%d" % i](
                h,
                edge_index,
                edge_attr=edge_attr,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )
        x = self._modules["gcl_equiv"](
            h,
            x,
            edge_index,
            coord_diff,
            coord_cross,
            edge_attr,
            node_mask,
            edge_mask,
            update_coords_mask=update_coords_mask,
        )
        if node_mask is not None:
            h = h * node_mask
        return h, x


class EGNN(nn.Module):
    """Three parallel EGNN stacks with periodic ligand-node fusion.

    Graph 1 is ligand+pocket, graph 2 ligand+H-bond particles, graph 3
    ligand+hydrophobic particles. The ligand block of ``h``/``x`` (the first
    ``len(mask_atom)`` rows of each graph) is averaged across graphs every
    layer -- pairwise onto graphs 2/3 normally, three-way at the midpoint and
    final layer.
    """

    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=3,
        attention=False,
        norm_diff=True,
        out_node_nf=None,
        tanh=False,
        coords_range=15,
        norm_constant=1,
        inv_sublayers=2,
        sin_embedding=False,
        normalization_factor=100,
        aggregation_method="sum",
        reflection_equiv=True,
    ):
        super().__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range / n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.reflection_equiv = reflection_equiv

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2
        edge_feat_nf = edge_feat_nf + in_edge_nf

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        self.embedding2 = nn.Linear(in_node_nf, self.hidden_nf)
        # embedding2_out/embedding3_out are unused by forward() but exist in
        # the released checkpoint -- kept so state_dict keys match exactly.
        self.embedding2_out = nn.Linear(self.hidden_nf, out_node_nf)
        self.embedding3 = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding3_out = nn.Linear(self.hidden_nf, out_node_nf)

        for i in range(n_layers):
            for g in (1, 2, 3):
                self.add_module(
                    "e_block_%d_%d" % (g, i),
                    EquivariantBlock(
                        hidden_nf,
                        edge_feat_nf=edge_feat_nf,
                        device=device,
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
                        reflection_equiv=self.reflection_equiv,
                    ),
                )
        self.to(self.device)

    def forward(
        self,
        h,
        x,
        h2,
        x2,
        h3,
        x3,
        edge_index,
        edge_index_2,
        edge_index_3,
        mask_atom=None,
        mask_intersh=None,
        mask_intershp=None,
        node_mask=None,
        edge_mask=None,
        update_coords_mask=None,
        update_coords_mask_2=None,
        update_coords_mask_3=None,
        batch_mask=None,
        batch_mask_2=None,
        batch_mask_3=None,
        edge_attr=None,
    ):
        edge_feat, _ = coord2diff(x, edge_index)
        edge_feat_2, _ = coord2diff(x2, edge_index_2)
        edge_feat_3, _ = coord2diff(x3, edge_index_3)
        if self.sin_embedding is not None:
            edge_feat = self.sin_embedding(edge_feat)
        if edge_attr is not None:
            edge_feat = torch.cat([edge_feat, edge_attr], dim=1)

        h = self.embedding(h)
        h2 = self.embedding2(h2)
        h3 = self.embedding3(h3)

        n_lig = len(mask_atom)
        for i in range(self.n_layers):
            h, x = self._modules["e_block_1_%d" % i](
                h,
                x,
                edge_index,
                node_mask=node_mask,
                edge_mask=edge_mask,
                edge_attr=edge_feat,
                update_coords_mask=update_coords_mask,
                batch_mask=batch_mask,
            )
            h2, x2 = self._modules["e_block_2_%d" % i](
                h2,
                x2,
                edge_index_2,
                node_mask=node_mask,
                edge_mask=edge_mask,
                edge_attr=edge_feat_2,
                update_coords_mask=update_coords_mask_2,
                batch_mask=batch_mask_2,
            )
            h3, x3 = self._modules["e_block_3_%d" % i](
                h3,
                x3,
                edge_index_3,
                node_mask=node_mask,
                edge_mask=edge_mask,
                edge_attr=edge_feat_3,
                update_coords_mask=update_coords_mask_3,
                batch_mask=batch_mask_3,
            )

            if i != (self.n_layers) / 2 - 1 and i != self.n_layers - 1:
                h_average2 = (h[:n_lig] + h2[:n_lig]) / 2
                x_average2 = (x[:n_lig] + x2[:n_lig]) / 2
                h2 = torch.cat([h_average2, h2[n_lig:]], dim=0)
                x2 = torch.cat([x_average2, x2[n_lig:]], dim=0)
                h_average3 = (h[:n_lig] + h3[:n_lig]) / 2
                x_average3 = (x[:n_lig] + x3[:n_lig]) / 2
                h3 = torch.cat([h_average3, h3[n_lig:]], dim=0)
                x3 = torch.cat([x_average3, x3[n_lig:]], dim=0)
            else:
                h_average = (h[:n_lig] + h2[:n_lig] + h3[:n_lig]) / 3
                x_average = (x[:n_lig] + x2[:n_lig] + x3[:n_lig]) / 3
                h = torch.cat([h_average, h[n_lig:]], dim=0)
                x = torch.cat([x_average, x[n_lig:]], dim=0)
                h2 = torch.cat([h_average, h2[n_lig:]], dim=0)
                x2 = torch.cat([x_average, x2[n_lig:]], dim=0)
                h3 = torch.cat([h_average, h3[n_lig:]], dim=0)
                x3 = torch.cat([x_average, x3[n_lig:]], dim=0)

        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        return h, x
