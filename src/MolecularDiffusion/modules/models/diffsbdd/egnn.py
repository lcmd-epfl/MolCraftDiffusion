"""EGNN backbone ported from DiffSBDD's ``equivariant_diffusion/egnn_new.py``.

Module and submodule names are **load-bearing**: they are the state-dict keys
of the released CrossDocked checkpoints (Zenodo 8183747), so ``embedding``,
``embedding_out``, ``e_block_<i>``, ``gcl_<i>``, ``gcl_equiv``, ``edge_mlp``,
``node_mlp``, ``att_mlp``, ``coord_mlp`` and ``cross_product_mlp`` must not be
renamed. ``docs/model_integrations/diffsbdd/scripts/convert_checkpoint.py``
asserts the whole key set, because ``cli/generate.py`` loads with
``strict=False`` and a bad remap would load nothing in silence.

Differences from upstream, all deliberate:

* ``GNN`` / ``mode='gnn_dynamics'`` is not ported -- no shipped DiffSBDD
  config selects it (``configs/*.yml`` all leave ``mode`` at its
  ``egnn_dynamics`` default).
* The flat-scatter batching, ``update_coords_mask`` and ``reflection_equiv``
  cross-product term are why this is a separate EGCL from the platform's
  existing one in ``modules/layers/`` rather than a reuse of it.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


def unsorted_segment_sum(
    data: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
    normalization_factor: Optional[float],
    aggregation_method: str,
) -> torch.Tensor:
    """TensorFlow's ``unsorted_segment_sum``, with 'sum' or 'mean' scaling."""
    result = data.new_zeros((num_segments, data.size(1)))
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


def coord2diff(
    x: torch.Tensor, edge_index: torch.Tensor, norm_constant: float = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    return radial, coord_diff / (norm + norm_constant)


def coord2cross(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    batch_mask: torch.Tensor,
    norm_constant: float = 1,
) -> torch.Tensor:
    mean = unsorted_segment_sum(
        x,
        batch_mask,
        num_segments=int(batch_mask.max()) + 1,
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
    """Sinusoidal distance embedding (off in every shipped config)."""

    def __init__(
        self,
        max_res: float = 15.0,
        min_res: float = 15.0 / 2000.0,
        div_factor: int = 4,
    ) -> None:
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = (
            2 * math.pi * div_factor ** torch.arange(self.n_frequencies) / max_res
        )
        self.dim = len(self.frequencies) * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        return torch.cat((emb.sin(), emb.cos()), dim=-1).detach()


class GCL(nn.Module):
    """Invariant message-passing layer."""

    def __init__(
        self,
        input_nf: int,
        output_nf: int,
        hidden_nf: int,
        normalization_factor: float,
        aggregation_method: str,
        edges_in_d: int = 0,
        act_fn: Optional[nn.Module] = None,
        attention: bool = False,
    ) -> None:
        super().__init__()
        act_fn = nn.SiLU() if act_fn is None else act_fn
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_nf * 2 + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )
        if attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        row, col = edge_index
        if edge_attr is None:
            out = torch.cat([h[row], h[col]], dim=1)
        else:
            out = torch.cat([h[row], h[col], edge_attr], dim=1)
        mij = self.edge_mlp(out)
        edge_feat = mij * self.att_mlp(mij) if self.attention else mij

        agg = unsorted_segment_sum(
            edge_feat,
            row,
            num_segments=h.size(0),
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method,
        )
        return h + self.node_mlp(torch.cat([h, agg], dim=1))


class EquivariantUpdate(nn.Module):
    """Coordinate update. ``update_coords_mask`` freezes the pocket nodes."""

    def __init__(
        self,
        hidden_nf: int,
        normalization_factor: float,
        aggregation_method: str,
        edges_in_d: int = 1,
        act_fn: Optional[nn.Module] = None,
        tanh: bool = False,
        coords_range: float = 10.0,
        reflection_equiv: bool = True,
    ) -> None:
        super().__init__()
        act_fn = nn.SiLU() if act_fn is None else act_fn
        self.tanh = tanh
        self.coords_range = coords_range
        self.reflection_equiv = reflection_equiv
        input_edge = hidden_nf * 2 + edges_in_d

        # NOTE: upstream deliberately shares this ONE Linear between both
        # MLPs (egnn_new.py:78-92), so `coord_mlp.4.weight` and
        # `cross_product_mlp.4.weight` are the same tensor. Keep it shared:
        # the released checkpoints store both keys and they are identical.
        layer = nn.Linear(hidden_nf, 1, bias=False)
        nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer,
        )
        self.cross_product_mlp = (
            None
            if reflection_equiv
            else nn.Sequential(
                nn.Linear(input_edge, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn,
                layer,
            )
        )
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def forward(
        self,
        h: torch.Tensor,
        coord: torch.Tensor,
        edge_index: torch.Tensor,
        coord_diff: torch.Tensor,
        coord_cross: Optional[torch.Tensor],
        edge_attr: torch.Tensor,
        update_coords_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        phi = self.coord_mlp(input_tensor)
        if self.tanh:
            trans = coord_diff * torch.tanh(phi) * self.coords_range
        else:
            trans = coord_diff * phi

        if not self.reflection_equiv:
            phi_cross = self.cross_product_mlp(input_tensor)
            if self.tanh:
                phi_cross = torch.tanh(phi_cross) * self.coords_range
            trans = trans + coord_cross * phi_cross

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


class EquivariantBlock(nn.Module):
    """``inv_sublayers`` GCLs followed by one coordinate update."""

    def __init__(
        self,
        hidden_nf: int,
        edge_feat_nf: int = 2,
        act_fn: Optional[nn.Module] = None,
        n_layers: int = 2,
        attention: bool = True,
        tanh: bool = False,
        coords_range: float = 15,
        norm_constant: float = 1,
        sin_embedding: Optional[nn.Module] = None,
        normalization_factor: float = 100,
        aggregation_method: str = "sum",
        reflection_equiv: bool = True,
    ) -> None:
        super().__init__()
        act_fn = nn.SiLU() if act_fn is None else act_fn
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.reflection_equiv = reflection_equiv

        for i in range(n_layers):
            self.add_module(
                f"gcl_{i}",
                GCL(
                    hidden_nf,
                    hidden_nf,
                    hidden_nf,
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
                reflection_equiv=reflection_equiv,
            ),
        )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        update_coords_mask: Optional[torch.Tensor] = None,
        batch_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            h = self._modules[f"gcl_{i}"](h, edge_index, edge_attr=edge_attr)
        x = self._modules["gcl_equiv"](
            h,
            x,
            edge_index,
            coord_diff,
            coord_cross,
            edge_attr,
            update_coords_mask=update_coords_mask,
        )
        return h, x


class EGNN(nn.Module):
    """E(n)-equivariant GNN over the flat ligand+pocket node set."""

    def __init__(
        self,
        in_node_nf: int,
        in_edge_nf: int,
        hidden_nf: int,
        act_fn: Optional[nn.Module] = None,
        n_layers: int = 3,
        attention: bool = False,
        out_node_nf: Optional[int] = None,
        tanh: bool = False,
        coords_range: float = 15,
        norm_constant: float = 1,
        inv_sublayers: int = 2,
        sin_embedding: bool = False,
        normalization_factor: float = 100,
        aggregation_method: str = "sum",
        reflection_equiv: bool = True,
    ) -> None:
        super().__init__()
        act_fn = nn.SiLU() if act_fn is None else act_fn
        out_node_nf = in_node_nf if out_node_nf is None else out_node_nf
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2
        edge_feat_nf = edge_feat_nf + in_edge_nf

        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.embedding_out = nn.Linear(hidden_nf, out_node_nf)
        for i in range(n_layers):
            self.add_module(
                f"e_block_{i}",
                EquivariantBlock(
                    hidden_nf,
                    edge_feat_nf=edge_feat_nf,
                    act_fn=act_fn,
                    n_layers=inv_sublayers,
                    attention=attention,
                    tanh=tanh,
                    coords_range=coords_range,
                    norm_constant=norm_constant,
                    sin_embedding=self.sin_embedding,
                    normalization_factor=normalization_factor,
                    aggregation_method=aggregation_method,
                    reflection_equiv=reflection_equiv,
                ),
            )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        update_coords_mask: Optional[torch.Tensor] = None,
        batch_mask: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_feat, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            edge_feat = self.sin_embedding(edge_feat)
        if edge_attr is not None:
            edge_feat = torch.cat([edge_feat, edge_attr], dim=1)

        h = self.embedding(h)
        for i in range(self.n_layers):
            h, x = self._modules[f"e_block_{i}"](
                h,
                x,
                edge_index,
                edge_attr=edge_feat,
                update_coords_mask=update_coords_mask,
                batch_mask=batch_mask,
            )
        return self.embedding_out(h), x
