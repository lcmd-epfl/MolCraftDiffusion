"""The denoiser wrapper: three per-object encoder/decoder pairs + LEFTNet.

Ported from ``oa_reactdiff/dynamics/_base.py`` and
``oa_reactdiff/dynamics/egnn_dynamics.py`` (commit 543aaa8, MIT).

What this layer actually does, since the name does not say it: the reactant,
transition state and product each get their **own** encoder MLP and their own
decoder MLP (``fragment_names = ["R", "TS", "P"]`` fixes the order and hence
which weights belong to which object). Their encoded node features are then
concatenated into one flat tensor, LEFTNet message-passes over the whole
thing with ``subgraph_mask`` telling it which edges are intra-object, and the
predicted displacement is split back apart and **centred per object** --
which is what keeps the three SE(3) frames independent.

Deviations from upstream, both non-behavioural:

* Upstream's ``EGNN`` backbone is not ported (OA-ReactDiff's released model is
  LEFTNet-only), so ``LEFTNet`` is the default ``model=``. The factory passes
  it explicitly anyway.
* Four unreachable methods are dropped: ``enpose_pbc`` (its only call site is
  commented out at ``egnn_dynamics.py:170``) and the edge-attribute rebuild
  trio ``adjust_edge_attr_on_new_eij`` / ``create_new_edge_attr`` /
  ``init_edge_attr``. The latter need edge features, and this model has none
  -- ``edge_nf`` is 0 in the checkpoint and every ``self.dynamics(...)`` call
  in ``en_diffusion.py`` passes ``edge_attr=None`` literally. The
  ``edge_encoder`` / ``edge_decoder`` build path is kept as-is.
"""

# ruff: noqa
# mypy: ignore-errors

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn, Tensor
from torch_scatter import scatter_mean

from .graph_tools import get_subgraph_mask
from .leftnet import MLP, LEFTNet


class BaseDynamics(nn.Module):
    def __init__(
        self,
        model_config: Dict,
        fragment_names: List[str],
        node_nfs: List[int],
        edge_nf: int,
        condition_nf: int = 0,
        pos_dim: int = 3,
        update_pocket_coords: bool = True,
        condition_time: bool = True,
        edge_cutoff: Optional[float] = None,
        model: Optional[nn.Module] = LEFTNet,
        device: torch.device = torch.device("cuda"),
        enforce_same_encoding: Optional[List] = None,
        source: Optional[Dict] = None,
    ) -> None:
        r"""Base dynamics class set up for denoising process.

        Args:
            model_config (Dict): config for the equivariant model.
            fragment_names (List[str]): list of names for fragments
            node_nfs (List[int]): list of number of input node attributues.
            edge_nf (int): number of input edge attributes.
            condition_nf (int): number of attributes for conditional generation.
            Defaults to 0.
            pos_dim (int): dimension for position vector. Defaults to 3.
            update_pocket_coords (bool): whether to update positions of everything.
                Defaults to True.
            condition_time (bool): whether to condition on time. Defaults to True.
            edge_cutoff (Optional[float]): cutoff for building intra-fragment edges.
                Defaults to None.
            model (Optional[nn.Module]): Module for equivariant model. Defaults to None.
        """
        super().__init__()
        assert len(node_nfs) == len(fragment_names)
        for nf in node_nfs:
            assert nf > pos_dim
        if "act_fn" not in model_config:
            model_config["act_fn"] = "swish"
        if "in_node_nf" not in model_config:
            model_config["in_node_nf"] = model_config["in_hidden_channels"]
        self.model_config = model_config
        self.node_nfs = node_nfs
        self.edge_nf = edge_nf
        self.condition_nf = condition_nf
        self.fragment_names = fragment_names
        self.pos_dim = pos_dim
        self.update_pocket_coords = update_pocket_coords
        self.condition_time = condition_time
        self.edge_cutoff = edge_cutoff
        self.device = device

        if model is None:
            model = LEFTNet
        self.model = model(**model_config)
        if source is not None:
            self.model.load_state_dict(source["model"])
        self.dist_dim = self.model.dist_dim if hasattr(self.model, "dist_dim") else 0

        self.embed_dim = model_config["in_node_nf"]
        self.edge_embed_dim = (
            model_config["in_edge_nf"] if "in_edge_nf" in model_config else 0
        )
        if condition_time:
            self.embed_dim -= 1
        if condition_nf > 0:
            self.embed_dim -= condition_nf
        assert self.embed_dim > 0

        self.build_encoders_decoders(enforce_same_encoding, source)
        del source

    def build_encoders_decoders(
        self,
        enfoce_name_encoding: Optional[List] = None,
        source: Optional[Dict] = None,
    ):
        r"""Build encoders and decoders for nodes and edges."""
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for ii, name in enumerate(self.fragment_names):
            self.encoders.append(
                MLP(
                    in_dim=self.node_nfs[ii] - self.pos_dim,
                    out_dims=[2 * (self.node_nfs[ii] - self.pos_dim), self.embed_dim],
                    activation=self.model_config["act_fn"],
                    last_layer_no_activation=True,
                )
            )
            self.decoders.append(
                MLP(
                    in_dim=self.embed_dim,
                    out_dims=[
                        2 * (self.node_nfs[ii] - self.pos_dim),
                        self.node_nfs[ii] - self.pos_dim,
                    ],
                    activation=self.model_config["act_fn"],
                    last_layer_no_activation=True,
                )
            )
        if enfoce_name_encoding is not None:
            for ii in enfoce_name_encoding:
                self.encoders[ii] = self.encoders[0]
                self.decoders[ii] = self.decoders[0]
        if source is not None:
            self.encoders.load_state_dict(source["encoders"])
            self.decoders.load_state_dict(source["decoders"])

        if self.edge_embed_dim > 0:
            self.edge_encoder = MLP(
                in_dim=self.edge_nf,
                out_dims=[2 * self.edge_nf, self.edge_embed_dim],
                activation=self.model_config["act_fn"],
                last_layer_no_activation=True,
            )
            self.edge_decoder = MLP(
                in_dim=self.edge_embed_dim + self.dist_dim,
                out_dims=[2 * self.edge_nf, self.edge_nf],
                activation=self.model_config["act_fn"],
                last_layer_no_activation=True,
            )
        else:
            self.edge_encoder, self.edge_decoder = None, None

    def forward(self):
        raise NotImplementedError


class EGNNDynamics(BaseDynamics):
    def __init__(
        self,
        model_config: Dict,
        fragment_names: List[str],
        node_nfs: List[int],
        edge_nf: int,
        condition_nf: int = 0,
        pos_dim: int = 3,
        update_pocket_coords: bool = True,
        condition_time: bool = True,
        edge_cutoff: Optional[float] = None,
        model: Optional[nn.Module] = LEFTNet,
        device: torch.device = torch.device("cuda"),
        enforce_same_encoding: Optional[List] = None,
        source: Optional[Dict] = None,
    ) -> None:
        r"""Base dynamics class set up for denoising process.

        Args:
            model_config (Dict): config for the equivariant model.
            fragment_names (List[str]): list of names for fragments
            node_nfs (List[int]): list of number of input node attributues.
            edge_nf (int): number of input edge attributes.
            condition_nf (int): number of attributes for conditional generation.
            Defaults to 0.
            pos_dim (int): dimension for position vector. Defaults to 3.
            update_pocket_coords (bool): whether to update positions of everything.
                Defaults to True.
            condition_time (bool): whether to condition on time. Defaults to True.
            edge_cutoff (Optional[float]): cutoff for building intra-fragment edges.
                Defaults to None.
            model (Optional[nn.Module]): Module for equivariant model. Defaults to None.
        """
        super().__init__(
            model_config,
            fragment_names,
            node_nfs,
            edge_nf,
            condition_nf,
            pos_dim,
            update_pocket_coords,
            condition_time,
            edge_cutoff,
            model,
            device,
            enforce_same_encoding,
            source=source,
        )

    def forward(
        self,
        xh: List[Tensor],
        edge_index: Tensor,
        t: Tensor,
        conditions: Tensor,
        n_frag_switch: Tensor,
        combined_mask: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tuple[List[Tensor], Tensor]:
        r"""predict noise /mu.

        Args:
            xh (List[Tensor]): list of concatenated tensors for pos and h
            edge_index (Tensor): [n_edge, 2]
            t (Tensor): time tensor. If dim is 1, same for all samples;
                otherwise different t for different samples
            conditions (Tensor): condition tensors
            n_frag_switch (Tensor): [n_nodes], fragment index for each nodes
            combined_mask (Tensor): [n_nodes], sample index for each node
            edge_attr (Optional[Tensor]): [n_edge, dim_edge_attribute]. Defaults to None.

        Raises:
            NotImplementedError: The fragement-position-fixed mode is not implement.

        Returns:
            Tuple[List[Tensor], Tensor]: updated pos-h and edge attributes
        """
        pos = torch.concat(
            [_xh[:, : self.pos_dim].clone() for _xh in xh],
            dim=0,
        )
        h = torch.concat(
            [
                self.encoders[ii](xh[ii][:, self.pos_dim :].clone())
                for ii, name in enumerate(self.fragment_names)
            ],
            dim=0,
        )
        if self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)

        condition_dim = 0
        if self.condition_time:
            if len(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t[combined_mask]
            h = torch.cat([h, h_time], dim=1)
            condition_dim += 1

        if self.condition_nf > 0:
            h_condition = conditions[combined_mask]
            h = torch.cat([h, h_condition], dim=1)
            condition_dim += self.condition_nf

        subgraph_mask = get_subgraph_mask(edge_index, n_frag_switch)
        if self.update_pocket_coords:
            update_coords_mask = None
        else:
            raise NotImplementedError  # no need to mask pos for inpainting mode.

        h_final, pos_final, edge_attr_final = self.model(
            h,
            pos,
            edge_index,
            edge_attr,
            node_mask=None,
            edge_mask=None,
            update_coords_mask=update_coords_mask,
            subgraph_mask=subgraph_mask[:, None],
        )
        vel = pos_final - pos
        if torch.any(torch.isnan(vel)):
            print("Warning: detected nan in pos, resetting LEFTNet output to randn.")
            vel = torch.randn_like(vel)
        if torch.any(torch.isnan(vel)):
            print("Warning: detected nan in h, resetting LEFTNet output to randn.")
            h_final = torch.randn_like(h_final)

        h_final = h_final[:, :-condition_dim]

        frag_index = self.compute_frag_index(n_frag_switch)
        xh_final = [
            torch.cat(
                [
                    self.remove_mean_batch(
                        vel[frag_index[ii] : frag_index[ii + 1]],
                        combined_mask[frag_index[ii] : frag_index[ii + 1]],
                    ),
                    self.decoders[ii](h_final[frag_index[ii] : frag_index[ii + 1]]),
                ],
                dim=-1,
            )
            for ii, name in enumerate(self.fragment_names)
        ]

        # xh_final = self.enpose_pbc(xh_final)

        if edge_attr_final is None or edge_attr_final.size(1) <= max(1, self.dist_dim):
            edge_attr_final = None
        else:
            edge_attr_final = self.edge_decoder(edge_attr_final)
        return xh_final, edge_attr_final

    @staticmethod
    def compute_frag_index(n_frag_switch: Tensor) -> np.ndarray:
        counts = [
            torch.where(n_frag_switch == ii)[0].numel()
            for ii in torch.unique(n_frag_switch)
        ]
        return np.concatenate([np.array([0]), np.cumsum(counts)])

    @staticmethod
    def remove_mean_batch(x, indices):
        mean = scatter_mean(x, indices, dim=0)
        x = x - mean[indices]
        return x
