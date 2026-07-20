"""Coordinate-only FlowMol GVP vector field (endpoint parameterization).

Ported and surgically reduced from FlowMol
(``flowmol/models/vector_field.py::EndpointVectorField``). Differences from
the upstream model, per the approved integration plan (coordinate-only):

- The ``e`` (bond) modality is REMOVED as a generated output. The generated
  modalities are ``x`` (coords), ``a`` (atom types), ``c`` (formal charge).
- Edges are KEPT purely as geometry-only message-passing channels: the edge
  feature channel is initialised from an RBF expansion of the current
  interatomic distance (``edge_embedding`` maps ``rbf_dim -> n_hidden_edge``)
  instead of from a noised bond token ``e_t``. ``EdgeUpdate`` (with
  ``update_edge_w_distance=True``) refines it through the network.
- ``to_edge_logits``, the edge interpolant, edge prior sampling, the edge
  loss term, self-conditioning, CTMC / Dirichlet, fake atoms, trajectory
  frames and guidance are all dropped.
"""

from typing import Union

import dgl
import dgl.function as fn
import torch
import torch.nn as nn

from MolecularDiffusion.modules.layers.gvp import (
    GVP,
    GVPConv,
    _norm_no_nan,
    _rbf,
    get_time_embedding,
)
from MolecularDiffusion.modules.models.flowmol.interpolant_scheduler import (
    InterpolantScheduler,
)


class EndpointVectorField(nn.Module):

    def __init__(
        self,
        n_atom_types: int,
        canonical_feat_order: list,
        interpolant_scheduler: InterpolantScheduler,
        n_charges: int = 6,
        n_vec_channels: int = 16,
        n_cp_feats: int = 0,
        n_hidden_scalars: int = 64,
        n_hidden_edge_feats: int = 64,
        n_recycles: int = 1,
        n_molecule_updates: int = 2,
        convs_per_update: int = 2,
        n_message_gvps: int = 3,
        n_update_gvps: int = 3,
        n_expansion_gvps: int = 3,
        separate_mol_updaters: bool = False,
        message_norm: Union[float, str] = 100,
        update_edge_w_distance: bool = True,
        rbf_dmax: float = 20,
        rbf_dim: int = 16,
        time_embedding_dim: int = 1,
        attention: bool = False,
        n_heads: int = 1,
        s_message_dim: int = None,
        v_message_dim: int = None,
        dropout: float = 0.0,
        use_dst_feats: bool = False,
        dst_feat_msg_reduction_factor: float = 4,
    ):
        super().__init__()

        self.n_atom_types = n_atom_types
        self.n_charges = n_charges
        self.n_hidden_scalars = n_hidden_scalars
        self.n_hidden_edge_feats = n_hidden_edge_feats
        self.n_vec_channels = n_vec_channels
        self.message_norm = message_norm
        self.n_recycles = n_recycles
        self.separate_mol_updaters = separate_mol_updaters
        self.interpolant_scheduler = interpolant_scheduler
        self.canonical_feat_order = canonical_feat_order
        self.time_embedding_dim = time_embedding_dim
        self.convs_per_update = convs_per_update
        self.n_molecule_updates = n_molecule_updates
        self.rbf_dmax = rbf_dmax
        self.rbf_dim = rbf_dim

        assert n_vec_channels >= 3, "n_vec_channels must be >= 3"

        # categorical node modalities carried as continuous simplex vectors
        self.n_cat_feats = {"a": n_atom_types, "c": n_charges}

        self.scalar_embedding = nn.Sequential(
            nn.Linear(
                n_atom_types + n_charges + self.time_embedding_dim,
                n_hidden_scalars,
            ),
            nn.SiLU(),
            nn.Linear(n_hidden_scalars, n_hidden_scalars),
            nn.SiLU(),
            nn.LayerNorm(n_hidden_scalars),
        )

        # geometry-only edge features: RBF(distance) -> edge feature channel
        # (replaces FlowMol's bond-token embedding)
        self.edge_embedding = nn.Sequential(
            nn.Linear(rbf_dim, n_hidden_edge_feats),
            nn.SiLU(),
            nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
            nn.SiLU(),
            nn.LayerNorm(n_hidden_edge_feats),
        )

        self.conv_layers = nn.ModuleList(
            [
                GVPConv(
                    scalar_size=n_hidden_scalars,
                    vector_size=n_vec_channels,
                    n_cp_feats=n_cp_feats,
                    edge_feat_size=n_hidden_edge_feats,
                    n_message_gvps=n_message_gvps,
                    n_update_gvps=n_update_gvps,
                    n_expansion_gvps=n_expansion_gvps,
                    message_norm=message_norm,
                    rbf_dmax=rbf_dmax,
                    rbf_dim=rbf_dim,
                    attention=attention,
                    n_heads=n_heads,
                    s_message_dim=s_message_dim,
                    v_message_dim=v_message_dim,
                    dropout=dropout,
                    use_dst_feats=use_dst_feats,
                    dst_feat_msg_reduction_factor=dst_feat_msg_reduction_factor,
                )
                for _ in range(convs_per_update * n_molecule_updates)
            ]
        )

        self.node_position_updaters = nn.ModuleList([])
        self.edge_updaters = nn.ModuleList([])
        n_updaters = n_molecule_updates if separate_mol_updaters else 1
        for _ in range(n_updaters):
            self.node_position_updaters.append(
                NodePositionUpdate(
                    n_hidden_scalars, n_vec_channels, n_gvps=3,
                    n_cp_feats=n_cp_feats,
                )
            )
            self.edge_updaters.append(
                EdgeUpdate(
                    n_hidden_scalars,
                    n_hidden_edge_feats,
                    update_edge_w_distance=update_edge_w_distance,
                    rbf_dim=rbf_dim,
                )
            )

        self.node_output_head = nn.Sequential(
            nn.Linear(n_hidden_scalars, n_hidden_scalars),
            nn.SiLU(),
            nn.Linear(n_hidden_scalars, n_atom_types + n_charges),
        )

    def forward(
        self,
        g: dgl.DGLGraph,
        t: torch.Tensor,
        node_batch_idx: torch.Tensor,
        apply_softmax: bool = False,
        remove_com: bool = False,
    ):
        """Predict the trajectory endpoint (x_1, a_1, c_1) given x_t."""
        device = g.device

        with g.local_scope():
            node_scalar_features = [g.ndata["a_t"], g.ndata["c_t"]]

            if self.time_embedding_dim == 1:
                node_scalar_features.append(t[node_batch_idx].unsqueeze(-1))
            else:
                t_emb = get_time_embedding(
                    t, embedding_dim=self.time_embedding_dim
                )
                node_scalar_features.append(t_emb[node_batch_idx])

            node_scalar_features = torch.cat(node_scalar_features, dim=-1)
            node_scalar_features = self.scalar_embedding(node_scalar_features)

            node_positions = g.ndata["x_t"]
            num_nodes = g.num_nodes()
            node_vec_features = torch.zeros(
                (num_nodes, self.n_vec_channels, 3), device=device
            )

        dst_dict = self.denoise_graph(
            g,
            node_scalar_features,
            node_vec_features,
            node_positions,
            node_batch_idx,
            apply_softmax,
            remove_com,
        )
        return dst_dict

    def denoise_graph(
        self,
        g: dgl.DGLGraph,
        node_scalar_features: torch.Tensor,
        node_vec_features: torch.Tensor,
        node_positions: torch.Tensor,
        node_batch_idx: torch.Tensor,
        apply_softmax: bool = False,
        remove_com: bool = False,
    ):
        x_diff, d = self.precompute_distances(g, node_positions)

        # geometry-only initial edge features from the interatomic distance RBF
        edge_features = self.edge_embedding(d)

        for _ in range(self.n_recycles):
            for conv_idx, conv in enumerate(self.conv_layers):
                node_scalar_features, node_vec_features = conv(
                    g,
                    scalar_feats=node_scalar_features,
                    coord_feats=node_positions,
                    vec_feats=node_vec_features,
                    edge_feats=edge_features,
                    x_diff=x_diff,
                    d=d,
                )

                if (
                    conv_idx != 0
                    and (conv_idx + 1) % self.convs_per_update == 0
                ):
                    if self.separate_mol_updaters:
                        updater_idx = conv_idx // self.convs_per_update
                    else:
                        updater_idx = 0

                    node_positions = self.node_position_updaters[updater_idx](
                        node_scalar_features, node_positions, node_vec_features
                    )
                    x_diff, d = self.precompute_distances(g, node_positions)
                    edge_features = self.edge_updaters[updater_idx](
                        g, node_scalar_features, edge_features, d=d
                    )

        node_out = self.node_output_head(node_scalar_features)
        atom_type_logits = node_out[:, : self.n_atom_types]
        atom_charge_logits = node_out[:, self.n_atom_types :]

        if remove_com:
            g.ndata["x_1_pred"] = node_positions
            com = dgl.readout_nodes(g, feat="x_1_pred", op="mean")
            node_positions = node_positions - com[node_batch_idx]

        dst_dict = {
            "x": node_positions,
            "a": atom_type_logits,
            "c": atom_charge_logits,
        }

        if apply_softmax:
            for feat in ("a", "c"):
                dst_dict[feat] = torch.softmax(dst_dict[feat], dim=-1)

        return dst_dict

    def precompute_distances(
        self, g: dgl.DGLGraph, node_positions=None
    ):
        """Precompute normalised displacement + RBF distance on every edge."""
        with g.local_scope():
            if node_positions is None:
                g.ndata["x_d"] = g.ndata["x_t"]
            else:
                g.ndata["x_d"] = node_positions

            g.apply_edges(fn.u_sub_v("x_d", "x_d", "x_diff"))
            dij = _norm_no_nan(g.edata["x_diff"], keepdims=True) + 1e-8
            x_diff = g.edata["x_diff"] / dij
            d = _rbf(dij.squeeze(1), D_max=self.rbf_dmax, D_count=self.rbf_dim)
        return x_diff, d

    def vector_field(self, x_t, x_1, alpha_t, alpha_t_prime):
        return alpha_t_prime / (1 - alpha_t) * (x_1 - x_t)

    def sample_conditional_path(
        self, g: dgl.DGLGraph, t: torch.Tensor, node_batch_idx: torch.Tensor
    ):
        """Interpolate x_t between prior (x_0) and data (x_1_true)."""
        src_weights, dst_weights = self.interpolant_scheduler.interpolant_weights(
            t
        )
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            src_weight = src_weights[:, feat_idx][node_batch_idx].unsqueeze(-1)
            dst_weight = dst_weights[:, feat_idx][node_batch_idx].unsqueeze(-1)
            g.ndata[f"{feat}_t"] = (
                src_weight * g.ndata[f"{feat}_0"]
                + dst_weight * g.ndata[f"{feat}_1_true"]
            )
        return g

    @torch.no_grad()
    def integrate(
        self,
        g: dgl.DGLGraph,
        node_batch_idx: torch.Tensor,
        n_timesteps: int,
    ):
        """Integrate the trajectory from prior (x_0) to endpoint along the VF."""
        t = torch.linspace(0, 1, n_timesteps, device=g.device)
        alpha_t = self.interpolant_scheduler.alpha_t(t)
        alpha_t_prime = self.interpolant_scheduler.alpha_t_prime(t)

        for feat in self.canonical_feat_order:
            g.ndata[f"{feat}_t"] = g.ndata[f"{feat}_0"]

        for s_idx in range(1, t.shape[0]):
            s_i = t[s_idx]
            t_i = t[s_idx - 1]
            alpha_t_i = alpha_t[s_idx - 1]
            alpha_t_prime_i = alpha_t_prime[s_idx - 1]
            g = self.step(
                g, s_i, t_i, alpha_t_i, alpha_t_prime_i, node_batch_idx
            )

        for feat in self.canonical_feat_order:
            g.ndata[f"{feat}_1"] = g.ndata[f"{feat}_t"]
        return g

    def step(
        self,
        g: dgl.DGLGraph,
        s_i: torch.Tensor,
        t_i: torch.Tensor,
        alpha_t_i: torch.Tensor,
        alpha_t_prime_i: torch.Tensor,
        node_batch_idx: torch.Tensor,
    ):
        dst_dict = self(
            g,
            t=torch.full((g.batch_size,), t_i, device=g.device),
            node_batch_idx=node_batch_idx,
            apply_softmax=True,
            remove_com=True,
        )

        for feat_idx, feat in enumerate(self.canonical_feat_order):
            x_t = g.ndata[f"{feat}_t"]
            x_1 = dst_dict[feat]
            vf = self.vector_field(
                x_t, x_1, alpha_t_i[feat_idx], alpha_t_prime_i[feat_idx]
            )
            g.ndata[f"{feat}_t"] = x_t + vf * (s_i - t_i)

        return g


class NodePositionUpdate(nn.Module):

    def __init__(
        self, n_scalars, n_vec_channels, n_gvps: int = 3, n_cp_feats: int = 0
    ):
        super().__init__()
        gvps = []
        for i in range(n_gvps):
            if i == n_gvps - 1:
                vectors_out = 1
                vectors_activation = nn.Identity()
            else:
                vectors_out = n_vec_channels
                vectors_activation = nn.Sigmoid()

            gvps.append(
                GVP(
                    dim_feats_in=n_scalars,
                    dim_feats_out=n_scalars,
                    dim_vectors_in=n_vec_channels,
                    dim_vectors_out=vectors_out,
                    n_cp_feats=n_cp_feats,
                    vectors_activation=vectors_activation,
                )
            )
        self.gvps = nn.Sequential(*gvps)

    def forward(self, scalars, positions, vectors):
        _, vector_updates = self.gvps((scalars, vectors))
        return positions + vector_updates.squeeze(1)


class EdgeUpdate(nn.Module):

    def __init__(
        self,
        n_node_scalars,
        n_edge_feats,
        update_edge_w_distance=False,
        rbf_dim=16,
    ):
        super().__init__()
        self.update_edge_w_distance = update_edge_w_distance

        input_dim = n_node_scalars * 2 + n_edge_feats
        if update_edge_w_distance:
            input_dim += rbf_dim

        self.edge_update_fn = nn.Sequential(
            nn.Linear(input_dim, n_edge_feats),
            nn.SiLU(),
            nn.Linear(n_edge_feats, n_edge_feats),
            nn.SiLU(),
        )
        self.edge_norm = nn.LayerNorm(n_edge_feats)

    def forward(self, g: dgl.DGLGraph, node_scalars, edge_feats, d):
        src_idxs, dst_idxs = g.edges()
        mlp_inputs = [
            node_scalars[src_idxs],
            node_scalars[dst_idxs],
            edge_feats,
        ]
        if self.update_edge_w_distance:
            mlp_inputs.append(d)
        edge_feats = self.edge_norm(
            edge_feats + self.edge_update_fn(torch.cat(mlp_inputs, dim=-1))
        )
        return edge_feats
