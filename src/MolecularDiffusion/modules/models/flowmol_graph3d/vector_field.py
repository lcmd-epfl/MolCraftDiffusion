"""Bond-carrying FlowMol endpoint vector field (GVP, SE(3)-equivariant).

Ported from FlowMol (``flowmol/models/vector_field.py``) **with the bond (``e``)
modality intact**, which is what makes this a separate package from the
platform's coordinate-only ``modules/models/flowmol/``: that one has no
``n_bond_types``, no ``token_embeddings``, no ``to_edge_logits`` and an
``edge_embedding`` re-sourced to an RBF of interatomic distance, so restoring
bonds there would mean changing its constructor and every public method (and
breaking the existing ``diffusion_flowmol`` task, which pins
``canonical_feat_order = ["x", "a", "c"]``).

Reused unchanged rather than re-ported:

- ``modules/layers/gvp`` -- ``GVPConv``, ``NodePositionUpdate``, ``EdgeUpdate``,
  ``_rbf``, ``_norm_no_nan``, ``get_time_embedding``. Diffed against upstream:
  the platform copy already supports every FlowMol3 setting, including
  ``message_norm: 'sum'``, ``n_expansion_gvps``, ``n_cp_feats`` and the
  ``edge_feat_size`` path.
- ``modules/models/flowmol.interpolant_scheduler.InterpolantScheduler`` --
  feature-agnostic (driven by ``canonical_feat_order``) and ``linear``-capable
  without ``cosine_params``.

Not ported: ``VectorField`` and ``DirichletVectorField`` (the ``vector-field``
and ``dirichlet`` parameterizations, both unreachable at ``parameterization:
ctmc``), and trajectory visualization (``n_frames`` is out of scope).

**Symmetry is structural, not asserted.** Loss and integration run on the upper
triangle only and are written back to both halves, and the backbone pools both
directions (``ue_feats + le_feats``) before the bond head. That makes the whole
thing depend on the edge *ordering* laid down by ``build_edge_idxs``, which is
why ``get_upper_edge_mask`` may only ever be applied to graphs built with it.
"""

from collections.abc import Callable

import dgl
import dgl.function as fn
import torch
from torch import nn

from MolecularDiffusion.modules.layers.gvp import (
    EdgeUpdate,
    GVPConv,
    NodePositionUpdate,
    _norm_no_nan,
    _rbf,
    get_time_embedding,
)
from MolecularDiffusion.modules.models.flowmol.interpolant_scheduler import (
    InterpolantScheduler,
)
from MolecularDiffusion.modules.models.flowmol_graph3d.self_conditioning import (
    SelfConditioningResidualLayer,
)

__all__ = ["EndpointVectorField"]


class EndpointVectorField(nn.Module):
    """Predicts the trajectory endpoint ``x_1`` for all four modalities.

    ``forward`` consumes ``g.ndata['x_t','a_t','c_t']`` and ``g.edata['e_t']``
    and returns a dict keyed ``{'x','a','c','e'}``. The ``e`` entry covers the
    **upper-triangle edges only** (that is what ``to_edge_logits`` is fed).
    """

    def __init__(  # noqa: PLR0913, PLR0915
        self,
        n_atom_types: int,
        canonical_feat_order: list,
        interpolant_scheduler: InterpolantScheduler,
        n_charges: int = 6,
        n_bond_types: int = 4,
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
        message_norm: float | str = 100,
        update_edge_w_distance: bool = False,
        rbf_dmax: float = 20,
        rbf_dim: int = 16,
        exclude_charges: bool = False,
        continuous_inv_temp_schedule=None,
        continuous_inv_temp_max: float = 10.0,
        time_embedding_dim: int = 1,
        a_token_dim: int = 0,
        c_token_dim: int = 0,
        e_token_dim: int = 0,
        attention: bool = False,
        n_heads: int = 1,
        s_message_dim: int = None,
        v_message_dim: int = None,
        dropout: float = 0.0,
        has_mask: bool = False,
        self_conditioning: bool = False,
        use_dst_feats: bool = False,
        dst_feat_msg_reduction_factor: float = 4,
        scprop: float = 0.5,
    ) -> None:
        super().__init__()

        if exclude_charges:
            # Upstream raises here too (vector_field.py:78-80).
            msg = "exclude_charges is deprecated upstream and not supported"
            raise ValueError(msg)

        self.n_atom_types = n_atom_types
        self.n_charges = n_charges
        self.n_bond_types = n_bond_types
        self.n_hidden_scalars = n_hidden_scalars
        self.n_hidden_edge_feats = n_hidden_edge_feats
        self.n_vec_channels = n_vec_channels
        self.message_norm = message_norm
        self.n_recycles = n_recycles
        self.separate_mol_updaters = separate_mol_updaters
        self.interpolant_scheduler = interpolant_scheduler
        self.canonical_feat_order = canonical_feat_order
        self.time_embedding_dim = time_embedding_dim
        self.self_conditioning = self_conditioning
        self.has_mask = has_mask
        self.scprop = scprop
        self.convs_per_update = convs_per_update
        self.n_molecule_updates = n_molecule_updates
        self.rbf_dmax = rbf_dmax
        self.rbf_dim = rbf_dim

        if n_vec_channels < 3:  # noqa: PLR2004
            msg = "n_vec_channels must be >= 3"
            raise ValueError(msg)

        self.continuous_inv_temp_schedule = continuous_inv_temp_schedule
        self.continuous_inv_temp_max = continuous_inv_temp_max
        self.continuous_inv_temp_func = self.build_continuous_inv_temp_func(
            continuous_inv_temp_schedule, continuous_inv_temp_max
        )

        #: real class counts, mask token NOT included
        self.n_cat_feats = {
            "a": n_atom_types,
            "c": n_charges,
            "e": n_bond_types,
        }

        n_mask_feats = int(has_mask)

        # Under CTMC the categorical `_t` tensors are argmaxed and looked up in
        # an nn.Embedding (rather than concatenated as one-hots), which is why
        # the mask column never reaches an nn.Linear. The embedding therefore
        # carries one extra row; the output heads do not.
        self.token_dims = {
            "a": a_token_dim,
            "c": c_token_dim,
            "e": e_token_dim,
        }
        self.token_embeddings = nn.ModuleDict()
        for feat, token_dim in self.token_dims.items():
            if token_dim == 0:
                self.token_embeddings[feat] = None
            else:
                self.token_embeddings[feat] = nn.Embedding(
                    self.n_cat_feats[feat] + n_mask_feats, token_dim
                )

        for modality, token_dim in self.token_dims.items():
            if token_dim == 0:
                self.token_dims[modality] = (
                    self.n_cat_feats[modality] + n_mask_feats
                )

        self.scalar_embedding = nn.Sequential(
            nn.Linear(
                self.token_dims["a"]
                + self.token_dims["c"]
                + self.time_embedding_dim,
                n_hidden_scalars,
            ),
            nn.SiLU(),
            nn.Linear(n_hidden_scalars, n_hidden_scalars),
            nn.SiLU(),
            nn.LayerNorm(n_hidden_scalars),
        )

        self.edge_embedding = nn.Sequential(
            nn.Linear(self.token_dims["e"], n_hidden_edge_feats),
            nn.SiLU(),
            nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
            nn.SiLU(),
            nn.LayerNorm(n_hidden_edge_feats),
        )

        conv_layers = []
        for _ in range(convs_per_update * n_molecule_updates):
            conv_layers.append(
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
            )
        self.conv_layers = nn.ModuleList(conv_layers)

        self.node_position_updaters = nn.ModuleList([])
        self.edge_updaters = nn.ModuleList([])
        n_updaters = n_molecule_updates if separate_mol_updaters else 1
        for _ in range(n_updaters):
            self.node_position_updaters.append(
                NodePositionUpdate(
                    n_hidden_scalars,
                    n_vec_channels,
                    n_gvps=3,
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

        # One head for atom types + charges, split at n_atom_types. Note
        # n_atom_types already includes the fake-atom column when fake atoms
        # are enabled; the mask token is never in an output.
        self.node_output_head = nn.Sequential(
            nn.Linear(n_hidden_scalars, n_hidden_scalars),
            nn.SiLU(),
            nn.Linear(n_hidden_scalars, n_atom_types + n_charges),
        )

        self.to_edge_logits = nn.Sequential(
            nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
            nn.SiLU(),
            nn.Linear(n_hidden_edge_feats, n_bond_types),
        )

        if self.self_conditioning:
            self.self_conditioning_residual_layer = (
                SelfConditioningResidualLayer(
                    n_atom_types=n_atom_types,
                    n_charges=n_charges,
                    n_bond_types=n_bond_types,
                    node_embedding_dim=n_hidden_scalars,
                    edge_embedding_dim=n_hidden_edge_feats,
                    rbf_dim=rbf_dim,
                    rbf_dmax=rbf_dmax,
                )
            )

    @staticmethod
    def build_continuous_inv_temp_func(
        schedule, max_inv_temp: float = None
    ) -> Callable:
        """Inverse-temperature schedule for the continuous (``x``) vector field."""
        if schedule is None:
            return lambda t: 1.0  # noqa: ARG005
        if schedule == "linear":
            return lambda t: max_inv_temp * (1 - t)
        if callable(schedule):
            return schedule
        msg = f"Invalid continuous_inv_temp_schedule: {schedule}"
        raise ValueError(msg)

    # -- forward ------------------------------------------------------------

    def forward(  # noqa: PLR0913
        self,
        g: dgl.DGLGraph,
        t: torch.Tensor,
        node_batch_idx: torch.Tensor,
        upper_edge_mask: torch.Tensor,
        apply_softmax: bool = False,
        remove_com: bool = False,
        prev_dst_dict: dict = None,
    ) -> dict:
        """Predict ``x_1`` given ``x_t`` (and optionally a previous endpoint)."""
        device = g.device

        with g.local_scope():
            node_scalar_features = []
            if self.token_embeddings["a"] is None:
                node_scalar_features.append(g.ndata["a_t"])
                node_scalar_features.append(g.ndata["c_t"])
            else:
                node_scalar_features.append(
                    self.token_embeddings["a"](g.ndata["a_t"].argmax(dim=-1))
                )
                node_scalar_features.append(
                    self.token_embeddings["c"](g.ndata["c_t"].argmax(dim=-1))
                )

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
            node_vec_features = torch.zeros(
                (g.num_nodes(), self.n_vec_channels, 3), device=device
            )

            if self.token_embeddings["e"] is None:
                edge_features = g.edata["e_t"]
            else:
                edge_features = self.token_embeddings["e"](
                    g.edata["e_t"].argmax(dim=-1)
                )
            edge_features = self.edge_embedding(edge_features)

        # Self-conditioning: with probability (1 - scprop) at training time, and
        # always on the first inference step, run a gradient-stopped pass to get
        # a predicted endpoint to condition on.
        if self.self_conditioning and prev_dst_dict is None:
            train_self_condition = (
                self.training and (torch.rand(1) > self.scprop).item()
            )
            inference_first_step = not self.training and (t == 0).all().item()

            if train_self_condition or inference_first_step:
                with torch.no_grad():
                    prev_dst_dict = self.denoise_graph(
                        g,
                        node_scalar_features.clone(),
                        node_vec_features.clone(),
                        node_positions.clone(),
                        edge_features.clone(),
                        node_batch_idx,
                        upper_edge_mask,
                        apply_softmax=True,
                        remove_com=False,
                    )

        if self.self_conditioning and prev_dst_dict is not None:
            (
                node_scalar_features,
                node_positions,
                node_vec_features,
                edge_features,
            ) = self.self_conditioning_residual_layer(
                g,
                node_scalar_features,
                node_positions,
                node_vec_features,
                edge_features,
                prev_dst_dict,
                node_batch_idx,
                upper_edge_mask,
            )

        return self.denoise_graph(
            g,
            node_scalar_features,
            node_vec_features,
            node_positions,
            edge_features,
            node_batch_idx,
            upper_edge_mask,
            apply_softmax,
            remove_com,
        )

    def denoise_graph(  # noqa: PLR0913
        self,
        g: dgl.DGLGraph,
        node_scalar_features: torch.Tensor,
        node_vec_features: torch.Tensor,
        node_positions: torch.Tensor,
        edge_features: torch.Tensor,
        node_batch_idx: torch.Tensor,
        upper_edge_mask: torch.Tensor,
        apply_softmax: bool = False,
        remove_com: bool = False,
    ) -> dict:
        """The GVP message-passing stack plus the four output heads."""
        x_diff, d = self.precompute_distances(g)
        for _recycle_idx in range(self.n_recycles):
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

        node_scalar_features = self.node_output_head(node_scalar_features)
        atom_type_logits = node_scalar_features[:, : self.n_atom_types]
        atom_charge_logits = node_scalar_features[:, self.n_atom_types :]

        # Pool both directions of each bond before the bond head -- this is
        # where bond symmetry is enforced structurally.
        ue_feats = edge_features[upper_edge_mask]
        le_feats = edge_features[~upper_edge_mask]
        edge_logits = self.to_edge_logits(ue_feats + le_feats)

        if remove_com:
            with g.local_scope():
                g.ndata["x_1_pred"] = node_positions
                com = dgl.readout_nodes(g, feat="x_1_pred", op="mean")
                node_positions = node_positions - com[node_batch_idx]

        dst_dict = {
            "x": node_positions,
            "a": atom_type_logits,
            "c": atom_charge_logits,
            "e": edge_logits,
        }

        # Training uses CrossEntropyLoss (which includes the softmax);
        # inference wants a point on the simplex.
        if apply_softmax:
            for feat in ("a", "c", "e"):
                dst_dict[feat] = torch.softmax(dst_dict[feat], dim=-1)

        return dst_dict

    def precompute_distances(
        self, g: dgl.DGLGraph, node_positions: torch.Tensor = None
    ):
        """Unit displacement vectors and RBF-embedded lengths for every edge."""
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

    # -- integration --------------------------------------------------------

    def integrate(
        self,
        g: dgl.DGLGraph,
        node_batch_idx: torch.Tensor,
        upper_edge_mask: torch.Tensor,
        n_timesteps: int,
        **kwargs,
    ) -> dgl.DGLGraph:
        """Euler-integrate all four modalities from the prior to the data.

        Overridden by :class:`~...ctmc_vector_field.CTMCVectorField`, which is
        the only parameterization in scope; this stays for the sake of a
        complete, reusable bond-carrying parent.
        """
        t = torch.linspace(0, 1, n_timesteps, device=g.device)
        alpha_t = self.interpolant_scheduler.alpha_t(t)
        alpha_t_prime = self.interpolant_scheduler.alpha_t_prime(t)

        for feat in self.canonical_feat_order:
            data_src = g.edata if feat == "e" else g.ndata
            data_src[f"{feat}_t"] = data_src[f"{feat}_0"]

        dst_dict = None
        for s_idx in range(1, t.shape[0]):
            g, dst_dict = self.step(
                g,
                t[s_idx],
                t[s_idx - 1],
                alpha_t[s_idx - 1],
                alpha_t[s_idx],
                alpha_t_prime[s_idx - 1],
                node_batch_idx,
                upper_edge_mask,
                prev_dst_dict=dst_dict,
                **kwargs,
            )

        for feat in self.canonical_feat_order:
            data_src = g.edata if feat == "e" else g.ndata
            data_src[f"{feat}_1"] = data_src[f"{feat}_t"]

        return g

    def step(  # noqa: PLR0913
        self,
        g: dgl.DGLGraph,
        s_i: torch.Tensor,
        t_i: torch.Tensor,
        alpha_t_i: torch.Tensor,
        alpha_s_i: torch.Tensor,  # noqa: ARG002 - signature parity with CTMC
        alpha_t_prime_i: torch.Tensor,
        node_batch_idx: torch.Tensor,
        upper_edge_mask: torch.Tensor,
        prev_dst_dict: dict = None,
        inv_temp_func: Callable = None,
        **kwargs,  # noqa: ARG002
    ):
        """One Euler step of the endpoint parameterization."""
        if inv_temp_func is None:
            inv_temp_func = self.continuous_inv_temp_func

        dst_dict = self(
            g,
            t=torch.full((g.batch_size,), t_i, device=g.device),
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=True,
            remove_com=True,
            prev_dst_dict=prev_dst_dict,
        )

        for feat_idx, feat in enumerate(self.canonical_feat_order):
            data_src = g.edata if feat == "e" else g.ndata

            x_t = data_src[f"{feat}_t"]
            x_1 = dst_dict[feat]
            if feat == "e":
                x_t = x_t[upper_edge_mask]

            vf = self.vector_field(
                x_t, x_1, alpha_t_i[feat_idx], alpha_t_prime_i[feat_idx]
            )
            vf = vf * inv_temp_func(t_i)
            x_s = x_t + vf * (s_i - t_i)

            if feat == "e":
                # mirror onto both edge directions
                e_s = torch.zeros_like(g.edata["e_0"])
                e_s[upper_edge_mask] = x_s
                e_s[~upper_edge_mask] = x_s
                x_s = e_s

            data_src[f"{feat}_t"] = x_s

        return g, dst_dict

    @staticmethod
    def vector_field(x_t, x_1, alpha_t, alpha_t_prime):
        """The endpoint-parameterized conditional vector field."""
        return alpha_t_prime / (1 - alpha_t) * (x_1 - x_t)

    # -- training-time interpolation ----------------------------------------

    def sample_conditional_path(
        self,
        g: dgl.DGLGraph,
        t: torch.Tensor,
        node_batch_idx: torch.Tensor,
        edge_batch_idx: torch.Tensor,
        upper_edge_mask: torch.Tensor,  # noqa: ARG002 - CTMC subclass needs it
    ) -> dgl.DGLGraph:
        """Linearly interpolate between the prior and the data at time ``t``."""
        src_weights, dst_weights = (
            self.interpolant_scheduler.interpolant_weights(t)
        )

        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if feat == "e":
                continue
            src_w = src_weights[:, feat_idx][node_batch_idx].unsqueeze(-1)
            dst_w = dst_weights[:, feat_idx][node_batch_idx].unsqueeze(-1)
            g.ndata[f"{feat}_t"] = (
                src_w * g.ndata[f"{feat}_0"]
                + dst_w * g.ndata[f"{feat}_1_true"]
            )

        e_idx = self.canonical_feat_order.index("e")
        src_w = src_weights[:, e_idx][edge_batch_idx].unsqueeze(-1)
        dst_w = dst_weights[:, e_idx][edge_batch_idx].unsqueeze(-1)
        g.edata["e_t"] = src_w * g.edata["e_0"] + dst_w * g.edata["e_1_true"]

        return g
