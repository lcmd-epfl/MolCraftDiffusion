"""Adapter binding GCDM's GCPNet denoiser to the platform's dense point-cloud
diffusion contract.

``EnVariationalDiffusion.phi`` (``modules/models/en_diffusion.py:234``) calls
``dynamics._forward(t, xh, node_mask, edge_mask, context)`` with **dense,
padded** tensors::

    xh        (B, N, 3 + in_node_nf)
    node_mask (B, N, 1)
    edge_mask (B, N*N, 1)      -- discarded here, see below
    context   (B, N, C) or None
    t         (B, 1) or a 0-d/1-element tensor

GCDM's ``GCPNetDynamics.atom_types_and_coords_forward``
(``src/models/components/gcpnet.py:1069``) instead consumes the **flat,
unpadded** node list of a PyG ``Batch``.  :class:`GCDMDynamics` bridges the
two by *compacting* the padded batch down to its real atoms, running the
ported network, and scattering the prediction back into the padded layout.

Compaction rather than a plain ``reshape(B*N, ...)`` is deliberate and
load-bearing for checkpoint fidelity: GCDM's equivariant node channel
``chi`` is ``_orientations`` -- forward/backward displacements between
*consecutive rows of the flat node list* -- so padding rows interleaved
between molecules would change those features.  Row-major compaction of a
dense ``(B, N, ...)`` batch reproduces exactly the concatenated node
ordering ``Batch.from_data_list`` gives upstream.

``edge_mask`` is discarded because GCPNet rebuilds its own fully-connected
intra-molecule edge list every step from the batch index
(``get_fully_connected_edge_index``, ``gcpnet.py:1056``); the platform's
``edge_mask`` is the identical mask outer-product.
"""

from typing import Any, Optional, Sequence

import torch
from torch import nn

from MolecularDiffusion.modules.models.gcdm.gcp_layers import (
    GCPEmbedding,
    GCPInteractions,
    GCPLayerConfig,
    GCPModuleConfig,
)
from MolecularDiffusion.modules.models.gcdm.gcp_utils import (
    ScalarVector,
    centralize,
    edge_features,
    localize,
    node_vector_features,
)


class _FlatBatch:
    """Minimal stand-in for the PyG ``Batch`` the ported blocks read.

    They only ever touch ``h``/``chi``/``e``/``xi``/``x``/``edge_index``/
    ``f_ij``/``mask``/``batch``/``num_nodes`` as attributes, plus ``batch[key]``
    item access from :func:`centralize`.  Importing ``torch_geometric`` just
    to get that would drag a real ``Batch``'s collation machinery in for no
    benefit.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)


class GCDMDynamics(nn.Module):
    """GCPNet denoiser, ported from ``GCPNetDynamics`` (``gcpnet.py:933``).

    All five upstream ``DictConfig`` arguments are flattened into named
    keyword arguments; the defaults below are the shipped QM9 preset
    (``configs/model/model_cfg/qm9_mol_gen_ddpm_gcp_model.yaml``).

    Parameters
    ----------
    in_node_nf:
        Number of node scalar channels the diffusion latent carries -- i.e.
        ``len(atom_vocab) + int(include_charges)``.  Upstream's
        ``num_atom_types + include_charges``.
    context_node_nf:
        Number of property-conditioning channels (upstream
        ``len(module_cfg.conditioning)``).  ``0`` disables context
        conditioning entirely.
    n_dims:
        Spatial dimensionality (3).
    num_encoder_layers:
        Number of :class:`GCPInteractions` blocks (QM9: 9, GEOM: 4).
    h_hidden_dim / chi_hidden_dim / e_hidden_dim / xi_hidden_dim:
        Hidden scalar/vector widths for nodes and edges.
    chi_input_dim / e_input_dim / xi_input_dim:
        Input widths of the geometric features GCPNet builds itself.
    self_condition:
        Ported for completeness; ``False`` in every shipped upstream config
        and doubles the input widths when on.
    """

    def __init__(
        self,
        in_node_nf: int,
        context_node_nf: int = 0,
        n_dims: int = 3,
        num_encoder_layers: int = 9,
        h_hidden_dim: int = 256,
        chi_hidden_dim: int = 32,
        e_hidden_dim: int = 64,
        xi_hidden_dim: int = 16,
        chi_input_dim: int = 2,
        e_input_dim: int = 1,
        xi_input_dim: int = 1,
        dropout: float = 0.0,
        condition_on_time: bool = True,
        self_condition: bool = False,
        module_cfg: Optional[GCPModuleConfig] = None,
        layer_cfg: Optional[GCPLayerConfig] = None,
    ) -> None:
        super().__init__()

        module_cfg = module_cfg or GCPModuleConfig()
        layer_cfg = layer_cfg or GCPLayerConfig()
        self.module_cfg = module_cfg
        self.layer_cfg = layer_cfg

        h_input_dim_ = in_node_nf
        h_input_conditioning_dim = int(condition_on_time)
        h_input_conditioning_dim += context_node_nf

        h_input_dim = h_input_dim_ * 2 if self_condition else h_input_dim_
        e_input_dim_ = e_input_dim * 2 if self_condition else e_input_dim
        chi_input_dim_ = chi_input_dim * 2 if self_condition else chi_input_dim
        xi_input_dim_ = xi_input_dim * 2 if self_condition else xi_input_dim

        self.edge_input_dims = ScalarVector(e_input_dim_, xi_input_dim_)
        self.node_input_dims = ScalarVector(
            h_input_dim + h_input_conditioning_dim, chi_input_dim_
        )
        self.edge_dims = ScalarVector(e_hidden_dim, xi_hidden_dim)
        self.node_dims = ScalarVector(h_hidden_dim, chi_hidden_dim)
        self.num_context_node_features = context_node_nf

        self.num_x_dims = n_dims
        self.n_dims = n_dims
        self.in_node_nf = in_node_nf
        self.norm_x_diff = module_cfg.norm_x_diff

        self.self_condition = self_condition
        self.condition_on_time = condition_on_time
        self.condition_on_context = context_node_nf > 0

        self.gcp_embedding = GCPEmbedding(
            self.edge_input_dims,
            self.node_input_dims,
            self.edge_dims,
            self.node_dims,
            num_atom_types=0,  # atom types arrive as float latents, not ids
            cfg=module_cfg,
            use_gcp_norm=layer_cfg.use_gcp_norm,
        )

        self.interaction_layers = nn.ModuleList(
            GCPInteractions(
                self.node_dims,
                self.edge_dims,
                cfg=module_cfg,
                layer_cfg=layer_cfg,
                dropout=dropout,
                update_node_positions=True,
            )
            for _ in range(num_encoder_layers)
        )

        h_input_dim_without_self_conditioning = (
            h_input_dim_ + h_input_conditioning_dim
        )
        self.scalar_node_projection_gcp = module_cfg.selected_gcp(
            self.node_dims,
            (h_input_dim_without_self_conditioning, 0),
            nonlinearities=(None, None),
            scalar_gate=module_cfg.scalar_gate,
            vector_gate=module_cfg.vector_gate,
            frame_gate=module_cfg.frame_gate,
            sigma_frame_gate=module_cfg.sigma_frame_gate,
            vector_frame_residual=module_cfg.vector_frame_residual,
            ablate_frame_updates=module_cfg.ablate_frame_updates,
            ablate_scalars=module_cfg.ablate_scalars,
            ablate_vectors=module_cfg.ablate_vectors,
        )

    # -- platform contract -------------------------------------------------
    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)

        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, t, xh, node_mask, edge_mask, context=None, **kwargs):
        """Dense ``(B, N, 3+nf)`` in, dense ``(B, N, 3+nf)`` out."""
        del edge_mask, kwargs  # rebuilt internally; see module docstring

        bs, n_nodes, n_feat = xh.shape
        keep = node_mask.reshape(bs * n_nodes).bool()

        xh_flat = xh.reshape(bs * n_nodes, n_feat)[keep]
        batch_index = torch.arange(bs, device=xh.device).repeat_interleave(
            n_nodes
        )[keep]

        if t.numel() == 1:
            t_flat = torch.full(
                (xh_flat.shape[0], 1),
                float(t.reshape(-1)[0]),
                device=xh.device,
                dtype=xh.dtype,
            )
        else:
            t_flat = t.reshape(bs, 1)[batch_index]

        if context is not None and self.condition_on_context:
            context_flat = context.reshape(bs * n_nodes, -1)[keep]
            if context_flat.shape[1] != self.num_context_node_features:
                raise ValueError(
                    f"GCDMDynamics was built for "
                    f"{self.num_context_node_features} context channel(s) "
                    f"but received {context_flat.shape[1]}."
                )
        else:
            context_flat = None

        net_out_flat = self._flat_forward(
            xh_flat, t_flat, batch_index, context_flat
        )

        net_out = torch.zeros(
            bs * n_nodes, n_feat, device=xh.device, dtype=net_out_flat.dtype
        )
        net_out[keep] = net_out_flat
        return net_out.reshape(bs, n_nodes, n_feat) * node_mask

    # -- ported network body -----------------------------------------------
    @staticmethod
    def get_fully_connected_edge_index(
        batch_index: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``gcpnet.py:1056``. Every intra-molecule ordered pair, self loops
        included."""
        adj = batch_index[:, None] == batch_index[None, :]
        edge_index = torch.stack(torch.where(adj), dim=0)
        if node_mask is not None:
            row, col = edge_index
            edge_mask = node_mask[row] & node_mask[col]
            edge_index = edge_index[:, edge_mask]
        return edge_index

    def _flat_forward(
        self,
        xh: torch.Tensor,
        t: torch.Tensor,
        batch_index: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        xh_self_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Port of ``atom_types_and_coords_forward`` (``gcpnet.py:1069``).

        Every node here is real, so the node mask is all-ones; it is still
        threaded through so the code path matches the one the published
        weights were trained under.
        """
        num_nodes = xh.shape[0]
        mask = torch.ones(
            num_nodes, dtype=torch.bool, device=xh.device
        )

        batch = _FlatBatch(
            batch=batch_index, mask=mask, num_nodes=num_nodes
        )

        xh = xh.clone() * mask.float().unsqueeze(-1)
        x_init = xh[:, : self.num_x_dims].clone()
        h_init = xh[:, self.num_x_dims :].clone()

        batch.edge_index = self.get_fully_connected_edge_index(
            batch_index=batch.batch, node_mask=batch.mask
        )

        batch.x = x_init
        batch.chi = node_vector_features(batch.x)
        batch.h = h_init
        batch.e, batch.xi = edge_features(batch.x, batch.edge_index)

        if self.self_condition:
            x_self_cond_ = (
                xh_self_cond[:, : self.num_x_dims].clone()
                if xh_self_cond is not None
                else torch.zeros_like(x_init)
            )
            h_self_cond = (
                xh_self_cond[:, self.num_x_dims :].clone()
                if xh_self_cond is not None
                else torch.zeros_like(h_init)
            )
            x_self_cond_chi = node_vector_features(x_self_cond_)
            x_self_cond_e, x_self_cond_xi = edge_features(
                x_self_cond_, batch.edge_index
            )
            batch.h = torch.cat((batch.h, h_self_cond), dim=-1)
            batch.chi = torch.cat((batch.chi, x_self_cond_chi), dim=1)
            batch.e = torch.cat((batch.e, x_self_cond_e), dim=-1)
            batch.xi = torch.cat((batch.xi, x_self_cond_xi), dim=1)

        if self.condition_on_time:
            batch.h = torch.cat((batch.h, t.view(num_nodes, 1)), dim=-1)

        if self.condition_on_context:
            if context is None:
                # Upstream raises here too (qm9_mol_gen_ddpm.py:657). Without
                # this the None flows into `.view()` and surfaces as an
                # opaque AttributeError several frames deep.
                raise ValueError(
                    "This GCDM checkpoint is context-conditional "
                    f"(num_context_node_features="
                    f"{self.num_context_node_features}) but no context was "
                    "supplied. Set `interference.target_values` (and "
                    "`property_names`) to the property value(s) you want, or "
                    "use an unconditional checkpoint."
                )
            batch.h = torch.cat(
                (
                    batch.h,
                    context.view(num_nodes, self.num_context_node_features),
                ),
                dim=-1,
            )

        # centralize so the process stays translation-invariant
        _, batch.x = centralize(
            batch,
            key="x",
            batch_index=batch.batch,
            node_mask=batch.mask,
            edm=True,
        )

        batch.f_ij = localize(
            batch.x,
            batch.edge_index,
            norm_x_diff=self.norm_x_diff,
            node_mask=batch.mask,
        )

        (h, chi), (e, xi) = self.gcp_embedding(batch)

        for layer in self.interaction_layers:
            (h, chi), batch.x = layer(
                (h, chi),
                (e, xi),
                batch.edge_index,
                batch.f_ij,
                node_mask=batch.mask,
                node_pos=batch.x,
            )

        h = self.scalar_node_projection_gcp(
            ScalarVector(h, chi),
            batch.edge_index,
            batch.f_ij,
            node_inputs=True,
            node_mask=batch.mask,
        )

        vel = (batch.x - x_init) * batch.mask.float().unsqueeze(-1)
        h_final = h

        if self.condition_on_context:
            h_final = h_final[:, : -self.num_context_node_features]
        if self.condition_on_time:
            h_final = h_final[:, :-1]

        if vel.isnan().any():
            vel = torch.zeros_like(vel)

        batch.vel = vel
        _, vel = centralize(
            batch,
            key="vel",
            batch_index=batch.batch,
            node_mask=batch.mask,
            edm=True,
        )

        return torch.cat((vel, h_final), dim=-1)


__all__: Sequence[str] = ["GCDMDynamics"]
