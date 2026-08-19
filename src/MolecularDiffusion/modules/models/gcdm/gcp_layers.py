# ruff: noqa
"""Geometry-Complete Perceptron (GCP) blocks.

Near-verbatim port of ``src/models/components/gcpnet.py:35-931`` from
https://github.com/BioinfoMachineLearning/bio-diffusion (commit
``a328950c``), which adapts
https://github.com/BioinfoMachineLearning/GCPNet.

Changes from upstream, all mechanical -- **no module is renamed and no
``nn.Parameter`` shape changes**, so the published GCDM checkpoints map onto
these classes one-to-one:

- ``torchtyping``/``typeguard`` decorators stripped.
- the five ``omegaconf.DictConfig`` objects upstream threads through every
  constructor are replaced by two plain dataclasses, :class:`GCPModuleConfig`
  and :class:`GCPLayerConfig`, with the shipped QM9/GEOM values as defaults.
  ``copy.copy`` + attribute assignment (which upstream relies on for its
  ``soft_cfg`` / ``ff_cfg`` variants) still works on them.
- ``module_cfg.selected_GCP``, upstream a Hydra ``_partial_`` pointing at
  :class:`GCP2`, becomes a plain ``gcp_version: "GCP" | "GCP2"`` string.
- upstream's ``mp_cfg`` sub-config is flattened into
  :class:`GCPLayerConfig` (it only ever carried three used fields).
"""

from copy import copy
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Optional, Tuple

import torch
from torch import nn
from torch_scatter import scatter

from MolecularDiffusion.modules.models.gcdm.gcp_utils import (
    GCPDropout,
    GCPLayerNorm,
    ScalarVector,
    get_nonlinearity,
    is_identity,
    safe_norm,
    scalarize,
    vectorize,
)


@dataclass
class GCPModuleConfig:
    """Flattened ``configs/model/module_cfg/*_gcp_module.yaml``.

    Defaults are the values shipped for **both** QM9 and GEOM (the two
    upstream files are identical apart from their ``conditioning`` list).
    """

    gcp_version: str = "GCP2"
    norm_x_diff: bool = True
    scalar_gate: int = 0
    vector_gate: bool = True
    vector_residual: bool = False
    vector_frame_residual: bool = False
    frame_gate: bool = False
    sigma_frame_gate: bool = False
    scalar_nonlinearity: Optional[str] = "silu"
    vector_nonlinearity: Optional[str] = "silu"
    bottleneck: int = 4
    default_vector_residual: bool = False
    default_bottleneck: int = 4
    node_positions_weight: float = 1.0
    update_positions_with_vector_sum: bool = False
    ablate_frame_updates: bool = False
    ablate_scalars: bool = False
    ablate_vectors: bool = False
    nonlinearities: Optional[Tuple[Optional[str], Optional[str]]] = None

    def __post_init__(self) -> None:
        if self.nonlinearities is None:
            self.nonlinearities = (
                self.scalar_nonlinearity,
                self.vector_nonlinearity,
            )

    @property
    def selected_gcp(self):
        return {"GCP": GCP, "GCP2": GCP2}[self.gcp_version]


@dataclass
class GCPLayerConfig:
    """Flattened ``layer_cfg/*_gcp_interaction_layer.yaml`` + its ``mp_cfg``.

    Defaults are the shipped QM9/GEOM values (the two files are identical).
    """

    pre_norm: bool = False
    use_gcp_norm: bool = False
    use_gcp_dropout: bool = False
    use_scalar_message_attention: bool = True
    num_feedforward_layers: int = 1
    dropout: float = 0.0
    nonlinearity_slope: float = 1e-2
    # mp_cfg
    num_message_layers: int = 4
    self_message: bool = True
    use_residual_message_gcp: bool = True


class GCP(nn.Module):
    def __init__(
        self,
        input_dims: ScalarVector,
        output_dims: ScalarVector,
        nonlinearities: Tuple[Optional[str], Optional[str]] = ("silu", "silu"),
        scalar_out_nonlinearity: Optional[str] = "silu",
        scalar_gate: int = 0,
        vector_gate: bool = True,
        frame_gate: bool = False,
        sigma_frame_gate: bool = False,
        feedforward_out: bool = False,
        bottleneck: int = 1,
        vector_residual: bool = False,
        vector_frame_residual: bool = False,
        ablate_frame_updates: bool = False,
        ablate_scalars: bool = False,
        ablate_vectors: bool = False,
        scalarization_vectorization_output_dim: int = 3,
        **kwargs,
    ):
        super().__init__()

        if nonlinearities is None:
            nonlinearities = (None, None)

        self.scalar_input_dim, self.vector_input_dim = input_dims
        self.scalar_output_dim, self.vector_output_dim = output_dims
        self.scalar_nonlinearity, self.vector_nonlinearity = (
            get_nonlinearity(nonlinearities[0], return_functional=True),
            get_nonlinearity(nonlinearities[1], return_functional=True),
        )
        self.scalar_gate, self.vector_gate, self.frame_gate, self.sigma_frame_gate = (
            scalar_gate,
            vector_gate,
            frame_gate,
            sigma_frame_gate,
        )
        self.vector_residual, self.vector_frame_residual = (
            vector_residual,
            vector_frame_residual,
        )
        self.ablate_frame_updates = ablate_frame_updates
        self.ablate_scalars, self.ablate_vectors = ablate_scalars, ablate_vectors

        if self.scalar_gate > 0:
            self.norm = nn.LayerNorm(self.scalar_output_dim)

        if self.vector_input_dim:
            assert self.vector_input_dim % bottleneck == 0, (
                f"Input channel of vector ({self.vector_input_dim}) must be "
                f"divisible with bottleneck factor ({bottleneck})"
            )

            self.hidden_dim = (
                self.vector_input_dim // bottleneck
                if bottleneck > 1
                else max(self.vector_input_dim, self.vector_output_dim)
            )

            self.vector_down = nn.Linear(
                self.vector_input_dim, self.hidden_dim, bias=False
            )
            self.scalar_out = (
                nn.Sequential(
                    nn.Linear(
                        self.hidden_dim + self.scalar_input_dim,
                        self.scalar_output_dim,
                    ),
                    get_nonlinearity(scalar_out_nonlinearity),
                    nn.Linear(self.scalar_output_dim, self.scalar_output_dim),
                )
                if feedforward_out
                else nn.Linear(
                    self.hidden_dim + self.scalar_input_dim,
                    self.scalar_output_dim,
                )
            )

            if self.vector_output_dim:
                self.vector_up = nn.Linear(
                    self.hidden_dim, self.vector_output_dim, bias=False
                )
                if self.vector_gate:
                    self.vector_out_scale = nn.Linear(
                        self.scalar_output_dim, self.vector_output_dim
                    )

            if not self.ablate_frame_updates:
                vector_down_frames_input_dim = (
                    self.hidden_dim
                    if not self.vector_output_dim
                    else self.vector_output_dim
                )
                self.vector_down_frames = nn.Linear(
                    vector_down_frames_input_dim,
                    scalarization_vectorization_output_dim,
                    bias=False,
                )
                self.scalar_out_frames = nn.Linear(
                    self.scalar_output_dim
                    + scalarization_vectorization_output_dim * 3,
                    self.scalar_output_dim,
                )

                if self.vector_output_dim and self.sigma_frame_gate:
                    self.vector_out_scale_sigma_frames = nn.Linear(
                        self.scalar_output_dim, self.vector_output_dim
                    )
                elif self.vector_output_dim and self.frame_gate:
                    self.vector_out_scale_frames = nn.Linear(
                        self.scalar_output_dim,
                        scalarization_vectorization_output_dim * 3,
                    )
                    self.vector_up_frames = nn.Linear(
                        scalarization_vectorization_output_dim,
                        self.vector_output_dim,
                        bias=False,
                    )
        else:
            self.scalar_out = (
                nn.Sequential(
                    nn.Linear(self.scalar_input_dim, self.scalar_output_dim),
                    get_nonlinearity(scalar_out_nonlinearity),
                    nn.Linear(self.scalar_output_dim, self.scalar_output_dim),
                )
                if feedforward_out
                else nn.Linear(self.scalar_input_dim, self.scalar_output_dim)
            )

    def process_vector(self, scalar_rep, v_pre, vector_hidden_rep):
        vector_rep = self.vector_up(vector_hidden_rep)
        if self.vector_residual:
            vector_rep = vector_rep + v_pre
        vector_rep = vector_rep.transpose(-1, -2)
        if self.vector_gate:
            gate = self.vector_out_scale(self.vector_nonlinearity(scalar_rep))
            vector_rep = vector_rep * torch.sigmoid(gate).unsqueeze(-1)
        elif not is_identity(self.vector_nonlinearity):
            vector_rep = vector_rep * self.vector_nonlinearity(
                safe_norm(vector_rep, dim=-1, keepdim=True)
            )
        return vector_rep

    def create_zero_vector(self, scalar_rep):
        return torch.zeros(
            scalar_rep.shape[0],
            self.vector_output_dim,
            3,
            device=scalar_rep.device,
        )

    def process_vector_frames(
        self,
        scalar_rep,
        v_pre,
        edge_index,
        frames,
        node_inputs,
        node_mask=None,
    ):
        vector_rep = v_pre.transpose(-1, -2)
        if self.sigma_frame_gate:
            gate = self.vector_out_scale_sigma_frames(
                self.vector_nonlinearity(scalar_rep)
            )
            vector_rep = vector_rep * torch.sigmoid(gate).unsqueeze(-1)
        elif self.frame_gate:
            gate = self.vector_out_scale_frames(
                self.vector_nonlinearity(scalar_rep)
            )
            gate_vector = vectorize(
                gate,
                edge_index,
                frames,
                node_inputs=node_inputs,
                dim_size=scalar_rep.shape[0],
                node_mask=node_mask,
            )
            gate_vector_rep = self.vector_up_frames(
                gate_vector.transpose(-1, -2)
            ).transpose(-1, -2)
            vector_rep = vector_rep * self.vector_nonlinearity(
                safe_norm(gate_vector_rep, dim=-1, keepdim=True)
            )
            if self.vector_frame_residual:
                vector_rep = vector_rep + v_pre.transpose(-1, -2)
        elif not is_identity(self.vector_nonlinearity):
            vector_rep = vector_rep * self.vector_nonlinearity(
                safe_norm(vector_rep, dim=-1, keepdim=True)
            )
        return vector_rep

    def forward(
        self,
        s_maybe_v,
        edge_index,
        frames,
        node_inputs: bool = False,
        node_mask=None,
    ):
        if self.vector_input_dim:
            scalar_rep, vector_rep = s_maybe_v
            scalar_rep = (
                torch.zeros_like(scalar_rep)
                if self.ablate_scalars
                else scalar_rep
            )
            vector_rep = (
                torch.zeros_like(vector_rep)
                if self.ablate_vectors
                else vector_rep
            )
            v_pre = vector_rep.transpose(-1, -2)

            vector_hidden_rep = self.vector_down(v_pre)
            vector_norm = safe_norm(vector_hidden_rep, dim=-2)
            merged = torch.cat((scalar_rep, vector_norm), dim=-1)
        else:
            merged = s_maybe_v
            merged = (
                torch.zeros_like(merged) if self.ablate_scalars else merged
            )

        scalar_rep = self.scalar_out(merged)

        if self.vector_input_dim and self.vector_output_dim:
            vector_rep = self.process_vector(
                scalar_rep, v_pre, vector_hidden_rep
            )

        scalar_rep = self.scalar_nonlinearity(scalar_rep)
        vector_rep = (
            self.create_zero_vector(scalar_rep)
            if self.vector_output_dim and not self.vector_input_dim
            else vector_rep
        )

        if self.ablate_frame_updates:
            return (
                ScalarVector(scalar_rep, vector_rep)
                if self.vector_output_dim
                else scalar_rep
            )

        # GCP: update scalar features using complete local frames
        v_pre = vector_rep.transpose(-1, -2)
        vector_hidden_rep = self.vector_down_frames(v_pre)
        scalar_hidden_rep = scalarize(
            vector_hidden_rep.transpose(-1, -2),
            edge_index,
            frames,
            node_inputs=node_inputs,
            dim_size=vector_hidden_rep.shape[0],
            node_mask=node_mask,
        )
        merged = torch.cat((scalar_rep, scalar_hidden_rep), dim=-1)

        scalar_rep = self.scalar_out_frames(merged)

        if not self.vector_output_dim:
            scalar_rep = (
                torch.zeros_like(scalar_rep)
                if self.ablate_scalars
                else scalar_rep
            )
            return self.scalar_nonlinearity(scalar_rep)

        # GCP: update vector features using complete local frames
        if self.vector_input_dim and self.vector_output_dim:
            vector_rep = self.process_vector_frames(
                scalar_rep,
                v_pre,
                edge_index,
                frames,
                node_inputs=node_inputs,
                node_mask=node_mask,
            )

        scalar_rep = self.scalar_nonlinearity(scalar_rep)
        scalar_rep = (
            torch.zeros_like(scalar_rep) if self.ablate_scalars else scalar_rep
        )
        vector_rep = (
            torch.zeros_like(vector_rep) if self.ablate_vectors else vector_rep
        )
        return ScalarVector(scalar_rep, vector_rep)


class GCP2(nn.Module):
    def __init__(
        self,
        input_dims: ScalarVector,
        output_dims: ScalarVector,
        nonlinearities: Tuple[Optional[str], Optional[str]] = ("silu", "silu"),
        scalar_out_nonlinearity: Optional[str] = "silu",
        scalar_gate: int = 0,
        vector_gate: bool = True,
        frame_gate: bool = False,
        sigma_frame_gate: bool = False,
        feedforward_out: bool = False,
        bottleneck: int = 1,
        vector_residual: bool = False,
        vector_frame_residual: bool = False,
        ablate_frame_updates: bool = False,
        ablate_scalars: bool = False,
        ablate_vectors: bool = False,
        scalarization_vectorization_output_dim: int = 3,
        **kwargs,
    ):
        super().__init__()

        if nonlinearities is None:
            nonlinearities = (None, None)

        self.scalar_input_dim, self.vector_input_dim = input_dims
        self.scalar_output_dim, self.vector_output_dim = output_dims
        self.scalar_nonlinearity, self.vector_nonlinearity = (
            get_nonlinearity(nonlinearities[0], return_functional=True),
            get_nonlinearity(nonlinearities[1], return_functional=True),
        )
        self.scalar_gate, self.vector_gate, self.frame_gate, self.sigma_frame_gate = (
            scalar_gate,
            vector_gate,
            frame_gate,
            sigma_frame_gate,
        )
        self.vector_residual, self.vector_frame_residual = (
            vector_residual,
            vector_frame_residual,
        )
        self.ablate_frame_updates = ablate_frame_updates
        self.ablate_scalars, self.ablate_vectors = ablate_scalars, ablate_vectors

        if self.scalar_gate > 0:
            self.norm = nn.LayerNorm(self.scalar_output_dim)

        if self.vector_input_dim:
            assert self.vector_input_dim % bottleneck == 0, (
                f"Input channel of vector ({self.vector_input_dim}) must be "
                f"divisible with bottleneck factor ({bottleneck})"
            )

            self.hidden_dim = (
                self.vector_input_dim // bottleneck
                if bottleneck > 1
                else max(self.vector_input_dim, self.vector_output_dim)
            )

            scalar_vector_frame_dim = (
                (scalarization_vectorization_output_dim * 3)
                if not self.ablate_frame_updates
                else 0
            )
            self.vector_down = nn.Linear(
                self.vector_input_dim, self.hidden_dim, bias=False
            )
            self.scalar_out = (
                nn.Sequential(
                    nn.Linear(
                        self.hidden_dim
                        + self.scalar_input_dim
                        + scalar_vector_frame_dim,
                        self.scalar_output_dim,
                    ),
                    get_nonlinearity(scalar_out_nonlinearity),
                    nn.Linear(self.scalar_output_dim, self.scalar_output_dim),
                )
                if feedforward_out
                else nn.Linear(
                    self.hidden_dim
                    + self.scalar_input_dim
                    + scalar_vector_frame_dim,
                    self.scalar_output_dim,
                )
            )

            if not self.ablate_frame_updates:
                self.vector_down_frames = nn.Linear(
                    self.vector_input_dim,
                    scalarization_vectorization_output_dim,
                    bias=False,
                )

            if self.vector_output_dim:
                self.vector_up = nn.Linear(
                    self.hidden_dim, self.vector_output_dim, bias=False
                )
                if not self.ablate_frame_updates:
                    if self.frame_gate:
                        self.vector_out_scale_frames = nn.Linear(
                            self.scalar_output_dim,
                            scalarization_vectorization_output_dim * 3,
                        )
                        self.vector_up_frames = nn.Linear(
                            scalarization_vectorization_output_dim,
                            self.vector_output_dim,
                            bias=False,
                        )
                    elif self.vector_gate:
                        self.vector_out_scale = nn.Linear(
                            self.scalar_output_dim, self.vector_output_dim
                        )
                elif self.vector_gate:
                    self.vector_out_scale = nn.Linear(
                        self.scalar_output_dim, self.vector_output_dim
                    )
        else:
            self.scalar_out = (
                nn.Sequential(
                    nn.Linear(self.scalar_input_dim, self.scalar_output_dim),
                    get_nonlinearity(scalar_out_nonlinearity),
                    nn.Linear(self.scalar_output_dim, self.scalar_output_dim),
                )
                if feedforward_out
                else nn.Linear(self.scalar_input_dim, self.scalar_output_dim)
            )

    def create_zero_vector(self, scalar_rep):
        return torch.zeros(
            scalar_rep.shape[0],
            self.vector_output_dim,
            3,
            device=scalar_rep.device,
        )

    def process_vector_without_frames(
        self, scalar_rep, v_pre, vector_hidden_rep
    ):
        vector_rep = self.vector_up(vector_hidden_rep)
        if self.vector_residual:
            vector_rep = vector_rep + v_pre
        vector_rep = vector_rep.transpose(-1, -2)

        if self.vector_gate:
            gate = self.vector_out_scale(self.vector_nonlinearity(scalar_rep))
            vector_rep = vector_rep * torch.sigmoid(gate).unsqueeze(-1)
        elif not is_identity(self.vector_nonlinearity):
            vector_rep = vector_rep * self.vector_nonlinearity(
                safe_norm(vector_rep, dim=-1, keepdim=True)
            )
        return vector_rep

    def process_vector_with_frames(
        self,
        scalar_rep,
        v_pre,
        vector_hidden_rep,
        edge_index,
        frames,
        node_inputs,
        node_mask=None,
    ):
        vector_rep = self.vector_up(vector_hidden_rep)
        if self.vector_residual:
            vector_rep = vector_rep + v_pre
        vector_rep = vector_rep.transpose(-1, -2)

        if self.frame_gate:
            gate = self.vector_out_scale_frames(
                self.vector_nonlinearity(scalar_rep)
            )
            gate_vector = vectorize(
                gate,
                edge_index,
                frames,
                node_inputs=node_inputs,
                dim_size=scalar_rep.shape[0],
                node_mask=node_mask,
            )
            gate_vector_rep = self.vector_up_frames(
                gate_vector.transpose(-1, -2)
            ).transpose(-1, -2)
            vector_rep = vector_rep * self.vector_nonlinearity(
                safe_norm(gate_vector_rep, dim=-1, keepdim=True)
            )
        elif self.vector_gate:
            gate = self.vector_out_scale(self.vector_nonlinearity(scalar_rep))
            vector_rep = vector_rep * torch.sigmoid(gate).unsqueeze(-1)
        elif not is_identity(self.vector_nonlinearity):
            vector_rep = vector_rep * self.vector_nonlinearity(
                safe_norm(vector_rep, dim=-1, keepdim=True)
            )
        return vector_rep

    def forward(
        self,
        s_maybe_v,
        edge_index,
        frames,
        node_inputs: bool = False,
        node_mask=None,
    ):
        if self.vector_input_dim:
            scalar_rep, vector_rep = s_maybe_v
            scalar_rep = (
                torch.zeros_like(scalar_rep)
                if self.ablate_scalars
                else scalar_rep
            )
            vector_rep = (
                torch.zeros_like(vector_rep)
                if self.ablate_vectors
                else vector_rep
            )
            v_pre = vector_rep.transpose(-1, -2)

            vector_hidden_rep = self.vector_down(v_pre)
            vector_norm = safe_norm(vector_hidden_rep, dim=-2)
            merged = torch.cat((scalar_rep, vector_norm), dim=-1)

            if not self.ablate_frame_updates:
                # GCP2: curate direction-robust scalar geometric features
                vector_down_frames_hidden_rep = self.vector_down_frames(v_pre)
                scalar_hidden_rep = scalarize(
                    vector_down_frames_hidden_rep.transpose(-1, -2),
                    edge_index,
                    frames,
                    node_inputs=node_inputs,
                    dim_size=vector_down_frames_hidden_rep.shape[0],
                    node_mask=node_mask,
                )
                merged = torch.cat((merged, scalar_hidden_rep), dim=-1)
        else:
            merged = s_maybe_v

        scalar_rep = self.scalar_out(merged)

        if not self.vector_output_dim:
            scalar_rep = (
                torch.zeros_like(scalar_rep)
                if self.ablate_scalars
                else scalar_rep
            )
            return self.scalar_nonlinearity(scalar_rep)
        elif self.vector_output_dim and not self.vector_input_dim:
            vector_rep = self.create_zero_vector(scalar_rep)
        elif self.ablate_frame_updates:
            vector_rep = self.process_vector_without_frames(
                scalar_rep, v_pre, vector_hidden_rep
            )
        else:
            vector_rep = self.process_vector_with_frames(
                scalar_rep,
                v_pre,
                vector_hidden_rep,
                edge_index,
                frames,
                node_inputs=node_inputs,
                node_mask=node_mask,
            )

        scalar_rep = self.scalar_nonlinearity(scalar_rep)
        scalar_rep = (
            torch.zeros_like(scalar_rep) if self.ablate_scalars else scalar_rep
        )
        vector_rep = (
            torch.zeros_like(vector_rep) if self.ablate_vectors else vector_rep
        )
        return ScalarVector(scalar_rep, vector_rep)


class GCPEmbedding(nn.Module):
    def __init__(
        self,
        edge_input_dims: ScalarVector,
        node_input_dims: ScalarVector,
        edge_hidden_dims: ScalarVector,
        node_hidden_dims: ScalarVector,
        num_atom_types: int,
        nonlinearities: Tuple[Optional[str], Optional[str]] = ("silu", "silu"),
        cfg: Optional[GCPModuleConfig] = None,
        pre_norm: bool = True,
        use_gcp_norm: bool = True,
    ):
        super().__init__()

        if num_atom_types > 0:
            self.atom_embedding = nn.Embedding(num_atom_types, num_atom_types)
        else:
            self.atom_embedding = None

        self.pre_norm = pre_norm
        if pre_norm:
            self.edge_normalization = GCPLayerNorm(
                edge_input_dims, use_gcp_norm=use_gcp_norm
            )
            self.node_normalization = GCPLayerNorm(
                node_input_dims, use_gcp_norm=use_gcp_norm
            )
        else:
            self.edge_normalization = GCPLayerNorm(
                edge_hidden_dims, use_gcp_norm=use_gcp_norm
            )
            self.node_normalization = GCPLayerNorm(
                node_hidden_dims, use_gcp_norm=use_gcp_norm
            )

        selected_gcp = cfg.selected_gcp
        self.edge_embedding = selected_gcp(
            edge_input_dims,
            edge_hidden_dims,
            nonlinearities=nonlinearities,
            scalar_gate=cfg.scalar_gate,
            vector_gate=cfg.vector_gate,
            frame_gate=cfg.frame_gate,
            sigma_frame_gate=cfg.sigma_frame_gate,
            vector_frame_residual=cfg.vector_frame_residual,
            ablate_frame_updates=cfg.ablate_frame_updates,
            ablate_scalars=cfg.ablate_scalars,
            ablate_vectors=cfg.ablate_vectors,
        )

        self.node_embedding = selected_gcp(
            node_input_dims,
            node_hidden_dims,
            nonlinearities=(None, None),
            scalar_gate=cfg.scalar_gate,
            vector_gate=cfg.vector_gate,
            frame_gate=cfg.frame_gate,
            sigma_frame_gate=cfg.sigma_frame_gate,
            vector_frame_residual=cfg.vector_frame_residual,
            ablate_frame_updates=cfg.ablate_frame_updates,
            ablate_scalars=cfg.ablate_scalars,
            ablate_vectors=cfg.ablate_vectors,
        )

    def forward(self, batch: Any):
        if self.atom_embedding is not None:
            node_rep = ScalarVector(self.atom_embedding(batch.h), batch.chi)
        else:
            node_rep = ScalarVector(batch.h, batch.chi)

        edge_rep = ScalarVector(batch.e, batch.xi)

        edge_rep = (
            edge_rep.scalar
            if not self.edge_embedding.vector_input_dim
            else edge_rep
        )
        node_rep = (
            node_rep.scalar
            if not self.node_embedding.vector_input_dim
            else node_rep
        )

        if self.pre_norm:
            edge_rep = self.edge_normalization(edge_rep)
            node_rep = self.node_normalization(node_rep)

        edge_rep = self.edge_embedding(
            edge_rep,
            batch.edge_index,
            batch.f_ij,
            node_inputs=False,
            node_mask=getattr(batch, "mask", None),
        )
        node_rep = self.node_embedding(
            node_rep,
            batch.edge_index,
            batch.f_ij,
            node_inputs=True,
            node_mask=getattr(batch, "mask", None),
        )

        if not self.pre_norm:
            edge_rep = self.edge_normalization(edge_rep)
            node_rep = self.node_normalization(node_rep)

        return node_rep, edge_rep


def get_GCP_with_custom_cfg(
    input_dims, output_dims, cfg: GCPModuleConfig, **kwargs
):
    """Upstream ``gcpnet.py:606``, with ``OmegaConf`` swapped for dataclasses."""
    cfg_dict = asdict(cfg)
    cfg_dict["nonlinearities"] = cfg.nonlinearities
    del cfg_dict["scalar_nonlinearity"]
    del cfg_dict["vector_nonlinearity"]
    del cfg_dict["gcp_version"]

    for key in kwargs:
        cfg_dict[key] = kwargs[key]

    return cfg.selected_gcp(input_dims, output_dims, **cfg_dict)


class GCPMessagePassing(nn.Module):
    def __init__(
        self,
        input_dims: ScalarVector,
        output_dims: ScalarVector,
        edge_dims: ScalarVector,
        cfg: GCPModuleConfig,
        mp_cfg: GCPLayerConfig,
        reduce_function: str = "sum",
        use_scalar_message_attention: bool = True,
    ):
        super().__init__()

        self.scalar_input_dim, self.vector_input_dim = input_dims
        self.scalar_output_dim, self.vector_output_dim = output_dims
        self.edge_scalar_dim, self.edge_vector_dim = edge_dims
        self.conv_cfg = mp_cfg
        self.self_message = self.conv_cfg.self_message
        self.reduce_function = reduce_function
        self.use_residual_message_gcp = self.conv_cfg.use_residual_message_gcp
        self.use_scalar_message_attention = use_scalar_message_attention

        scalars_in_dim = 2 * self.scalar_input_dim + self.edge_scalar_dim
        vectors_in_dim = 2 * self.vector_input_dim + self.edge_vector_dim

        soft_cfg = copy(cfg)
        soft_cfg.bottleneck, soft_cfg.vector_residual = (
            cfg.default_bottleneck,
            cfg.default_vector_residual,
        )

        primary_cfg_GCP = partial(get_GCP_with_custom_cfg, cfg=soft_cfg)
        secondary_cfg_GCP = partial(get_GCP_with_custom_cfg, cfg=cfg)

        module_list = [
            primary_cfg_GCP(
                (scalars_in_dim, vectors_in_dim),
                output_dims,
                nonlinearities=cfg.nonlinearities,
            )
        ]

        for _ in range(self.conv_cfg.num_message_layers - 2):
            module_list.append(secondary_cfg_GCP(output_dims, output_dims))

        if self.conv_cfg.num_message_layers > 1:
            module_list.append(
                primary_cfg_GCP(
                    output_dims,
                    output_dims,
                    nonlinearities=cfg.nonlinearities,
                )
            )

        self.message_fusion = nn.ModuleList(module_list)

        if use_scalar_message_attention:
            self.scalar_message_attention = nn.Sequential(
                nn.Linear(output_dims.scalar, 1), nn.Sigmoid()
            )

    def message(
        self, node_rep, edge_rep, edge_index, frames, node_mask=None
    ):
        row, col = edge_index
        vector = node_rep.vector.reshape(
            node_rep.vector.shape[0],
            node_rep.vector.shape[1] * node_rep.vector.shape[2],
        )
        vector_reshaped = ScalarVector(node_rep.scalar, vector)

        s_row, v_row = vector_reshaped.idx(row)
        s_col, v_col = vector_reshaped.idx(col)

        v_row = v_row.reshape(v_row.shape[0], v_row.shape[1] // 3, 3)
        v_col = v_col.reshape(v_col.shape[0], v_col.shape[1] // 3, 3)

        message = ScalarVector(s_row, v_row).concat(
            (edge_rep, ScalarVector(s_col, v_col))
        )

        if self.use_residual_message_gcp:
            message_residual = self.message_fusion[0](
                message,
                edge_index,
                frames,
                node_inputs=False,
                node_mask=node_mask,
            )
            for module in self.message_fusion[1:]:
                # ResGCP: exchange geometric messages while maintaining a
                # residual connection to the original message
                new_message = module(
                    message_residual,
                    edge_index,
                    frames,
                    node_inputs=False,
                    node_mask=node_mask,
                )
                message_residual = message_residual + new_message
        else:
            message_residual = message
            for module in self.message_fusion:
                message_residual = module(
                    message_residual,
                    edge_index,
                    frames,
                    node_inputs=False,
                    node_mask=node_mask,
                )

        if self.use_scalar_message_attention:
            message_residual_attn = self.scalar_message_attention(
                message_residual.scalar
            )
            message_residual = ScalarVector(
                message_residual.scalar * message_residual_attn,
                message_residual.vector,
            )

        return message_residual.flatten()

    def aggregate(self, message, edge_index, dim_size: int):
        row, col = edge_index
        return scatter(
            message,
            row,
            dim=0,
            dim_size=dim_size,
            reduce=self.reduce_function,
        )

    def forward(
        self, node_rep, edge_rep, edge_index, frames, node_mask=None
    ):
        message = self.message(
            node_rep, edge_rep, edge_index, frames, node_mask=node_mask
        )
        aggregate = self.aggregate(
            message, edge_index, dim_size=node_rep.scalar.shape[0]
        )
        return ScalarVector.recover(aggregate, self.vector_output_dim)


class GCPInteractions(nn.Module):
    def __init__(
        self,
        node_dims: ScalarVector,
        edge_dims: ScalarVector,
        cfg: GCPModuleConfig,
        layer_cfg: GCPLayerConfig,
        dropout: float = 0.0,
        nonlinearities: Optional[Tuple[Any, Any]] = None,
        update_node_positions: bool = False,
    ):
        super().__init__()

        if nonlinearities is None:
            nonlinearities = cfg.nonlinearities
        self.pre_norm = layer_cfg.pre_norm
        self.update_node_positions = update_node_positions
        self.node_positions_weight = getattr(cfg, "node_positions_weight", 1.0)
        self.update_positions_with_vector_sum = getattr(
            cfg, "update_positions_with_vector_sum", False
        )
        reduce_function = "sum"

        self.interaction = GCPMessagePassing(
            node_dims,
            node_dims,
            edge_dims,
            cfg=cfg,
            mp_cfg=layer_cfg,
            reduce_function=reduce_function,
            use_scalar_message_attention=layer_cfg.use_scalar_message_attention,
        )

        ff_cfg = copy(cfg)
        ff_cfg.nonlinearities = nonlinearities
        ff_without_res_cfg = copy(cfg)
        ff_without_res_cfg.vector_residual = False

        ff_GCP = partial(get_GCP_with_custom_cfg, cfg=ff_cfg)
        ff_without_res_GCP = partial(
            get_GCP_with_custom_cfg, cfg=ff_without_res_cfg
        )

        self.gcp_norm = nn.ModuleList(
            [GCPLayerNorm(node_dims, use_gcp_norm=layer_cfg.use_gcp_norm)]
        )
        self.gcp_dropout = nn.ModuleList(
            [GCPDropout(dropout, use_gcp_dropout=layer_cfg.use_gcp_dropout)]
        )

        hidden_dims = (
            (node_dims.scalar, node_dims.vector)
            if layer_cfg.num_feedforward_layers == 1
            else (4 * node_dims.scalar, 2 * node_dims.vector)
        )
        ff_interaction_layers = [
            ff_without_res_GCP(
                (node_dims.scalar * 2, node_dims.vector * 2),
                hidden_dims,
                nonlinearities=(None, None)
                if layer_cfg.num_feedforward_layers == 1
                else cfg.nonlinearities,
                feedforward_out=layer_cfg.num_feedforward_layers == 1,
            )
        ]

        interaction_layers = [
            ff_GCP(hidden_dims, hidden_dims)
            for _ in range(layer_cfg.num_feedforward_layers - 2)
        ]
        ff_interaction_layers.extend(interaction_layers)

        if layer_cfg.num_feedforward_layers > 1:
            ff_interaction_layers.append(
                ff_without_res_GCP(
                    hidden_dims,
                    node_dims,
                    nonlinearities=(None, None),
                    feedforward_out=True,
                )
            )

        self.feedforward_network = nn.ModuleList(ff_interaction_layers)

        if update_node_positions:
            position_output_dims = (
                node_dims
                if getattr(cfg, "update_positions_with_vector_sum", False)
                else (node_dims.scalar, 1)
            )
            self.node_position_update_gcp = ff_without_res_GCP(
                node_dims,
                position_output_dims,
                nonlinearities=cfg.nonlinearities,
            )

    def derive_x_update(self, node_rep, edge_index, f_ij, node_mask=None):
        # VectorUpdate: use vector-valued features to derive position updates
        node_rep_update = self.node_position_update_gcp(
            node_rep, edge_index, f_ij, node_inputs=True, node_mask=node_mask
        )
        if self.update_positions_with_vector_sum:
            x_vector_update = node_rep_update.vector.sum(1)
        else:
            x_vector_update = node_rep_update.vector.squeeze(1)

        return x_vector_update * self.node_positions_weight

    def forward(
        self,
        node_rep,
        edge_rep,
        edge_index,
        frames,
        node_mask=None,
        node_pos=None,
    ):
        node_rep = ScalarVector(node_rep[0], node_rep[1])
        edge_rep = ScalarVector(edge_rep[0], edge_rep[1])

        if self.pre_norm:
            node_rep = self.gcp_norm[0](node_rep)

        hidden_residual = self.interaction(
            node_rep, edge_rep, edge_index, frames, node_mask=node_mask
        )

        hidden_residual = ScalarVector(*hidden_residual.concat((node_rep,)))

        for module in self.feedforward_network:
            hidden_residual = module(
                hidden_residual,
                edge_index,
                frames,
                node_inputs=True,
                node_mask=node_mask,
            )

        node_rep = node_rep + self.gcp_dropout[0](hidden_residual)

        if not self.pre_norm:
            node_rep = self.gcp_norm[0](node_rep)

        if node_mask is not None:
            node_rep = node_rep.mask(node_mask.float())

        if not self.update_node_positions:
            return node_rep

        node_pos = node_pos + self.derive_x_update(
            node_rep, edge_index, frames, node_mask=node_mask
        )

        if node_mask is not None:
            node_pos = node_pos * node_mask.float().unsqueeze(-1)

        return node_rep, node_pos
