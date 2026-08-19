# ruff: noqa
"""Geometry helpers and small containers used by the GCP blocks.

Near-verbatim port of the pieces of the GCDM repo that ``gcpnet.py`` imports:

- ``src/models/components/__init__.py`` -> ``centralize``, ``localize``,
  ``scalarize``, ``vectorize``, ``safe_norm``, ``norm_no_nan``,
  ``is_identity``, ``ScalarVector``, ``VectorDropout``, ``GCPDropout``,
  ``GCPLayerNorm``
- ``src/models/__init__.py`` -> ``get_nonlinearity``
- ``src/datamodules/components/helper.py`` -> ``_normalize``
- ``src/datamodules/components/protein_graph_dataset.py`` -> ``_orientations``
- ``src/datamodules/components/edm_dataset.py`` -> ``_node_features`` /
  ``_edge_features``

Changes from upstream, all mechanical:
- ``torchtyping`` / ``typeguard`` decorators stripped (neither is a platform
  dependency); the shapes they documented are kept as docstrings.
- the plotting / PyMOL / wandb / ProDy helpers that shared the upstream module
  are not ported -- nothing in the denoiser touches them.
- ``_node_features`` / ``_edge_features`` are reduced to the ``edm_sampling``
  path, the only one the denoiser calls.
"""

from copy import copy
from functools import partial
from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter


# --------------------------------------------------------------------------
# nonlinearities (src/models/__init__.py:30)
# --------------------------------------------------------------------------
def get_nonlinearity(
    nonlinearity: Optional[str] = None,
    slope: float = 1e-2,
    return_functional: bool = False,
) -> Any:
    nonlinearity = (
        nonlinearity if nonlinearity is None else nonlinearity.lower().strip()
    )
    if nonlinearity == "relu":
        return F.relu if return_functional else nn.ReLU()
    elif nonlinearity == "leakyrelu":
        return (
            partial(F.leaky_relu, negative_slope=slope)
            if return_functional
            else nn.LeakyReLU(negative_slope=slope)
        )
    elif nonlinearity == "selu":
        return partial(F.selu) if return_functional else nn.SELU()
    elif nonlinearity == "silu":
        return partial(F.silu) if return_functional else nn.SiLU()
    elif nonlinearity == "sigmoid":
        return torch.sigmoid if return_functional else nn.Sigmoid()
    elif nonlinearity is None:
        return nn.Identity()
    raise NotImplementedError(
        f"The nonlinearity {nonlinearity} is currently not implemented."
    )


def is_identity(
    nonlinearity: Optional[Union[Callable, nn.Module]] = None,
) -> bool:
    return nonlinearity is None or isinstance(nonlinearity, nn.Identity)


def safe_norm(
    x: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
    keepdim: bool = False,
    sqrt: bool = True,
) -> torch.Tensor:
    norm = torch.sum(x**2, dim=dim, keepdim=keepdim)
    if sqrt:
        norm = torch.sqrt(norm + eps)
    return norm + eps


def norm_no_nan(
    x: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
    eps: float = 1e-8,
    sqrt: bool = True,
) -> torch.Tensor:
    """From https://github.com/drorlab/gvp-pytorch."""
    out = torch.clamp(
        torch.sum(torch.square(x), dim=dim, keepdim=keepdim), min=eps
    )
    return torch.sqrt(out) if sqrt else out


def _normalize(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """From https://github.com/drorlab/gvp-pytorch."""
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
    )


# --------------------------------------------------------------------------
# geometry (src/models/components/__init__.py:46-272)
# --------------------------------------------------------------------------
def centralize(
    batch: Any,
    key: str,
    batch_index: torch.Tensor,
    node_mask: Optional[torch.Tensor] = None,
    edm: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(centroid, centered)`` for ``batch[key]``.

    ``node_mask``: ``(batch_num_nodes,)`` bool.
    """
    if node_mask is not None:
        if edm:
            masked_max_abs_value = (
                batch[key][~node_mask].abs().sum().item()
            )
            assert (
                masked_max_abs_value < 1e-5
            ), f"Masked CoG error {masked_max_abs_value} is too high"

            num_entities = scatter(
                node_mask.float(), batch_index, dim=0, reduce="sum"
            ).unsqueeze(-1)
            entities_sum = scatter(
                batch[key], batch_index, dim=0, reduce="sum"
            )
            entities_centroid = entities_sum / num_entities
        else:
            entities_centroid = scatter(
                batch[key][node_mask],
                batch_index[node_mask],
                dim=0,
                reduce="mean",
            )

        if edm:
            entities_centered = batch[key] - (
                entities_centroid[batch_index] * node_mask.float().unsqueeze(-1)
            )
        else:
            masked_values = torch.ones_like(batch[key]) * torch.inf
            values = batch[key][node_mask]
            masked_values[node_mask] = (
                values - entities_centroid[batch_index][node_mask]
            )
            entities_centered = masked_values
    else:
        entities_centroid = scatter(
            batch[key], batch_index, dim=0, reduce="mean"
        )
        entities_centered = batch[key] - entities_centroid[batch_index]

    return entities_centroid, entities_centered


def localize(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    norm_x_diff: bool = True,
    node_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build the complete local frame of every edge.

    ``x``: ``(batch_num_nodes, 3)``; ``edge_index``: ``(2, batch_num_edges)``.
    Returns ``(batch_num_edges, 3, 3)``.
    """
    row, col = edge_index[0], edge_index[1]

    if node_mask is not None:
        edge_mask = node_mask[row] & node_mask[col]

        x_diff = (
            torch.ones((edge_index.shape[1], 3), device=edge_index.device)
            * torch.inf
        )
        x_diff[edge_mask] = x[row][edge_mask] - x[col][edge_mask]

        x_cross = (
            torch.ones((edge_index.shape[1], 3), device=edge_index.device)
            * torch.inf
        )
        x_cross[edge_mask] = torch.cross(x[row][edge_mask], x[col][edge_mask])
    else:
        x_diff = x[row] - x[col]
        x_cross = torch.cross(x[row], x[col])

    if norm_x_diff:
        if node_mask is not None:
            norm = torch.ones((edge_index.shape[1], 1), device=x_diff.device)
            norm[edge_mask] = (
                torch.sqrt(
                    torch.sum((x_diff[edge_mask] ** 2), dim=1).unsqueeze(1)
                )
            ) + 1
        else:
            norm = (
                torch.sqrt(torch.sum((x_diff) ** 2, dim=1).unsqueeze(1)) + 1
            )
        x_diff = x_diff / norm

        if node_mask is not None:
            cross_norm = torch.ones(
                (edge_index.shape[1], 1), device=x_cross.device
            )
            cross_norm[edge_mask] = (
                torch.sqrt(
                    torch.sum((x_cross[edge_mask]) ** 2, dim=1).unsqueeze(1)
                )
            ) + 1
        else:
            cross_norm = (
                torch.sqrt(torch.sum((x_cross) ** 2, dim=1).unsqueeze(1))
            ) + 1
        x_cross = x_cross / cross_norm

    if node_mask is not None:
        x_vertical = (
            torch.ones((edge_index.shape[1], 3), device=edge_index.device)
            * torch.inf
        )
        x_vertical[edge_mask] = torch.cross(
            x_diff[edge_mask], x_cross[edge_mask]
        )
    else:
        x_vertical = torch.cross(x_diff, x_cross)

    return torch.cat(
        (
            x_diff.unsqueeze(1),
            x_cross.unsqueeze(1),
            x_vertical.unsqueeze(1),
        ),
        dim=1,
    )


def scalarize(
    vector_rep: torch.Tensor,
    edge_index: torch.Tensor,
    frames: torch.Tensor,
    node_inputs: bool,
    dim_size: int,
    node_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Project ``(N, 3, 3)`` equivariant values onto local frames -> ``(N, 9)``."""
    row, col = edge_index[0], edge_index[1]

    vector_rep_i = vector_rep[row] if node_inputs else vector_rep

    if vector_rep_i.ndim == 2:
        vector_rep_i = vector_rep_i.unsqueeze(-1)
    elif vector_rep_i.ndim == 3:
        vector_rep_i = vector_rep_i.transpose(-1, -2)

    if node_mask is not None:
        edge_mask = node_mask[row] & node_mask[col]
        local_scalar_rep_i = torch.zeros(
            (edge_index.shape[1], 3, 3), device=edge_index.device
        )
        local_scalar_rep_i[edge_mask] = torch.matmul(
            frames[edge_mask], vector_rep_i[edge_mask]
        )
        local_scalar_rep_i = local_scalar_rep_i.transpose(-1, -2)
    else:
        local_scalar_rep_i = torch.matmul(frames, vector_rep_i).transpose(
            -1, -2
        )

    local_scalar_rep_i = local_scalar_rep_i.reshape(vector_rep_i.shape[0], 9)

    if node_inputs:
        # summarize according to source node indices, because GCP2's
        # equivariant frames are directional
        return scatter(
            local_scalar_rep_i,
            row,
            dim=0,
            dim_size=dim_size,
            reduce="mean",
        )

    return local_scalar_rep_i


def vectorize(
    gate: torch.Tensor,
    edge_index: torch.Tensor,
    frames: torch.Tensor,
    node_inputs: bool,
    dim_size: int,
    node_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Turn ``(N, 9)`` frame-space gates back into ``(N, 3, 3)`` vectors."""
    row, col = edge_index

    frames = frames.reshape(frames.shape[0], 1, 9)
    x_diff, x_cross, x_vertical = (
        frames[:, :, :3].squeeze(),
        frames[:, :, 3:6].squeeze(),
        frames[:, :, 6:].squeeze(),
    )

    gate = gate[row] if node_inputs else gate

    if node_mask is not None:
        edge_mask = node_mask[row] & node_mask[col]

    gate_vector = torch.zeros_like(gate)
    for i in range(0, gate.shape[-1], 3):
        if node_mask is not None:
            gate_vector[edge_mask, i : i + 3] = (
                gate[edge_mask, i : i + 1] * x_diff[edge_mask]
                + gate[edge_mask, i + 1 : i + 2] * x_cross[edge_mask]
                + gate[edge_mask, i + 2 : i + 3] * x_vertical[edge_mask]
            )
        else:
            gate_vector[:, i : i + 3] = (
                gate[:, i : i + 1] * x_diff
                + gate[:, i + 1 : i + 2] * x_cross
                + gate[:, i + 2 : i + 3] * x_vertical
            )
    gate_vector = gate_vector.reshape(gate_vector.shape[0], 3, 3)

    if node_inputs:
        return scatter(
            gate_vector, row, dim=0, dim_size=dim_size, reduce="mean"
        )

    return gate_vector


# --------------------------------------------------------------------------
# input featurization (src/datamodules/components/edm_dataset.py:22-76)
# --------------------------------------------------------------------------
def _orientations(x: torch.Tensor) -> torch.Tensor:
    """``(num_nodes, 3)`` -> ``(num_nodes, 2, 3)``.

    Forward/backward unit displacements along the *flat node ordering*.
    Upstream borrows this from its protein pipeline
    (``protein_graph_dataset.py:218``) and applies it to the concatenated,
    unpadded node list of a PyG ``Batch`` -- which is why the GCDM adapter
    compacts the platform's padded dense batch before calling it.
    """
    forward = _normalize(x[1:] - x[:-1])
    backward = _normalize(x[:-1] - x[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat(
        (forward.unsqueeze(-2), backward.unsqueeze(-2)), dim=-2
    )


def node_vector_features(coords: torch.Tensor) -> torch.Tensor:
    """``chi`` -- the 2 equivariant node channels GCPNet embeds."""
    return torch.nan_to_num(_orientations(coords))


def edge_features(
    coords: torch.Tensor, edge_index: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(e, xi)`` -- squared distance and unit displacement per edge."""
    e_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    radial = torch.sum(e_vectors**2, dim=1).unsqueeze(-1)
    edge_s = radial
    edge_v = _normalize(e_vectors).unsqueeze(-2)
    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))
    return edge_s, edge_v


# --------------------------------------------------------------------------
# containers (src/models/components/__init__.py:658-809)
# --------------------------------------------------------------------------
class ScalarVector(tuple):
    """From https://github.com/sarpaykent/GBPNet."""

    def __new__(cls, scalar, vector):
        return tuple.__new__(cls, (scalar, vector))

    def __getnewargs__(self):
        return self.scalar, self.vector

    @property
    def scalar(self):
        return self[0]

    @property
    def vector(self):
        return self[1]

    def __add__(self, other):
        if isinstance(other, tuple):
            scalar_other = other[0]
            vector_other = other[1]
        else:
            scalar_other = other.scalar
            vector_other = other.vector
        return ScalarVector(
            self.scalar + scalar_other, self.vector + vector_other
        )

    def __mul__(self, other):
        if isinstance(other, tuple):
            other = ScalarVector(other[0], other[1])
        if isinstance(other, ScalarVector):
            return ScalarVector(
                self.scalar * other.scalar, self.vector * other.vector
            )
        return ScalarVector(self.scalar * other, self.vector * other)

    def concat(self, others, dim=-1):
        dim %= len(self.scalar.shape)
        s_args, v_args = list(zip(*(self, *others)))
        return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)

    def flatten(self):
        flat_vector = torch.reshape(
            self.vector,
            self.vector.shape[:-2] + (3 * self.vector.shape[-2],),
        )
        return torch.cat((self.scalar, flat_vector), dim=-1)

    @staticmethod
    def recover(x, vector_dim):
        v = torch.reshape(
            x[..., -3 * vector_dim :], x.shape[:-1] + (vector_dim, 3)
        )
        s = x[..., : -3 * vector_dim]
        return ScalarVector(s, v)

    def vs(self):
        return self.scalar, self.vector

    def idx(self, idx):
        return ScalarVector(self.scalar[idx], self.vector[idx])

    def repeat(self, n, c=1, y=1):
        return ScalarVector(
            self.scalar.repeat(n, c), self.vector.repeat(n, y, c)
        )

    def clone(self):
        return ScalarVector(self.scalar.clone(), self.vector.clone())

    def mask(self, node_mask: torch.Tensor):
        return ScalarVector(
            self.scalar * node_mask[:, None],
            self.vector * node_mask[:, None, None],
        )

    def __setitem__(self, key, value):
        self.scalar[key] = value.scalar
        self.vector[key] = value.vector

    def __repr__(self):
        return f"ScalarVector({self.scalar}, {self.vector})"


class VectorDropout(nn.Module):
    """From https://github.com/drorlab/gvp-pytorch."""

    def __init__(self, drop_rate: float):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        device = x[0].device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        return mask * x / (1 - self.drop_rate)


class GCPDropout(nn.Module):
    """From https://github.com/drorlab/gvp-pytorch."""

    def __init__(self, drop_rate: float, use_gcp_dropout: bool = True):
        super().__init__()
        self.scalar_dropout = (
            nn.Dropout(drop_rate) if use_gcp_dropout else nn.Identity()
        )
        self.vector_dropout = (
            VectorDropout(drop_rate) if use_gcp_dropout else nn.Identity()
        )

    def forward(self, x: Union[torch.Tensor, ScalarVector]):
        if isinstance(x, torch.Tensor) and x.shape[0] == 0:
            return x
        elif isinstance(x, ScalarVector) and (
            x.scalar.shape[0] == 0 or x.vector.shape[0] == 0
        ):
            return x
        elif isinstance(x, torch.Tensor):
            return self.scalar_dropout(x)
        return ScalarVector(
            self.scalar_dropout(x[0]), self.vector_dropout(x[1])
        )


class GCPLayerNorm(nn.Module):
    """From https://github.com/drorlab/gvp-pytorch."""

    def __init__(
        self,
        dims: ScalarVector,
        eps: float = 1e-8,
        use_gcp_norm: bool = True,
    ):
        super().__init__()
        self.scalar_dims, self.vector_dims = dims
        self.scalar_norm = (
            nn.LayerNorm(self.scalar_dims) if use_gcp_norm else nn.Identity()
        )
        self.use_gcp_norm = use_gcp_norm
        self.eps = eps

    @staticmethod
    def norm_vector(
        v: torch.Tensor, use_gcp_norm: bool = True, eps: float = 1e-8
    ) -> torch.Tensor:
        v_norm = v
        if use_gcp_norm:
            vector_norm = torch.clamp(
                torch.sum(torch.square(v), dim=-1, keepdim=True), min=eps
            )
            vector_norm = torch.sqrt(
                torch.mean(vector_norm, dim=-2, keepdim=True)
            )
            v_norm = v / vector_norm
        return v_norm

    def forward(self, x: Union[torch.Tensor, ScalarVector]):
        if isinstance(x, torch.Tensor) and x.shape[0] == 0:
            return x
        elif isinstance(x, ScalarVector) and (
            x.scalar.shape[0] == 0 or x.vector.shape[0] == 0
        ):
            return x
        elif not self.vector_dims:
            return self.scalar_norm(x)
        s, v = x
        return ScalarVector(
            self.scalar_norm(s),
            self.norm_vector(
                v, use_gcp_norm=self.use_gcp_norm, eps=self.eps
            ),
        )


__all__ = [
    "GCPDropout",
    "GCPLayerNorm",
    "ScalarVector",
    "VectorDropout",
    "centralize",
    "edge_features",
    "get_nonlinearity",
    "is_identity",
    "localize",
    "node_vector_features",
    "norm_no_nan",
    "safe_norm",
    "scalarize",
    "vectorize",
]
