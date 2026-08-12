"""LigandDiff denoiser wrapper.

Port of the target repo's ``src/egnn.py``, restricted to the
``gvp_dynamics`` backbone.

ponytail: the ``egnn_dynamics`` branch (``GCL``/``EquivariantBlock``/``EGNN``
/``SinusoidsEmbeddingNew``/``coord2diff``, ~200 lines) is deliberately not
ported. Upstream's ``config.yml:28`` selects ``gvp_dynamics`` and the released
``model/pretrained.ckpt`` was trained with it (``hyper_parameters['model'] ==
'gvp_dynamics'``), so the EGNN branch is unreachable for every artifact this
integration ships. Port it from ``others/LigandDiff/src/egnn.py:69-279`` if a
future run actually wants it.
"""

import math
from typing import Callable, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_, zeros_
from torch_geometric.nn import radius_graph

from MolecularDiffusion.modules.models.ligandiff import utils
from MolecularDiffusion.modules.models.ligandiff.gvp_model import GVPNetwork


class ScaledSiLU(torch.nn.Module):
    """SiLU rescaled by 1/0.6 to keep unit variance."""

    def __init__(self) -> None:
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = nn.SiLU()

    def forward(self, x):
        return self._activation(x) * self.scale_factor


class DenseLayer(nn.Linear):
    """``nn.Linear`` with kaiming init and an optional fused activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module, None] = None,
        weight_init: Callable = kaiming_uniform_,
        bias_init: Callable = zeros_,
    ) -> None:
        self.weight_init = weight_init
        self.bias_init = bias_init
        super().__init__(in_features, out_features, bias)

        if isinstance(activation, str):
            activation = activation.lower()
        if activation in ["swish", "silu"]:
            self._activation = ScaledSiLU()
        elif activation is None:
            self._activation = nn.Identity()
        else:
            raise NotImplementedError("Activation function not implemented.")

    def reset_parameters(self) -> None:
        self.weight_init(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            self.bias_init(self.bias)


class Dynamics(nn.Module):
    """Epsilon-prediction denoiser over a flat (ragged) atom batch.

    Consumes ``xh = cat(x, h)`` with membership given by ``batch_seg``; edges
    are a runtime fully-connected geometric graph, never stored data.
    """

    def __init__(
        self,
        in_node_nf: int,
        n_dims: int,
        ligand_group_node_nf: int,
        hidden_nf: int = 32,
        activation: str = "silu",
        n_layers: int = 2,
        attention: bool = False,
        tanh: bool = True,
        norm_constant: float = 0.00001,
        inv_sublayers: int = 2,
        sin_embedding: bool = False,
        normalization_factor: float = 100,
        aggregation_method: str = "sum",
        drop_rate: float = 0.0,
        device: str = "cpu",
        model: str = "gvp_dynamics",
        normalization: str = "batch_norm",
        condition_time: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.n_dims = n_dims
        self.ligand_group_node_nf = ligand_group_node_nf + 1
        self.model = model

        self.ligand_group_embedding = DenseLayer(
            ligand_group_node_nf + 1, hidden_nf, activation=activation
        )
        self.h_embedding = DenseLayer(
            in_node_nf, hidden_nf, activation=activation
        )
        # Upstream recomputes the output width as
        # 8 + ligand_group_node_nf(6) + condition_time(1) = 15 (src/egnn.py:300)
        # and trims the trailing 7 back off after the backbone (:370). The
        # released checkpoint's h_embedding_out.weight is (15, 192) for exactly
        # this reason -- it is not a defect.
        in_node_nf = in_node_nf + ligand_group_node_nf + condition_time
        self.h_embedding_out = DenseLayer(
            hidden_nf, in_node_nf, activation=None
        )

        if self.model != "gvp_dynamics":
            raise NotImplementedError(
                f"LigandDiff port supports model='gvp_dynamics' only, got "
                f"{model!r}. See this module's docstring."
            )
        self.dynamics = GVPNetwork(
            in_dims=(hidden_nf * 2, 0),  # (scalar_features, vector_features)
            out_dims=(hidden_nf, 1),
            hidden_dims=(hidden_nf, hidden_nf // 2),
            drop_rate=drop_rate,
            vector_gate=True,
            num_layers=n_layers,
            attention=attention,
            normalization_factor=normalization_factor,
        )

    def forward(self, xh, t, ligand_diff, ligand_group, batch_seg):
        x = xh[:, : self.n_dims].clone()  # (N_total, 3)
        h = xh[:, self.n_dims :].clone()  # (N_total, nf)
        edge_index = radius_graph(
            x, r=1e50, batch=batch_seg, loop=False, max_num_neighbors=100
        )

        if np.prod(t.size()) == 1:
            h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
        else:
            h_time = t[batch_seg]

        ligand_group_with_time = torch.cat([ligand_group, h_time], dim=-1)
        ligand_group_with_time = self.ligand_group_embedding(
            ligand_group_with_time
        )
        h = self.h_embedding(h)
        h = torch.cat([h, ligand_group_with_time], dim=-1)

        h_final, vel = self.dynamics(h, x, edge_index)
        h_final = self.h_embedding_out(h_final)
        # Upstream is a bare `.squeeze()` (src/egnn.py:363); squeeze(-2) is
        # the same for the (N, 1, 3) output but does not also collapse the
        # atom axis when a batch happens to hold a single atom.
        vel = vel.squeeze(-2)

        if ligand_group is not None:
            h_final = h_final[:, : -self.ligand_group_node_nf]
        if torch.any(torch.isnan(vel)) or torch.any(torch.isnan(h_final)):
            raise utils.FoundNaNException(vel, h_final)

        return torch.cat([vel, h_final], dim=1)
