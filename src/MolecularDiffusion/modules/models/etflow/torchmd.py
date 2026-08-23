"""TorchMD-ET equivariant transformer, time-conditioned (ET-Flow's backbone).

Ported near-verbatim from ET-Flow's ``etflow/networks/torchmd_net/`` (MIT,
(c) 2024 Majdi Hassan, Nikhil Shenoy, Jungyoon Lee), which is itself a fork of
torchmd-net's ``TorchMD_ET``. **Module names and attribute names are load
bearing**: they are exactly the ones in the released Zenodo checkpoints
(``network.representation_model.*`` / ``network.output_model.*``), so the
conversion in ``docs/model_integrations/etflow/scripts/convert_checkpoint.py``
is an identity remap and can assert a strict bijection. Renaming anything here
breaks that.

What differs from upstream's file, and why:

* ``Scalar``, ``EquivariantVectorAndScalarOutput``, ``Distance``,
  ``GaussianSmearing``, ``ShiftedSoftplus`` and ``CoorsNorm`` are dropped.
  None of them is reachable from ``ETFlowTask``: the output head is fixed to
  ``EquivariantVectorOutput``, ``edge_index`` is always supplied, and
  upstream's own checkpoint schema types ``rbf_type``/``activation`` as
  ``Literal["expnorm"]``/``Literal["silu"]`` (``commons/configs.py:102,104``).
  ``norm_coors`` was never settable from ``BaseFlow`` either
  (``models/model.py:74-95`` does not pass it), so ``coors_norm`` was always
  ``nn.Identity``. Dropping them changes no ``state_dict`` key.
* The abstract ``OutputModel`` base is flattened into
  ``EquivariantVectorOutput`` -- it carried no parameters.

Two upstream quirks are kept deliberately, because the published weights were
trained with them and "fixing" either would silently invalidate the
checkpoints:

* ``edge_weight`` is the SQUARED interatomic distance, not the distance
  (``r_ij`` below). It is what feeds the RBF expansion and the cosine cutoff.
* ``EquivariantVectorOutput.pre_reduce`` adds ``pos`` and
  ``TorchMDDynamics.forward`` immediately subtracts it again.
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

#: ``activation`` / ``attn_activation`` strings -> module classes. Upstream
#: ships ``silu`` in every config and types the field ``Literal["silu"]``.
act_class_mapping = {
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


def center(pos: Tensor, batch: Tensor) -> Tensor:
    """Subtract each graph's centre of mass."""
    return pos - scatter(pos, batch, dim=0, reduce="mean")[batch]


class CosineCutoff(nn.Module):
    """Smooth 1 -> 0 envelope. Parameter-free."""

    def __init__(self, cutoff_lower: float = 0.0, cutoff_upper: float = 5.0):
        super().__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances: Tensor) -> Tensor:
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    torch.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs * (distances > self.cutoff_lower).float()
        cutoffs = 0.5 * (
            torch.cos(distances * torch.pi / self.cutoff_upper) + 1.0
        )
        return cutoffs * (distances < self.cutoff_upper).float()


class ExpNormalSmearing(nn.Module):
    """PhysNet-style exponential-normal radial basis.

    ``trainable=True`` (upstream's setting) registers ``means``/``betas`` as
    parameters, which is why they appear in the released checkpoints.
    """

    def __init__(
        self,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        num_rbf: int = 50,
        trainable: bool = True,
    ):
        super().__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self) -> tuple[Tensor, Tensor]:
        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self) -> None:
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means)
            ** 2
        )


class NeighborEmbedding(MessagePassing):
    """One distance-weighted neighbour aggregation before the transformer."""

    def __init__(
        self,
        hidden_channels: int,
        num_rbf: int,
        cutoff_lower: float,
        cutoff_upper: float,
        max_z: int = 100,
    ):
        super().__init__(aggr="add")
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.fill_(0)
        self.combine.bias.data.fill_(0)

    def forward(self, z, x, edge_index, edge_weight, edge_attr):
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        c = self.cutoff(edge_weight)
        w = self.distance_proj(edge_attr) * c.view(-1, 1)

        x_neighbors = self.embedding(z)
        x_neighbors = self.propagate(edge_index, x=x_neighbors, W=w, size=None)
        return self.combine(torch.cat([x, x_neighbors], dim=1))

    def message(self, x_j, W):
        return x_j * W


class GatedEquivariantBlock(nn.Module):
    """Gated equivariant block (Schuett et al. 2021), TorchMD-ET variant."""

    def __init__(  # noqa: PLR0913
        self,
        hidden_channels: int,
        out_channels: int,
        intermediate_channels: int | None = None,
        activation: str = "silu",
        scalar_activation: bool = False,
        vector_output: bool = False,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.vector_output = vector_output
        self.layer_norm = layer_norm

        proj_out_channels = 1 if vector_output else out_channels
        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, proj_out_channels, bias=False)

        act_class = act_class_mapping[activation]
        last_out = out_channels + 1 if vector_output else out_channels * 2
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            act_class(),
            nn.Linear(intermediate_channels, last_out),
        )
        if layer_norm:
            # NOTE the resulting index layout -- Linear(0), LayerNorm(1),
            # act(2), Linear(3) -- is what qm9-o3's checkpoint keys encode
            # (`update_net.1.*` / `update_net.3.*`), while drugs-o3, trained
            # with output_layer_norm=False, has `update_net.0/.2`.
            self.update_net = nn.Sequential(
                self.update_net[0],
                nn.LayerNorm(intermediate_channels),
                self.update_net[1],
                self.update_net[2],
            )

        self.act = act_class() if scalar_activation else None

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[-1].weight)
        self.update_net[-1].bias.data.fill_(0)

    def forward(self, x: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        vec1_buffer = self.vec1_proj(v)

        # Detach all-zero rows to avoid NaN gradients through torch.norm.
        vec1 = torch.zeros(
            vec1_buffer.size(0), vec1_buffer.size(2), device=vec1_buffer.device
        )
        mask = (vec1_buffer != 0).view(vec1_buffer.size(0), -1).all(dim=1)
        if not mask.all():
            warnings.warn(
                f"Skipping gradients for {(~mask).sum()} atoms whose vector "
                "features are zero (typically atoms outside every cutoff).",
                stacklevel=2,
            )
        vec1[mask] = torch.norm(vec1_buffer[mask], dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        if self.vector_output:
            out = self.update_net(x)
            x, v = out[:, : self.out_channels], out[:, self.out_channels :]
        else:
            x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)

        v = v.unsqueeze(1) * vec2
        if self.act is not None:
            x = self.act(x)
        return x, v


class EquivariantVectorOutput(nn.Module):
    """Two gated blocks mapping (scalar, vector) features to one 3-vector."""

    def __init__(
        self,
        hidden_channels: int,
        activation: str = "silu",
        reduce_op: str = "sum",  # noqa: ARG002 - kept for signature parity
        layer_norm: bool = False,
    ):
        super().__init__()
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                    layer_norm=layer_norm,
                ),
                GatedEquivariantBlock(
                    hidden_channels // 2,
                    hidden_channels,
                    activation=activation,
                    vector_output=True,
                    layer_norm=layer_norm,
                ),
            ]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x, v, z, pos, batch):  # noqa: ARG002
        for layer in self.output_network:
            x, v = layer(x, v)
        # `+ pos` here and `- pos` in TorchMDDynamics.forward cancel. Kept
        # verbatim: it is what the released weights were trained through.
        return x, v.squeeze() + pos


class EquivariantMultiHeadAttention(MessagePassing):
    """One time-conditioned equivariant attention block.

    Time does not get its own embedding: it is concatenated with the node
    features and the projected ``node_attr`` and mixed by ``mixing_mlp``
    INSIDE every block (upstream ``model_dynamics.py:116-118``).

    ``so3_equivariant`` splits the value projection into 4 parts instead of 3,
    adding a cross-product term -- SO(3)- rather than O(3)-equivariant, i.e.
    chirality-aware by construction. False in both ``o3`` checkpoints.
    """

    def __init__(  # noqa: PLR0913
        self,
        hidden_channels: int,
        num_rbf: int,
        distance_influence: str,
        num_heads: int,
        activation,
        attn_activation: str,
        cutoff_lower: float,
        cutoff_upper: float,
        node_attr_dim: int = 0,
        qk_norm: bool = False,
        so3_equivariant: bool = False,
    ):
        super().__init__(aggr="add", node_dim=0)
        if hidden_channels % num_heads != 0:
            msg = (
                f"hidden_channels ({hidden_channels}) must be divisible by "
                f"num_heads ({num_heads})"
            )
            raise ValueError(msg)

        self.so3_equivariant = so3_equivariant
        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.node_attr_dim = node_attr_dim
        self.qk_norm = qk_norm

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.act = activation()
        self.attn_activation = act_class_mapping[attn_activation]()
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        input_channels = (
            hidden_channels + 1 + (hidden_channels if node_attr_dim > 0 else 0)
        )
        self.mixing_mlp = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        if qk_norm:
            self.q_proj = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
            )
            self.k_proj = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
            )
        else:
            self.q_proj = nn.Linear(hidden_channels, hidden_channels)
            self.k_proj = nn.Linear(hidden_channels, hidden_channels)

        n_split = 3 + int(so3_equivariant)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels * n_split)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias=False)
        self.dk_proj = nn.Linear(num_rbf, hidden_channels)
        self.dv_proj = nn.Linear(num_rbf, hidden_channels * n_split)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.layernorm.reset_parameters()
        if self.qk_norm:
            self.q_proj[0].bias.data.fill_(0)
            nn.init.xavier_uniform_(self.q_proj[0].weight)
            self.k_proj[0].bias.data.fill_(0)
            nn.init.xavier_uniform_(self.k_proj[0].weight)
        else:
            self.q_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.q_proj.weight)
            self.k_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.k_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.xavier_uniform_(self.dk_proj.weight)
        self.dk_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.dv_proj.weight)
        self.dv_proj.bias.data.fill_(0)

    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij, t, node_attr):  # noqa: PLR0913
        x = self.mixing_mlp(torch.cat([x, t, node_attr], dim=1))
        x = self.layernorm(x)

        n_split = 3 + int(self.so3_equivariant)
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(
            -1, self.num_heads, self.head_dim * n_split
        )

        vec1, vec2, vec3 = torch.split(
            self.vec_proj(vec), self.hidden_channels, dim=-1
        )
        vec = vec.reshape(-1, 3, self.num_heads, self.head_dim)
        vec_dot = (vec1 * vec2).sum(dim=1)

        dk = self.act(self.dk_proj(f_ij)).reshape(
            -1, self.num_heads, self.head_dim
        )
        dv = self.act(self.dv_proj(f_ij)).reshape(
            -1, self.num_heads, self.head_dim * n_split
        )

        x, vec = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            vec=vec,
            dk=dk,
            dv=dv,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )
        x = x.reshape(-1, self.hidden_channels)
        vec = vec.reshape(-1, 3, self.hidden_channels)

        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dvec = vec3 * o1.unsqueeze(1) + vec
        dx = vec_dot * o2 + o3
        return dx, dvec

    def message(self, q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij):  # noqa: PLR0913
        attn = (q_i * k_j * dk).sum(dim=-1)
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        v_j = v_j * dv
        if self.so3_equivariant:
            x, vec1, vec2, vec3 = torch.split(v_j, self.head_dim, dim=2)
        else:
            x, vec1, vec2 = torch.split(v_j, self.head_dim, dim=2)
            vec3 = None

        x = x * attn.unsqueeze(2)
        d = d_ij.unsqueeze(2).unsqueeze(3)
        vec = vec_j * vec1.unsqueeze(1) + vec2.unsqueeze(1) * d
        if self.so3_equivariant:
            vec = vec + vec3.unsqueeze(1) * torch.cross(d, vec_j, dim=1)
        return x, vec

    def aggregate(self, features, index, ptr, dim_size):  # noqa: ARG002
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(self, inputs):
        return inputs


class TorchMD_ET_dynamics(nn.Module):  # noqa: N801 - upstream name, checkpoint key
    """The stack of time-conditioned equivariant attention blocks."""

    def __init__(  # noqa: PLR0913
        self,
        hidden_channels: int = 128,
        num_layers: int = 6,
        num_rbf: int = 50,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = True,
        activation: str = "silu",
        attn_activation: str = "silu",
        neighbor_embedding: bool = True,
        num_heads: int = 8,
        distance_influence: str = "both",
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 10.0,
        max_z: int = 100,
        node_attr_dim: int = 0,
        edge_attr_dim: int = 0,
        qk_norm: bool = False,
        clip_during_norm: bool = False,
        so3_equivariant: bool = False,
    ):
        super().__init__()
        if distance_influence not in ("keys", "values", "both", "none"):
            msg = f"unknown distance_influence {distance_influence!r}"
            raise ValueError(msg)
        if rbf_type != "expnorm":
            msg = (
                f"rbf_type {rbf_type!r} is not ported; upstream ships and types "
                "only 'expnorm' (commons/configs.py:102)."
            )
            raise ValueError(msg)
        for name, value in (
            ("activation", activation),
            ("attn_activation", attn_activation),
        ):
            if value not in act_class_mapping:
                msg = (
                    f"unknown {name} {value!r}; choose from "
                    f"{sorted(act_class_mapping)}"
                )
                raise ValueError(msg)

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.num_heads = num_heads
        self.distance_influence = distance_influence
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_z = max_z
        self.node_attr_dim = node_attr_dim
        self.edge_attr_dim = edge_attr_dim
        self.clip_during_norm = clip_during_norm

        act_class = act_class_mapping[activation]
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance_expansion = ExpNormalSmearing(
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        self.neighbor_embedding = (
            NeighborEmbedding(
                hidden_channels,
                num_rbf + edge_attr_dim,
                cutoff_lower,
                cutoff_upper,
                max_z,
            )
            if neighbor_embedding
            else None
        )

        if node_attr_dim > 0:
            self.node_mlp = nn.Sequential(
                nn.Linear(node_attr_dim, hidden_channels),
                act_class(),
                nn.LayerNorm(hidden_channels),
                nn.Linear(hidden_channels, hidden_channels),
            )

        self.attention_layers = nn.ModuleList(
            EquivariantMultiHeadAttention(
                hidden_channels,
                num_rbf + edge_attr_dim,
                distance_influence,
                num_heads,
                act_class,
                attn_activation,
                cutoff_lower,
                cutoff_upper,
                node_attr_dim=node_attr_dim,
                qk_norm=qk_norm,
                so3_equivariant=so3_equivariant,
            )
            for _ in range(num_layers)
        )
        self.out_norm = nn.LayerNorm(hidden_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        if self.neighbor_embedding is not None:
            self.neighbor_embedding.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()
        self.out_norm.reset_parameters()

    def forward(  # noqa: PLR0913
        self,
        z: Tensor,
        t: Tensor,
        pos: Tensor,
        batch: Tensor,
        edge_index: Optional[Tensor] = None,
        node_attr: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
    ):
        if z.dim() > 1:
            z = z.squeeze()
        x = self.embedding(z)

        node_attr = self.node_mlp(node_attr) if self.node_attr_dim > 0 else None

        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        # Upstream quirk, kept: this is the SQUARED distance, and it is what
        # the RBF expansion and the cosine cutoff both consume.
        edge_weight = (edge_vec**2).sum(dim=-1, keepdim=False)

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(1)
            edge_attr = torch.cat(
                [self.distance_expansion(edge_weight), edge_attr], dim=-1
            )
        else:
            edge_attr = self.distance_expansion(edge_weight)

        mask = edge_index[0] == edge_index[1]
        masked_edge_weight = edge_weight.masked_fill(mask, 1).unsqueeze(1)
        if self.clip_during_norm:
            masked_edge_weight = masked_edge_weight.clamp(min=1.0e-2)
        edge_vec = edge_vec / masked_edge_weight

        if self.neighbor_embedding is not None:
            x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)

        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)
        for attn in self.attention_layers:
            dx, dvec = attn(
                x,
                vec,
                edge_index,
                edge_weight,
                edge_attr,
                edge_vec,
                node_attr=node_attr,
                t=t,
            )
            x = x + dx
            vec = vec + dvec
        x = self.out_norm(x)
        return x, vec, z, pos, batch


class TorchMDDynamics(nn.Module):
    """ET-Flow's vector field: ``(z, t, pos, edge_index) -> (N, 3)``.

    The returned field is centre-of-mass free per graph, which is what makes
    the flow stay in the zero-COM subspace the harmonic prior lives in.
    """

    def __init__(  # noqa: PLR0913
        self,
        hidden_channels: int = 128,
        num_layers: int = 8,
        num_rbf: int = 64,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = False,
        activation: str = "silu",
        neighbor_embedding: bool = True,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 10.0,
        max_z: int = 100,
        node_attr_dim: int = 0,
        edge_attr_dim: int = 0,
        attn_activation: str = "silu",
        num_heads: int = 8,
        distance_influence: str = "both",
        reduce_op: str = "sum",
        qk_norm: bool = False,
        output_layer_norm: bool = True,
        clip_during_norm: bool = False,
        so3_equivariant: bool = False,
    ):
        super().__init__()
        self.representation_model = TorchMD_ET_dynamics(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            rbf_type=rbf_type,
            trainable_rbf=trainable_rbf,
            activation=activation,
            neighbor_embedding=neighbor_embedding,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_z=max_z,
            attn_activation=attn_activation,
            num_heads=num_heads,
            distance_influence=distance_influence,
            node_attr_dim=node_attr_dim,
            edge_attr_dim=edge_attr_dim,
            qk_norm=qk_norm,
            clip_during_norm=clip_during_norm,
            so3_equivariant=so3_equivariant,
        )
        self.output_model = EquivariantVectorOutput(
            hidden_channels=hidden_channels,
            activation=activation,
            reduce_op=reduce_op,
            layer_norm=output_layer_norm,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()

    def forward(  # noqa: PLR0913
        self,
        z: Tensor,
        t: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Optional[Tensor] = None,
        node_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """Args mirror upstream's; ``t`` arrives already broadcast per atom."""
        x, v, z, pos, batch = self.representation_model(
            z=z,
            t=t,
            pos=pos,
            batch=batch,
            node_attr=node_attr,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        _, v = self.output_model.pre_reduce(x, v, z, pos, batch)
        return center(v - pos, batch)
