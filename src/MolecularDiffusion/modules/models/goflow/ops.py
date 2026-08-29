"""Supporting primitives GotenNet's forward path actually reaches.

Ported from ``gotennet/models/components/ops.py`` (1644 lines upstream; that
file also holds machinery for upstream's *other*, non-flow ``GotenNetModule``
energy-prediction model, which GoFlow's own ``configs/train.yaml`` never
selects). See ``docs/model_integrations/goflow/INTEGRATION_PLAN.md``,
Integration Plan, for the exact line ranges this was ported from.

Two reductions versus upstream, both because no shipped GoFlow config ever
selects the dropped branch (ponytail: only the reachable path is ported;
the line above names where to find the rest if that ever changes):

* ``str2act`` supports only ``""`` (-> ``None``) and ``"swish"`` (->
  ``nn.SiLU()``, upstream's own mapping for that string --
  ``ops.py:1319-1344``, ``get_activations``/``dictionary_to_option``).
  ``activation: swish`` is the only value any shipped GoFlow config ever
  sets.
* ``str2basis`` supports only ``"expnorm"`` (case-insensitive), so
  ``BesselBasis``/``GaussianRBF`` (``ops.py:281-343``) are not ported.
  ``radial_basis: expnorm`` is hardcoded in every shipped experiment config.

``NodeInit`` similarly drops the ``concat=True`` branch (a second embedding
table plus a differently-shaped ``distance_proj``/``message``): GotenNet
always constructs it with ``concat=False`` (``gotennet.py:506-509``), and
the dropped branch is otherwise unreachable code, not a fidelity gap.

``Distance`` (upstream ``ops.py:1473-1497``) is not ported at all: it is
constructed by ``GotenNet.__init__`` but its ``forward`` is never called --
``GotenNet.forward`` builds edges via ``_extend_condensed_graph_edge``
instead. Confirmed by reading the whole of ``gotennet.py``'s ``forward``:
no ``self.distance(`` call site exists. It carries no parameters, so
dropping it changes no checkpoint shape either way.

``.jittable()`` (called on every ``MessagePassing`` subclass in upstream's
``GotenNet.__init__``) is a documented no-op in the installed PyG (2.7):
``'X.jittable' is deprecated and a no-op. Please remove its usage.`` -- so
those calls are simply not reproduced here.
"""

from __future__ import annotations

import math
from functools import partial
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.init import constant_, xavier_uniform_
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot_orthogonal

zeros_initializer = partial(constant_, val=0.0)


def parse_update_info(edge_updates):
    """Decode the ``edge_updates`` config string into flag bits.

    Verbatim from ``ops.py:48-86``. GoFlow's shipped config sets
    ``edge_updates: norej`` (``configs/model/flow.yaml:39``), which flips
    only ``rej`` to ``False``.
    """
    update_info = {
        "gated": False,
        "rej": True,
        "vec_norm": False,
        "mlp": False,
        "mlpa": False,
        "lin_w": 0,
        "drej": False,
    }
    if isinstance(edge_updates, str):
        update_parts = edge_updates.split("_")
    else:
        update_parts = []

    allowed_parts = [
        "gated",
        "gatedt",
        "norej",
        "mlp",
        "mlpa",
        "act",
        "linw",
        "linwa",
        "drej",
    ]
    if not all(part in allowed_parts for part in update_parts):
        raise ValueError(
            f"Invalid edge update parts. Allowed parts are {allowed_parts}"
        )

    if "gated" in update_parts:
        update_info["gated"] = "gated"
    if "gatedt" in update_parts:
        update_info["gated"] = "gatedt"
    if "act" in update_parts:
        update_info["gated"] = "act"
    if "norej" in update_parts:
        update_info["rej"] = False
    if "mlp" in update_parts:
        update_info["mlp"] = True
    if "mlpa" in update_parts:
        update_info["mlpa"] = True
    if "linw" in update_parts:
        update_info["lin_w"] = 1
    if "linwa" in update_parts:
        update_info["lin_w"] = 2
    if "drej" in update_parts:
        update_info["drej"] = True
    return update_info


def shifted_softplus(x: torch.Tensor) -> torch.Tensor:
    """Verbatim from ``ops.py:103-104``."""
    return F.softplus(x) - math.log(2.0)


class CosineCutoff(nn.Module):
    """Verbatim from ``ops.py:147-158``.

    Instantiated directly by ``configs/tasks/diffusion_goflow.yaml``'s
    ``representation.cutoff_fn`` block (``_target_`` points here), exactly
    mirroring upstream's own ``configs/model/flow.yaml:26-29``.
    """

    def __init__(self, cutoff: float, scaling: float) -> None:
        super().__init__()
        if isinstance(cutoff, torch.Tensor):
            cutoff = cutoff.item()
        self.cutoff = cutoff
        self.scaling = scaling

    def forward(self, distances: Tensor) -> Tensor:
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        return self.scaling * cutoffs * (distances < self.cutoff).float()


def glorot_orthogonal_wrapper_(tensor: Tensor, scale: float = 2.0) -> Tensor:
    """Verbatim from ``ops.py:344-345``. Not selected by any shipped config
    (``weight_init: xavier_uniform``), ported for the switch's completeness.
    """
    return glorot_orthogonal(tensor, scale=scale)


def _standardize(kernel: Tensor) -> Tensor:
    """Verbatim from ``ops.py:348-361``."""
    eps = 1e-6
    axis = [0, 1] if len(kernel.shape) == 3 else 1
    var, mean = torch.var_mean(kernel, dim=axis, unbiased=True, keepdim=True)
    return (kernel - mean) / (var + eps) ** 0.5


def he_orthogonal_init(tensor: Tensor) -> Tensor:
    """Verbatim from ``ops.py:364-384``. Not selected by any shipped config."""
    tensor = torch.nn.init.orthogonal_(tensor)
    fan_in = (
        tensor.shape[:-1].numel() if len(tensor.shape) == 3 else tensor.shape[1]
    )
    with torch.no_grad():
        tensor.data = _standardize(tensor.data)
        tensor.data *= (1 / fan_in) ** 0.5
    return tensor


def get_weight_init_by_string(init_str: str):
    """Verbatim from ``ops.py:387-400``."""
    if init_str == "":
        return lambda x: x
    if init_str == "zeros":
        return torch.nn.init.zeros_
    if init_str == "xavier_uniform":
        return torch.nn.init.xavier_uniform_
    if init_str == "glo_orthogonal":
        return glorot_orthogonal_wrapper_
    if init_str == "he_orthogonal":
        return he_orthogonal_init
    raise ValueError(f"Unknown initialization {init_str}")


class Dense(nn.Linear):
    """Verbatim from ``ops.py:403-476``."""

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=None,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
        norm=None,
        gain=None,
    ) -> None:
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.gain = gain
        super().__init__(in_features, out_features, bias)
        self.activation = activation

        if norm == "layer":
            self.norm = nn.LayerNorm(out_features)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(out_features)
        elif norm == "instance":
            self.norm = nn.InstanceNorm1d(out_features)
        else:
            self.norm = None

    def reset_parameters(self) -> None:
        if self.gain:
            self.weight_init(self.weight, gain=self.gain)
        else:
            self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        y = super().forward(inputs)
        if self.norm is not None:
            y = self.norm(y)
        if self.activation:
            y = self.activation(y)
        return y


class TensorInit(nn.Module):
    """Real spherical-harmonic components up to ``l=2``.

    Verbatim from ``ops.py:543-575`` (the ``lmax in {1, 2}`` branches only).
    Every shipped GoFlow config fixes ``lmax: 2``
    (``configs/model/flow.yaml:43``, restated across every model-size
    ablation in the Hyperparameter Provenance table), so the ``lmax >= 3``
    branches (``ops.py:577-``, another ~80 lines of higher-order e3nn-style
    coefficients) are not ported -- ``forward`` raises for any other value
    rather than silently truncating.
    """

    def __init__(self, l: int = 2) -> None:
        super().__init__()
        if l not in (1, 2):
            raise NotImplementedError(
                f"TensorInit(l={l}): only l in {{1, 2}} is ported (every "
                "shipped GoFlow config fixes lmax=2). See ops.py:577- "
                "upstream for the l>=3 spherical-harmonic coefficients if "
                "a future config needs them."
            )
        self.l = l

    def forward(self, edge_vec: Tensor) -> Tensor:
        return self._calculate_components(
            self.l, edge_vec[..., 0], edge_vec[..., 1], edge_vec[..., 2]
        )

    @property
    def tensor_size(self) -> int:
        return ((self.l + 1) ** 2) - 1

    @staticmethod
    def _calculate_components(
        lmax: int, x: Tensor, y: Tensor, z: Tensor
    ) -> Tensor:
        sh_1_0, sh_1_1, sh_1_2 = x, y, z
        if lmax == 1:
            return torch.stack([sh_1_0, sh_1_1, sh_1_2], dim=-1)

        sh_2_0 = math.sqrt(3.0) * x * z
        sh_2_1 = math.sqrt(3.0) * x * y
        y2 = y.pow(2)
        x2z2 = x.pow(2) + z.pow(2)
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = math.sqrt(3.0) * y * z
        sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))
        return torch.stack(
            [sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4],
            dim=-1,
        )


class VecLayerNorm(nn.Module):
    """Verbatim from ``ops.py:1234-1304``.

    Every shipped GoFlow config leaves ``int_layer_norm``/``int_vector_norm``
    at ``""`` (``configs/model/flow.yaml:35-36``), so ``GATA`` builds
    ``nn.Identity()`` instead of this class in practice -- ported anyway
    since the switch (``rms``/``max_min``/none) is cheap and self-contained.
    """

    def __init__(self, hidden_channels, trainable, norm_type="max_min") -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.eps = 1e-12

        weight = torch.ones(self.hidden_channels)
        if trainable:
            self.register_parameter("weight", nn.Parameter(weight))
        else:
            self.register_buffer("weight", weight)

        if norm_type == "rms":
            self.norm = self.rms_norm
        elif norm_type == "max_min":
            self.norm = self.max_min_norm
        else:
            self.norm = self.none_norm

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight.data.copy_(torch.ones(self.hidden_channels))

    def none_norm(self, vec: Tensor) -> Tensor:
        return vec

    def rms_norm(self, vec: Tensor) -> Tensor:
        dist = torch.norm(vec, dim=1)
        if (dist == 0).all():
            return torch.zeros_like(vec)
        dist = dist.clamp(min=self.eps)
        dist = torch.sqrt(torch.mean(dist**2, dim=-1))
        return vec / F.relu(dist).unsqueeze(-1).unsqueeze(-1)

    def max_min_norm(self, vec: Tensor) -> Tensor:
        dist = torch.norm(vec, dim=1, keepdim=True)
        if (dist == 0).all():
            return torch.zeros_like(vec)
        dist = dist.clamp(min=self.eps)
        direct = vec / dist

        max_val, _ = torch.max(dist, dim=-1)
        min_val, _ = torch.min(dist, dim=-1)
        delta = (max_val - min_val).view(-1)
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        dist = (dist - min_val.view(-1, 1, 1)) / delta.view(-1, 1, 1)
        return F.relu(dist) * direct

    def forward(self, vec: Tensor) -> Tensor:
        if vec.shape[1] == 3:
            vec = self.norm(vec)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        if vec.shape[1] == 8:
            vec1, vec2 = torch.split(vec, [3, 5], dim=1)
            vec1 = self.norm(vec1)
            vec2 = self.norm(vec2)
            vec = torch.cat([vec1, vec2], dim=1)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        raise ValueError("VecLayerNorm only support 3 or 8 channels")


def str2act(input_str: str):
    """Reduced dispatcher -- see the module docstring for why."""
    if input_str == "":
        return None
    if input_str == "swish":
        return nn.SiLU()
    raise ValueError(
        f'Invalid activation "{input_str}": only "" and "swish" are ported '
        "(see the module docstring). Extend from ops.py:1319-1385 upstream "
        "if another value is genuinely needed."
    )


class ExpNormalSmearing(nn.Module):
    """Verbatim from ``ops.py:1388-1421``. The only radial basis any shipped
    GoFlow config selects (``radial_basis: expnorm``)."""

    def __init__(
        self, cutoff: float = 5.0, scaling: float = 1.0, n_rbf: int = 50,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(cutoff, torch.Tensor):
            cutoff = cutoff.item()
        self.cutoff = cutoff
        self.n_rbf = n_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(cutoff, scaling)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.n_rbf)
        betas = torch.tensor(
            [(2 / self.n_rbf * (1 - start_value)) ** -2] * self.n_rbf
        )
        return means, betas

    def reset_parameters(self) -> None:
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas * (torch.exp(self.alpha * (-dist)) - self.means) ** 2
        )


def str2basis(input_str: str):
    """Reduced dispatcher -- see the module docstring for why."""
    if not isinstance(input_str, str):
        return input_str
    if input_str.lower() == "expnorm":
        return ExpNormalSmearing
    raise ValueError(
        f'Unknown radial basis "{input_str}": only "expnorm" is ported '
        "(see the module docstring). BesselBasis/GaussianRBF are in "
        "ops.py:281-343 upstream if another basis is genuinely needed."
    )


class MLP(nn.Module):
    """Verbatim from ``ops.py:1440-1470``."""

    def __init__(
        self,
        hidden_dims: List[int],
        bias=True,
        activation=None,
        last_activation=None,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
        norm="",
    ) -> None:
        super().__init__()
        dims = hidden_dims
        n_layers = len(dims)
        dense_mlp = partial(Dense, bias=bias, weight_init=weight_init, bias_init=bias_init)

        self.dense_layers = nn.ModuleList(
            [
                dense_mlp(dims[i], dims[i + 1], activation=activation, norm=norm)
                for i in range(n_layers - 2)
            ]
            + [dense_mlp(dims[-2], dims[-1], activation=last_activation)]
        )
        self.layers = nn.Sequential(*self.dense_layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.dense_layers:
            m.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class AtomCGREmbedding(nn.Module):
    """Verbatim from ``ops.py:1500-1513``, with the ``einops.rearrange``
    replaced by the plain-``torch.cat`` equivalent (see
    ``INTEGRATION_PLAN.md``, Naming: for the ``a=2`` case used here,
    ``rearrange([z1, z2], 'a n d -> n (a d)', a=2)`` is exactly
    ``torch.cat([z1, z2], dim=-1)``)."""

    def __init__(self, n_atom_rdkit_feats: int, last_channel: int) -> None:
        super().__init__()
        self.half_last_channel_dim = last_channel // 2
        self.atom_embedding = nn.Embedding(100, self.half_last_channel_dim)
        self.atom_feat_embedding = nn.Linear(
            n_atom_rdkit_feats, self.half_last_channel_dim, bias=False
        )

    def forward(self, z_N: Tensor, r_feat_N_F: Tensor, p_feat_N_F: Tensor) -> Tensor:
        a_emb = self.atom_embedding(z_N)
        af_emb_r = self.atom_feat_embedding(r_feat_N_F.float())
        af_emb_p = self.atom_feat_embedding(p_feat_N_F.float())
        z1 = a_emb + af_emb_r
        z2 = af_emb_p - af_emb_r
        return torch.cat([z1, z2], dim=-1)


class swish(nn.Module):  # noqa: N801 - upstream's own (lowercase) class name
    """Verbatim from ``ops.py:1517-1522``. Functionally SiLU; kept as its
    own class because it lives inside ``EdgeCGREmbedding.edge_cat``'s
    ``nn.Sequential``, whose state-dict has no parameters here either way."""

    def forward(self, x: Tensor) -> Tensor:
        return x * x.sigmoid()


class EdgeCGREmbedding(nn.Module):
    """Verbatim from ``ops.py:1525-1539``, ``einops.rearrange`` replaced the
    same way as :class:`AtomCGREmbedding`."""

    def __init__(self, hidden_dim: int = 100) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bond_emb = nn.Embedding(100, hidden_dim)
        self.edge_cat = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            swish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, edge_type_r: Tensor, edge_type_p: Tensor) -> Tensor:
        edge_attr_r = self.bond_emb(edge_type_r)
        edge_attr_p = self.bond_emb(edge_type_p)
        return self.edge_cat(torch.cat([edge_attr_r, edge_attr_p], dim=-1))


class NodeInit(MessagePassing):
    """``concat=False`` only -- verbatim from ``ops.py:1542-1610`` with the
    ``concat=True`` branch dropped (see the module docstring: GotenNet never
    constructs this with ``concat=True``)."""

    def __init__(
        self,
        hidden_channels,
        n_atom_rdkit_feats,
        num_rbf,
        cutoff,
        scaling,
        max_z=100,
        activation=F.silu,
        proj_ln="",
        last_activation=False,
        weight_init=nn.init.xavier_uniform_,
        bias_init=nn.init.zeros_,
    ) -> None:
        super().__init__(aggr="add")
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]
        last_channel = hidden_channels[-1]

        self.atom_cgr_embedding = AtomCGREmbedding(n_atom_rdkit_feats, last_channel)
        self.edge_cgr_embedding = EdgeCGREmbedding(last_channel)
        self.distance_proj = MLP(
            [num_rbf] + [last_channel],
            activation=None,
            norm="",
            weight_init=weight_init,
            bias_init=bias_init,
            last_activation=None,
        )
        self.combine = MLP(
            [2 * last_channel] + hidden_channels,
            activation=activation,
            norm=proj_ln,
            weight_init=weight_init,
            bias_init=bias_init,
            last_activation=activation if last_activation else None,
        )
        self.cutoff = CosineCutoff(cutoff, scaling)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.distance_proj.reset_parameters()
        self.combine.reset_parameters()

    def forward(
        self, z, r_feat, p_feat, x, edge_index, edge_weight, edge_attr,
        edge_type_r, edge_type_p,
    ):
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        x_neighbors = self.atom_cgr_embedding(z, r_feat, p_feat)
        cutoff = self.cutoff(edge_weight)
        w = (
            self.edge_cgr_embedding(edge_type_r, edge_type_p)
            * self.distance_proj(edge_attr)
            * cutoff.view(-1, 1)
        )
        x_neighbors = self.propagate(edge_index, x=x_neighbors, s=x_neighbors, W=w, size=None)
        return self.combine(torch.cat([x, x_neighbors], dim=1))

    def message(self, s_i, x_j, W):  # noqa: N803 - upstream's own names
        return x_j * W


class EdgeInit(MessagePassing):
    """Verbatim from ``ops.py:1613-1644``. ``aggr=None``: this never
    actually calls ``propagate`` (its ``forward`` is a plain function call,
    kept as a ``MessagePassing`` subclass only because upstream jittable'd
    it alongside the others)."""

    def __init__(
        self,
        num_rbf,
        hidden_channels,
        activation=F.silu,
        proj_ln="",
        last_activation=False,
        weight_init=nn.init.xavier_uniform_,
        bias_init=nn.init.zeros_,
    ) -> None:
        super().__init__(aggr=None)
        self.activation = activation
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]
        self.edge_up = MLP(
            [num_rbf] + hidden_channels,
            activation=activation,
            norm=proj_ln,
            weight_init=weight_init,
            bias_init=bias_init,
            last_activation=activation if last_activation else None,
        )
        self.edge_cgr_embedding = EdgeCGREmbedding(hidden_channels[-1])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.edge_up.reset_parameters()

    def forward(self, edge_index, edge_attr, edge_type_r, edge_type_p):  # noqa: ARG002
        return self.edge_cgr_embedding(edge_type_r, edge_type_p) * self.edge_up(edge_attr)
