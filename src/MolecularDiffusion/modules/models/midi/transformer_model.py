"""MiDi's relational graph transformer with an EGNN-style coordinate update.

Ported verbatim in behaviour from ``midi/models/transformer_model.py``; the
only changes are import paths, the ``Dims`` dims container and docstrings.

Two upstream oddities are deliberately preserved because the released
checkpoints depend on them:

* ``NodeEdgeBlock.pre_softmax`` is never used in ``forward`` but is still
  registered (upstream comment: "Unused, but needed to load old checkpoints").
* every layer is built with ``last_layer=False``, including the last one, so
  the ``y``-branch modules exist in all 12 layers (upstream line 339).

Removing either would change the parameter set and silently break the weight
mapping, since ``cli/generate.py`` loads with ``strict=False``.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

from .diffusion_utils import assert_correctly_masked
from .layers import (
    EtoX,
    Etoy,
    PositionsMLP,
    SE3Norm,
    Xtoy,
    masked_softmax,
)
from .placeholder import Dims, PlaceHolder, remove_mean_with_mask


class NodeEdgeBlock(nn.Module):
    """Self-attention over nodes that also updates edges, ``y`` and positions."""

    def __init__(
        self, dx: int, de: int, dy: int, n_head: int, *, last_layer: bool = False
    ) -> None:
        super().__init__()
        if dx % n_head != 0:
            msg = f"dx: {dx} -- nhead: {n_head}"
            raise ValueError(msg)
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        self.in_E = Linear(de, de)

        # FiLM X to E
        self.x_e_mul1 = Linear(dx, de)
        self.x_e_mul2 = Linear(dx, de)

        # Distance encoding
        self.lin_dist1 = Linear(2, de)
        self.lin_norm_pos1 = Linear(1, de)
        self.lin_norm_pos2 = Linear(1, de)

        self.dist_add_e = Linear(de, de)
        self.dist_mul_e = Linear(de, de)

        # Attention
        self.k = Linear(dx, dx)
        self.q = Linear(dx, dx)
        self.v = Linear(dx, dx)
        self.a = Linear(dx, n_head, bias=False)
        self.out = Linear(dx * n_head, dx)

        # Incorporate E to X
        self.e_att_mul = Linear(de, n_head)
        self.pos_att_mul = Linear(de, n_head)
        self.e_x_mul = EtoX(de, dx)
        self.pos_x_mul = EtoX(de, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, de)
        self.y_e_add = Linear(dy, de)

        # Unused in forward; kept so released checkpoints map 1:1.
        self.pre_softmax = Linear(de, dx)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.last_layer = last_layer
        if not last_layer:
            self.y_y = Linear(dy, dy)
            self.x_y = Xtoy(dx, dy)
            self.e_y = Etoy(de, dy)
            self.dist_y = Etoy(de, dy)

        # Process positions
        self.e_pos1 = Linear(de, de, bias=False)
        self.e_pos2 = Linear(de, 1, bias=False)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(de, de)
        if not last_layer:
            self.y_out = nn.Sequential(
                nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy)
            )

    def forward(  # noqa: PLR0914
        self,
        X: torch.Tensor,  # noqa: N803
        E: torch.Tensor,  # noqa: N803
        y: torch.Tensor,
        pos: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Return ``(Xout, Eout, y_out, vel)``, all with the input shapes."""
        _bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        # 0. Geometry: pairwise distances and cosines.
        pos = pos * x_mask
        norm_pos = torch.norm(pos, dim=-1, keepdim=True)  # bs, n, 1
        normalized_pos = pos / (norm_pos + 1e-7)  # bs, n, 3

        pairwise_dist = torch.cdist(pos, pos).unsqueeze(-1).float()
        cosines = torch.sum(
            normalized_pos.unsqueeze(1) * normalized_pos.unsqueeze(2),
            dim=-1,
            keepdim=True,
        )
        pos_info = torch.cat((pairwise_dist, cosines), dim=-1)

        norm1 = self.lin_norm_pos1(norm_pos)  # bs, n, de
        norm2 = self.lin_norm_pos2(norm_pos)  # bs, n, de
        dist1 = (
            F.relu(
                self.lin_dist1(pos_info)
                + norm1.unsqueeze(2)
                + norm2.unsqueeze(1)
            )
            * e_mask1
            * e_mask2
        )

        # 1. Process E.
        Y = self.in_E(E)  # noqa: N806

        x_e_mul1 = self.x_e_mul1(X) * x_mask
        x_e_mul2 = self.x_e_mul2(X) * x_mask
        Y = (  # noqa: N806
            Y * x_e_mul1.unsqueeze(1) * x_e_mul2.unsqueeze(2) * e_mask1 * e_mask2
        )

        dist_add = self.dist_add_e(dist1)
        dist_mul = self.dist_mul_e(dist1)
        Y = (Y + dist_add + Y * dist_mul) * e_mask1 * e_mask2  # noqa: N806

        y_e_add = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        y_e_mul = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        E = (Y + y_e_add + Y * y_e_mul) * e_mask1 * e_mask2  # noqa: N806

        Eout = self.e_out(E) * e_mask1 * e_mask2  # noqa: N806
        assert_correctly_masked(Eout, e_mask1 * e_mask2)

        # 2. Node features / attention.
        Q = (self.q(X) * x_mask).unsqueeze(2)  # noqa: N806
        K = (self.k(X) * x_mask).unsqueeze(1)  # noqa: N806
        prod = Q * K / math.sqrt(Y.size(-1))
        a = self.a(prod) * e_mask1 * e_mask2  # bs, n, n, n_head

        a = a + self.e_att_mul(E) * a
        a = a + self.pos_att_mul(dist1) * a
        a = a * e_mask1 * e_mask2

        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)
        alpha = masked_softmax(a, softmax_mask, dim=2).unsqueeze(-1)
        V = (self.v(X) * x_mask).unsqueeze(1).unsqueeze(3)  # noqa: N806
        weighted_V = (alpha * V).sum(dim=2)  # noqa: N806
        weighted_V = weighted_V.flatten(start_dim=2)  # noqa: N806
        weighted_V = self.out(weighted_V) * x_mask  # noqa: N806

        weighted_V = weighted_V + self.e_x_mul(E, e_mask2) * weighted_V  # noqa: N806
        weighted_V = weighted_V + self.pos_x_mul(dist1, e_mask2) * weighted_V  # noqa: N806

        yx1 = self.y_x_add(y).unsqueeze(1)  # bs, 1, dx
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = weighted_V * (yx2 + 1) + yx1  # noqa: N806

        Xout = self.x_out(newX) * x_mask  # noqa: N806
        assert_correctly_masked(Xout, x_mask)

        # 3. Global feature.
        if self.last_layer:
            y_out = None
        else:
            y = self.y_y(y)
            e_y = self.e_y(Y, e_mask1, e_mask2)
            x_y = self.x_y(newX, x_mask)
            dist_y = self.dist_y(dist1, e_mask1, e_mask2)
            y_out = self.y_out(y + x_y + e_y + dist_y)  # bs, dy

        # 4. Coordinate update (EGNN-style, mean removed to stay equivariant).
        pos1 = pos.unsqueeze(1).expand(-1, n, -1, -1)
        pos2 = pos.unsqueeze(2).expand(-1, -1, n, -1)
        delta_pos = pos2 - pos1  # bs, n, n, 3

        messages = self.e_pos2(F.relu(self.e_pos1(Y)))  # bs, n, n, 1
        vel = (messages * delta_pos).sum(dim=2) * x_mask
        vel = remove_mean_with_mask(vel, node_mask)
        return Xout, Eout, y_out, vel


class XEyTransformerLayer(nn.Module):
    """One transformer layer updating nodes, edges, positions and ``y``."""

    def __init__(  # noqa: PLR0913
        self,
        dx: int,
        de: int,
        dy: int,
        n_head: int,
        dim_ffX: int = 2048,  # noqa: N803
        dim_ffE: int = 128,  # noqa: N803
        dim_ffy: int = 2048,  # noqa: N803
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        *,
        last_layer: bool = False,
    ) -> None:
        kw = {"device": device, "dtype": dtype}
        super().__init__()

        self.self_attn = NodeEdgeBlock(
            dx, de, dy, n_head, last_layer=last_layer
        )

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.norm_pos1 = SE3Norm(eps=layer_norm_eps, **kw)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.last_layer = last_layer
        if not last_layer:
            self.lin_y1 = Linear(dy, dim_ffy, **kw)
            self.lin_y2 = Linear(dim_ffy, dy, **kw)
            self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
            self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
            self.dropout_y1 = Dropout(dropout)
            self.dropout_y2 = Dropout(dropout)
            self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, features: PlaceHolder) -> PlaceHolder:
        """Residual update of ``X``, ``E``, ``y`` and ``pos``."""
        X = features.X  # noqa: N806
        E = features.E  # noqa: N806
        y = features.y
        pos = features.pos
        node_mask = features.node_mask
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1

        newX, newE, new_y, vel = self.self_attn(  # noqa: N806
            X, E, y, pos, node_mask=node_mask
        )

        X = self.normX1(X + self.dropoutX1(newX))  # noqa: N806
        new_pos = self.norm_pos1(vel, x_mask) + pos
        if torch.isnan(new_pos).any():
            msg = "NaN in new_pos"
            raise ValueError(msg)

        E = self.normE1(E + self.dropoutE1(newE))  # noqa: N806

        if not self.last_layer:
            y = self.norm_y1(y + self.dropout_y1(new_y))

        ff_outputX = self.linX2(  # noqa: N806
            self.dropoutX2(self.activation(self.linX1(X)))
        )
        X = self.normX2(X + self.dropoutX3(ff_outputX))  # noqa: N806

        ff_outputE = self.linE2(  # noqa: N806
            self.dropoutE2(self.activation(self.linE1(E)))
        )
        E = self.normE2(E + self.dropoutE3(ff_outputE))  # noqa: N806
        E = 0.5 * (E + torch.transpose(E, 1, 2))  # noqa: N806

        if not self.last_layer:
            ff_output_y = self.lin_y2(
                self.dropout_y2(self.activation(self.lin_y1(y)))
            )
            y = self.norm_y2(y + self.dropout_y3(ff_output_y))

        return PlaceHolder(
            X=X, E=E, y=y, pos=new_pos, charges=None, node_mask=node_mask
        ).mask()


class GraphTransformer(nn.Module):
    """MiDi's denoiser: ``PlaceHolder -> PlaceHolder``.

    Args:
        input_dims: channel counts of the noised input (``y`` includes the
            timestep column).
        n_layers: number of :class:`XEyTransformerLayer` blocks.
        hidden_mlp_dims: widths of the input/output MLPs, keys
            ``X``/``E``/``y``/``pos``.
        hidden_dims: transformer widths, keys ``dx``/``de``/``dy``/``n_head``/
            ``dim_ffX``/``dim_ffE``/``dim_ffy``.
        output_dims: channel counts of the prediction (``y`` is 0 -- MiDi is
            unconditional).
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dims: Dims,
        n_layers: int,
        hidden_mlp_dims: dict,
        hidden_dims: dict,
        output_dims: Dims,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims.X
        self.out_dim_E = output_dims.E
        self.out_dim_y = output_dims.y
        self.out_dim_charges = output_dims.charges

        act_fn_in = nn.ReLU()
        act_fn_out = nn.ReLU()

        self.mlp_in_X = nn.Sequential(
            nn.Linear(input_dims.X + input_dims.charges, hidden_mlp_dims["X"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["X"], hidden_dims["dx"]),
            act_fn_in,
        )
        self.mlp_in_E = nn.Sequential(
            nn.Linear(input_dims.E, hidden_mlp_dims["E"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["E"], hidden_dims["de"]),
            act_fn_in,
        )
        self.mlp_in_y = nn.Sequential(
            nn.Linear(input_dims.y, hidden_mlp_dims["y"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["y"], hidden_dims["dy"]),
            act_fn_in,
        )
        self.mlp_in_pos = PositionsMLP(hidden_mlp_dims["pos"])

        # last_layer=False for every layer, upstream line 339: the released
        # checkpoints contain the y-branch weights of the final layer too.
        self.tf_layers = nn.ModuleList(
            [
                XEyTransformerLayer(
                    dx=hidden_dims["dx"],
                    de=hidden_dims["de"],
                    dy=hidden_dims["dy"],
                    n_head=hidden_dims["n_head"],
                    dim_ffX=hidden_dims["dim_ffX"],
                    dim_ffE=hidden_dims["dim_ffE"],
                    last_layer=False,
                )
                for _ in range(n_layers)
            ]
        )

        self.mlp_out_X = nn.Sequential(
            nn.Linear(hidden_dims["dx"], hidden_mlp_dims["X"]),
            act_fn_out,
            nn.Linear(
                hidden_mlp_dims["X"], output_dims.X + output_dims.charges
            ),
        )
        self.mlp_out_E = nn.Sequential(
            nn.Linear(hidden_dims["de"], hidden_mlp_dims["E"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["E"], output_dims.E),
        )
        self.mlp_out_pos = PositionsMLP(hidden_mlp_dims["pos"])

    def forward(self, data: PlaceHolder) -> PlaceHolder:
        """Denoise one batch; returns logits for the categorical modalities."""
        bs, n = data.X.shape[0], data.X.shape[1]
        node_mask = data.node_mask

        diag_mask = ~torch.eye(n, device=data.X.device, dtype=torch.bool)
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)
        X = torch.cat((data.X, data.charges), dim=-1)  # noqa: N806

        X_to_out = X[..., : self.out_dim_X + self.out_dim_charges]  # noqa: N806
        E_to_out = data.E[..., : self.out_dim_E]  # noqa: N806
        y_to_out = data.y[..., : self.out_dim_y]

        new_E = self.mlp_in_E(data.E)  # noqa: N806
        new_E = (new_E + new_E.transpose(1, 2)) / 2  # noqa: N806
        features = PlaceHolder(
            X=self.mlp_in_X(X),
            E=new_E,
            y=self.mlp_in_y(data.y),
            charges=None,
            pos=self.mlp_in_pos(data.pos, node_mask),
            node_mask=node_mask,
        ).mask()

        for layer in self.tf_layers:
            features = layer(features)

        X = self.mlp_out_X(features.X)  # noqa: N806
        E = self.mlp_out_E(features.E)  # noqa: N806
        pos = self.mlp_out_pos(features.pos, node_mask)

        X = X + X_to_out  # noqa: N806
        E = (E + E_to_out) * diag_mask  # noqa: N806
        E = 0.5 * (E + torch.transpose(E, 1, 2))  # noqa: N806

        return PlaceHolder(
            pos=pos,
            X=X[..., : self.out_dim_X],
            charges=X[..., self.out_dim_X :],
            E=E,
            y=y_to_out,
            node_mask=node_mask,
        ).mask()
