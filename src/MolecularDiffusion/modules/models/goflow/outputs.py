"""The atom-wise 3D output head that turns ``(q, mu)`` into a velocity.

Ported from ``gotennet/models/components/outputs.py:21-49`` (``SNNDense``,
what ``GatedEquivariantBlock`` needs) and ``:52-189`` (``GatedEquivariantBlock``,
``Atomwise3DOut``). ``Atomwise3DOutRTSP`` (upstream's own "TODO: modify to
accept r/ts/p", never imported by ``flow_matching/flow_module.py``) is not
ported -- dead code, not a fidelity gap.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.init import xavier_uniform_, zeros_

from .ops import Dense, shifted_softplus, str2act


class SNNDense(nn.Linear):
    """Fully connected linear layer with activation. Verbatim from
    ``outputs.py:21-49``."""

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True,
        activation: Union[Callable, nn.Module, None] = None,
        weight_init: Callable = xavier_uniform_, bias_init: Callable = zeros_,
    ) -> None:
        self.weight_init = weight_init
        self.bias_init = bias_init
        super().__init__(in_features, out_features, bias)
        self.activation = activation if activation is not None else nn.Identity()

    def reset_parameters(self) -> None:
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, input: Tensor) -> Tensor:  # noqa: A002 - upstream's own name
        y = F.linear(input, self.weight, self.bias)
        return self.activation(y)


class GatedEquivariantBlock(nn.Module):
    """Rotationally invariant/equivariant tensorial feature mixing.

    Verbatim from ``outputs.py:52-85``.
    """

    def __init__(self, n_sin, n_vin, n_sout, n_vout, n_hidden, activation=F.silu, sactivation=None) -> None:
        super().__init__()
        self.n_sin = n_sin
        self.n_vin = n_vin
        self.n_sout = n_sout
        self.n_vout = n_vout
        self.n_hidden = n_hidden
        self.mix_vectors = SNNDense(n_vin, 2 * n_vout, activation=None, bias=False)
        self.scalar_net = nn.Sequential(
            Dense(n_sin + n_vout, n_hidden, activation=activation),
            Dense(n_hidden, n_sout + n_vout, activation=None),
        )
        self.sactivation = sactivation

    def forward(self, scalars: Tensor, vectors: Tensor):
        vmix = self.mix_vectors(vectors)
        vectors_v, vectors_w = torch.split(vmix, self.n_vout, dim=-1)
        vectors_vn = torch.norm(vectors_v, dim=-2)

        ctx = torch.cat([scalars, vectors_vn], dim=-1)
        x = self.scalar_net(ctx)
        s_out, x = torch.split(x, [self.n_sout, self.n_vout], dim=-1)
        v_out = x.unsqueeze(-2) * vectors_w

        if self.sactivation:
            s_out = self.sactivation(s_out)

        return s_out, v_out


class Atomwise3DOut(nn.Module):
    """Two stacked :class:`GatedEquivariantBlock`\\ s mapping ``(q, mu[:,
    :3, :])`` to a per-atom 3D velocity. Verbatim from ``outputs.py:161-189``.
    """

    def __init__(self, n_in, n_hidden: Optional[int] = None, activation=shifted_softplus) -> None:
        super().__init__()
        if isinstance(activation, str):
            activation = str2act(activation)

        self.out_net = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    n_sin=n_in, n_vin=n_in, n_sout=n_hidden, n_vout=n_hidden,
                    n_hidden=n_hidden, activation=activation, sactivation=activation,
                ),
                GatedEquivariantBlock(
                    n_sin=n_hidden, n_vin=n_hidden, n_sout=1, n_vout=1,
                    n_hidden=n_hidden, activation=activation,
                ),
            ]
        )

    def forward(self, l0: Tensor, l1: Tensor) -> Tensor:
        for layer in self.out_net:
            l0, l1 = layer(l0, l1)
        return l1.squeeze()
