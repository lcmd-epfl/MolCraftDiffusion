"""LoQI self-conditioning module (diffusion variant only).

Ported from ``others/LoQI/src/megalodon/models/self_conditioning.py`` (NVIDIA,
Apache-2.0), narrowed to the single variable LoQI self-conditions on: ``x``,
with ``vector: True``. In that mode the fusion is a 2-in / 1-out linear stack
applied along a stacked last axis -- i.e. a learned mix of the current noisy
coordinates and the previous step's prediction. ``loqi_flow.yaml`` has no
``self_conditioning`` block at all, so the flow task builds this as ``None``.

``modules_dict`` and its ``x`` key are load-bearing: they are the
``self_conditioning_module.*.modules_dict.x.*`` keys of the released weights.
"""

from __future__ import annotations

import torch
from torch import nn


class BaseSelfConditioningModule(nn.Module):
    """Fuse ``<var>_t`` with the previous step's ``<var>_hat``."""

    def __init__(self, variables: list[dict]) -> None:
        super().__init__()
        self.modules_dict = nn.ModuleDict()
        self.keys: list[str] = []
        self.vector_mask: list[bool] = []
        self.fuse_softmax: list[bool] = []
        self.clamps: list[tuple] = []

        for var in variables:
            self.keys.append(var["variable_name"])
            self.vector_mask.append(bool(var["vector"]))
            self.fuse_softmax.append(bool(var["fuse_softmax"]))
            self.clamps.append((var.get("clamp_min"), var.get("clamp_max")))

            if not self.vector_mask[-1]:
                inp, out = 2 * var["inp_dim"], var["inp_dim"]
                activation = nn.SiLU() if var["fuse_softmax"] else nn.Identity()
                bias = True
            else:
                inp, out = 2, 1  # stacked (x_t, x_cond) along a new last axis
                activation = nn.Identity()
                bias = False
            self.modules_dict[self.keys[-1]] = nn.Sequential(
                nn.Linear(inp, var["hidden_dims"], bias=bias),
                activation,
                nn.Linear(var["hidden_dims"], out),
            )

    def forward(self, batch: dict, cond_batch: dict):
        non_fused_variables = {}
        for key, vec, fuse, clamp in zip(
            self.keys, self.vector_mask, self.fuse_softmax, self.clamps
        ):
            non_fused_variables[f"{key}_t"] = batch[f"{key}_t"].clone()
            x = batch[f"{key}_t"]

            if f"{key}_logits" in cond_batch and not fuse:
                x_cond = cond_batch[f"{key}_logits"]
            else:
                x_cond = cond_batch[f"{key}_hat"]

            if clamp[0] is not None or clamp[1] is not None:
                x_cond = torch.clamp(x_cond, min=clamp[0], max=clamp[1])

            if not vec:
                x = self.modules_dict[key](torch.cat([x, x_cond], dim=-1))
            else:
                x = self.modules_dict[key](torch.stack([x, x_cond], dim=-1))[..., 0]
            batch[f"{key}_t"] = batch[f"{key}_t"] + x

        return batch, non_fused_variables
