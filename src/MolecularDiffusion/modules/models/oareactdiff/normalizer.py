"""Feature normalisation for OA-ReactDiff.

Ported verbatim from ``oa_reactdiff/diffusion/_normalizer.py``
(commit 543aaa8, MIT).

``FEATURE_MAPPING`` is load-bearing beyond normalisation: it also fixes the
**column order** of the flat ``xh`` tensor every part of this package speaks
in -- ``[pos (3) | one_hot (5) | charge (1)]`` = 9 columns, which is the
``node_nfs: [9, 9, 9]`` in the task config.

The released checkpoint was trained with ``norm_values=(1,1,1)`` and
``norm_biases=(0,0,0)``, i.e. **no normalisation at all**; the class is kept
because the diffusion loss divides by ``norm_values`` in several places and
would need those branches anyway.
"""

# ruff: noqa
# mypy: ignore-errors

from typing import Tuple, List, Dict


import torch
from torch import nn, Tensor

FEATURE_MAPPING = ["pos", "one_hot", "charge"]


class Normalizer(nn.Module):
    def __init__(
        self,
        norm_values: Tuple = (1.0, 1.0, 1.0),
        norm_biases: Tuple = (0.0, 0.0, 0.0),
        pos_dim: int = 3,
    ) -> None:
        super().__init__()
        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.pos_dim = pos_dim

    def normalize(self, representations: List[Dict]) -> List[Dict]:
        for ii in range(len(representations)):
            for jj, feature_type in enumerate(FEATURE_MAPPING):
                representations[ii][feature_type] = (
                    representations[ii][feature_type] - self.norm_biases[jj]
                ) / self.norm_values[jj]
        return representations

    def unnormalize(self, x: Tensor, ind: int) -> Tensor:
        return x * self.norm_values[ind] + self.norm_biases[ind]

    def unnormalize_z(self, z_combined: List[Tensor]) -> List[Tensor]:
        for ii in range(len(z_combined)):
            z_combined[ii][:, : self.pos_dim] = self.unnormalize(
                z_combined[ii][:, : self.pos_dim], 0
            )
            z_combined[ii][:, self.pos_dim : -1] = self.unnormalize(
                z_combined[ii][:, self.pos_dim : -1], 1
            )
            z_combined[ii][:, -1:] = self.unnormalize(z_combined[ii][:, -1:], 2)
        return z_combined
