"""Retrieval-augmented linear attention over the ligand token block.

Ported from ``others/Apo2Mol/models/attn.py:17-44``.

**The retrieval path itself is out of scope** (``topk_prompt: 0`` in the
released ``configs/training.yaml``, so
``prompt_hbap_ligand_batch_all_list`` is always empty and the module is
called with ``h_retrieved = h``, i.e. self-attention). The module is still
ported because its four weight matrices are in the released checkpoint and
they do affect the forward pass in that degenerate configuration.

``CrossAttention`` from the same upstream file is not ported: it is imported
by ``models/molopt_score_model.py:15`` but never instantiated, so it
contributes no checkpoint tensors and has no call site.
"""

from __future__ import annotations

import torch
from torch import nn


class RetAugmentationLinearAttention(nn.Module):
    """Linear attention mixing a token block with a retrieved block.

    ``h`` and ``h_retrieved`` are ``(B, N, in_dim)`` -- note this is the one
    place in the model that uses a *padded* ligand block rather than the flat
    scatter layout (``molopt_score_model.py:390-414`` pads to 150 atoms and
    unpads afterwards).
    """

    def __init__(self, in_dim: int, d: int, context_dim: int) -> None:
        super().__init__()

        self.cond_flag = False
        if context_dim != in_dim:
            self.cond_flag = True
            self.to_cond = nn.Linear(context_dim, in_dim, bias=False)

        self.linear_attn = nn.Linear(in_dim, d, bias=False)
        self.to_k = nn.Linear(in_dim, d, bias=False)
        self.to_v = nn.Linear(in_dim, d, bias=False)
        self.out = nn.Linear(d, in_dim, bias=False)

    def forward(self, h: torch.Tensor, h_retrieved: torch.Tensor) -> torch.Tensor:
        if self.cond_flag:
            h_retrieved = self.to_cond(h_retrieved)

        attn = torch.softmax(self.linear_attn(h), dim=-1)
        if h.shape[-1] != h_retrieved.shape[-1]:
            raise ValueError(
                "h and h_retrieved must share their last dim; got "
                f"{h.shape[-1]} and {h_retrieved.shape[-1]}."
            )
        kv_in = torch.cat([h, h_retrieved], dim=1)
        k = self.to_k(kv_in)
        v = self.to_v(kv_in)
        f = torch.bmm(k.permute(0, 2, 1), v)

        h_aug = torch.bmm(attn, f)
        return h + self.out(h_aug)
