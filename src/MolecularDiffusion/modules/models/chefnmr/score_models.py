"""ChefNMR denoiser: a DiT over atom coordinates (MIT, (c) 2025 Ziyu Xiong).

Upstream: ``src/model/modules/score_models.py``, based on DiT
(facebookresearch/DiT); the ``x_embedder`` is copied from NExT-Mol.

The only per-atom input is ``cat([noisy_coords, atom_one_hot])`` -- there is
no edge tensor, no adjacency and no valency table anywhere in the forward
path. Conditioning (noise level + spectrum embedding) enters through
adaLN-zero in every block; the atom axis is masked attention. Equivariance
is **not** built in, it is trained in via
:func:`~MolecularDiffusion.modules.models.chefnmr.utils.center_random_augmentation`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
from torch import nn

from MolecularDiffusion.modules.models.chefnmr.embedders import NMRSpectraEmbedder
from MolecularDiffusion.modules.models.chefnmr.layers import (
    DiTBlock,
    FinalLayer,
    TimestepEmbedder,
)

_KNOWN_CONDITIONS = ("H1NMRSpectrum", "C13NMRSpectrum", "H1C13NMRSpectrum")


class DiffusionModuleTransformer(nn.Module):
    """Noisy coords + known formula + spectrum -> coordinate update."""

    def __init__(  # noqa: PLR0913
        self,
        in_atom_feature_size: int = 10,
        out_atom_coords_size: int = 3,
        condition: str = "H1C13NMRSpectrum",
        in_condition_size: Union[int, List[int]] = (10000, 80),
        max_n_atoms: int = 300,
        drop_transform: str = "zero",
        n_blocks: int = 10,
        n_heads: int = 8,
        hidden_size: int = 512,
        mlp_ratio: float = 4.0,
        embedder_args: Optional[Dict] = None,
        **kwargs,  # noqa: ARG002
    ) -> None:
        super().__init__()

        self.x_embedder = nn.Sequential(
            nn.Linear(in_atom_feature_size + 3, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.condition = condition
        if condition not in _KNOWN_CONDITIONS:
            msg = f"Condition embedding {condition} not implemented."
            raise NotImplementedError(msg)

        use_hnmr = "H1" in condition
        use_cnmr = "C13" in condition
        hnmr_dim = 0
        cnmr_dim = 0
        if use_hnmr and use_cnmr:
            if isinstance(in_condition_size, int):
                msg = (
                    f"in_condition_size must be a 2-element sequence for "
                    f"{condition}, got {in_condition_size!r}"
                )
                raise TypeError(msg)
            hnmr_dim, cnmr_dim = int(in_condition_size[0]), int(in_condition_size[1])
        elif use_hnmr:
            hnmr_dim = int(in_condition_size)
        else:
            cnmr_dim = int(in_condition_size)

        self.y_embedder = NMRSpectraEmbedder(
            use_hnmr=use_hnmr,
            use_cnmr=use_cnmr,
            hnmr_dim=hnmr_dim,
            cnmr_dim=cnmr_dim,
            hidden_dim=embedder_args["hidden_dim"],
            output_dim=hidden_size,
            dropout=embedder_args["dropout"],
            pooling=embedder_args["pooling"],
            tokenizer_args=embedder_args["tokenizer_args"],
            transformer_args=embedder_args["transformer_args"],
        )

        # DEAD ON PURPOSE, KEEP IT. Never read in any forward path (upstream
        # or here), frozen, all-zero -- but it is in the released state_dict
        # and it is the one tensor whose shape pins max_n_atoms. It is also
        # the one parameter EXCLUDED from the EMA `shadow_params` list
        # (requires_grad=False), which is what makes the positional zip in
        # scripts/convert_checkpoint.py checkable. 310 kB.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_n_atoms, hidden_size), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, n_heads, mlp_ratio=mlp_ratio)
                for _ in range(n_blocks)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, out_atom_coords_size)
        self.initialize_weights()

        self.drop_transform = drop_transform

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        r_noisy: torch.Tensor,
        times: torch.Tensor,
        model_inputs: Dict[str, torch.Tensor],
        multiplicity: int = 1,
        guidance_scale: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Args:
            r_noisy: noisy atom coordinates ``(B, N, 3)``.
            times: preconditioned noise levels ``(B,)``.
            model_inputs: ``atom_mask``, ``atom_one_hot``, ``condition``.
            multiplicity: candidates per input (a batch dimension).
            guidance_scale: CFG ``w``. ``0.0`` = conditional pass only,
                which is also what training uses.
        """
        atom_mask = (
            model_inputs["atom_mask"].repeat_interleave(multiplicity, 0).bool()
        )
        padded_atom_mask = atom_mask[..., None]
        atom_one_hot = model_inputs["atom_one_hot"].repeat_interleave(multiplicity, 0)
        condition = model_inputs["condition"].repeat_interleave(multiplicity, 0)

        if guidance_scale != 0.0:
            r_noisy = torch.cat([r_noisy, r_noisy], dim=0)
            times = torch.cat([times, times], dim=0)
            atom_mask = torch.cat([atom_mask, atom_mask], dim=0)
            padded_atom_mask = torch.cat([padded_atom_mask, padded_atom_mask], dim=0)
            atom_one_hot = torch.cat([atom_one_hot, atom_one_hot], dim=0)

            if self.drop_transform != "zero":
                msg = f"Unsupported drop_transform: {self.drop_transform}"
                raise ValueError(msg)
            condition = torch.cat([condition, torch.zeros_like(condition)], dim=0)

        x = torch.cat([r_noisy, atom_one_hot], dim=-1)
        x = self.x_embedder(x) * padded_atom_mask

        c = self.t_embedder(times) + self.y_embedder(condition)

        for block in self.blocks:
            x = block(x, c, atom_mask)

        x = self.final_layer(x, c)
        x = x * padded_atom_mask

        if guidance_scale != 0.0:
            half = x.shape[0] // 2
            x = (1 + guidance_scale) * x[:half] - guidance_scale * x[half:]

        return {"r_update": x}
