"""MiDi's multi-modality tensor container.

Trimmed from upstream ``midi/utils.py``: the wandb setup, the torchmetrics
``NoSync*`` wrappers and ``to_dense`` (the PyG -> dense bridge) are all gone.
``to_dense`` in particular is replaced wholesale by the platform's
``graph3d_dense_collate``, which already emits these exact shapes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Dims:
    """Per-modality channel counts.

    Upstream reuses ``PlaceHolder`` itself as the dims container, which reads
    badly (a ``pos`` field holding the integer 3). A dataclass says the same
    thing without pretending to be a batch.
    """

    X: int
    charges: int
    E: int
    y: int
    pos: int = 3


def remove_mean_with_mask(x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Subtract the per-molecule mean over real nodes only.

    Args:
        x: ``(B, N, D)`` tensor, already zeroed on padded rows.
        node_mask: ``(B, N)`` bool, ``True`` for real atoms.

    Returns:
        ``x`` with the masked mean removed from every real row.
    """
    if node_mask.dtype != torch.bool:
        msg = f"node_mask must be bool, got {node_mask.dtype}"
        raise TypeError(msg)
    node_mask = node_mask.unsqueeze(-1)
    masked_max_abs_value = (x * (~node_mask)).abs().sum().item()
    if masked_max_abs_value >= 1e-5:
        msg = f"padded rows are not zero (sum |x| = {masked_max_abs_value})"
        raise ValueError(msg)
    n = node_mask.sum(1, keepdims=True)
    mean = torch.sum(x, dim=1, keepdim=True) / n
    return x - mean * node_mask


class PlaceHolder:
    """Dense batch of the four diffused modalities plus the global feature.

    ``pos (B,N,3)``, ``X (B,N,K)``, ``charges (B,N,C)``, ``E (B,N,N,5)``,
    ``y (B,dy)``, ``node_mask (B,N)`` bool.
    """

    def __init__(  # noqa: PLR0913
        self,
        pos: torch.Tensor | None,
        X: torch.Tensor | None,  # noqa: N803 - upstream modality name
        charges: torch.Tensor | None,
        E: torch.Tensor | None,  # noqa: N803 - upstream modality name
        y: torch.Tensor | None,
        t_int: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> None:
        self.pos = pos
        self.X = X
        self.charges = charges
        self.E = E
        self.y = y
        self.t_int = t_int
        self.t = t
        self.node_mask = node_mask

    def device_as(self, x: torch.Tensor) -> PlaceHolder:
        """Move every present modality onto ``x``'s device."""
        self.pos = self.pos.to(x.device) if self.pos is not None else None
        self.X = self.X.to(x.device) if self.X is not None else None
        self.charges = (
            self.charges.to(x.device) if self.charges is not None else None
        )
        self.E = self.E.to(x.device) if self.E is not None else None
        self.y = self.y.to(x.device) if self.y is not None else None
        return self

    def mask(self, node_mask: torch.Tensor | None = None) -> PlaceHolder:
        """Zero padded rows/columns and the diagonal, and re-centre ``pos``.

        The trailing symmetry check is upstream's and is kept: MiDi's whole
        edge pipeline assumes ``E == E^T``, and a violation shows up as bad
        molecules rather than as an exception anywhere else.
        """
        if node_mask is None:
            if self.node_mask is None:
                msg = "PlaceHolder.mask() needs a node_mask"
                raise ValueError(msg)
            node_mask = self.node_mask
        bs, n = node_mask.shape
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1
        diag_mask = ~torch.eye(
            n, dtype=torch.bool, device=node_mask.device
        ).unsqueeze(0).expand(bs, -1, -1).unsqueeze(-1)

        if self.X is not None:
            self.X = self.X * x_mask
        if self.charges is not None:
            self.charges = self.charges * x_mask
        if self.E is not None:
            self.E = self.E * e_mask1 * e_mask2 * diag_mask
        if self.pos is not None:
            self.pos = self.pos * x_mask
            self.pos = self.pos - self.pos.mean(dim=1, keepdim=True)
        if self.E is not None and not torch.allclose(
            self.E.float(), torch.transpose(self.E, 1, 2).float()
        ):
            msg = "edge tensor is not symmetric"
            raise ValueError(msg)
        return self

    def collapse(self, collapse_charges: torch.Tensor) -> PlaceHolder:
        """One-hot/logit modalities -> integer class ids (charges decoded).

        Padded entries are marked out of range exactly as upstream does
        (``X = -1``, ``charges = 1000``, ``E = -1``); the task clamps them
        before handing anything to the platform.
        """
        copy = self.copy()
        copy.X = torch.argmax(self.X, dim=-1)
        copy.charges = collapse_charges.to(self.charges.device)[
            torch.argmax(self.charges, dim=-1)
        ]
        copy.E = torch.argmax(self.E, dim=-1)
        x_mask = self.node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1
        copy.X[self.node_mask == 0] = -1
        copy.charges[self.node_mask == 0] = 1000
        copy.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        return copy

    def copy(self) -> PlaceHolder:
        """Shallow copy sharing the underlying tensors."""
        return PlaceHolder(
            X=self.X,
            charges=self.charges,
            E=self.E,
            y=self.y,
            pos=self.pos,
            t_int=self.t_int,
            t=self.t,
            node_mask=self.node_mask,
        )

    def __repr__(self) -> str:
        def shape(v: torch.Tensor | None) -> object:
            return v.shape if isinstance(v, torch.Tensor) else v

        return (
            f"pos: {shape(self.pos)} -- X: {shape(self.X)} -- "
            f"charges: {shape(self.charges)} -- E: {shape(self.E)} -- "
            f"y: {shape(self.y)}"
        )
