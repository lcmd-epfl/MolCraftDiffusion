"""Pure-torch stand-in for ``torch_scatter`` (macOS / Apple MPS only).

Same call signatures and results as torch_scatter 2.1 for the functions
MolCraftDiffusion and torch_geometric use, built on native
``scatter_add_`` / ``scatter_reduce_`` so every op runs on MPS as well as CPU.
"""

from .scatter import (
    broadcast,
    scatter,
    scatter_add,
    scatter_max,
    scatter_mean,
    scatter_min,
    scatter_mul,
    scatter_sum,
)
from .segment import segment_coo, segment_csr
from .composite import scatter_log_softmax, scatter_logsumexp, scatter_softmax

__version__ = "2.1.2+molcraft.shim"

__all__ = [
    "broadcast",
    "scatter",
    "scatter_add",
    "scatter_sum",
    "scatter_mul",
    "scatter_mean",
    "scatter_min",
    "scatter_max",
    "segment_coo",
    "segment_csr",
    "scatter_softmax",
    "scatter_log_softmax",
    "scatter_logsumexp",
]
