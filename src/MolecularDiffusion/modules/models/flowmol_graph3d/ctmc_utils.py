"""Purity sampling for CTMC discrete flow matching.

Ported verbatim from FlowMol (``flowmol/utils/ctmc_utils.py``). Reached only
when ``high_confidence_threshold > 0``; the released FlowMol3 config sets
``0.9``, so this is on the live path.
"""

import torch
from torch_scatter import segment_csr

__all__ = ["purity_sampling"]


def purity_sampling(  # noqa: PLR0913
    xt: torch.Tensor,
    x1: torch.Tensor,  # noqa: ARG001 - kept for signature parity with upstream
    x1_probs: torch.Tensor,
    unmask_prob: torch.Tensor,
    mask_index: int,
    batch_size: int,
    batch_num_nodes: torch.Tensor,
    node_batch_idx: torch.Tensor,
    hc_thresh: float,
    device: torch.device,
) -> torch.Tensor:
    """Bias unmasking toward high-confidence predictions.

    Instead of unmasking a uniformly random subset of the still-masked
    positions, allocate the per-graph unmasking budget preferentially to
    positions whose predicted class probability exceeds ``hc_thresh``, spending
    the remainder on the low-confidence ones.
    """
    masked_nodes = xt == mask_index
    purities = x1_probs.max(-1)[0]

    hc_mask = purities >= hc_thresh
    hc_mask = hc_mask * masked_nodes

    indptr = torch.zeros(batch_size + 1, device=device, dtype=torch.long)
    indptr[1:] = batch_num_nodes.cumsum(0)
    hc_nodes_per_graph = segment_csr(hc_mask.long(), indptr)
    masked_nodes_per_graph = segment_csr(masked_nodes.long(), indptr)

    ph_max = unmask_prob * masked_nodes_per_graph / hc_nodes_per_graph
    ph_max[hc_nodes_per_graph == 0] = torch.inf

    ph = torch.minimum(ph_max, torch.full_like(ph_max, 1.0))
    pl = (unmask_prob * masked_nodes_per_graph - ph * hc_nodes_per_graph) / (
        masked_nodes_per_graph - hc_nodes_per_graph
    )

    node_unmask_prob = torch.zeros_like(xt).float()
    node_unmask_prob[hc_mask] = ph[node_batch_idx[hc_mask]]
    lc_mask = (purities < hc_thresh) * masked_nodes
    node_unmask_prob[lc_mask] = pl[node_batch_idx[lc_mask]]

    return torch.rand(xt.shape[0], device=device) < node_unmask_prob
