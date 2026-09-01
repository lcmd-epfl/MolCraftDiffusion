"""Size GNN: an auxiliary, fragment-geometry-conditioned linker-size
classifier, ported from DiffLinker's ``src/linker_size.py``
(``SizeGNN``) and ``src/linker_size_lightning.py`` (``SizeClassifier``'s
inference path only -- training machinery is out of scope).

See docs/model_integrations/difflinker/INTEGRATION_PLAN.md's "Revision 8 --
Size GNN" section for the full design (why this is a small helper class, not
a new ``Task``; the exact upstream inference contract this ports; the
documented per-retry-resampling simplification). Used only as an optional,
additive size predictor plugged into
``MolecularDiffusion.modules.tasks.diffusion_difflinker.DiffLinkerTask.sample()``
-- unrelated to, and architecturally distinct from, ``linker_size.py``'s
``DistributionNodes`` (unconditional histogram sampling).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .egnn import GCL, coord2diff


class SizeGNN(nn.Module):
    """Ported near-verbatim from ``others/difflinker/src/linker_size.py:45-91``.

    A plain stack of message-passing ``GCL`` layers (imported from the
    already-ported ``egnn.py``, not reimplemented) producing a per-node
    hidden vector -- no coordinate output, no coordinate-update layer
    anywhere (unlike ``Dynamics``/``EGNN``).
    """

    def __init__(
        self,
        in_node_nf: int,
        hidden_nf: int,
        out_node_nf: int,
        n_layers: int,
        normalization: Optional[str] = None,
    ):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.out_node_nf = out_node_nf

        self.embedding_in = nn.Linear(in_node_nf, hidden_nf)
        self.gcl1 = GCL(
            input_nf=hidden_nf,
            output_nf=hidden_nf,
            hidden_nf=hidden_nf,
            normalization_factor=1,
            aggregation_method="sum",
            edges_in_d=1,
            activation=nn.ReLU(),
            attention=False,
            normalization=normalization,
        )
        self.gcl_layers = nn.ModuleList(
            [
                GCL(
                    input_nf=hidden_nf,
                    output_nf=hidden_nf,
                    hidden_nf=hidden_nf,
                    normalization_factor=1,
                    aggregation_method="sum",
                    edges_in_d=1,
                    activation=nn.ReLU(),
                    attention=False,
                    normalization=normalization,
                )
                for _ in range(n_layers - 1)
            ]
        )
        self.embedding_out = nn.Linear(hidden_nf, out_node_nf)

    def forward(self, h, edges, distances, node_mask, edge_mask):
        h = self.embedding_in(h)
        h, _ = self.gcl1(h, edges, edge_attr=distances, node_mask=node_mask, edge_mask=edge_mask)
        for gcl in self.gcl_layers:
            h, _ = gcl(h, edges, edge_attr=distances, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)
        return h


def _fully_connected_edges(n_nodes: int, batch_size: int, device) -> list:
    """Batch-flat fully-connected edge index -- the same ~10-line pattern
    already duplicated in this codebase (``Dynamics.get_edges``,
    ``egnn.py``) and upstream (``collate_with_fragment_edges``,
    ``others/difflinker/src/datasets.py:404-413``); a third small inline
    copy here is consistent with existing precedent (see
    INTEGRATION_PLAN.md Revision 8's Naming section)."""
    rows, cols = [], []
    for batch_idx in range(batch_size):
        for i in range(n_nodes):
            for j in range(n_nodes):
                rows.append(i + batch_idx * n_nodes)
                cols.append(j + batch_idx * n_nodes)
    return [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]


class LinkerSizePredictor(nn.Module):
    """Thin inference-only wrapper matching ``SizeClassifier``'s own
    state-dict key layout exactly (``self.gnn = SizeGNN(...)`` -- every
    checkpoint key is prefixed ``gnn.``, confirmed in
    INTEGRATION_PLAN.md Revision 8's Phase A6). Never routed through
    ``cli/generate.py``'s ``Task``-loading machinery -- this is not a
    ``Task``, so its own converted-checkpoint format is a plain,
    self-describing ``{"state_dict", "hyper_parameters"}`` dict.
    """

    def __init__(
        self,
        in_node_nf: int,
        hidden_nf: int,
        out_node_nf: int,
        n_layers: int,
        normalization: Optional[str],
        linker_id2size: list,
    ):
        super().__init__()
        self.gnn = SizeGNN(
            in_node_nf=in_node_nf,
            hidden_nf=hidden_nf,
            out_node_nf=out_node_nf,
            n_layers=n_layers,
            normalization=normalization,
        )
        self.linker_id2size = list(linker_id2size)

    @torch.no_grad()
    def predict(
        self,
        one_hot: torch.Tensor,
        positions: torch.Tensor,
        fragment_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reproduces ``SizeClassifier.forward(..., return_loss=False)``
        (``linker_size_lightning.py:83-109``) + ``generate.py``'s
        softmax/Categorical/id-to-size sampling (``generate.py:90-99``) in
        one pass. ``one_hot``/``positions`` are the fragment-only tensors
        (``(B, N, n_vocab)``/``(B, N, 3)``); with no ``fragment_mask``
        supplied, every atom in ``N`` is treated as a real fragment atom
        (the reference structure *is* the fragment, by construction --
        matches ``DiffLinkerTask.sample()``'s own ``ref_onehot``/
        ``ref_positions``). Returns a ``(B,)`` ``torch.LongTensor`` of
        predicted linker-only atom counts.
        """
        device = positions.device
        bsz, n_nodes = positions.shape[0], positions.shape[1]
        if fragment_mask is None:
            fragment_mask = torch.ones(bsz, n_nodes, 1, device=device)

        x = (positions * fragment_mask).reshape(bsz * n_nodes, -1)
        h = (one_hot * fragment_mask).reshape(bsz * n_nodes, -1)
        flat_fragment_mask = fragment_mask.reshape(bsz * n_nodes, 1)

        edges = _fully_connected_edges(n_nodes, bsz, device)
        distances, _ = coord2diff(x, edges)

        am = flat_fragment_mask.reshape(bsz, n_nodes)
        edge_mask_full = am.unsqueeze(2) * am.unsqueeze(1)
        diag = ~torch.eye(n_nodes, dtype=torch.bool, device=device)
        edge_mask_full = edge_mask_full * diag.unsqueeze(0)
        edge_mask = edge_mask_full.reshape(-1, 1)
        distance_edge_mask = (edge_mask.bool() & (distances < 6)).float()

        logits = self.gnn(h, edges, distances, flat_fragment_mask, distance_edge_mask)
        logits = logits.reshape(bsz, n_nodes, -1).mean(1)

        probabilities = torch.softmax(logits, dim=1)
        samples = torch.distributions.Categorical(probs=probabilities).sample()
        id2size = torch.tensor(self.linker_id2size, device=device, dtype=torch.long)
        return id2size[samples]

    @classmethod
    def from_checkpoint(cls, path: str, map_location="cpu") -> "LinkerSizePredictor":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        hp = ckpt["hyper_parameters"]
        model = cls(
            in_node_nf=hp["in_node_nf"],
            hidden_nf=hp["hidden_nf"],
            out_node_nf=hp["out_node_nf"],
            n_layers=hp["n_layers"],
            normalization=hp["normalization"],
            linker_id2size=hp["linker_id2size"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
