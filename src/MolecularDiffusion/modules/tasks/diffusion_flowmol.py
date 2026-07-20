"""Coordinate-only FlowMol flow-matching task integration.

Wraps the SE(3)-equivariant GVP endpoint vector field
(``modules/models/flowmol``) in the duck-typed ``Task`` contract
(docs/adding_new_models.md §2.1) and adapts the platform's PointCloud dict
batch to / from FlowMol's batched-DGL-graph container.

Coordinate-only scope (per the approved INTEGRATION_PLAN): the generated
modalities are ``x`` (coords), ``a`` (atom types), ``c`` (formal charge).
Bonds (the ``e`` modality) are NOT generated -- edges are a geometry-only
message-passing convenience. Since the PointCloud pipeline carries no
formal-charge channel, ``c`` is trained against a constant neutral target
(QM9-style neutral molecules) and is not surfaced in the sample() return.
"""

from collections import Counter
from typing import Dict, List, Optional

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from MolecularDiffusion.modules.models.flowmol.graph_utils import (
    build_edge_idxs,
    get_node_batch_idxs,
)
from MolecularDiffusion.modules.models.flowmol.interpolant_scheduler import (
    InterpolantScheduler,
)
from MolecularDiffusion.modules.models.flowmol.priors import (
    centered_normal_prior_batched_graph,
    uniform_simplex_prior,
)
from MolecularDiffusion.modules.models.flowmol.vector_field import (
    EndpointVectorField,
)

# Histogram-backed size sampler already implemented for TABASCO; reuse it
# rather than re-implementing the same interface.
from MolecularDiffusion.modules.tasks.diffusion_tabasco import (
    TabascoNodeDistribution,
)


def _atom_onehot(batch: Dict[str, torch.Tensor], n_atom_types: int):
    """Pull the atom-type one-hot out of a PointCloud batch."""
    if "node_feature" in batch:
        feat = batch["node_feature"]
    elif "node_features" in batch:
        feat = batch["node_features"]
    elif "x" in batch:
        feat = batch["x"]
    else:
        feat = F.one_hot(batch["charges"].long(), n_atom_types).float()
    # keep only the atom-type OHE columns (drop any extra per-atom features)
    return feat[..., :n_atom_types].float()


class PointCloudToDGLAdapter(nn.Module):
    """PointCloud dict batch -> batched, fully-connected DGL graph.

    Padding is stripped via ``node_mask``; per-molecule COM is removed from
    coordinates so ``x_1_true`` lives in the same zero-COM subspace as the
    Gaussian prior. No ``edata`` is set -- bonds are dropped.
    """

    def __init__(
        self, n_atom_types: int, n_charges: int, neutral_charge_index: int
    ):
        super().__init__()
        self.n_atom_types = n_atom_types
        self.n_charges = n_charges
        self.neutral_charge_index = neutral_charge_index

    def forward(self, batch: Dict[str, torch.Tensor]) -> dgl.DGLGraph:
        coords = batch["coords"]
        node_mask = batch["node_mask"].bool()
        atom_oh = _atom_onehot(batch, self.n_atom_types)
        device = coords.device

        graphs = []
        for b in range(coords.size(0)):
            mask_b = node_mask[b]
            n = int(mask_b.sum().item())
            if n == 0:
                continue
            coords_b = coords[b][mask_b]
            coords_b = coords_b - coords_b.mean(dim=0, keepdim=True)
            a_b = atom_oh[b][mask_b]
            c_b = F.one_hot(
                torch.full((n,), self.neutral_charge_index, device=device),
                self.n_charges,
            ).float()

            edges = build_edge_idxs(n).to(device)
            g_i = dgl.graph(
                (edges[0], edges[1]), num_nodes=n, device=device
            )
            g_i.ndata["x_1_true"] = coords_b
            g_i.ndata["a_1_true"] = a_b
            g_i.ndata["c_1_true"] = c_b
            graphs.append(g_i)

        return dgl.batch(graphs)


class DGLToPointCloudAdapter(nn.Module):
    """Integrated DGL graph -> padded PointCloud sample tuple pieces."""

    def forward(
        self, g: dgl.DGLGraph, n_atom_types: int
    ) -> Dict[str, torch.Tensor]:
        device = g.device
        mols = dgl.unbatch(g)
        n_max = max(int(gi.num_nodes()) for gi in mols)
        bsz = len(mols)

        one_hot = torch.zeros(bsz, n_max, n_atom_types, device=device)
        coords = torch.zeros(bsz, n_max, 3, device=device)
        charges = torch.zeros(bsz, n_max, dtype=torch.long, device=device)
        node_mask = torch.zeros(bsz, n_max, device=device)

        for i, gi in enumerate(mols):
            n = int(gi.num_nodes())
            a_idx = gi.ndata["a_1"].argmax(dim=-1)
            one_hot[i, :n] = F.one_hot(a_idx, n_atom_types).float()
            coords[i, :n] = gi.ndata["x_1"]
            charges[i, :n] = a_idx
            node_mask[i, :n] = 1

        return {
            "one_hot": one_hot,
            "coords": coords,
            "charges": charges,
            "node_mask": node_mask,
        }


class FlowMolTaskFactory:
    """Factory instantiated by ``cli/train.py`` (declares ``train_set`` so the
    declarative seam injects the dataset for atom-count histogram stats)."""

    def __init__(
        self,
        task_type: str,
        interpolant_scheduler_config: dict,
        vector_field_config: dict,
        n_charges: int = 6,
        neutral_charge_index: int = 0,
        prior_std: float = 1.0,
        time_scaled_loss: bool = True,
        total_loss_weights: Optional[dict] = None,
        default_n_timesteps: int = 250,
        dataset_stats: Optional[dict] = None,
        atom_vocab: Optional[list] = None,
        train_set: Optional[torch.utils.data.Dataset] = None,
        **kwargs,
    ):
        self.task_type = task_type
        self.interpolant_scheduler_config = interpolant_scheduler_config
        self.vector_field_config = vector_field_config
        self.n_charges = n_charges
        self.neutral_charge_index = neutral_charge_index
        self.prior_std = prior_std
        self.time_scaled_loss = time_scaled_loss
        self.total_loss_weights = total_loss_weights or {}
        self.default_n_timesteps = default_n_timesteps
        self.dataset_stats = dataset_stats or {}
        self.atom_vocab = atom_vocab or kwargs.get("atom_vocab", None)
        self.train_set = train_set
        self.kwargs = kwargs

    def compute_dataset_stats(self, dataset):
        """Build the atom-count histogram from the training set (like
        TABASCO's ``compute_dataset_stats``)."""
        if hasattr(dataset, "n_atoms"):
            num_atoms_list = list(dataset.n_atoms)
        else:
            num_atoms_list = []
            for i in range(len(dataset)):
                item = dataset[i]
                if "natoms" in item:
                    n = item["natoms"]
                    num_atoms_list.append(
                        int(n.item()) if torch.is_tensor(n) else int(n)
                    )
                elif "node_mask" in item:
                    num_atoms_list.append(int(item["node_mask"].sum().item()))
        histogram = {int(k): int(v) for k, v in Counter(num_atoms_list).items()}
        self.dataset_stats["num_atoms_histogram"] = histogram
        print(
            f"[flowmol] atom-count histogram: {len(histogram)} unique sizes "
            f"from {len(num_atoms_list)} molecules."
        )

    def build(self):
        if not self.dataset_stats.get("num_atoms_histogram"):
            if self.train_set is not None:
                self.compute_dataset_stats(self.train_set)
            else:
                print(
                    "[flowmol] WARNING: no train_set and no histogram; "
                    "generation size sampling will fall back to a default."
                )

        if not self.atom_vocab:
            raise ValueError(
                "FlowMolTaskFactory requires atom_vocab (injected from "
                "data.atom_vocab) to size the atom-type head."
            )

        self.task = FlowMolFlowMatchingTask(
            n_atom_types=len(self.atom_vocab),
            n_charges=self.n_charges,
            neutral_charge_index=self.neutral_charge_index,
            interpolant_scheduler_config=self.interpolant_scheduler_config,
            vector_field_config=self.vector_field_config,
            prior_std=self.prior_std,
            time_scaled_loss=self.time_scaled_loss,
            total_loss_weights=self.total_loss_weights,
            default_n_timesteps=self.default_n_timesteps,
            dataset_stats=self.dataset_stats,
            atom_vocab=self.atom_vocab,
        )
        return self.task


class FlowMolFlowMatchingTask(nn.Module):

    def __init__(
        self,
        n_atom_types: int,
        n_charges: int,
        neutral_charge_index: int,
        interpolant_scheduler_config: dict,
        vector_field_config: dict,
        prior_std: float,
        time_scaled_loss: bool,
        total_loss_weights: dict,
        default_n_timesteps: int,
        dataset_stats: dict,
        atom_vocab: Optional[list] = None,
    ):
        super().__init__()
        self.canonical_feat_order = ["x", "a", "c"]
        self.n_atom_types = n_atom_types
        self.n_charges = n_charges
        self.neutral_charge_index = neutral_charge_index
        self.prior_std = prior_std
        self.time_scaled_loss = time_scaled_loss
        self.task_type = "diffusion_flowmol"
        self.fm_num_timesteps = default_n_timesteps
        self.atom_vocab = atom_vocab

        self.total_loss_weights = {"x": 1.0, "a": 1.0, "c": 1.0}
        self.total_loss_weights.update(total_loss_weights or {})

        self.interpolant_scheduler = InterpolantScheduler(
            canonical_feat_order=self.canonical_feat_order,
            **interpolant_scheduler_config,
        )
        self.vector_field = EndpointVectorField(
            n_atom_types=n_atom_types,
            canonical_feat_order=self.canonical_feat_order,
            interpolant_scheduler=self.interpolant_scheduler,
            n_charges=n_charges,
            **vector_field_config,
        )

        self.to_dgl = PointCloudToDGLAdapter(
            n_atom_types, n_charges, neutral_charge_index
        )
        self.to_pc = DGLToPointCloudAdapter()

        self.loss_x = nn.MSELoss(reduction="none")
        self.loss_a = nn.CrossEntropyLoss(reduction="none")
        self.loss_c = nn.CrossEntropyLoss(reduction="none")

        self.prop_dist_model = None
        self._dataset_stats = dataset_stats
        self._node_dist_model = None

    # ------------------------------------------------------------------ #
    # prior sampling                                                     #
    # ------------------------------------------------------------------ #
    def _sample_prior(self, g: dgl.DGLGraph, node_batch_idx: torch.Tensor):
        n = g.num_nodes()
        g.ndata["x_0"] = centered_normal_prior_batched_graph(
            g, node_batch_idx, std=self.prior_std
        ).to(g.device)
        g.ndata["a_0"] = uniform_simplex_prior(n, self.n_atom_types).to(
            g.device
        )
        g.ndata["c_0"] = uniform_simplex_prior(n, self.n_charges).to(g.device)
        return g

    # ------------------------------------------------------------------ #
    # training / evaluation                                              #
    # ------------------------------------------------------------------ #
    def forward(self, batch: Dict[str, torch.Tensor]):
        g = self.to_dgl(batch)
        node_batch_idx = get_node_batch_idxs(g)
        batch_size = g.batch_size

        g = self._sample_prior(g, node_batch_idx)

        t = torch.rand(batch_size, device=g.device).float()
        g = self.vector_field.sample_conditional_path(g, t, node_batch_idx)

        vf_output = self.vector_field(g, t, node_batch_idx)

        # endpoint targets
        x_target = g.ndata["x_1_true"]
        a_target = g.ndata["a_1_true"].argmax(dim=-1)
        c_target = g.ndata["c_1_true"].argmax(dim=-1)

        raw = {
            "x": self.loss_x(vf_output["x"], x_target).mean(dim=-1),
            "a": self.loss_a(vf_output["a"], a_target),
            "c": self.loss_c(vf_output["c"], c_target),
        }

        if self.time_scaled_loss:
            time_weights = self.interpolant_scheduler.loss_weights(t)
        losses = {}
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if self.time_scaled_loss:
                w = time_weights[:, feat_idx][node_batch_idx]
                losses[feat] = (raw[feat] * w).mean()
            else:
                losses[feat] = raw[feat].mean()

        total_loss = sum(
            self.total_loss_weights[feat] * losses[feat]
            for feat in self.canonical_feat_order
        )
        stats = {f"{feat}_loss": losses[feat].detach() for feat in losses}
        stats["total_loss"] = total_loss.detach()
        return total_loss, stats

    def predict_and_target(self, batch: Dict[str, torch.Tensor]):
        loss, _ = self.forward(batch)
        if loss.dim() == 0:
            loss = loss.unsqueeze(0)
        return loss.detach(), torch.zeros_like(loss)

    def evaluate(self, pred: torch.Tensor, target: torch.Tensor):
        return {"val_loss": pred.mean()}

    # ------------------------------------------------------------------ #
    # generation                                                         #
    # ------------------------------------------------------------------ #
    def _build_graphs(self, sizes: torch.Tensor) -> dgl.DGLGraph:
        graphs = []
        for n in sizes.tolist():
            n = int(n)
            edges = build_edge_idxs(n).to(self.device)
            graphs.append(
                dgl.graph((edges[0], edges[1]), num_nodes=n, device=self.device)
            )
        return dgl.batch(graphs)

    @torch.no_grad()
    def sample(
        self,
        batch_size: Optional[int] = None,
        nodesxsample: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        batch: Optional[Dict[str, torch.Tensor]] = None,
        mode=None,
        n_frames: int = 0,
        **kwargs,
    ):
        if num_steps is None:
            num_steps = self.fm_num_timesteps

        if nodesxsample is not None:
            sizes = nodesxsample.to(self.device).long()
        elif batch is not None:
            sizes = batch["natoms"].to(self.device).long()
        else:
            if batch_size is None:
                raise ValueError(
                    "sample() needs nodesxsample, batch, or batch_size."
                )
            sizes = self.node_dist_model.sample(batch_size).to(self.device)

        g = self._build_graphs(sizes)
        node_batch_idx = get_node_batch_idxs(g)
        g = self._sample_prior(g, node_batch_idx)

        g = self.vector_field.integrate(g, node_batch_idx, n_timesteps=num_steps)

        pc = self.to_pc(g, self.n_atom_types)
        return pc["one_hot"], pc["charges"], pc["coords"], pc["node_mask"]

    # ------------------------------------------------------------------ #
    # properties (generation contract)                                   #
    # ------------------------------------------------------------------ #
    @property
    def model(self):
        return self

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def node_dist_model(self):
        if self._node_dist_model is None:
            self._node_dist_model = TabascoNodeDistribution(
                self._dataset_stats
            )
        return self._node_dist_model

    @property
    def n_node_dist(self):
        return self.node_dist_model.n_node_dist
