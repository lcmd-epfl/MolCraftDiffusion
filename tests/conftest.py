"""Shared fixtures for compatibility tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@pytest.fixture(autouse=True)
def _deterministic_torch_seed():
    torch.manual_seed(7)


class TinyRegressionDataset(list):
    """List-like dataset exposing the target metadata used by regression tasks."""

    def __init__(self, samples, targets):
        super().__init__(samples)
        self.targets = targets


class TinyBackbone(nn.Module):
    """Minimal pyG-compatible backbone returning per-node embeddings."""

    hidden_nf = 4

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, self.hidden_nf)

    def forward(self, graph, use_embed=True):  # noqa: ARG002
        return self.proj(graph.x.float()), None


class TinyTask(nn.Module):
    """Small task object for engine wrapper tests."""

    def __init__(self, loss_value=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.loss_value = float(loss_value)
        self.preprocess_calls = 0

    def preprocess(self, train_set):
        self.preprocess_calls += 1
        return train_set, ["valid-replaced"], ["test-replaced"]

    def forward(self, batch):  # noqa: ARG002
        loss = self.weight * self.loss_value
        return loss, {"loss": loss.detach()}


class TinyGenerationTask(nn.Module):
    """Task carrying generation metadata saved in checkpoints."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.task_type = "diffusion"
        self.condition = ["energy"]
        self.node_dist_model = None
        self.n_node_dist = {3: 2, 5: 1}
        self.prop_dist_model = None
        self.reference_indices = [0, 2]
        self.reference_freeze_mode = "features_only"
        self.reference_feature_stats = {"node_feature": torch.ones(1, 2, 1)}
        self.reference_scaffold = torch.zeros(1, 2, 4)

    def forward(self, batch):  # noqa: ARG002
        loss = self.weight.square()
        return loss, {"loss": loss.detach()}


@pytest.fixture
def pyg_data_cls():
    return pytest.importorskip("torch_geometric.data").Data


@pytest.fixture
def pyg_batch_cls():
    return pytest.importorskip("torch_geometric.data").Batch


@pytest.fixture
def tiny_pyg_graphs(pyg_data_cls):
    graph_a = pyg_data_cls(
        x=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        atomic_numbers=torch.tensor([6, 1]),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        natoms=2,
        smiles="C",
    )
    graph_b = pyg_data_cls(
        x=torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        atomic_numbers=torch.tensor([8, 1, 1]),
        edge_index=torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]], dtype=torch.long),
        natoms=3,
        smiles="O",
    )
    return [graph_a, graph_b]


@pytest.fixture
def tiny_regression_dataset():
    samples = [
        {"energy": 1.0, "labeled": True},
        {"energy": 100.0, "labeled": False},
        {"energy": float("nan"), "labeled": True},
        {"energy": 3.0, "labeled": True},
    ]
    return TinyRegressionDataset(samples, targets={"energy": [1.0, 3.0]})


@pytest.fixture
def namespace_factory():
    return SimpleNamespace
