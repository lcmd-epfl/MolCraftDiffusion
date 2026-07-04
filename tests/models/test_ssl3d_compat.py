"""Model-specific compatibility tests for SSL3D tasks."""

from __future__ import annotations

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.ssl3d


class DummySSLBackbone(nn.Module):
    hidden_nf = 4

    def forward(self, graph):
        if graph.x.shape[1] >= self.hidden_nf:
            return {"x": graph.x[:, : self.hidden_nf]}
        pad = torch.zeros(
            graph.x.shape[0],
            self.hidden_nf - graph.x.shape[1],
            device=graph.x.device,
        )
        return {"x": torch.cat([graph.x, pad], dim=-1)}


def test_ssl3d_masked_atom_objective_masks_only_atom_vocab_columns(tiny_pyg_graphs):
    from MolecularDiffusion.modules.tasks.ssl3d import MaskedAtomTypeObjective

    objective = MaskedAtomTypeObjective(mask_rate=1.0, atom_vocab_size=2)
    objective.build_head(hidden_nf=4)
    batch_work = {"graph": tiny_pyg_graphs[0].clone()}
    batch_work["graph"].x = torch.cat(
        [batch_work["graph"].x, torch.ones(batch_work["graph"].x.shape[0], 1)],
        dim=-1,
    )

    aux = objective.corrupt(batch_work, device=torch.device("cpu"))
    h = torch.ones(batch_work["graph"].x.shape[0], 4)
    loss, metrics = objective.compute_loss(h, None, {}, batch_work, aux)

    assert aux["mask"].tolist() == [True, True]
    assert torch.equal(batch_work["graph"].x[:, :2], torch.zeros(2, 2))
    assert torch.equal(batch_work["graph"].x[:, 2], torch.ones(2))
    assert torch.isfinite(loss)
    assert "ssl/mtype_loss" in metrics


def test_ssl3d_coord_denoise_objective_builds_finite_invariant_loss(tiny_pyg_graphs):
    from MolecularDiffusion.modules.tasks.ssl3d import CoordDenoiseObjective

    objective = CoordDenoiseObjective(sigma_min=0.5, sigma_max=0.5)
    objective.build_head(hidden_nf=4)
    graph = tiny_pyg_graphs[1].clone()
    graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
    batch_work = {"graph": graph}

    aux = objective.corrupt(batch_work, device=torch.device("cpu"))
    h = torch.ones(graph.x.shape[0], 4)
    loss, metrics = objective.compute_loss(h, None, {"graph": graph}, batch_work, aux)

    assert aux["sigma"].shape == (graph.x.shape[0], 1)
    assert torch.allclose(aux["sigma"], torch.full_like(aux["sigma"], 0.5))
    assert torch.isfinite(loss)
    assert "ssl/denoise_loss" in metrics


def test_ssl3d_forward_is_finite_and_does_not_mutate_original_graph(
    tiny_pyg_graphs, pyg_batch_cls
):
    from MolecularDiffusion.modules.tasks.ssl3d import MaskedAtomTypeObjective, SSL3D

    graph = pyg_batch_cls.from_data_list(tiny_pyg_graphs)
    original_x = graph.x.clone()
    task = SSL3D(
        DummySSLBackbone(),
        [MaskedAtomTypeObjective(mask_rate=1.0, atom_vocab_size=2)],
        include_charge=True,
    )

    loss, metrics = task({"graph": graph})

    assert torch.isfinite(loss)
    assert "ssl/mtype_loss" in metrics
    assert "ssl/total_loss" in metrics
    assert torch.equal(graph.x, original_x)


def test_ssl3d_predict_and_evaluate_use_loss_as_compatibility_metric(
    tiny_pyg_graphs, pyg_batch_cls
):
    from MolecularDiffusion.modules.tasks.ssl3d import MaskedAtomTypeObjective, SSL3D

    task = SSL3D(
        DummySSLBackbone(),
        [MaskedAtomTypeObjective(mask_rate=1.0, atom_vocab_size=2)],
        include_charge=False,
    )

    pred, target = task.predict_and_target(
        {"graph": pyg_batch_cls.from_data_list(tiny_pyg_graphs)}
    )
    metrics = task.evaluate(pred, target)

    assert pred.shape == (1,)
    assert torch.equal(target, torch.zeros_like(pred))
    assert torch.equal(metrics["ssl/total_loss"], pred.mean())
