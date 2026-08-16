"""A NaN/Inf gradient must drop the optimizer step, not poison the weights."""

import torch
from torch import nn

from MolecularDiffusion.core.engine_lightning import EngineLightning


class _Stub(EngineLightning):
    """Bypass EngineLightning.__init__ -- only the grad guard is under test."""

    def __init__(self, task: nn.Module) -> None:
        nn.Module.__init__(self)
        self.task = task
        self._nonfinite_grad_steps = 0


def _run(bad: float | None) -> tuple[bool, torch.Tensor]:
    task = nn.Linear(2, 1)
    task.weight.grad = torch.ones_like(task.weight)
    if bad is not None:
        task.weight.grad[0, 0] = bad
    opt = torch.optim.SGD(task.parameters(), lr=1.0)
    skipped = _Stub(task)._skip_nonfinite_grads()
    opt.step()
    return skipped, task.weight.detach().clone()


def test_finite_grads_step_normally() -> None:
    skipped, w = _run(None)
    assert not skipped
    assert torch.isfinite(w).all()


def test_nonfinite_grads_are_skipped() -> None:
    for bad in (float("nan"), float("inf"), -float("inf")):
        skipped, w = _run(bad)
        assert skipped, bad
        assert torch.isfinite(w).all(), bad


def test_no_grads_yet_is_not_a_skip() -> None:
    assert not _Stub(nn.Linear(2, 1))._skip_nonfinite_grads()




class _BadTask(nn.Module):
    """Raises like a backbone's own masking assert for the first `n_bad` calls."""

    split = "train"

    def __init__(self, n_bad: int) -> None:
        super().__init__()
        self.n_bad = n_bad
        self.calls = 0

    def forward(self, batch: object) -> tuple[torch.Tensor, dict]:
        self.calls += 1
        if self.calls <= self.n_bad:
            msg = "NaN in masked tensor of shape torch.Size([48, 23, 23, 64])"
            raise ValueError(msg)
        return torch.tensor(1.0), {"loss": torch.tensor(1.0)}


class _StepStub(EngineLightning):
    def __init__(self, task: nn.Module, tolerance: int = 20) -> None:
        nn.Module.__init__(self)
        self.task = task
        self.monitor_metric = None
        self._consecutive_bad_batches = 0
        self.max_consecutive_bad_batches = tolerance

    def log(self, *args: object, **kwargs: object) -> None:
        pass


def test_isolated_bad_batches_are_skipped() -> None:
    stub = _StepStub(_BadTask(n_bad=3))
    assert [stub.training_step([0], i) for i in range(3)] == [None] * 3
    assert stub.training_step([0], 3) is not None
    assert stub._consecutive_bad_batches == 0


def test_unrecoverable_run_still_dies() -> None:
    stub = _StepStub(_BadTask(n_bad=99), tolerance=3)
    for i in range(3):
        stub.training_step([0], i)
    try:
        stub.training_step([0], 3)
    except RuntimeError:
        return
    raise AssertionError("should have raised after the tolerance")
