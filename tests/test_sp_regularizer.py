import torch

from MolecularDiffusion.callbacks.train_helper import (
    Queue,
    SP_regularizer,
    gradient_clipping,
)


def test_hard_drops_outliers_without_nan() -> None:
    sp = SP_regularizer("hard", lambda_=5, warm_up_steps=1)
    losses = torch.tensor([1.0, 1.0, float("inf"), float("nan")])
    out = sp(losses)
    assert torch.isfinite(out).all()
    # mean over the batch == mean over the two kept samples
    assert out.mean().item() == 1.0


def test_hard_all_dropped_is_zero() -> None:
    sp = SP_regularizer("hard", lambda_=0, warm_up_steps=1)
    assert sp(torch.tensor([1.0, 2.0])).sum().item() == 0.0


def test_zero_grad_steps_do_not_collapse_clipping() -> None:
    lin = torch.nn.Linear(4, 1)
    queue = Queue(max_len=50)
    queue.add(3000.0)
    clip = gradient_clipping(m=1)
    for _ in range(60):  # every batch dropped by SP -> zero grads
        lin.zero_grad()
        (lin(torch.randn(8, 4)).sum() * 0).backward()
        clip(lin, queue)
    lin.zero_grad()
    lin(torch.randn(8, 4)).sum().backward()
    clip(lin, queue)
    assert clip.max_grad_norm > 0
    assert lin.weight.grad.norm() > 0
