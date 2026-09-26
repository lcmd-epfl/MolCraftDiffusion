"""Checks for the pure-torch torch_scatter / torch_cluster shims.

The shims are loaded under private names so they can sit next to the real
extensions: when real ``torch_scatter`` / ``torch_cluster`` are importable
(Linux, or the PyG macOS CPU wheels on PYTHONPATH) every function is compared
against them on CPU. Independently, each op is compared to a plain-loop
reference, and MPS results are compared to CPU results when MPS is available.

    pytest mac_shims/tests
"""

import importlib.util
import itertools
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load(name, alias):
    spec = importlib.util.spec_from_file_location(
        alias, ROOT / name / "__init__.py", submodule_search_locations=[str(ROOT / name)]
    )
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


ts = _load("torch_scatter", "_shim_torch_scatter")
tc = _load("torch_cluster", "_shim_torch_cluster")


def _real(name):
    try:
        mod = importlib.import_module(name)
    except Exception:
        return None
    # Never compare the shim against itself (shim installed as the real name).
    return None if "molcraft.shim" in getattr(mod, "__version__", "") else mod


real_ts = _real("torch_scatter")
real_tc = _real("torch_cluster")

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def _close(a, b):
    a, b = a.cpu(), b.cpu()
    assert a.shape == b.shape, (a.shape, b.shape)
    if a.is_floating_point():
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    else:
        assert torch.equal(a, b)


# --------------------------------------------------------------------- scatter

CASES = [
    # (src shape, index len, dim, dim_size)
    ((10,), 10, 0, None),
    ((10,), 10, -1, 7),  # empty segments at the tail
    ((10, 4), 10, 0, None),  # 1-D index broadcast over N-D src
    ((10, 4, 3), 10, 0, 12),
    ((3, 10), 10, -1, None),  # default dim=-1 usage (kgdiff, metrics.py)
    ((3, 10, 2), 10, 1, None),
]


def _inputs(shape, n, dim, seed=0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randn(shape, generator=g)
    index = torch.randint(0, 5, (n,), generator=g)
    index[0] = 6  # leaves segment 5 empty
    return src, index


def _loop_reduce(src, index, dim, dim_size, fn, empty):
    dim = dim % src.dim()
    size = dim_size if dim_size is not None else int(index.max()) + 1
    moved = src.movedim(dim, 0)
    out = []
    for s in range(size):
        sel = moved[index == s]
        out.append(fn(sel) if len(sel) else torch.full(moved.shape[1:], empty, dtype=src.dtype))
    return torch.stack(out).movedim(0, dim)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape,n,dim,dim_size", CASES)
def test_scatter_reductions(device, shape, n, dim, dim_size):
    src, index = _inputs(shape, n, dim)
    refs = {
        "sum": _loop_reduce(src, index, dim, dim_size, lambda t: t.sum(0), 0.0),
        "mean": _loop_reduce(src, index, dim, dim_size, lambda t: t.mean(0), 0.0),
        "max": _loop_reduce(src, index, dim, dim_size, lambda t: t.max(0).values, 0.0),
        "min": _loop_reduce(src, index, dim, dim_size, lambda t: t.min(0).values, 0.0),
        "mul": _loop_reduce(src, index, dim, dim_size, lambda t: t.prod(0), 1.0),
    }
    s, i = src.to(device), index.to(device)
    for reduce, ref in refs.items():
        _close(ts.scatter(s, i, dim=dim, dim_size=dim_size, reduce=reduce), ref)
    _close(ts.scatter_add(s, i, dim, dim_size=dim_size), refs["sum"])
    _close(ts.scatter_sum(s, i, dim, dim_size=dim_size), refs["sum"])
    _close(ts.scatter_mean(s, i, dim, dim_size=dim_size), refs["mean"])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape,n,dim,dim_size", CASES)
def test_scatter_arg_convention(device, shape, n, dim, dim_size):
    src, index = _inputs(shape, n, dim)
    val, arg = ts.scatter_max(src.to(device), index.to(device), dim, dim_size=dim_size)
    val, arg = val.cpu(), arg.cpu()
    d = dim % src.dim()
    empty = arg == src.size(d)
    # Non-empty segments: the arg points at an element equal to the max.
    picked = src.gather(d, arg.clamp(max=src.size(d) - 1))
    assert torch.equal(picked[~empty], val[~empty])
    assert (val[empty] == 0).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape,n,dim,dim_size", CASES)
def test_softmax_family(device, shape, n, dim, dim_size):
    src, index = _inputs(shape, n, dim)
    d = dim % src.dim()
    sm = ts.scatter_softmax(src.to(device), index.to(device), dim, dim_size=dim_size).cpu()
    moved, smm = src.movedim(d, 0), sm.movedim(d, 0)
    for s in index.unique():
        mask = index == s
        _close(smm[mask], torch.softmax(moved[mask], dim=0))
    lsm = ts.scatter_log_softmax(src.to(device), index.to(device), dim, dim_size=dim_size).cpu()
    _close(lsm.exp(), sm)


@pytest.mark.parametrize("device", DEVICES)
def test_segments(device):
    src = torch.arange(1, 11).float().to(device)
    indptr = torch.tensor([0, 3, 3, 10]).to(device)
    _close(ts.segment_csr(src, indptr), torch.tensor([6.0, 0.0, 49.0]))
    index = torch.tensor([0, 0, 0, 2, 2, 2, 2, 2, 2, 2]).to(device)
    _close(ts.segment_coo(src, index, dim_size=4), torch.tensor([6.0, 0.0, 49.0, 0.0]))
    ones = index.new_ones(1).expand_as(index)  # the graph_module.py idiom
    _close(ts.segment_coo(ones, index, dim_size=3), torch.tensor([3, 0, 7]))


@pytest.mark.skipif(real_ts is None, reason="real torch_scatter not importable")
@pytest.mark.parametrize("shape,n,dim,dim_size", CASES)
def test_matches_real_torch_scatter(shape, n, dim, dim_size):
    src, index = _inputs(shape, n, dim, seed=1)
    for reduce in ["sum", "add", "mean", "max", "min", "mul"]:
        _close(
            ts.scatter(src, index, dim, dim_size=dim_size, reduce=reduce),
            real_ts.scatter(src, index, dim, dim_size=dim_size, reduce=reduce),
        )
    for fn in ["scatter_max", "scatter_min"]:
        for a, b in zip(getattr(ts, fn)(src, index, dim, dim_size=dim_size),
                        getattr(real_ts, fn)(src, index, dim, dim_size=dim_size)):
            _close(a, b)
    _close(ts.scatter_softmax(src, index, dim, dim_size=dim_size),
           real_ts.scatter_softmax(src, index, dim, dim_size=dim_size))
    _close(ts.scatter_log_softmax(src, index, dim, dim_size=dim_size),
           real_ts.composite.scatter_log_softmax(src, index, dim, dim_size=dim_size))
    ptr = torch.tensor([0, 2, 2, 7, 10])
    flat = torch.randn(10, 3)
    for reduce in ["sum", "mean", "max", "min"]:
        _close(ts.segment_csr(flat, ptr, reduce=reduce), real_ts.segment_csr(flat, ptr, reduce=reduce))


# --------------------------------------------------------------------- cluster

def _cloud(seed=0, n=60, nb=3):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, generator=g) * 2
    batch = torch.sort(torch.randint(0, nb, (n,), generator=g)).values
    return x, batch


def _edge_set(ei):
    return set(map(tuple, ei.cpu().t().tolist()))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("loop,flow", list(itertools.product([False, True], ["source_to_target", "target_to_source"])))
def test_radius_graph_reference(device, loop, flow):
    x, batch = _cloud()
    r = 2.0
    ei = tc.radius_graph(x.to(device), r, batch.to(device), loop=loop, max_num_neighbors=1000, flow=flow)
    d = torch.cdist(x, x)
    ref = {(j, i) if flow == "source_to_target" else (i, j)
           for i in range(len(x)) for j in range(len(x))
           if d[i, j] <= r and batch[i] == batch[j] and (loop or i != j)}
    assert _edge_set(ei) == ref


@pytest.mark.parametrize("device", DEVICES)
def test_radius_graph_neighbor_cap(device):
    x, batch = _cloud(n=80, nb=1)
    ei = tc.radius_graph(x.to(device), 100.0, batch.to(device), max_num_neighbors=5)
    counts = torch.bincount(ei[1].cpu(), minlength=len(x))
    assert counts.max() <= 5


@pytest.mark.parametrize("device", DEVICES)
def test_knn_graph_reference(device):
    x, batch = _cloud()
    k = 4
    ei = tc.knn_graph(x.to(device), k, batch.to(device))
    d = torch.cdist(x, x).masked_fill(batch[:, None] != batch[None, :], float("inf"))
    d.fill_diagonal_(float("inf"))
    ref = set()
    for i in range(len(x)):
        for j in d[i].topk(k, largest=False).indices.tolist():
            if torch.isfinite(d[i, j]):
                ref.add((j, i))
    assert _edge_set(ei) == ref


@pytest.mark.skipif(real_tc is None, reason="real torch_cluster not importable")
@pytest.mark.parametrize("loop", [False, True])
def test_matches_real_torch_cluster(loop):
    x, batch = _cloud(seed=3)
    for cap in [1000, 32]:  # uncapped and the PyG default
        assert _edge_set(tc.radius_graph(x, 2.0, batch, loop=loop, max_num_neighbors=cap)) == \
            _edge_set(real_tc.radius_graph(x, 2.0, batch, loop=loop, max_num_neighbors=cap))
    # When the cap bites, WHICH neighbours survive is kernel-specific (real CPU:
    # KD-tree order, real CUDA: index order, shim: nearest). Only check the
    # guarantees they share.
    full = _edge_set(real_tc.radius_graph(x, 2.0, batch, loop=loop, max_num_neighbors=1000))
    shim = tc.radius_graph(x, 2.0, batch, loop=loop, max_num_neighbors=3)
    real = real_tc.radius_graph(x, 2.0, batch, loop=loop, max_num_neighbors=3)
    assert _edge_set(shim) <= full
    assert torch.bincount(shim[1], minlength=len(x)).max() <= torch.bincount(real[1], minlength=len(x)).max()
    assert _edge_set(tc.knn_graph(x, 5, batch, loop=loop)) == _edge_set(real_tc.knn_graph(x, 5, batch, loop=loop))
    y, by = _cloud(seed=4, n=20)
    assert _edge_set(tc.radius(x, y, 2.0, batch, by, max_num_neighbors=1000)) == \
        _edge_set(real_tc.radius(x, y, 2.0, batch, by, max_num_neighbors=1000))


def test_pyg_probe_sees_batch_size():
    # torch_geometric < 2.8: WITH_TORCH_CLUSTER_BATCH_SIZE = 'batch_size' in knn.__doc__
    assert "batch_size" in tc.knn.__doc__
