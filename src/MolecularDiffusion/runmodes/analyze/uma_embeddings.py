"""UMA backbone embedding extraction — package-native implementation.

fairchem is NOT installed as a pip package. The vendored source tree at
``<repo_root>/fairchem/src`` is prepended to ``sys.path`` at import time.

The repo root is resolved in this order:
1. ``MOLCRAFT_REPO_ROOT`` environment variable (explicit override).
2. Walk up from ``cwd`` until a directory containing ``fairchem/src`` is found.

Import stubs for ``ray``, ``torchtnt``, and ``monty`` are installed into
``sys.modules`` so that fairchem's module-level imports succeed without those
packages being present.
"""

from __future__ import annotations

import os
import sys
import types
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import torch
from ase import Atoms

from ._io import DEFAULT_STRUCTURE_EXTENSIONS, iter_structure_files, read_atoms

DEFAULT_UMA_CHECKPOINT = "training_outputs/uma-s-1p2.pt"


# ---------------------------------------------------------------------------
# Repo-root / fairchem path resolution
# ---------------------------------------------------------------------------

def _find_repo_root() -> Path:
    """Return the repo root that contains fairchem/src."""
    env = os.environ.get("MOLCRAFT_REPO_ROOT")
    if env:
        root = Path(env).resolve()
        if (root / "fairchem" / "src").is_dir():
            return root
        raise FileNotFoundError(
            f"MOLCRAFT_REPO_ROOT={env!r} does not contain fairchem/src"
        )

    # walk up from cwd
    candidate = Path.cwd()
    for _ in range(10):
        if (candidate / "fairchem" / "src").is_dir():
            return candidate
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent

    raise FileNotFoundError(
        "Could not locate fairchem/src.\n"
        "The UMA backend requires the fairchem source tree to be present at "
        "<repo_root>/fairchem/src.\n\n"
        "Clone it with:\n"
        "  git clone https://github.com/pregHosh/fairchem fairchem\n\n"
        "Then either:\n"
        "  (a) run MolCraftDiff from the repository root, or\n"
        "  (b) set the environment variable:\n"
        "      export MOLCRAFT_REPO_ROOT=/path/to/MolCraftDiffusion"
    )


def _ensure_local_fairchem_importable() -> None:
    os.environ.setdefault("FAIRCHEM_CACHE_DIR", "/tmp/fairchem_cache")
    fairchem_src = str(_find_repo_root() / "fairchem" / "src")
    sys.path = [p for p in sys.path if p != fairchem_src]
    sys.path.insert(0, fairchem_src)


# ---------------------------------------------------------------------------
# Import stubs
# ---------------------------------------------------------------------------

def _passthrough_decorator(*decorator_args, **decorator_kwargs):
    if decorator_args and callable(decorator_args[0]) and not decorator_kwargs:
        return decorator_args[0]
    def decorator(obj):
        return obj
    return decorator


def _install_ray_import_stub() -> None:
    if find_spec("ray") is not None:
        return

    def unsupported(*args, **kwargs):
        raise RuntimeError(
            "Ray is not installed. Only single-process UMA embedding "
            "extraction is supported via the vendored fairchem checkout."
        )

    ray = types.ModuleType("ray")
    ray.remote = _passthrough_decorator
    ray.is_initialized = lambda: False
    ray.shutdown = lambda: None
    ray.init = unsupported
    ray.get = lambda obj: obj
    ray.put = lambda obj: obj
    ray.get_gpu_ids = lambda: []

    util = types.ModuleType("ray.util")
    util.get_node_ip_address = lambda: "127.0.0.1"
    util.placement_group = unsupported

    scheduling = types.ModuleType("ray.util.scheduling_strategies")

    class PlacementGroupSchedulingStrategy:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    scheduling.PlacementGroupSchedulingStrategy = PlacementGroupSchedulingStrategy
    util.scheduling_strategies = scheduling
    ray.util = util

    serve = types.ModuleType("ray.serve")
    serve.deployment = _passthrough_decorator
    serve.batch = _passthrough_decorator

    class LoggingConfig:
        def __init__(self, *args, **kwargs):
            pass

    serve.schema = types.SimpleNamespace(LoggingConfig=LoggingConfig)
    ray.serve = serve

    sys.modules["ray"] = ray
    sys.modules["ray.util"] = util
    sys.modules["ray.util.scheduling_strategies"] = scheduling
    sys.modules["ray.serve"] = serve


def _install_torchtnt_import_stub() -> None:
    if find_spec("torchtnt") is not None:
        return

    torchtnt = types.ModuleType("torchtnt")
    framework = types.ModuleType("torchtnt.framework")
    callback_mod = types.ModuleType("torchtnt.framework.callback")
    fit_mod = types.ModuleType("torchtnt.framework.fit")
    state_mod = types.ModuleType("torchtnt.framework.state")
    unit_mod = types.ModuleType("torchtnt.framework.unit")
    utils = types.ModuleType("torchtnt.utils")
    prepare_module_mod = types.ModuleType("torchtnt.utils.prepare_module")
    distributed = types.ModuleType("torchtnt.utils.distributed")

    class _PredictUnit:
        def __class_getitem__(cls, item): return cls

    class _EvalUnit:
        def __class_getitem__(cls, item): return cls

    class _TrainUnit:
        def __class_getitem__(cls, item): return cls

    class State: pass
    class Callback: pass

    def fit(*args, **kwargs):
        raise RuntimeError("TorchTNT training is unavailable.")

    framework.PredictUnit = _PredictUnit
    framework.EvalUnit = _EvalUnit
    framework.TrainUnit = _TrainUnit
    framework.State = State
    callback_mod.Callback = Callback
    fit_mod.fit = fit
    state_mod.State = State
    unit_mod.TTrainUnit = _TrainUnit
    prepare_module_mod.prepare_module = lambda module, *args, **kwargs: module
    distributed.get_file_init_method = lambda *a, **k: "file:///tmp/fairchem"
    distributed.get_tcp_init_method = lambda *a, **k: "tcp://127.0.0.1:13356"
    utils.prepare_module = prepare_module_mod
    utils.distributed = distributed
    torchtnt.framework = framework
    torchtnt.utils = utils

    sys.modules["torchtnt"] = torchtnt
    sys.modules["torchtnt.framework"] = framework
    sys.modules["torchtnt.framework.callback"] = callback_mod
    sys.modules["torchtnt.framework.fit"] = fit_mod
    sys.modules["torchtnt.framework.state"] = state_mod
    sys.modules["torchtnt.framework.unit"] = unit_mod
    sys.modules["torchtnt.utils"] = utils
    sys.modules["torchtnt.utils.prepare_module"] = prepare_module_mod
    sys.modules["torchtnt.utils.distributed"] = distributed


def _install_monty_import_stub() -> None:
    if find_spec("monty") is not None:
        return

    monty = types.ModuleType("monty")
    dev = types.ModuleType("monty.dev")

    def requires(condition, message=None):
        def decorator(func):
            if condition:
                return func
            def unavailable(*args, **kwargs):
                raise ImportError(message or "Required optional dependency missing")
            return unavailable
        return decorator

    dev.requires = requires
    monty.dev = dev
    sys.modules["monty"] = monty
    sys.modules["monty.dev"] = dev


# ---------------------------------------------------------------------------
# Pooling / backbone helpers
# ---------------------------------------------------------------------------

def _pool_node_embedding(
    node_embedding: torch.Tensor,
    batch: torch.Tensor,
    num_systems: int,
    pooling: Literal["mean", "sum"],
) -> torch.Tensor:
    pooled = torch.zeros(
        num_systems,
        node_embedding.shape[-1],
        dtype=node_embedding.dtype,
        device=node_embedding.device,
    )
    pooled.index_add_(0, batch, node_embedding)
    if pooling == "mean":
        counts = torch.bincount(batch, minlength=num_systems).clamp_min(1)
        pooled = pooled / counts.to(dtype=pooled.dtype, device=pooled.device).unsqueeze(1)
    elif pooling != "sum":
        raise ValueError(f"pooling must be 'mean' or 'sum', got {pooling!r}")
    return pooled


@contextmanager
def _embedding_backbone_mode(backbone):
    regress_config = getattr(backbone, "regress_config", None)
    if regress_config is None:
        yield
        return
    original = {
        "forces": regress_config.forces,
        "stress": regress_config.stress,
        "hessian": regress_config.hessian,
    }
    regress_config.forces = False
    regress_config.stress = False
    regress_config.hessian = False
    try:
        yield
    finally:
        regress_config.forces = original["forces"]
        regress_config.stress = original["stress"]
        regress_config.hessian = original["hessian"]


def _batch_iter(items: Sequence, batch_size: int) -> Iterable[Sequence]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    for start in range(0, len(items), batch_size):
        yield items[start: start + batch_size]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_uma_molecule_embeddings(
    source: str | Path | list[Atoms],
    checkpoint_path: str | Path = DEFAULT_UMA_CHECKPOINT,
    output_path: str | Path | None = None,
    task_name: str = "omol",
    device: Literal["cuda", "cpu"] | None = None,
    batch_size: int = 8,
    pooling: Literal["mean", "sum"] = "mean",
    scalar_only: bool = True,
    read_all_frames: bool = False,
    recursive: bool = False,
    extensions: Sequence[str] = DEFAULT_STRUCTURE_EXTENSIONS,
    charge: int = 0,
    spin: int = 1,
) -> list[dict[str, object]]:
    """Extract UMA backbone embeddings.

    Parameters
    ----------
    source:
        Directory of structure files **or** a pre-loaded ``list[Atoms]``.
        When a list is passed, file discovery and ase_read are skipped;
        ``path`` and ``frame`` in each result are ``None``.
    charge:
        Total molecular charge applied to all structures (default: 0).
    spin:
        Spin multiplicity applied to all structures (default: 1).
    """
    _ensure_local_fairchem_importable()
    _install_ray_import_stub()
    _install_torchtnt_import_stub()
    _install_monty_import_stub()

    try:
        from fairchem.core.datasets.atomic_data import (
            AtomicData,
            atomicdata_list_to_batch,
        )
        from fairchem.core.units.mlip_unit import load_predict_unit
    except ImportError as exc:
        raise ImportError(
            "Could not import fairchem UMA utilities from the vendored checkout. "
            f"Missing import: {exc.name!r}. "
            "Ensure fairchem/src is present in the repo root."
        ) from exc

    # --- build structures list ---
    if isinstance(source, list):
        # pre-loaded atoms; path/frame are unknown
        structures: list[tuple[Path | None, int | None, Atoms]] = [
            (None, None, atoms) for atoms in source
        ]
    else:
        paths = iter_structure_files(source, extensions=extensions, recursive=recursive)
        if not paths:
            raise FileNotFoundError(f"No supported molecule files found in {source}")
        structures = []
        for path in paths:
            for frame, atoms in enumerate(read_atoms(path, read_all_frames=read_all_frames)):
                structures.append((path, frame, atoms))

    checkpoint = Path(checkpoint_path)
    predictor = load_predict_unit(checkpoint, device=device)
    settings = predictor.inference_settings
    dtype = settings.base_precision_dtype

    # stamp charge/spin so fairchem omol task doesn't warn and default silently
    for _, _, atoms in structures:
        atoms.info.setdefault("charge", charge)
        atoms.info.setdefault("spin", spin)
        predictor.validate_atoms_data(atoms, task_name)

    results: list[dict[str, object]] = []
    for chunk in _batch_iter(structures, batch_size):
        data_list = [
            AtomicData.from_ase(
                atoms,
                task_name=task_name,
                r_edges=settings.external_graph_gen,
                r_data_keys=["spin", "charge"],
                max_neigh=300 if settings.external_graph_gen else None,
                radius=6.0,
                target_dtype=dtype,
                sid=f"{path.name if path else 'atoms'}:{frame}",
            )
            for path, frame, atoms in chunk
        ]
        batch = atomicdata_list_to_batch(data_list)

        if not predictor.lazy_model_intialized:
            predictor._lazy_init(batch)

        data = batch.to(predictor.device).clone()
        for key, value in data:
            if torch.is_tensor(value) and value.is_floating_point():
                data[key] = value.to(dtype)

        predictor.model.module.on_predict_check(data)
        backbone = predictor.model.module.backbone
        with torch.no_grad(), _embedding_backbone_mode(backbone):
            embeddings = backbone(data)["node_embedding"].detach()

        node_features = (
            embeddings[:, 0, :] if scalar_only else embeddings.flatten(start_dim=1)
        )
        molecule_features = _pool_node_embedding(
            node_features, data.batch, len(chunk), pooling
        )

        node_features = node_features.cpu()
        molecule_features = molecule_features.cpu()
        atomic_numbers = data.atomic_numbers.cpu()
        batch_index = data.batch.cpu()

        for item_index, (path, frame, _atoms) in enumerate(chunk):
            node_mask = batch_index == item_index
            results.append({
                "path": str(path) if path is not None else None,
                "frame": frame,
                "atomic_numbers": atomic_numbers[node_mask].clone(),
                "node_embedding": node_features[node_mask].clone(),
                "molecule_embedding": molecule_features[item_index].clone(),
            })

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "checkpoint_path": str(checkpoint),
                "task_name": task_name,
                "pooling": pooling,
                "scalar_only": scalar_only,
                "embeddings": results,
            },
            out,
        )

    return results
