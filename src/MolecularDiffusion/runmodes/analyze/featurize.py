"""Molecular featurization backends for 3D XYZ structures.

Backends
--------
soap  SOAP descriptor via dscribe — CPU, no external model needed.
uma   UMA backbone embeddings — requires vendored fairchem/src + checkpoint.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase import Atoms

from ._io import DEFAULT_STRUCTURE_EXTENSIONS, iter_structure_files, read_atoms

DEFAULT_SOAP_SPECIES = [
    "H", "B", "C", "N", "O", "F",
    "Al", "Si", "P", "S", "Cl",
    "As", "Se", "Br", "I", "Hg", "Bi",
]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseFeaturizer(ABC):
    @abstractmethod
    def featurize(self, atoms_list: list[Atoms]) -> np.ndarray:
        """Return float32 array of shape (N, D)."""


# ---------------------------------------------------------------------------
# SOAP backend
# ---------------------------------------------------------------------------

class SOAPFeaturizer(BaseFeaturizer):
    def __init__(
        self,
        species: list[str],
        r_cut: float = 6.0,
        n_max: int = 8,
        l_max: int = 6,
        sigma: float = 0.1,
        pooling: str = "mean",
        n_jobs: int = 1,
    ) -> None:
        try:
            from dscribe.descriptors import SOAP
        except ImportError as exc:
            raise ImportError(
                "dscribe is required for the SOAP backend. "
                "Install it with: pip install dscribe"
            ) from exc

        self.desc = SOAP(
            species=species,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            sigma=sigma,
            periodic=False,
        )
        if pooling not in ("mean", "sum"):
            raise ValueError(f"pooling must be 'mean' or 'sum', got {pooling!r}")
        self.pooling = pooling
        self.n_jobs = n_jobs

    def featurize(self, atoms_list: list[Atoms]) -> np.ndarray:
        per_atom = self.desc.create(atoms_list, n_jobs=self.n_jobs)
        # dscribe returns a list when input is a list
        if isinstance(per_atom, np.ndarray) and per_atom.ndim == 2:
            # single molecule edge case
            per_atom = [per_atom]
        if self.pooling == "mean":
            pooled = [a.mean(axis=0) for a in per_atom]
        else:
            pooled = [a.sum(axis=0) for a in per_atom]
        return np.vstack(pooled).astype(np.float32)


# ---------------------------------------------------------------------------
# UMA backend
# ---------------------------------------------------------------------------

class UMAFeaturizer(BaseFeaturizer):
    def __init__(
        self,
        checkpoint: str | Path,
        task_name: str = "omol",
        device: str | None = None,
        batch_size: int = 8,
        pooling: str = "mean",
        scalar_only: bool = True,
        charge: int = 0,
        spin: int = 1,
    ) -> None:
        self.kwargs = dict(
            checkpoint_path=checkpoint,
            task_name=task_name,
            device=device,
            batch_size=batch_size,
            pooling=pooling,
            scalar_only=scalar_only,
            charge=charge,
            spin=spin,
        )

    def featurize(self, atoms_list: list[Atoms]) -> np.ndarray:
        from .uma_embeddings import get_uma_molecule_embeddings
        results = get_uma_molecule_embeddings(source=atoms_list, **self.kwargs)
        return np.vstack(
            [r["molecule_embedding"].numpy() for r in results]
        ).astype(np.float32)


# ---------------------------------------------------------------------------
# SSL3D backend
# ---------------------------------------------------------------------------

class SSL3DFeaturizer(BaseFeaturizer):
    def __init__(
        self,
        checkpoint: str | Path,
        device: str | None = None,
        batch_size: int = 16,
        pooling: str = "mean",
        edge_radius: float = 5.0,
        atom_vocab: list[str] | None = None,
    ) -> None:
        import torch
        from .ssl3d_embed import load_ssl3d_task

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.pooling = pooling
        self.edge_radius = edge_radius

        print(f"Loading SSL3D checkpoint: {checkpoint}")
        self.task, vocab_from_ckpt = load_ssl3d_task(checkpoint, device=self.device)
        self.atom_vocab = atom_vocab if atom_vocab is not None else vocab_from_ckpt
        print(f"  atom_vocab : {self.atom_vocab}")
        print(f"  device     : {self.device}")

    def featurize(self, atoms_list: list[Atoms]) -> np.ndarray:
        import torch
        from .ssl3d_embed import atoms_to_batch, pool_nodes

        results = []
        n_total = len(atoms_list)
        for start in range(0, n_total, self.batch_size):
            chunk = atoms_list[start : start + self.batch_size]
            batch_dict = atoms_to_batch(
                chunk, self.atom_vocab, self.edge_radius, self.device
            )
            graph = batch_dict["graph"]
            n_atoms = graph.x.shape[0]
            sigma = torch.zeros(n_atoms, 1, device=self.device)

            with torch.no_grad():
                h, _ = self.task._forward_backbone(batch_dict, sigma)

            mol_embs = pool_nodes(h, graph.batch, self.pooling)
            results.append(mol_embs.cpu().numpy())

        return np.vstack(results).astype(np.float32)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, type[BaseFeaturizer]] = {
    "soap": SOAPFeaturizer,
    "uma": UMAFeaturizer,
    "ssl3d": SSL3DFeaturizer,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_featurize(
    input_dir: str | Path,
    backend: str = "soap",
    output_path: str | Path | None = None,
    extensions: Sequence[str] = DEFAULT_STRUCTURE_EXTENSIONS,
    recursive: bool = False,
    # SOAP kwargs
    species: list[str] | None = None,   # None → use DEFAULT_SOAP_SPECIES
    autodetect_species: bool = False,   # True → detect from files, ignores species
    r_cut: float = 6.0,
    n_max: int = 8,
    l_max: int = 6,
    sigma: float = 0.1,
    pooling: str = "mean",
    n_jobs: int = 1,
    # UMA kwargs
    checkpoint: str | Path = "training_outputs/uma-s-1p2.pt",
    task_name: str = "omol",
    device: str | None = None,
    batch_size: int = 8,
    scalar_only: bool = True,
    charge: int = 0,
    spin: int = 1,
    # SSL3D kwargs
    ssl3d_checkpoint: str | Path | None = None,
    ssl3d_edge_radius: float = 5.0,
    ssl3d_atom_vocab: list[str] | None = None,
) -> np.ndarray:
    """Discover XYZ files, featurize, save outputs, return (N, D) array."""
    input_dir = Path(input_dir)

    # --- discover files ---
    paths = iter_structure_files(input_dir, extensions=extensions, recursive=recursive)
    if not paths:
        raise FileNotFoundError(f"No structure files found in {input_dir}")

    print(f"Found {len(paths)} structure files.")

    # --- load atoms ---
    records: list[tuple[Path, int]] = []
    atoms_list: list[Atoms] = []
    for path in paths:
        for frame, atoms in enumerate(read_atoms(path)):
            records.append((path, frame))
            atoms_list.append(atoms)

    # --- build featurizer ---
    if backend == "soap":
        # collect all elements present in the dataset
        found: set[str] = set()
        for atoms in atoms_list:
            found.update(atoms.get_chemical_symbols())

        if autodetect_species:
            species = sorted(found)
            print(f"Auto-detected species: {species}")
        else:
            if species is None:
                species = list(DEFAULT_SOAP_SPECIES)
            extra = sorted(found - set(species))
            if extra:
                print(f"Warning: elements {extra} not in species list — adding them.")
                species = sorted(set(species) | set(extra))
            print(f"Using species: {species}")

        featurizer: BaseFeaturizer = SOAPFeaturizer(
            species=species,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            sigma=sigma,
            pooling=pooling,
            n_jobs=n_jobs,
        )
        meta_params = dict(
            species=species, r_cut=r_cut, n_max=n_max,
            l_max=l_max, sigma=sigma, pooling=pooling, n_jobs=n_jobs,
        )

    elif backend == "uma":
        featurizer = UMAFeaturizer(
            checkpoint=checkpoint,
            task_name=task_name,
            device=device,
            batch_size=batch_size,
            pooling=pooling,
            scalar_only=scalar_only,
            charge=charge,
            spin=spin,
        )
        meta_params = dict(
            checkpoint=str(checkpoint), task_name=task_name,
            device=device, batch_size=batch_size,
            pooling=pooling, scalar_only=scalar_only,
            charge=charge, spin=spin,
        )

    elif backend == "ssl3d":
        if ssl3d_checkpoint is None:
            raise ValueError("ssl3d_checkpoint is required for the ssl3d backend.")
        featurizer = SSL3DFeaturizer(
            checkpoint=ssl3d_checkpoint,
            device=device,
            batch_size=batch_size,
            pooling=pooling,
            edge_radius=ssl3d_edge_radius,
            atom_vocab=ssl3d_atom_vocab,
        )
        meta_params = dict(
            checkpoint=str(ssl3d_checkpoint),
            device=device, batch_size=batch_size,
            pooling=pooling, edge_radius=ssl3d_edge_radius,
        )

    else:
        raise ValueError(f"Unknown backend {backend!r}. Choose from: {list(_BACKENDS)}")

    # --- featurize ---
    print(f"Running {backend} featurization on {len(atoms_list)} structures...")
    features = featurizer.featurize(atoms_list)  # (N, D)

    # --- output paths ---
    stem = Path(output_path) if output_path else input_dir / "features"
    stem = stem.with_suffix("")  # strip any extension the user may have added
    stem.parent.mkdir(parents=True, exist_ok=True)

    npy_path = stem.with_suffix(".npy")
    csv_path = stem.with_suffix(".csv")
    meta_path = Path(str(stem) + "_meta.json")

    # --- save ---
    np.save(npy_path, features)

    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "file", "frame"])
        for i, (path, frame) in enumerate(records):
            writer.writerow([i, str(path), frame])

    meta = {
        "backend": backend,
        "n_structures": len(atoms_list),
        "feature_dim": features.shape[1],
        "params": meta_params,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n--- Summary ---")
    print(f"Structures : {len(atoms_list)}")
    print(f"Feature dim: {features.shape[1]}")
    print(f"Outputs    : {npy_path}")
    print(f"             {csv_path}")
    print(f"             {meta_path}")

    return features
