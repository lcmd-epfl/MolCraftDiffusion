"""Shared I/O helpers for the analyze runmodes."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ase import Atoms
from ase.io import read as ase_read

DEFAULT_STRUCTURE_EXTENSIONS = (
    ".xyz",
    ".extxyz",
    ".sdf",
    ".mol",
    ".mol2",
    ".pdb",
    ".cif",
    ".traj",
)


def iter_structure_files(
    directory: str | Path,
    extensions: Sequence[str] = DEFAULT_STRUCTURE_EXTENSIONS,
    recursive: bool = False,
) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")
    suffixes = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in extensions
    }
    files = root.rglob("*") if recursive else root.iterdir()
    return sorted(p for p in files if p.is_file() and p.suffix.lower() in suffixes)


def read_atoms(path: Path, read_all_frames: bool = False) -> list[Atoms]:
    if read_all_frames:
        result = ase_read(path, index=":")
        return [result] if isinstance(result, Atoms) else list(result)
    return [ase_read(path)]
