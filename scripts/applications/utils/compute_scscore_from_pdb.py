#!/usr/bin/env python3
"""
Batch-compute SCScore values for every PDB file inside a directory.


Need scscore in scripts/applications/utils/, can be pulled from https://github.com/connorcoley/scscore, 
The script loads the pretrained SCScorer model shipped under `scscore/`,
converts each PDB into a SMILES string via RDKit, and computes the
corresponding synthetic complexity score. Results are written to a CSV
and basic distribution plots (KDE + histogram) are exported as PNGs.

Example:
    python compute_scscore_from_pdb.py ./molecules --output-csv ./scscore.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise ImportError("matplotlib is required for plotting SCScore distributions.") from exc

try:
    import seaborn as sns
except ImportError as exc:  # pragma: no cover
    raise ImportError("seaborn is required for KDE plotting. Install with `pip install seaborn`.") from exc

try:
    from rdkit import Chem
except ImportError as exc:  # pragma: no cover - clearly actionable error.
    raise ImportError("RDKit is required for SCScore computation.") from exc

# Make the local `scscore/` package importable without installation.
REPO_ROOT = Path(__file__).resolve().parents[0]
print(REPO_ROOT)
SCScore_SRC = REPO_ROOT / "scscore"
sys.path.append(str(SCScore_SRC))

from scscore import standalone_model_numpy  # type: ignore  # noqa: E402

LOGGER = logging.getLogger("compute_scscore_from_pdb")
LOGGER.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)-15s %(levelname)s: %(message)s"))
LOGGER.addHandler(_handler)


def plot_kde(data: np.ndarray, task_name: str, output_filepath: Path) -> None:
    """
    Generates and saves a Kernel Density Estimate (KDE) plot.

    Args:
        data (np.ndarray): Numerical data to plot.
        task_name (str): Label for the x-axis and plot title.
        output_filepath (Path): Full path to save the plot (e.g., 'kde_plot.png').
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.kdeplot(
        data,
        label=task_name,
        color="#008fd5",
        fill=True,
        linewidth=3,
        alpha=0.5,
        zorder=2,
        ax=ax,
    )

    y_max = ax.get_ylim()[1] * 1.05
    ax.set_ylim(0, y_max)
    ax.set_xlabel(task_name, fontsize=36)
    ax.set_ylabel("Frequency", fontsize=36)
    ax.tick_params(axis="both", labelsize=32)
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=300)
    plt.close(fig)


def plot_hist(data: np.ndarray, task_name: str, output_filepath: Path, bins: int = 30) -> None:
    """
    Generates and saves a histogram plot.

    Args:
        data (np.ndarray): Numerical data to plot.
        task_name (str): Label for the x-axis and plot title.
        output_filepath (Path): Full path to save the plot (e.g., 'hist_plot.png').
        bins (int): Number of bins for the histogram.
    """
    is_int = np.allclose(data, data.astype(int))
    if is_int:
        min_val = int(data.min())
        max_val = int(data.max())
        bin_edges = np.arange(min_val - 0.5, max_val + 1.5, 1.0)
        xticks = np.arange(min_val, max_val + 1)
    else:
        bin_edges = bins
        xticks = None

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(
        data,
        bins=bin_edges,
        color="#008fd5",
        alpha=0.5,
        edgecolor="black",
        linewidth=2,
        zorder=2,
    )

    if is_int:
        ax.set_xticks(xticks)

    y_max = ax.get_ylim()[1] * 1.05
    ax.set_ylim(0, y_max)

    ax.set_xlabel(task_name, fontsize=36)
    ax.set_ylabel("Frequency", fontsize=36)
    ax.tick_params(axis="both", labelsize=32)

    plt.tight_layout()
    plt.savefig(output_filepath, dpi=300)
    plt.close(fig)


def load_model() -> standalone_model_numpy.SCScorer:
    """Instantiate and restore the pretrained SCScore model."""
    model = standalone_model_numpy.SCScorer()
    model_path = SCScore_SRC / "models" / "full_reaxys_model_1024bool" / "model.ckpt-10654.as_numpy.json.gz"
    model.restore(str(model_path))
    return model


def iter_pdb_files(pdb_dir: Path, recursive: bool) -> Iterable[Path]:
    """Yield all matching PDB files from the directory."""
    pattern = "**/*.pdb" if recursive else "*.pdb"
    yield from sorted(p for p in pdb_dir.glob(pattern) if p.is_file())


def parse_arguments() -> argparse.Namespace:
    """CLI definition."""
    parser = argparse.ArgumentParser(
        description="Compute SCScore for every PDB file in a directory and export plots + CSV."
    )
    parser.add_argument("pdb_dir", type=Path, help="Directory that contains PDB files.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Destination CSV (default: <pdb_dir>/scscore_results.csv).",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Directory to store distribution plots (default: same folder as the CSV).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Whether to recurse into subdirectories when searching for PDBs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if not args.pdb_dir.is_dir():
        raise NotADirectoryError(f"PDB directory not found: {args.pdb_dir}")

    output_csv = args.output_csv or (args.pdb_dir / "scscore_results.csv")
    plots_dir = args.plots_dir or output_csv.parent
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading SCScore model from %s", SCScore_SRC)
    model = load_model()
    LOGGER.info("Model ready.")

    pdb_files = list(iter_pdb_files(args.pdb_dir, args.recursive))
    if not pdb_files:
        raise FileNotFoundError(f"No PDB files found in {args.pdb_dir} (recursive={args.recursive}).")
    LOGGER.info("Found %d PDB files.", len(pdb_files))

    records: list[tuple[str, float]] = []
    errors: list[tuple[Path, str]] = []

    for pdb_path in tqdm(pdb_files, desc="Computing SCScore"):
        # try:
        mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=True)
        if mol is None:
            errors.append((pdb_path, "RDKit returned None"))
            continue
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        _, score = model.get_score_from_smi(smiles)
        records.append((pdb_path.name, float(score)))
        # except Exception as exc:  # pylint: disable=broad-except
        #     errors.append((pdb_path, str(exc)))

    if not records:
        raise RuntimeError("SCScore computation failed for every file. Check RDKit parsing.")

    df = pd.DataFrame(records, columns=["filename", "scscore"]).sort_values("filename")
    df.to_csv(output_csv, index=False)
    LOGGER.info("Wrote %d rows to %s", len(df), output_csv)

    scores = df["scscore"].to_numpy()
    LOGGER.info(
        "SCScore stats -> count=%d, mean=%.3f, min=%.3f, max=%.3f, std=%.3f",
        len(scores),
        float(np.mean(scores)),
        float(np.min(scores)),
        float(np.max(scores)),
        float(np.std(scores)),
    )
    if errors:
        LOGGER.warning("Encountered %d failures; see below.", len(errors))
        for path, msg in errors:
            LOGGER.warning("  %s -> %s", path, msg)

    plot_kde(scores, "SCScore", plots_dir / "scscore_kde.png")
    plot_hist(scores, "SCScore", plots_dir / "scscore_hist.png")
    LOGGER.info("Saved plots to %s", plots_dir)


if __name__ == "__main__":
    main()
