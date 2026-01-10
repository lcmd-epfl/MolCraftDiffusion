#!/usr/bin/env python3
"""
Compute RMSD and xTB energy difference between initial XYZ files and their
optimized counterparts in the `optimized_xyz` subdirectory.

No xyz2mol or force fields - uses xTB (GFN2) for energy calculations.

Usage:
    python compute_energy_rmsd.py <input_dir> [--n_subsets 5] [--csv output.csv]
"""

from __future__ import annotations

import argparse
import subprocess as sp
import tempfile
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from MolecularDiffusion.utils.geom_utils import read_xyz_file


def get_xtb_energy(xyz_path: str, charge: int = 0, level: str = "gfn2", timeout: int = 120) -> float:
    """
    Compute xTB single-point energy for an XYZ file.
    
    Args:
        xyz_path: Path to XYZ file
        charge: Molecular charge
        level: xTB level (gfn1, gfn2, gfn-ff)
        timeout: Timeout in seconds
        
    Returns:
        Energy in Hartree, or None if failed
    """
    cmd = ["xtb", xyz_path, f"-{level}", "-c", str(charge), "--sp"]
    
    try:
        result = sp.run(cmd, capture_output=True, text=True, timeout=timeout)
        
        # Parse energy from output
        for line in result.stdout.split("\n"):
            if "TOTAL ENERGY" in line:
                # Format: "          | TOTAL ENERGY              -XX.XXXXXX Eh"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "Eh" and i > 0:
                        return float(parts[i - 1])
        
        return None
    except (sp.TimeoutExpired, Exception) as e:
        return None


def compute_coord_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute RMSD between two coordinate sets (same atom order assumed)."""
    diff = coords1 - coords2
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def load_pairs(directory: Path) -> List[Tuple]:
    """Load coordinate pairs from input directory and optimized_xyz subdirectory."""
    optimized_dir = directory / "optimized_xyz"
    if not optimized_dir.is_dir():
        raise SystemExit(f"Error: Subdirectory 'optimized_xyz' not found in {directory}")

    pairs = []
    xyz_files = sorted(directory.glob("*.xyz"))

    for xyz_file in tqdm(xyz_files, desc="Finding XYZ pairs"):
        if xyz_file.stem.endswith("_opt"):
            continue

        opt_file = optimized_dir / f"{xyz_file.stem}_opt.xyz"
        if not opt_file.exists():
            continue

        pairs.append((xyz_file, opt_file))

    return pairs


def compute_metrics_for_pairs(
    pairs: List[Tuple],
    charge: int = 0,
    level: str = "gfn2",
    timeout: int = 120
) -> dict:
    """Compute RMSD and xTB energy difference for XYZ pairs."""
    rmsds, energy_diffs = [], []
    records = []

    for init_xyz, opt_xyz in tqdm(pairs, desc="Computing metrics"):
        try:
            # Read coordinates
            init_coords, init_Z = read_xyz_file(init_xyz)
            opt_coords, opt_Z = read_xyz_file(opt_xyz)

            init_np = init_coords.numpy() if hasattr(init_coords, 'numpy') else np.array(init_coords)
            opt_np = opt_coords.numpy() if hasattr(opt_coords, 'numpy') else np.array(opt_coords)

            # Compute RMSD
            rmsd = compute_coord_rmsd(init_np, opt_np)
            rmsds.append(rmsd)

            # Compute xTB energies
            e_init = get_xtb_energy(str(init_xyz), charge=charge, level=level, timeout=timeout)
            e_opt = get_xtb_energy(str(opt_xyz), charge=charge, level=level, timeout=timeout)

            if e_init is not None and e_opt is not None:
                # Energy drop = E_init - E_opt (positive means optimization lowered energy)
                energy_diff = (e_init - e_opt) * 627.5  # Convert Hartree to kcal/mol
                energy_diffs.append(energy_diff)
            else:
                energy_diff = None

            records.append({
                "file": init_xyz.name,
                "rmsd": rmsd,
                "energy_diff_kcal": energy_diff,
                "e_init_Ha": e_init,
                "e_opt_Ha": e_opt,
            })
        except Exception as e:
            tqdm.write(f"[WARN] Failed {init_xyz.name}: {e}")
            continue

    return {
        "avg_rmsd": np.mean(rmsds) if rmsds else 0.0,
        "med_rmsd": np.median(rmsds) if rmsds else 0.0,
        "avg_energy_diff": np.mean(energy_diffs) if energy_diffs else 0.0,
        "med_energy_diff": np.median(energy_diffs) if energy_diffs else 0.0,
        "n_rmsd": len(rmsds),
        "n_energy": len(energy_diffs),
        "records": records,
    }


def split_into_subsets(pairs: List, n_subsets: int) -> List[List]:
    """Split pairs into n_subsets with approximately equal size."""
    subset_size = len(pairs) // n_subsets
    return [pairs[i * subset_size:(i + 1) * subset_size] for i in range(n_subsets)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute RMSD and xTB energy difference between XYZ files and optimized counterparts."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing XYZ files (optimized in 'optimized_xyz' subdir)."
    )
    parser.add_argument(
        "--n_subsets",
        type=int,
        default=5,
        help="Number of subsets for std calculation (default: 5)."
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        type=Path,
        default=None,
        help="Output CSV filename for detailed results."
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=0,
        help="Molecular charge for xTB calculation (default: 0)."
    )
    parser.add_argument(
        "--level",
        type=str,
        default="gfn2",
        choices=["gfn1", "gfn2", "gfn-ff"],
        help="xTB level (default: gfn2)."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout per xTB calculation in seconds (default: 120)."
    )
    args = parser.parse_args()

    directory = args.directory.resolve()
    if not directory.is_dir():
        raise SystemExit(f"{directory} is not a directory")

    pairs = load_pairs(directory)
    print(f"\nFound {len(pairs)} XYZ pairs.\n")

    if len(pairs) == 0:
        raise SystemExit("No valid pairs found.")

    csv_path = args.csv_path if args.csv_path else directory / "energy_rmsd_results.csv"

    if args.n_subsets <= 1:
        results = compute_metrics_for_pairs(
            pairs, charge=args.charge, level=args.level, timeout=args.timeout
        )
        print(f"\nProcessed {results['n_rmsd']} RMSD, {results['n_energy']} energy pairs.")
        print(f"Avg RMSD: {results['avg_rmsd']:.4f} Å")
        print(f"Med RMSD: {results['med_rmsd']:.4f} Å")
        print(f"Avg Energy Diff: {results['avg_energy_diff']:.4f} kcal/mol")
        print(f"Med Energy Diff: {results['med_energy_diff']:.4f} kcal/mol")

        df = pd.DataFrame(results["records"])
        df.to_csv(csv_path, index=False)
        print(f"\nDetailed results saved to {csv_path}")
    else:
        print(f"Running analysis in {args.n_subsets} subsets...\n")

        subset_metrics = defaultdict(list)
        subsets = split_into_subsets(pairs, args.n_subsets)

        for subset in subsets:
            result = compute_metrics_for_pairs(
                subset, charge=args.charge, level=args.level, timeout=args.timeout
            )
            for key in ["avg_rmsd", "med_rmsd", "avg_energy_diff", "med_energy_diff"]:
                subset_metrics[key].append(result[key])

        n_total = len(pairs)
        print(f"\nProcessed {n_total} pairs across {args.n_subsets} subsets.\n")

        for key, unit in [("avg_rmsd", "Å"), ("med_rmsd", "Å"), 
                          ("avg_energy_diff", "kcal/mol"), ("med_energy_diff", "kcal/mol")]:
            values = subset_metrics[key]
            mean = np.mean(values)
            std = np.std(values)
            print(f"{key.replace('_', ' ').title()}: {mean:.4f} ± {std:.4f} {unit}")

        # Save summary
        summary_df = pd.DataFrame([{
            "rmsd_mean": np.mean(subset_metrics["avg_rmsd"]),
            "rmsd_std": np.std(subset_metrics["avg_rmsd"]),
            "energy_diff_mean": np.mean(subset_metrics["avg_energy_diff"]),
            "energy_diff_std": np.std(subset_metrics["avg_energy_diff"]),
        }])
        summary_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

    print("\nAnalysis complete.")
