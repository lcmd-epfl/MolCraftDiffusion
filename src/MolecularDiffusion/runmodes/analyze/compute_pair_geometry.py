#!/usr/bin/env python3
"""
Compute pair geometry metrics (bond lengths, angles, torsions) between
initial XYZ files and their optimized counterparts in the `optimized_xyz` subdirectory.

Two modes are available:
  - bond: Uses xyz2mol to infer bonds, then computes bond-specific geometry differences
  - geometry: Direct coordinate comparison without bond perception (RMSD-based)

Usage:
    python compute_pair_geometry.py <input_dir> [--mode bond] [--n_subsets 5]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from MolecularDiffusion.utils.geom_utils import read_xyz_file


# ============================================================================
# Mode: Bond-aware (requires xyz2mol)
# ============================================================================

def load_mol_pairs_bond_mode(directory: Path, timeout: int = 10) -> List[Tuple]:
    """Load molecule pairs using xyz2mol for bond perception."""
    from MolecularDiffusion.utils.smilify import smilify_cell2mol
    
    optimized_dir = directory / "optimized_xyz"
    if not optimized_dir.is_dir():
        raise SystemExit(f"Error: Subdirectory 'optimized_xyz' not found in {directory}")

    pairs = []
    xyz_files = sorted(directory.glob("*.xyz"))

    for xyz_file in tqdm(xyz_files, desc="Loading molecule pairs (bond mode)"):
        if xyz_file.stem.endswith("_opt"):
            continue

        opt_file = optimized_dir / f"{xyz_file.stem}_opt.xyz"
        if not opt_file.exists():
            continue

        try:
            _, init_mol = smilify_cell2mol(str(xyz_file), timeout=timeout)
            _, opt_mol = smilify_cell2mol(str(opt_file), timeout=timeout)

            if init_mol is not None and opt_mol is not None:
                pairs.append((xyz_file.name, init_mol, opt_mol))
        except Exception as e:
            tqdm.write(f"[WARN] Failed to load {xyz_file.name}: {e}")

    return pairs


def run_bond_analysis(pairs: List[Tuple], analysis_type: str = "bond_length"):
    """Run bond-aware geometry analysis on molecule pairs."""
    from MolecularDiffusion.utils.geom_stability import (
        compute_bond_lengths_diff,
        compute_bond_angles_diff,
        compute_torsion_angles_diff,
        compute_differences,
    )
    
    accumulated_results = defaultdict(lambda: ([], []))
    mol_pairs = [(p[1], p[2]) for p in pairs]  # Extract (init_mol, opt_mol)

    is_not_none = lambda x: (x[0] is not None) and (x[1] is not None)
    mol_pairs = list(filter(is_not_none, mol_pairs))

    if analysis_type == "bond_length":
        results = compute_differences(mol_pairs, compute_bond_lengths_diff)
    elif analysis_type == "bond_angle":
        results = compute_differences(mol_pairs, compute_bond_angles_diff)
    elif analysis_type == "torsion_angle":
        results = compute_differences(mol_pairs, compute_torsion_angles_diff)
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")

    for key, (avg_diff, std_dev, weight) in results.items():
        accumulated_results[key][0].append(avg_diff)
        accumulated_results[key][1].append(weight)

    return accumulated_results


def summarize_bond_results(results):
    """Compute weighted mean from accumulated bond results."""
    total_weighted_diffs = []
    total_weight = 0

    for key, (avg_diff_list, weight_list) in results.items():
        weighted_diffs = np.array(avg_diff_list) * np.array(weight_list)
        total_weighted_diffs.append(np.sum(weighted_diffs))
        total_weight += np.sum(weight_list)

    return np.sum(total_weighted_diffs) / total_weight if total_weight > 0 else 0


# ============================================================================
# Mode: Geometry-only (no xyz2mol, direct coordinate comparison)
# ============================================================================

def load_coord_pairs_geometry_mode(directory: Path) -> List[Tuple]:
    """Load coordinate pairs directly from XYZ files (no bond perception)."""
    optimized_dir = directory / "optimized_xyz"
    if not optimized_dir.is_dir():
        raise SystemExit(f"Error: Subdirectory 'optimized_xyz' not found in {directory}")

    pairs = []
    xyz_files = sorted(directory.glob("*.xyz"))

    for xyz_file in tqdm(xyz_files, desc="Loading coordinate pairs (geometry mode)"):
        if xyz_file.stem.endswith("_opt"):
            continue

        opt_file = optimized_dir / f"{xyz_file.stem}_opt.xyz"
        if not opt_file.exists():
            continue

        try:
            init_coords, init_Z = read_xyz_file(xyz_file)
            opt_coords, opt_Z = read_xyz_file(opt_file)

            # Ensure same atom count
            if init_coords.shape[0] == opt_coords.shape[0]:
                pairs.append((xyz_file.name, init_coords, opt_coords, init_Z))
        except Exception as e:
            tqdm.write(f"[WARN] Failed to load {xyz_file.name}: {e}")

    return pairs


def compute_geometry_rmsd(init_coords: np.ndarray, opt_coords: np.ndarray) -> float:
    """Compute RMSD between two coordinate sets (assumes same atom order)."""
    diff = init_coords - opt_coords
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def compute_max_displacement(init_coords: np.ndarray, opt_coords: np.ndarray) -> float:
    """Compute max atom displacement between two coordinate sets."""
    diff = init_coords - opt_coords
    return float(np.max(np.sqrt(np.sum(diff ** 2, axis=1))))


def compute_mean_displacement(init_coords: np.ndarray, opt_coords: np.ndarray) -> float:
    """Compute mean atom displacement between two coordinate sets."""
    diff = init_coords - opt_coords
    return float(np.mean(np.sqrt(np.sum(diff ** 2, axis=1))))


def run_geometry_analysis(pairs: List[Tuple]) -> dict:
    """Run geometry-only analysis (RMSD, max/mean displacement)."""
    rmsds, max_disps, mean_disps = [], [], []
    records = []

    for name, init_coords, opt_coords, _ in pairs:
        init_np = init_coords.numpy() if hasattr(init_coords, 'numpy') else np.array(init_coords)
        opt_np = opt_coords.numpy() if hasattr(opt_coords, 'numpy') else np.array(opt_coords)
        
        rmsd = compute_geometry_rmsd(init_np, opt_np)
        max_disp = compute_max_displacement(init_np, opt_np)
        mean_disp = compute_mean_displacement(init_np, opt_np)

        rmsds.append(rmsd)
        max_disps.append(max_disp)
        mean_disps.append(mean_disp)
        records.append({
            "file": name,
            "rmsd": rmsd,
            "max_displacement": max_disp,
            "mean_displacement": mean_disp,
        })

    return {
        "avg_rmsd": np.mean(rmsds) if rmsds else 0.0,
        "avg_max_disp": np.mean(max_disps) if max_disps else 0.0,
        "avg_mean_disp": np.mean(mean_disps) if mean_disps else 0.0,
        "n": len(rmsds),
        "records": records,
    }


# ============================================================================
# Subset Analysis Helpers
# ============================================================================

def run_subsets_bond_analysis(pairs: List[Tuple], analysis_type: str, n_subsets: int):
    """Run bond analysis on subsets and return list of scores."""
    fold_size = len(pairs) // n_subsets
    scores = []

    for i in range(n_subsets):
        if i < n_subsets - 1:
            fold_pairs = pairs[i * fold_size: (i + 1) * fold_size]
        else:
            fold_pairs = pairs[i * fold_size:]

        results = run_bond_analysis(fold_pairs, analysis_type=analysis_type)
        score = summarize_bond_results(results)
        scores.append(score)

    return scores


def run_subsets_geometry_analysis(pairs: List[Tuple], n_subsets: int):
    """Run geometry analysis on subsets and return dict of score lists."""
    fold_size = len(pairs) // n_subsets
    all_results = defaultdict(list)

    for i in range(n_subsets):
        if i < n_subsets - 1:
            fold_pairs = pairs[i * fold_size: (i + 1) * fold_size]
        else:
            fold_pairs = pairs[i * fold_size:]

        results = run_geometry_analysis(fold_pairs)
        for key in ["avg_rmsd", "avg_max_disp", "avg_mean_disp"]:
            all_results[key].append(results[key])

    return all_results


def print_scores(name: str, scores: List[float], unit: str = ""):
    """Print mean and std of scores."""
    mean = np.mean(scores)
    std = np.std(scores)
    print(f"{name}: {mean:.4f} ± {std:.4f} {unit}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute pair geometry metrics between XYZ files and optimized counterparts."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing XYZ files (optimized in 'optimized_xyz' subdir)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="bond",
        choices=["bond", "geometry"],
        help="Analysis mode: 'bond' (uses xyz2mol) or 'geometry' (direct coords). Default: bond."
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
        "--timeout",
        type=int,
        default=10,
        help="Timeout for cell2mol conversion in bond mode (default: 10s)."
    )
    args = parser.parse_args()

    directory = args.directory.resolve()
    if not directory.is_dir():
        raise SystemExit(f"{directory} is not a directory")

    csv_path = args.csv_path if args.csv_path else directory / "pair_geometry_results.csv"

    # ========== BOND MODE ==========
    if args.mode == "bond":
        pairs = load_mol_pairs_bond_mode(directory, timeout=args.timeout)
        print(f"\nLoaded {len(pairs)} molecule pairs (bond mode).\n")

        if len(pairs) == 0:
            raise SystemExit("No valid molecule pairs found.")

        if args.n_subsets <= 1:
            print("Bond Lengths")
            lengths = run_bond_analysis(pairs, analysis_type="bond_length")
            print(f"  Weighted Mean: {summarize_bond_results(lengths):.4f} Å")

            print("\nBond Angles")
            angles = run_bond_analysis(pairs, analysis_type="bond_angle")
            print(f"  Weighted Mean: {summarize_bond_results(angles):.4f}°")

            print("\nTorsions")
            torsions = run_bond_analysis(pairs, analysis_type="torsion_angle")
            print(f"  Weighted Mean: {summarize_bond_results(torsions):.4f}°")

            summary_df = pd.DataFrame([{
                "mode": "bond",
                "bond_length_mean": summarize_bond_results(lengths),
                "bond_angle_mean": summarize_bond_results(angles),
                "torsion_mean": summarize_bond_results(torsions),
            }])
            summary_df.to_csv(csv_path, index=False)
            print(f"\nResults saved to {csv_path}")
        else:
            print(f"Running bond analysis in {args.n_subsets} subsets...\n")

            bond_length_scores = run_subsets_bond_analysis(pairs, "bond_length", args.n_subsets)
            bond_angle_scores = run_subsets_bond_analysis(pairs, "bond_angle", args.n_subsets)
            torsion_scores = run_subsets_bond_analysis(pairs, "torsion_angle", args.n_subsets)

            print_scores("Bond Lengths", bond_length_scores, "Å")
            print_scores("Bond Angles", bond_angle_scores, "°")
            print_scores("Torsions", torsion_scores, "°")

            summary_df = pd.DataFrame([{
                "mode": "bond",
                "bond_length_mean": np.mean(bond_length_scores),
                "bond_length_std": np.std(bond_length_scores),
                "bond_angle_mean": np.mean(bond_angle_scores),
                "bond_angle_std": np.std(bond_angle_scores),
                "torsion_mean": np.mean(torsion_scores),
                "torsion_std": np.std(torsion_scores),
            }])
            summary_df.to_csv(csv_path, index=False)
            print(f"\nResults saved to {csv_path}")

    # ========== GEOMETRY MODE ==========
    else:
        pairs = load_coord_pairs_geometry_mode(directory)
        print(f"\nLoaded {len(pairs)} coordinate pairs (geometry mode).\n")

        if len(pairs) == 0:
            raise SystemExit("No valid coordinate pairs found.")

        if args.n_subsets <= 1:
            results = run_geometry_analysis(pairs)
            print(f"Processed {results['n']} molecules.")
            print(f"Avg RMSD: {results['avg_rmsd']:.4f} Å")
            print(f"Avg Max Displacement: {results['avg_max_disp']:.4f} Å")
            print(f"Avg Mean Displacement: {results['avg_mean_disp']:.4f} Å")

            df = pd.DataFrame(results["records"])
            df.to_csv(csv_path, index=False)
            print(f"\nDetailed results saved to {csv_path}")
        else:
            print(f"Running geometry analysis in {args.n_subsets} subsets...\n")

            subset_results = run_subsets_geometry_analysis(pairs, args.n_subsets)

            print_scores("Avg RMSD", subset_results["avg_rmsd"], "Å")
            print_scores("Avg Max Displacement", subset_results["avg_max_disp"], "Å")
            print_scores("Avg Mean Displacement", subset_results["avg_mean_disp"], "Å")

            summary_df = pd.DataFrame([{
                "mode": "geometry",
                "rmsd_mean": np.mean(subset_results["avg_rmsd"]),
                "rmsd_std": np.std(subset_results["avg_rmsd"]),
                "max_disp_mean": np.mean(subset_results["avg_max_disp"]),
                "max_disp_std": np.std(subset_results["avg_max_disp"]),
                "mean_disp_mean": np.mean(subset_results["avg_mean_disp"]),
                "mean_disp_std": np.std(subset_results["avg_mean_disp"]),
            }])
            summary_df.to_csv(csv_path, index=False)
            print(f"\nResults saved to {csv_path}")

    print("\nAnalysis complete.")
