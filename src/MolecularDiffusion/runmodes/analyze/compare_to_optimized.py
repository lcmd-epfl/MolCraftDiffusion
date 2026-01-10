#!/usr/bin/env python3
"""
Compute unified comparison metrics (RMSD, xTB Energy, Bond Geometry) between
initial XYZ files and their optimized counterparts in the `optimized_xyz` subdirectory.

Features:
- Validates connectivity (skip disjoint graphs)
- Computes RMSD and Energy difference (via xTB)
- Computes Bond Length, Angle, and Torsion differences (via OpenBabel/cell2mol)
"""

from __future__ import annotations

import argparse
import os
import subprocess as sp
import shutil
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem

try:
    from openbabel import pybel
except ImportError:
    pass

from MolecularDiffusion.utils.geom_utils import read_xyz_file
# Import from local package assuming this script is in src/MolecularDiffusion/runmodes/analyze/
from MolecularDiffusion.runmodes.analyze.xtb_optimization import check_xyz
from MolecularDiffusion.utils.geom_stability import (
    compute_bond_lengths_diff,
    compute_bond_angles_diff,
    compute_torsion_angles_diff,
    compute_differences,
)


def get_xtb_energy(xyz_path: str, charge: int = 0, level: str = "gfn2", timeout: int = 120) -> Optional[float]:
    """Compute xTB single-point energy for an XYZ file (returns Hartree)."""
    cmd = ["xtb", xyz_path, f"-{level}", "-c", str(charge), "--sp"]
    try:
        result = sp.run(cmd, capture_output=True, text=True, timeout=timeout)
        for line in result.stdout.split("\n"):
            if "TOTAL ENERGY" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "Eh" and i > 0:
                        return float(parts[i - 1])
        return None
    except (sp.TimeoutExpired, Exception):
        return None


def xyz2mol_openbabel(xyz_file: str) -> Optional[Chem.Mol]:
    """Convert XYZ to RDKit Mol using OpenBabel (via pybel)."""
    try:
        mol_pb = next(pybel.readfile("xyz", str(xyz_file)))
        mol_sdf = mol_pb.write("sdf")
        mol = Chem.MolFromMolBlock(mol_sdf, removeHs=False, sanitize=False)
        if mol is not None:
            Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def xyz2mol_cell2mol(xyz_file: str, timeout: int = 10) -> Optional[Chem.Mol]:
    """Convert XYZ to RDKit Mol using cell2mol logic."""
    from MolecularDiffusion.utils.smilify import smilify_cell2mol
    try:
        _, mol = smilify_cell2mol(str(xyz_file), timeout=timeout)
        return mol
    except Exception:
        return None


def compute_coord_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute RMSD between two coordinate sets."""
    diff = coords1 - coords2
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def compute_all_metrics(
    init_file: Path,
    opt_file: Path,
    args: argparse.Namespace
) -> dict:
    """Compute all available metrics for a single pair."""
    results = {}
    
    # 1. Connectivity Check
    if check_xyz is not None:
        is_conn_init, n_comp_init, _ = check_xyz(str(init_file), scale_factor=1.3)
        if not (is_conn_init and n_comp_init == 1):
            return {"error": "Initial molecule not fully connected"}
        
        is_conn_opt, n_comp_opt, _ = check_xyz(str(opt_file), scale_factor=1.3)
        if not (is_conn_opt and n_comp_opt == 1):
            return {"error": "Optimized molecule not fully connected"}
    
    # 2. RMSD & Energy
    try:
        init_coords, _ = read_xyz_file(init_file)
        opt_coords, _ = read_xyz_file(opt_file)
        
        init_np = init_coords.numpy() if hasattr(init_coords, 'numpy') else np.array(init_coords)
        opt_np = opt_coords.numpy() if hasattr(opt_coords, 'numpy') else np.array(opt_coords)
        
        results["rmsd"] = compute_coord_rmsd(init_np, opt_np)
        
        e_init = get_xtb_energy(str(init_file), args.charge, args.level, args.timeout)
        e_opt = get_xtb_energy(str(opt_file), args.charge, args.level, args.timeout)
        
        results["e_init_Ha"] = e_init
        results["e_opt_Ha"] = e_opt
        
        if e_init is not None and e_opt is not None:
            results["energy_diff_kcal"] = (e_init - e_opt) * 627.5
        else:
            results["energy_diff_kcal"] = None
            
    except Exception as e:
        return {"error": f"RMSD/Energy failed: {e}"}

    # 3. Bond Geometry
    try:
        if args.mol_converter == "openbabel":
            init_mol = xyz2mol_openbabel(str(init_file))
            opt_mol = xyz2mol_openbabel(str(opt_file))
        elif args.mol_converter == "cell2mol":
            init_mol = xyz2mol_cell2mol(str(init_file), timeout=args.timeout)
            opt_mol = xyz2mol_cell2mol(str(opt_file), timeout=args.timeout)
        else:
            return {"error": f"Unknown converter {args.mol_converter}"}
            
        if init_mol is None or opt_mol is None:
             return {"error": "Failed to load molecules for bond analysis"}
             
        pair = (init_mol, opt_mol)

        # Helper to extract mean diff from result tuple (avg_diff, std, weight)
        def get_diff(res_dict):
            # Typically returns a dict key -> (list of diffs, list of weights) in run_bond_analysis 
            # But compute_differences returns {k: (avg, std, weight)} ??
            # Wait, check `compute_pair_geometry.py` logic:
            # compute_differences returns dict: key -> (avg_diff, std_dev, weight)
            # Actually compute_differences returns {bond_type: (avg_diff, std, weight)}
            # We want the weighted mean across all bond types.
            total_w_diff = 0
            total_weight = 0
            for k, (avg, s, w) in res_dict.items():
                total_w_diff += avg * w
                total_weight += w
            return total_w_diff / total_weight if total_weight > 0 else None

        # Check `geom_stability.py` - checking `compute_pair_geometry.py` usage:
        # It aggregates results. Here we are doing per-pair.
        # `compute_differences` takes a List of pairs. We pass list of 1 pair.
        
        b_len_res = compute_differences([pair], compute_bond_lengths_diff)
        b_ang_res = compute_differences([pair], compute_bond_angles_diff)
        tor_res = compute_differences([pair], compute_torsion_angles_diff)
        
        results["bond_length_mean"] = get_diff(b_len_res)
        results["bond_angle_mean"] = get_diff(b_ang_res)
        results["torsion_angle_mean"] = get_diff(tor_res)

    except Exception as e:
         return {"error": f"Bond analysis failed: {e}"}

    return results


def run_compare_analysis(args: argparse.Namespace):
    directory = Path(args.directory).resolve()
    optimized_dir = directory / "optimized_xyz"
    csv_path = args.csv_path if args.csv_path else directory / "compare_results.csv"

    if not optimized_dir.is_dir():
        print(f"Error: {optimized_dir} does not exist.")
        return

    xyz_files = sorted(directory.glob("*.xyz"))
    pairs = []
    
    for f in xyz_files:
        if f.stem.endswith("_opt"): continue
        opt = optimized_dir / f"{f.stem}_opt.xyz"
        if opt.exists():
            pairs.append((f, opt))

    print(f"Found {len(pairs)} pairs in {directory}")
    print(f"Running analysis with converter={args.mol_converter}, level={args.level}")

    all_records = []
    
    # Process pairs
    for init_f, opt_f in tqdm(pairs, desc="Analyzing"):
        res = compute_all_metrics(init_f, opt_f, args)
        if "error" in res:
            # print(f"Skipping {init_f.name}: {res['error']}")
            continue
        
        rec = {"file": init_f.name, **res}
        all_records.append(rec)

    if not all_records:
        print("No valid results found.")
        return

    df = pd.DataFrame(all_records)
    
    # Save detailed results
    df.to_csv(csv_path, index=False)
    print(f"Saved detailed results to {csv_path}")

    # Summary Stats
    summary = {}
    metric_cols = [
        "rmsd", "energy_diff_kcal", 
        "bond_length_mean", "bond_angle_mean", "torsion_angle_mean"
    ]
    
    print("\n--- Summary ---")
    for col in metric_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            mean_val = df[col].mean()
            std_val = df[col].std()
            print(f"{col}: {mean_val:.4f} ± {std_val:.4f}")
            summary[f"{col}_mean"] = mean_val
            summary[f"{col}_std"] = std_val
    
    summary_path = directory / "compare_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified comparison of molecular geometries.")
    parser.add_argument("directory", type=str)
    parser.add_argument("--mol-converter", choices=["openbabel", "cell2mol"], default="openbabel")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--level", default="gfn2")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--csv", "csv_path", type=Path, default=None)
    parser.add_argument("--n-subsets", type=int, default=5) # Kept for CLI compat, primarily used for summary stats

    args = parser.parse_args()
    run_compare_analysis(args)
