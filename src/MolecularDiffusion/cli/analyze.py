"""Analyze CLI subcommands for 3D molecule analysis.

Provides subcommands for:
- optimize: XTB geometry optimization
- metrics: Validity/connectivity metrics
- compare: RMSD, energy, and optional bond analysis
- xyz2mol: XYZ to SMILES conversion + fingerprints
"""

import os
from pathlib import Path
from typing import Optional

import click

# Enable -h as alias for --help
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def analyze():
    """Analyze 3D molecular structures.
    
    \b
    Subcommands:
      optimize  XTB geometry optimization
      metrics   Validity/connectivity metrics
      compare   RMSD, energy, and bond analysis
      xyz2mol   Convert XYZ to SMILES + fingerprints
    """
    pass


# ============================================================================
# OPTIMIZE: XTB geometry optimization
# ============================================================================

@analyze.command("optimize", context_settings=CONTEXT_SETTINGS)
@click.option("-i", "--input-dir", required=True, type=click.Path(exists=True),
              help="Input directory containing XYZ files")
@click.option("-o", "--output-dir", default=None, type=click.Path(),
              help="Output directory for optimized files (default: input_dir/optimized_xyz)")
@click.option("-c", "--charge", default=0, type=int,
              help="Molecular charge for xTB (default: 0)")
@click.option("-l", "--level", default="gfn1", type=click.Choice(["gfn1", "gfn2", "gfn-ff"]),
              help="xTB calculation level (default: gfn1)")
@click.option("-t", "--timeout", default=240, type=int,
              help="Timeout per molecule in seconds (default: 240)")
@click.option("-s", "--scale-factor", default=1.3, type=float,
              help="Scale factor for covalent radii (default: 1.3)")
@click.option("--csv", "csv_path", default=None, type=click.Path(),
              help="CSV file to filter which files to optimize")
@click.option("--filter-column", default=None, type=str,
              help="Column name in CSV to filter by (values must be 1)")
def optimize(input_dir, output_dir, charge, level, timeout, scale_factor, csv_path, filter_column):
    """Optimize XYZ geometries using xTB.
    
    \b
    Examples:
        MolCraftDiff analyze optimize -i gen_xyz/
        MolCraftDiff analyze optimize -i gen_xyz/ -o optimized/ --level gfn2
    """
    from scripts.applications.utils.xtb_optimization import get_xtb_optimized_xyz
    
    output_dir = output_dir or os.path.join(input_dir, "optimized_xyz")
    
    click.echo(f"Optimizing XYZ files from: {input_dir}")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"xTB level: {level}, charge: {charge}")
    
    optimized_files = get_xtb_optimized_xyz(
        input_directory=input_dir,
        output_directory=output_dir,
        charge=charge,
        level=level,
        timeout=timeout,
        scale_factor=scale_factor,
        csv_path=csv_path,
        filter_column=filter_column,
    )
    
    click.echo(f"\nSuccessfully optimized {len(optimized_files)} files.")


# ============================================================================
# METRICS: Validity/connectivity metrics
# ============================================================================

@analyze.command("metrics", context_settings=CONTEXT_SETTINGS)
@click.option("-i", "--input-dir", required=True, type=click.Path(exists=True),
              help="Input directory containing XYZ files")
@click.option("-o", "--output", default=None, type=click.Path(),
              help="Output CSV file for results")
@click.option("--n-subsets", default=5, type=int,
              help="Number of subsets for std calculation (default: 5)")
@click.option("--scale-factor", default=1.2, type=float,
              help="Scale factor for edge detection (default: 1.2)")
def metrics(input_dir, output, n_subsets, scale_factor):
    """Compute validity and connectivity metrics for XYZ files.
    
    \b
    Examples:
        MolCraftDiff analyze metrics -i gen_xyz/
        MolCraftDiff analyze metrics -i gen_xyz/ --n-subsets 10
    """
    import argparse
    from scripts.analyze.compute_metrics import runner
    
    args = argparse.Namespace(
        input=input_dir,
        output=output or os.path.join(input_dir, "metrics_results.csv"),
        n_subsets=n_subsets,
        scale_factor=scale_factor,
    )
    
    click.echo(f"Computing metrics for: {input_dir}")
    runner(args)


# ============================================================================
# COMPARE: Unified RMSD, energy, and bond analysis
# ============================================================================

@analyze.command("compare", context_settings=CONTEXT_SETTINGS)
@click.argument("directory", type=click.Path(exists=True))
@click.option("--bonds", is_flag=True, default=False,
              help="Include bond length/angle/torsion analysis (slower, uses xyz2mol)")
@click.option("--n-subsets", default=5, type=int,
              help="Number of subsets for std calculation (default: 5)")
@click.option("--csv", "csv_path", default=None, type=click.Path(),
              help="Output CSV filename for results")
@click.option("--charge", default=0, type=int,
              help="Molecular charge for xTB energy (default: 0)")
@click.option("--level", default="gfn2", type=click.Choice(["gfn1", "gfn2", "gfn-ff"]),
              help="xTB level for energy calculation (default: gfn2)")
@click.option("--timeout", default=120, type=int,
              help="Timeout per xTB calculation in seconds (default: 120)")
def compare(directory, bonds, n_subsets, csv_path, charge, level, timeout):
    """Compare XYZ files with their optimized counterparts.
    
    Computes RMSD and xTB energy difference by default.
    Use --bonds for additional bond length/angle/torsion analysis.
    
    Requires 'optimized_xyz' subdirectory with *_opt.xyz files.
    
    \b
    Examples:
        MolCraftDiff analyze compare gen_xyz/
        MolCraftDiff analyze compare gen_xyz/ --bonds
        MolCraftDiff analyze compare gen_xyz/ --level gfn2 --csv results.csv
    """
    from pathlib import Path
    from collections import defaultdict
    import numpy as np
    import pandas as pd
    
    directory = Path(directory).resolve()
    csv_path = Path(csv_path) if csv_path else directory / "compare_results.csv"
    
    # ==================== RMSD + ENERGY ====================
    click.echo(f"Loading XYZ pairs from: {directory}")
    
    from scripts.analyze.compute_energy_rmsd import load_pairs, compute_metrics_for_pairs, split_into_subsets
    
    pairs = load_pairs(directory)
    click.echo(f"Found {len(pairs)} XYZ pairs.\n")
    
    if len(pairs) == 0:
        click.echo("No valid pairs found.", err=True)
        return
    
    click.echo("=" * 50)
    click.echo("RMSD & ENERGY ANALYSIS")
    click.echo("=" * 50)
    
    if n_subsets <= 1:
        results = compute_metrics_for_pairs(pairs, charge=charge, level=level, timeout=timeout)
        click.echo(f"Processed {results['n_rmsd']} RMSD, {results['n_energy']} energy pairs.")
        click.echo(f"Avg RMSD: {results['avg_rmsd']:.4f} Å")
        click.echo(f"Med RMSD: {results['med_rmsd']:.4f} Å")
        click.echo(f"Avg Energy Diff: {results['avg_energy_diff']:.4f} kcal/mol")
        click.echo(f"Med Energy Diff: {results['med_energy_diff']:.4f} kcal/mol")
        
        df = pd.DataFrame(results["records"])
        df.to_csv(csv_path, index=False)
    else:
        click.echo(f"Running in {n_subsets} subsets...\n")
        subset_metrics = defaultdict(list)
        subsets = split_into_subsets(pairs, n_subsets)
        
        for subset in subsets:
            result = compute_metrics_for_pairs(subset, charge=charge, level=level, timeout=timeout)
            for key in ["avg_rmsd", "med_rmsd", "avg_energy_diff", "med_energy_diff"]:
                subset_metrics[key].append(result[key])
        
        for key, unit in [("avg_rmsd", "Å"), ("med_rmsd", "Å"), 
                          ("avg_energy_diff", "kcal/mol"), ("med_energy_diff", "kcal/mol")]:
            values = subset_metrics[key]
            click.echo(f"{key.replace('_', ' ').title()}: {np.mean(values):.4f} ± {np.std(values):.4f} {unit}")
        
        summary_data = {
            "rmsd_mean": np.mean(subset_metrics["avg_rmsd"]),
            "rmsd_std": np.std(subset_metrics["avg_rmsd"]),
            "energy_diff_mean": np.mean(subset_metrics["avg_energy_diff"]),
            "energy_diff_std": np.std(subset_metrics["avg_energy_diff"]),
        }
    
    # ==================== BOND ANALYSIS (optional) ====================
    if bonds:
        click.echo("\n" + "=" * 50)
        click.echo("BOND/ANGLE/TORSION ANALYSIS")
        click.echo("=" * 50)
        
        from scripts.analyze.compute_pair_geometry import (
            load_mol_pairs_bond_mode, run_bond_analysis, summarize_bond_results,
            run_subsets_bond_analysis
        )
        
        bond_pairs = load_mol_pairs_bond_mode(directory, timeout=10)
        click.echo(f"Loaded {len(bond_pairs)} molecule pairs for bond analysis.\n")
        
        if len(bond_pairs) > 0:
            if n_subsets <= 1:
                lengths = run_bond_analysis(bond_pairs, analysis_type="bond_length")
                angles = run_bond_analysis(bond_pairs, analysis_type="bond_angle")
                torsions = run_bond_analysis(bond_pairs, analysis_type="torsion_angle")
                
                click.echo(f"Bond Lengths: {summarize_bond_results(lengths):.4f} Å")
                click.echo(f"Bond Angles: {summarize_bond_results(angles):.4f}°")
                click.echo(f"Torsions: {summarize_bond_results(torsions):.4f}°")
            else:
                bond_scores = run_subsets_bond_analysis(bond_pairs, "bond_length", n_subsets)
                angle_scores = run_subsets_bond_analysis(bond_pairs, "bond_angle", n_subsets)
                torsion_scores = run_subsets_bond_analysis(bond_pairs, "torsion_angle", n_subsets)
                
                click.echo(f"Bond Lengths: {np.mean(bond_scores):.4f} ± {np.std(bond_scores):.4f} Å")
                click.echo(f"Bond Angles: {np.mean(angle_scores):.4f} ± {np.std(angle_scores):.4f}°")
                click.echo(f"Torsions: {np.mean(torsion_scores):.4f} ± {np.std(torsion_scores):.4f}°")
        else:
            click.echo("No valid pairs for bond analysis.", err=True)
    
    click.echo(f"\nResults saved to {csv_path}")


# ============================================================================
# XYZ2MOL: Convert XYZ to SMILES + fingerprints
# ============================================================================

@analyze.command("xyz2mol", context_settings=CONTEXT_SETTINGS)
@click.option("-x", "--xyz-dir", required=True, type=click.Path(exists=True),
              help="Directory containing XYZ files")
@click.option("-i", "--input-csv", default=None, type=click.Path(),
              help="Optional input CSV with xyz file list")
@click.option("-l", "--label", default=None, type=str,
              help="Label for processed files")
@click.option("-t", "--timeout", default=30, type=int,
              help="Timeout per conversion in seconds (default: 30)")
@click.option("--bits", default=2048, type=int,
              help="Number of bits for Morgan fingerprint (default: 2048)")
@click.option("-v", "--verbose", is_flag=True,
              help="Enable verbose output")
def xyz2mol(xyz_dir, input_csv, label, timeout, bits, verbose):
    """Convert XYZ files to SMILES and extract fingerprints/scaffolds.
    
    Outputs are saved to xyz_dir/2d_reprs/:
      - smiles_processed.csv
      - fingerprints.npy
      - scaffolds.txt
      - substructures.json
    
    \b
    Examples:
        MolCraftDiff analyze xyz2mol -x gen_xyz/
        MolCraftDiff analyze xyz2mol -x gen_xyz/ --bits 1024 -v
    """
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import json
    import logging
    
    from scripts.applications.utils.xyz2mol import (
        load_file_list_from_dir, run_processing, extract_scaffold_and_fingerprints
    )
    
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    xyz_dir = Path(xyz_dir)
    two_d_reprs_dir = xyz_dir / "2d_reprs"
    two_d_reprs_dir.mkdir(parents=True, exist_ok=True)
    
    smiles_csv_output = two_d_reprs_dir / "smiles_processed.csv"
    
    click.echo(f"Processing XYZ files from: {xyz_dir}")
    click.echo(f"Output directory: {two_d_reprs_dir}")
    
    # Load file list
    if input_csv:
        df = pd.read_csv(input_csv)
    else:
        df = load_file_list_from_dir(str(xyz_dir))
    
    # Generate SMILES
    df_smiles = run_processing(df, str(xyz_dir), label, smiles_csv_output, timeout=timeout, verbose=verbose)
    
    if df_smiles is None or 'smiles' not in df_smiles.columns or df_smiles['smiles'].isnull().all():
        click.echo("No valid SMILES generated.", err=True)
        return
    
    # Extract fingerprints and scaffolds
    click.echo("\nExtracting fingerprints and scaffolds...")
    fps, scaffolds, clean_smiles, n_fail, substruct_counts = \
        extract_scaffold_and_fingerprints(df_smiles["smiles"].dropna().values, fp_bits=bits)
    
    np.save(two_d_reprs_dir / "fingerprints.npy", fps)
    with open(two_d_reprs_dir / "scaffolds.txt", "w") as f:
        f.write("\n".join(scaffolds))
    with open(two_d_reprs_dir / "smiles_cleaned.txt", "w") as f:
        f.write("\n".join(clean_smiles))
    with open(two_d_reprs_dir / "substructures.json", "w") as f:
        json.dump(substruct_counts, f, indent=2)
    
    total = len(df_smiles["smiles"].dropna())
    click.echo(f"\n--- Summary ---")
    click.echo(f"Total SMILES: {total}")
    click.echo(f"Failed FP extraction: {n_fail}")
    click.echo(f"Unique substructures: {len(substruct_counts)}")
    click.echo(f"Outputs saved to: {two_d_reprs_dir}")
