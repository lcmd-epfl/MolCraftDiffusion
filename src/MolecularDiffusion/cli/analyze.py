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
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("--output-dir", "-o", "--o", default=None, type=click.Path(),
              help="Output directory for optimized files (default: input_dir/optimized_xyz)")
@click.option("--charge", "-c", "--c", default=0, type=int,
              help="Molecular charge for xTB (default: 0)")
@click.option("--level", "-l", "--l", default="gfn1", type=click.Choice(["gfn1", "gfn2", "gfn-ff", "mmff94"]),
              help="Optimization level (default: gfn1)")
@click.option("--timeout", "-t", "--t", default=240, type=int,
              help="Timeout per molecule in seconds (default: 240)")
@click.option("--scale-factor", "-s", "--s", default=1.3, type=float,
              help="Scale factor for covalent radii (default: 1.3)")
@click.option("--csv", "csv_path", default=None, type=click.Path(),
              help="CSV file to filter which files to optimize")
@click.option("--filter-column", default=None, type=str,
              help="Column name in CSV to filter by (values must be 1)")
def optimize(input_dir, output_dir, charge, level, timeout, scale_factor, csv_path, filter_column):
    """Optimize XYZ geometries using xTB.
    
    \b
    Examples:
        MolCraftDiff analyze optimize gen_xyz/
        MolCraftDiff analyze optimize gen_xyz/ --o optimized/ --level gfn2
    """
    from MolecularDiffusion.runmodes.analyze.xtb_optimization import get_xtb_optimized_xyz
    
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
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("-o", "--o", "--output", "--output-csv", default=None, type=click.Path(),
              help="Output CSV file for results")
@click.option("--metrics", "-m", "--m", "metrics_type", default="all",
              type=click.Choice(["all", "core", "posebuster", "geom_revised"]),
              help="Which metrics to compute (default: all)")
@click.option("--recheck-topo", is_flag=True, default=False,
              help="Recheck topology using RDKit")
@click.option("--check-strain", is_flag=True, default=False,
              help="Check strain via XTB optimization")
@click.option("--portion", "-p", "--p", default=1.0, type=float,
              help="Portion of XYZ files to process (default: 1.0 = all)")
@click.option("--mol-converter", default="cell2mol",
              type=click.Choice(["cell2mol", "openbabel"]),
              help="XYZ to mol converter (default: cell2mol)")
@click.option("--skip-atoms", multiple=True, type=int,
              help="Atom indices to skip in validation")
@click.option("--n-subsets", "-n", "--n", default=5, type=int,
              help="Number of subsets for std calculation (default: 5)")
@click.option("--timeout", "-t", "--t", default=10, type=int,
              help="Timeout per xyz2mol conversion in seconds (default: 10)")
def metrics(input_dir, output, metrics_type, recheck_topo, check_strain, portion, mol_converter, skip_atoms, n_subsets, timeout):
    """Compute validity and connectivity metrics for XYZ files.
    
    \b
    Metrics types:
      all          Run all metrics (core + posebuster + geom_revised)
      core         Basic validity checks (connectivity, atom stability)
      posebuster   PoseBusters checks (bond lengths, angles, clashes)
      geom_revised Aromatic-aware stability metrics
    
    \b
    Examples:
        MolCraftDiff analyze metrics gen_xyz/
        MolCraftDiff analyze metrics gen_xyz/ --metrics posebuster
        MolCraftDiff analyze metrics gen_xyz/ --metrics geom_revised --mol-converter openbabel
    """
    import argparse
    from MolecularDiffusion.runmodes.analyze.compute_metrics import runner
    
    args = argparse.Namespace(
        input=input_dir,
        output=output,
        metrics=metrics_type,
        recheck_topo=recheck_topo,
        check_strain=check_strain,
        portion=portion,
        mol_converter=mol_converter,
        skip_atoms=list(skip_atoms) if skip_atoms else None,
        n_subsets=n_subsets,
        timeout=timeout,
    )
    
    click.echo(f"Computing {metrics_type} metrics for: {input_dir}")
    runner(args)


# ============================================================================
# COMPARE: Unified RMSD, energy, and bond analysis
# ============================================================================

@analyze.command("compare", context_settings=CONTEXT_SETTINGS)
@click.argument("directory", type=click.Path(exists=True))
@click.option("--mol-converter", default="openbabel", type=click.Choice(["openbabel", "cell2mol"]),
              help="Converter for bond perception (default: openbabel)")
@click.option("--n-subsets", "-n", "--n", default=5, type=int,
              help="Number of subsets for std calculation (default: 5)")
@click.option("--output", "-o", "--o", "--csv", "csv_path", default=None, type=click.Path(),
              help="Output CSV filename for results")
@click.option("--charge", "-c", "--c", default=0, type=int,
              help="Molecular charge for xTB energy (default: 0)")
@click.option("--level", "-l", "--l", default="gfn2", type=click.Choice(["gfn1", "gfn2", "gfn-ff", "mmff94"]),
              help="xTB level for energy calculation (default: gfn2)")
@click.option("--timeout", "-t", "--t", default=120, type=int,
              help="Timeout per xTB calculation in seconds (default: 120)")
def compare(directory, mol_converter, n_subsets, csv_path, charge, level, timeout):
    """Compare XYZ files with their optimized counterparts.
    
    Computes RMSD, xTB Energy Difference, and Bond Geometry Metrics.
    Enforces strict connectivity checks.
    
    Requires 'optimized_xyz' subdirectory with *_opt.xyz files.
    """
    import argparse
    from MolecularDiffusion.runmodes.analyze.compare_to_optimized import run_compare_analysis
    
    # Construct args namespace to pass to run_compare_analysis
    args = argparse.Namespace(
        directory=directory,
        mol_converter=mol_converter,
        n_subsets=n_subsets,
        csv_path=csv_path,
        charge=charge,
        level=level,
        timeout=timeout
    )
    
    run_compare_analysis(args)


# ============================================================================
# XYZ2MOL: Convert XYZ to SMILES + fingerprints
# ============================================================================

@analyze.command("xyz2mol", context_settings=CONTEXT_SETTINGS)
@click.argument("xyz_dir", type=click.Path(exists=True))
@click.option("--input-csv", "-i", "--i", default=None, type=click.Path(),
              help="Optional input CSV with xyz file list")
@click.option("--label", "-l", "--l", default=None, type=str,
              help="Label for processed files")
@click.option("--timeout", "-t", "--t", default=30, type=int,
              help="Timeout per conversion in seconds (default: 30)")
@click.option("--bits", "-b", "--b", default=2048, type=int,
              help="Number of bits for Morgan fingerprint (default: 2048)")
@click.option("--verbose", "-v", "--v", is_flag=True,
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
        MolCraftDiff analyze xyz2mol gen_xyz/
        MolCraftDiff analyze xyz2mol gen_xyz/ --bits 1024 -v
    """
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import json
    import logging
    
    from MolecularDiffusion.runmodes.analyze.xyz2mol import (
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
