"""Analysis utilities for 3D molecular structures.

This module provides tools for:
- XTB geometry optimization
- Validity/connectivity metrics
- Energy and RMSD computation
- Bond/angle/torsion analysis
- XYZ to SMILES conversion
"""

from MolecularDiffusion.runmodes.analyze.xtb_optimization import (
    get_xtb_optimized_xyz,
    check_xyz,
    check_neutrality,
    xyz2mol_xtb,
)

from MolecularDiffusion.runmodes.analyze.compute_energy_rmsd import (
    load_pairs,
    compute_metrics_for_pairs,
    split_into_subsets,
    get_xtb_energy,
    compute_coord_rmsd,
)

from MolecularDiffusion.runmodes.analyze.compute_pair_geometry import (
    load_mol_pairs_bond_mode,
    load_coord_pairs_geometry_mode,
    run_bond_analysis,
    run_geometry_analysis,
    summarize_bond_results,
    run_subsets_bond_analysis,
    run_subsets_geometry_analysis,
)

from MolecularDiffusion.runmodes.analyze.xyz2mol import (
    load_file_list_from_dir,
    run_processing,
    extract_scaffold_and_fingerprints,
    sanitize_smiles,
)

from MolecularDiffusion.runmodes.analyze.compute_metrics import runner as run_metrics

__all__ = [
    # XTB optimization
    "get_xtb_optimized_xyz",
    "check_xyz",
    "check_neutrality",
    "xyz2mol_xtb",
    # Energy/RMSD
    "load_pairs",
    "compute_metrics_for_pairs",
    "split_into_subsets",
    "get_xtb_energy",
    "compute_coord_rmsd",
    # Pair geometry
    "load_mol_pairs_bond_mode",
    "load_coord_pairs_geometry_mode",
    "run_bond_analysis",
    "run_geometry_analysis",
    "summarize_bond_results",
    "run_subsets_bond_analysis",
    "run_subsets_geometry_analysis",
    # xyz2mol
    "load_file_list_from_dir",
    "run_processing",
    "extract_scaffold_and_fingerprints",
    "sanitize_smiles",
    # Metrics
    "run_metrics",
]
