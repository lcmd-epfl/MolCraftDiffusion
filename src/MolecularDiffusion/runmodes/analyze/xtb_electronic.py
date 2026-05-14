"""XTB Electronic Property Computation using morfeus.

Provides functions to compute electronic properties for XYZ files using
the morfeus XTB interface, with support for batch processing and multiple
output formats (CSV, JSON, ASE database).
"""

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)
HARTREE_TO_EV = 27.211386245988

# Property group definitions (based on actual morfeus XTB API)
# Available methods: get_homo, get_lumo, get_charges, get_dipole, get_ip, get_ea,
#                    get_fukui, get_global_descriptor, get_bond_order(s)
PROPERTY_GROUPS = {
    "energy": ["homo", "lumo", "homo_lumo_gap"],  # Compute gap from homo/lumo
    "dipole": ["dipole"],
    "reactivity": ["ip", "ea"],
    "global": ["electrophilicity", "nucleophilicity", "electrofugality", "nucleofugality"],
    "charges": ["charges"],  # Atomic-level
    "fukui": ["fukui_plus", "fukui_minus", "fukui_radical", "fukui_dual"],  # Atomic-level
    "bond_orders": ["bond_orders"],  # Atomic-level
}

ATOMIC_LEVEL_GROUPS = {"charges", "fukui", "bond_orders"}
MOLECULAR_LEVEL_GROUPS = {"energy", "dipole", "reactivity", "global"}

# Note: Solvation requires morfeus-ml[xtb] with xtb binary installed
# and is version-dependent, so we omit it for now


def _check_morfeus_available():
    """Check if morfeus and xtb are available."""
    try:
        from morfeus import XTB, read_xyz
        return True
    except ImportError:
        logger.error("morfeus is not installed. Install with: pip install morfeus-ml")
        return False


def compute_xtb_electronic(
    xyz_path: str,
    method: int | str = 2,
    charge: int = 0,
    n_unpaired: int = 0,
    solvent: str | None = None,
    properties: list[str] | None = None,
    corrected: bool = True,
) -> dict[str, Any]:
    """Compute XTB electronic properties for a single XYZ file.
    
    Args:
        xyz_path: Path to XYZ file
        method: XTB method (1=GFN1, 2=GFN2, "ptb"=PTB)
        charge: Molecular charge
        n_unpaired: Number of unpaired electrons
        solvent: Solvent name for implicit solvation (e.g., "water", "thf")
        properties: List of property groups to compute. If None, computes "energy"
        corrected: Whether to apply empirical IP/EA correction
        
    Returns:
        Dictionary containing computed properties and metadata
    """
    from morfeus import XTB, read_xyz
    
    # Parse property groups
    if properties is None:
        properties = ["energy"]
    
    requested_props = set()
    for group in properties:
        if group == "all":
            for g in PROPERTY_GROUPS:
                requested_props.update(PROPERTY_GROUPS[g])
        elif group in PROPERTY_GROUPS:
            requested_props.update(PROPERTY_GROUPS[group])
    
    # Read XYZ file
    elements, coordinates = read_xyz(xyz_path)
    
    # Morfeus XTB selects GFN-xTB through `version`, not `method`.
    xtb_version = str(method)
    if xtb_version == "ptb":
        raise ValueError("The morfeus XTB backend supports only GFN1 and GFN2 for electronic properties.")

    xtb = XTB(
        elements=elements,
        coordinates=coordinates,
        version=xtb_version,
        charge=charge,
        n_unpaired=n_unpaired,
        solvent=solvent,
    )
    
    result = {
        "filename": os.path.basename(xyz_path),
        "n_atoms": len(elements),
        "method": method,
        "charge": charge,
        "solvent": solvent,
    }
    
    # Compute molecular-level properties
    try:
        # Energy properties are returned by Morfeus/XTB in Hartree.
        if "homo" in requested_props:
            homo = xtb.get_homo()
            result["homo"] = homo
            result["homo_ev"] = homo * HARTREE_TO_EV
        
        if "lumo" in requested_props:
            lumo = xtb.get_lumo()
            result["lumo"] = lumo
            result["lumo_ev"] = lumo * HARTREE_TO_EV
        
        if "homo_lumo_gap" in requested_props:
            homo = xtb.get_homo()
            lumo = xtb.get_lumo()
            gap = lumo - homo
            result["homo_lumo_gap"] = gap
            result["homo_lumo_gap_ev"] = gap * HARTREE_TO_EV
        
        # Dipole (in atomic units)
        if "dipole" in requested_props:
            dipole_vec = xtb.get_dipole()
            result["dipole"] = dipole_vec.tolist()
            result["dipole_magnitude"] = float(np.linalg.norm(dipole_vec))
        
        # Reactivity descriptors (IP and EA in eV with optional correction)
        if "ip" in requested_props:
            result["ip"] = xtb.get_ip(corrected=corrected)
        
        if "ea" in requested_props:
            result["ea"] = xtb.get_ea(corrected=corrected)
        
        # Global descriptors
        for desc in ["electrophilicity", "nucleophilicity", "electrofugality", "nucleofugality"]:
            if desc in requested_props:
                result[desc] = xtb.get_global_descriptor(variety=desc, corrected=corrected)
        
        # Atomic-level properties
        if "charges" in requested_props:
            result["charges"] = xtb.get_charges()
        
        if "bond_orders" in requested_props:
            # get_bond_orders() returns NxN numpy matrix
            # Store as sparse representation with bond order > 0.1
            bo_matrix = xtb.get_bond_orders()
            bond_orders = {}
            n_atoms = bo_matrix.shape[0]
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    if bo_matrix[i, j] > 0.1:
                        bond_orders[f"{i+1}-{j+1}"] = float(bo_matrix[i, j])
            result["bond_orders"] = bond_orders
        
        # Fukui indices
        for fukui_type in ["plus", "minus", "radical", "dual"]:
            prop_name = f"fukui_{fukui_type}"
            if prop_name in requested_props:
                variety = {"plus": "electrophilicity", "minus": "nucleophilicity", 
                          "radical": "radical", "dual": "dual"}[fukui_type]
                result[prop_name] = xtb.get_fukui(variety=variety)
        
        result["success"] = True
        
    except Exception as e:
        logger.warning(f"Error computing properties for {xyz_path}: {e}")
        result["success"] = False
        result["error"] = str(e)
    
    return result


def _compute_single_wrapper(args):
    """Wrapper for multiprocessing."""
    return compute_xtb_electronic(*args)


def batch_xtb_electronic(
    input_dir: str,
    output_path: str | None = None,
    output_format: str = "csv",
    method: int | str = 2,
    charge: int = 0,
    n_unpaired: int = 0,
    solvent: str | None = None,
    properties: list[str] | None = None,
    corrected: bool = True,
    timeout: int = 120,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Batch compute XTB electronic properties for XYZ files in a directory.
    
    Args:
        input_dir: Directory containing XYZ files
        output_path: Output file path (without extension for 'all' format)
        output_format: Output format ("csv", "json", "ase", or "all")
        method: XTB method (1=GFN1, 2=GFN2, "ptb"=PTB)
        charge: Molecular charge
        n_unpaired: Number of unpaired electrons
        solvent: Solvent name for implicit solvation
        properties: List of property groups to compute
        corrected: Whether to apply empirical IP/EA correction
        timeout: Timeout per molecule in seconds
        n_jobs: Number of parallel jobs
        
    Returns:
        DataFrame with results
    """
    if not _check_morfeus_available():
        raise RuntimeError("morfeus is not available")
    
    # Get XYZ files
    input_path = Path(input_dir)
    xyz_files = sorted(input_path.glob("*.xyz"))
    
    if not xyz_files:
        logger.warning(f"No XYZ files found in {input_dir}")
        return pd.DataFrame()
    
    logger.info(f"Processing {len(xyz_files)} XYZ files...")
    
    # Compute properties
    results = []
    
    if n_jobs == 1:
        # Sequential processing
        for xyz_file in tqdm(xyz_files, desc="Computing XTB properties"):
            try:
                result = compute_xtb_electronic(
                    str(xyz_file),
                    method=method,
                    charge=charge,
                    n_unpaired=n_unpaired,
                    solvent=solvent,
                    properties=properties,
                    corrected=corrected,
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process {xyz_file}: {e}")
                results.append({"filename": xyz_file.name, "success": False, "error": str(e)})
    else:
        # Parallel processing
        args_list = [
            (str(xyz_file), method, charge, n_unpaired, solvent, properties, corrected)
            for xyz_file in xyz_files
        ]
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_compute_single_wrapper, args): args[0] for args in args_list}
            for future in tqdm(futures, desc="Computing XTB properties"):
                try:
                    result = future.result(timeout=timeout)
                    results.append(result)
                except FutureTimeoutError:
                    fname = Path(futures[future]).name
                    logger.warning(f"Timeout for {fname}")
                    results.append({"filename": fname, "success": False, "error": "timeout"})
                except Exception as e:
                    fname = Path(futures[future]).name
                    logger.warning(f"Error for {fname}: {e}")
                    results.append({"filename": fname, "success": False, "error": str(e)})
    
    # Check which property groups have atomic-level data
    has_atomic = any(p in ATOMIC_LEVEL_GROUPS for p in (properties or ["energy"]))
    if "all" in (properties or []):
        has_atomic = True
    
    # Create outputs
    if output_path:
        _save_outputs(results, output_path, output_format, has_atomic, input_dir)
    
    # Create molecular-level DataFrame
    df = _results_to_dataframe(results)
    
    return df


def _results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert results to DataFrame, excluding atomic-level data."""
    atomic_keys = {"charges", "bond_orders", "fukui_plus", "fukui_minus", 
                   "fukui_radical", "fukui_dual", "dipole_vector"}
    
    df_rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k not in atomic_keys}
        df_rows.append(row)
    
    return pd.DataFrame(df_rows)


def _save_outputs(results: list[dict], output_path: str, output_format: str, 
                  has_atomic: bool, input_dir: str):
    """Save results in specified format(s)."""
    output_path = Path(output_path)
    base_path = output_path.with_suffix("")
    
    formats = [output_format] if output_format != "all" else ["csv", "json", "ase"]
    
    for fmt in formats:
        if fmt == "csv":
            csv_path = base_path.with_suffix(".csv")
            df = _results_to_dataframe(results)
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV to {csv_path}")
        
        elif fmt == "json":
            json_path = base_path.with_suffix(".json")
            # Convert numpy arrays to lists for JSON serialization
            json_results = []
            for r in results:
                jr = {}
                for k, v in r.items():
                    if isinstance(v, np.ndarray):
                        jr[k] = v.tolist()
                    elif isinstance(v, dict):
                        # Convert tuple keys to strings
                        jr[k] = {str(kk): vv for kk, vv in v.items()}
                    else:
                        jr[k] = v
                json_results.append(jr)
            with open(json_path, "w") as f:
                json.dump(json_results, f, indent=2)
            logger.info(f"Saved JSON to {json_path}")
        
        elif fmt == "ase":
            db_path = base_path.with_suffix(".db")
            _save_to_ase_db(results, db_path, input_dir)
            logger.info(f"Saved ASE database to {db_path}")


def _save_to_ase_db(results: list[dict], db_path: Path, input_dir: str):
    """Save results to ASE database with properties in atoms.info and arrays."""
    from ase import Atoms
    from ase.db import connect
    from ase.io import read
    
    # Remove if exists
    if db_path.exists():
        db_path.unlink()
    
    db = connect(str(db_path))
    input_path = Path(input_dir)
    
    for result in results:
        filename = result.get("filename")
        if not filename or not result.get("success", False):
            continue
        
        xyz_file = input_path / filename
        if not xyz_file.exists():
            continue
        
        try:
            atoms = read(str(xyz_file))
        except Exception as e:
            logger.warning(f"Failed to read {xyz_file} for ASE db: {e}")
            continue
        
        # Separate molecular-level and atomic-level properties
        mol_props = {}
        for key, value in result.items():
            if key in {"filename", "success", "error", "n_atoms"}:
                continue
            
            if key in {"charges", "fukui_plus", "fukui_minus", "fukui_radical", "fukui_dual"}:
                # These are dict[int, float] - convert to arrays
                if isinstance(value, dict):
                    arr = np.zeros(len(atoms))
                    for idx, val in value.items():
                        arr[int(idx) - 1] = val  # 1-indexed to 0-indexed
                    atoms.arrays[key] = arr
            elif key == "bond_orders":
                # Store as JSON string in info
                mol_props[key] = json.dumps({str(k): v for k, v in value.items()})
            elif key == "dipole_vector":
                mol_props[key] = json.dumps(value)
            elif isinstance(value, (int, float, str, bool)):
                mol_props[key] = value
        
        # Write to database
        db.write(atoms, data=mol_props)
