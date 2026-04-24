

"""
ASE database operations module.
Handles merging, inspecting, splitting, and sampling.
"""

import logging
import random
from pathlib import Path
from tqdm import tqdm
from typing import List

from ase.db import connect

# RDKit optional
try:
    from rdkit import Chem
except ImportError:
    Chem = None

logger = logging.getLogger(__name__)

# --- Merge Logic ---

def verify_datapoint(atoms, mol_block):
    """
    Verifies that ASE Atoms match RDKit Mol block.
    """
    if not mol_block: return False
    
    if isinstance(mol_block, bytes):
        mol_block = mol_block.decode('utf-8')

    if Chem is None: return True # Skip verification if RDKit missing
        
    mol = Chem.MolFromMolBlock(mol_block, removeHs=False, sanitize=False)
    if not mol: return False
    
    ase_sym = atoms.get_chemical_symbols()
    rd_sym = [a.GetSymbol() for a in mol.GetAtoms()]
    
    return ase_sym == rd_sym

def merge_dbs(
    input_dir: Path,
    output_db: Path,
    recursive: bool = False,
    pattern: str = "*.db"
):
    """
    Merges multiple ASE databases into one.
    """
    input_dir = Path(input_dir)
    output_db = Path(output_db)
    
    if recursive:
        files = sorted(list(input_dir.rglob(pattern)))
    else:
        files = sorted(list(input_dir.glob(pattern)))
        
    target_db = connect(str(output_db))
    # Track unique IDs
    seen_ids = {row.data.get("unique_id") for row in target_db.select() if row.data.get("unique_id")}
    
    merged_count = 0
    skipped_count = 0
    
    for db_file in tqdm(files, desc="Merging DBs"):
        try:
            source_db = connect(str(db_file))
            for row in source_db.select():
                uid = row.data.get("unique_id")
                # Resolve duplicates or missing IDs
                if not uid:
                    uid = f"merged_{row.id}"
                
                original_uid = uid
                copy_count = 0
                while uid in seen_ids:
                    copy_count += 1
                    uid = f"{original_uid}_copy_{copy_count}"
                
                if copy_count > 0:
                     logger.info(f"Resolved duplicate ID {original_uid} -> {uid}")
                     
                # Update the row data with the new unique_id
                if uid != row.data.get("unique_id"):
                    row.data["unique_id"] = uid
                
                # Verify
                atoms = row.toatoms()
                mol_block = row.data.get("mol_block")
                if verify_datapoint(atoms, mol_block):
                    data = row.data.copy()
                    data['source_db'] = str(db_file)
                    # Ensure no KVPs leak into write via kwargs
                    target_db.write(atoms, data=data)
                    seen_ids.add(uid)
                    merged_count += 1
                else:
                    logger.warning(f"Verification failed for {uid} in {db_file}")

        except Exception as e:
            logger.error(f"Failed to read {db_file}: {e}")
            
    logger.info(f"Merged {merged_count} entries. Skipped {skipped_count}.")


# --- Sample Logic ---

def is_clean(row):
    """
    Verifies that the atom order in ASE atoms and RDKit mol from mol_block are identical.
    """
    try:
        atoms = row.toatoms()
        mol_block = row.data.get('mol_block')

        if not mol_block:
            return False

        if isinstance(mol_block, bytes):
            mol_block = mol_block.decode('utf-8')

        if Chem is None: return True

        mol = Chem.MolFromMolBlock(mol_block, removeHs=False)

        if not mol:
            return False

        ase_symbols = atoms.get_chemical_symbols()
        rdkit_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

        return ase_symbols == rdkit_symbols
    except Exception:
        return False

def sample_db(
    input_db: Path,
    output: Path,
    output_type: str = 'db',  # 'db', 'xyz', or 'npy'
    fraction: float = None,
    number: int = None,
    seed: int = None,
    verify_clean: bool = False
):
    """
    Samples a random fraction or number of entries from an ASE database.

    output_type:
        'db'  – write to an ASE SQLite database (default)
        'xyz' – write one XYZ file per molecule into the output directory
        'npy' – write positions.npy (M,N,3), numbers.npy (M,N), and
                natoms.npy (M,) arrays into the output directory, where
                M is the number of sampled entries and N is padded to the
                maximum atom count in the sample.
    """
    import random
    import numpy as np
    from ase.io import write as ase_write

    VALID_TYPES = ('db', 'xyz', 'npy')
    if output_type not in VALID_TYPES:
        raise ValueError(f"output_type must be one of {VALID_TYPES}, got '{output_type}'")

    if seed is not None:
        random.seed(seed)

    source_db = connect(str(input_db))
    num_total = len(source_db)

    if num_total == 0:
        logger.warning("Source database is empty.")
        return

    # Determine num_to_sample
    if fraction is not None:
        num_to_sample = int(num_total * fraction)
    elif number is not None:
        num_to_sample = number
    else:
        raise ValueError("Either fraction or number must be provided.")

    if num_to_sample > num_total:
        logger.warning(f"Requested {num_to_sample} but DB only has {num_total}. Sampling all.")
        num_to_sample = num_total

    output_path = Path(output)

    all_ids = [row.id for row in source_db.select()]
    random.shuffle(all_ids)

    # Prepare output destinations
    output_db = None
    written_ids: set = set()

    if output_type == 'db':
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_db = connect(str(output_path))
        written_ids = {
            row.data.get("unique_id")
            for row in output_db.select()
            if "unique_id" in row.data
        }
    else:  # xyz or npy – both write into a directory
        output_path.mkdir(parents=True, exist_ok=True)

    # For npy we collect arrays and write at the end
    npy_positions: list = []
    npy_numbers: list = []
    npy_natoms: list = []

    num_written = 0
    with tqdm(total=num_to_sample, desc="Sampling") as pbar:
        for row_id in all_ids:
            if num_written >= num_to_sample:
                break

            row = source_db.get(id=row_id)
            uid = row.data.get("unique_id")

            if uid and uid in written_ids:
                continue

            if verify_clean and not is_clean(row):
                continue

            try:
                atoms = row.toatoms()

                if output_type == 'db':
                    output_db.write(atoms, data=row.data)
                    if uid:
                        written_ids.add(uid)

                elif output_type == 'xyz':
                    safe_uid = str(uid).replace(':', '_').replace('/', '_') if uid else f'row_{row.id}'
                    ase_write(output_path / f"{safe_uid}.xyz", atoms)

                elif output_type == 'npy':
                    npy_positions.append(atoms.get_positions())
                    npy_numbers.append(atoms.get_atomic_numbers())
                    npy_natoms.append(len(atoms))

                num_written += 1
                pbar.update(1)
            except Exception as e:
                logger.error(f"Failed to write row {row_id}: {e}")

    # Finalise npy output with zero-padding to maximum atom count
    if output_type == 'npy' and npy_natoms:
        max_n = max(npy_natoms)
        pos_arr = np.zeros((num_written, max_n, 3), dtype=np.float32)
        num_arr = np.zeros((num_written, max_n), dtype=np.int32)
        nat_arr = np.array(npy_natoms, dtype=np.int32)
        for i, (pos, nums) in enumerate(zip(npy_positions, npy_numbers)):
            n = len(nums)
            pos_arr[i, :n] = pos
            num_arr[i, :n] = nums
        np.save(output_path / 'positions.npy', pos_arr)
        np.save(output_path / 'numbers.npy', num_arr)
        np.save(output_path / 'natoms.npy', nat_arr)
        logger.info(
            f"Saved npy arrays: positions {pos_arr.shape}, "
            f"numbers {num_arr.shape}, natoms {nat_arr.shape}"
        )

    logger.info(f"Sampled {num_written} entries.")


# --- Inspection Logic ---

def inspect_db(
    db_path: Path, 
    output_dir: Path = None, 
    keys_to_plot: List[str] = None, 
    sample_size: int = 5000,
    limit_print: int = 10,
    check_nan: bool = False,
    nan_key: str = None,
    discard_nan: bool = False,
    detect_outliers: bool = False,
    outlier_threshold: float = 3.0,
    discard_outliers: bool = False,
    outlier_key: str = None,
    clean_db_path: Path = None
):
    """
    Inspects an ASE DB, printing stats and optionally plotting distributions.
    Allows identifying NaNs and outliers, and optionally saving a cleaned DB.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict, Counter
    from MolecularDiffusion.utils import plot_function as pf

    db = connect(str(db_path))
    n_rows = len(db)
    logger.info(f"Total entries: {n_rows}")

    if n_rows == 0: return

    # Simple print inspection
    keys = set()
    sample_ids = random.sample(range(1, n_rows + 1), min(n_rows, limit_print))
    for rid in sample_ids:
        row = db.get(rid)
        keys.update(row.key_value_pairs.keys())
        keys.update(row.data.keys())
    logger.info(f"Available Keys (sampled {min(n_rows, limit_print)}): {keys}")

    # Discovery logic if keys_to_plot is empty
    if not keys_to_plot and (output_dir or detect_outliers or (check_nan and not nan_key) or (discard_outliers and not outlier_key)):
        sample_ids = random.sample(range(1, n_rows + 1), min(n_rows, sample_size))
        discovered = set()
        for rid in sample_ids:
            row = db.get(rid)
            for k, v in row.data.items():
                if isinstance(v, (int, float, np.number)): discovered.add(k)
            for k, v in row.key_value_pairs.items():
                if isinstance(v, (int, float, np.number)): discovered.add(k)
            if hasattr(row, 'natoms'): discovered.add('natoms')
        keys_to_plot = sorted(list(discovered))
        logger.info(f"Discovered numeric keys: {keys_to_plot}")
    elif keys_to_plot:
        # Process specified keys (handle strings/commas/tuples from CLI)
        if not isinstance(keys_to_plot, (list, tuple)):
            keys_to_plot = [keys_to_plot]
            
        final_keys = []
        for k in keys_to_plot:
            if not isinstance(k, str):
                if isinstance(k, (list, tuple)):
                    final_keys.extend([str(ki).strip("'\"()[] ") for ki in k])
                else:
                    final_keys.append(k)
                continue
            
            k = k.strip("'\"()[] ")
            if ',' in k:
                final_keys.extend([ki.strip("'\"()[] ") for ki in k.split(',')])
            elif k:
                final_keys.append(k)
        keys_to_plot = final_keys

    # Data Collection
    ids_to_fetch = random.sample(range(1, n_rows + 1), min(n_rows, sample_size))
    
    data_map = defaultdict(list)
    row_id_map = defaultdict(list) # Track which row had which value
    atom_counts = Counter()
    
    nan_rows = []
    keys_to_check = keys_to_plot or []
    
    for rid in tqdm(ids_to_fetch, desc="Collecting Stats"):
        row = db.get(rid)
        atoms = row.toatoms()
        atom_counts.update(atoms.get_chemical_symbols())
        
        row_has_nan = False
        
        # Determine value for nan_key check
        if check_nan and nan_key:
            val = None
            if hasattr(row, nan_key): val = getattr(row, nan_key)
            elif nan_key in row.data: val = row.data[nan_key]
            elif nan_key in row.key_value_pairs: val = row.key_value_pairs[nan_key]
            
            if val is None or (isinstance(val, (float, np.number)) and np.isnan(val)):
                row_has_nan = True
        
        for k in keys_to_check:
            val = None
            if k == 'natoms': val = len(atoms)
            elif k == 'mol_weight': val = sum(atoms.get_masses())
            elif k == 'num_heteroatoms': val = sum(1 for sym in atoms.get_chemical_symbols() if sym not in ['C', 'H'])
            elif hasattr(row, k): val = getattr(row, k)
            elif k in row.data: val = row.data[k]
            elif k in row.key_value_pairs: val = row.key_value_pairs[k]
            
            if val is not None and isinstance(val, (int, float, np.number)):
                if np.isnan(val):
                    row_has_nan = True
                data_map[k].append(val)
                row_id_map[k].append(rid)
        
        if row_has_nan:
            nan_rows.append(rid)

    if check_nan:
        logger.info(f"--- NaN Detection ---")
        logger.info(f"Rows with NaNs: {len(nan_rows)}")
        if nan_rows:
            logger.info(f"Sample NaN Row IDs: {nan_rows[:limit_print]}")

    # Outlier Detection
    outlier_map = defaultdict(list)
    all_outlier_ids = set()
    if detect_outliers or discard_outliers:
        logger.info(f"--- Outlier Detection (Z-score threshold={outlier_threshold}) ---")
        for k, vals in data_map.items():
            vals = np.array(vals)
            finite_mask = np.isfinite(vals)
            f_vals = vals[finite_mask]
            f_ids = np.array(row_id_map[k])[finite_mask]
            
            if len(f_vals) > 0:
                mean = np.mean(f_vals)
                std = np.std(f_vals)
                if std > 0:
                    z_scores = np.abs((f_vals - mean) / std)
                    outlier_indices = np.where(z_scores > outlier_threshold)[0]
                    for idx in outlier_indices:
                        oid = f_ids[idx]
                        oval = f_vals[idx]
                        outlier_map[k].append((oid, oval))
                        # If a specific key is provided for outlier discard, only track that
                        if not outlier_key or k == outlier_key:
                            all_outlier_ids.add(oid)
                    
                    if len(outlier_indices) > 0:
                        logger.info(f"Key '{k}': Found {len(outlier_indices)} outliers.")
                        sample_outliers = outlier_map[k][:5]
                        logger.info(f"  Sample (ID, Val): {sample_outliers}")

    # Discard / Clean DB logic
    if (discard_nan or discard_outliers) and clean_db_path:
        clean_db_path = Path(clean_db_path)
        logger.info(f"Saving cleaned database to {clean_db_path}...")
        
        # Calculate stats for outliers if we need to check ALL rows during cleaning
        # Note: outliers are currently only estimated from IDs to fetch.
        # If discard_outliers is True, we might need stats from the sample to apply to ALL rows.
        outlier_stats = {}
        if discard_outliers:
            for k, vals in data_map.items():
                vals = np.array(vals)
                finite_vals = vals[np.isfinite(vals)]
                if len(finite_vals) > 0:
                    outlier_stats[k] = (np.mean(finite_vals), np.std(finite_vals))

        clean_db = connect(str(clean_db_path))
        
        discard_count = 0
        with clean_db:
            for row in tqdm(db.select(), total=n_rows, desc="Cleaning DB"):
                discard = False
                
                # Check NaNs
                if discard_nan:
                    row_has_nan = False
                    if nan_key:
                        val = row.data.get(nan_key) or row.key_value_pairs.get(nan_key)
                        if val is None or (isinstance(val, (float, np.number)) and np.isnan(val)):
                            row_has_nan = True
                    else:
                        for k in (keys_to_check or []):
                            val = None
                            if k == 'natoms': val = len(row.toatoms())
                            elif hasattr(row, k): val = getattr(row, k)
                            elif k in row.data: val = row.data[k]
                            elif k in row.key_value_pairs: val = row.key_value_pairs[k]
                            if val is not None and isinstance(val, (float, np.number)) and np.isnan(val):
                                row_has_nan = True; break
                    if row_has_nan: discard = True
                
                # Check Outliers
                if not discard and discard_outliers:
                    row_has_outlier = False
                    keys_to_verify = [outlier_key] if outlier_key else (keys_to_check or [])
                    for k in keys_to_verify:
                        if k not in outlier_stats: continue
                        mean, std = outlier_stats[k]
                        if std == 0: continue
                        
                        val = None
                        if k == 'natoms': val = len(row.toatoms())
                        elif hasattr(row, k): val = getattr(row, k)
                        elif k in row.data: val = row.data[k]
                        elif k in row.key_value_pairs: val = row.key_value_pairs[k]
                        
                        if val is not None and isinstance(val, (int, float, np.number)) and not np.isnan(val):
                            if np.abs((val - mean) / std) > outlier_threshold:
                                row_has_outlier = True; break
                    if row_has_outlier: discard = True

                if not discard:
                    clean_db.write(row.toatoms(), key_value_pairs=row.key_value_pairs, data=row.data)
                else:
                    discard_count += 1
        logger.info(f"Cleaned DB saved at {clean_db_path}. Discarded {discard_count} rows.")

    # Plotting & Stats (filtered if discard_nan is True)
    if not output_dir:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Atom stats
    if atom_counts:
        total_atoms = sum(atom_counts.values())
        logger.info("--- Atom Type Statistics ---")
        for sym, count in atom_counts.most_common():
            logger.info(f"{sym}: {count} ({count/total_atoms*100:.2f}%)")
            
        try:
            symbols, counts = zip(*atom_counts.most_common())
            plt.figure(figsize=(10, 6))
            plt.bar(symbols, counts)
            plt.xlabel("Atom Type")
            plt.ylabel("Count")
            plt.title("Atom Type Distribution")
            plt.savefig(output_dir / "hist_atom_types.png")
            plt.close()
        except Exception as e:
            logger.error(f"Failed to plot atom types: {e}")

    for k, vals in data_map.items():
        vals = np.array(vals)
        # Filter NaNs for plotting/stats unless we want to see them (usually we don't)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0: continue
        
        try:
            pf.plot_histogram_distribution(vals, task_name=k, output_path=str(output_dir / f"hist_{k}.png"))
            pf.plot_kde_distribution(vals, task_name=k, output_path=str(output_dir / f"kde_{k}.png"))
        except Exception as e:
            logger.error(f"Failed to plot {k}: {e}")

    # Print Stats Table
    logger.info("--- Statistics ---")
    for k, vals in data_map.items():
        vals = np.array(vals)
        finite_vals = vals[np.isfinite(vals)]
        if len(finite_vals) > 0:
             logger.info(f"{k}: Mean={np.mean(finite_vals):.4f}, Std={np.std(finite_vals):.4f}, Min={np.min(finite_vals)}, Max={np.max(finite_vals)}")



# --- Split Logic ---

def split_db(db_path: Path, output_dir: Path, n_splits: int = 2):
    """
    Splits a DB into N smaller DBs.
    """
    db = connect(str(db_path))
    ids = [row.id for row in db.select()]
    # Simple split logic
    chunk_size = len(ids) // n_splits + 1
    
    output_dir.mkdir(exist_ok=True)
    
    for i in range(n_splits):
        chunk = ids[i*chunk_size : (i+1)*chunk_size]
        new_db_path = output_dir / f"split_{i}.db"
        new_db = connect(str(new_db_path))
        
        for row_id in tqdm(chunk, desc=f"Split {i}"):
            row = db.get(row_id)
            new_db.write(row.toatoms(), key_value_pairs=row.key_value_pairs, data=row.data)


# --- Rename Logic ---

def rename_db_attribute(db_path: Path, old_name: str, new_name: str):
    """
    Renames a data attribute for all rows in an ASE database.
    """
    db = connect(str(db_path))
    rows = list(db.select())
    
    renamed_count = 0
    with db:
        for row in tqdm(rows, desc=f"Renaming '{old_name}' to '{new_name}'"):
            update_data = {}
            renamed = False
            
            # Check in .data or .key_value_pairs
            kvp = row.key_value_pairs.copy()
            data = row.data.copy()
            
            if old_name in data:
                data[new_name] = data.pop(old_name)
                renamed = True
            
            if old_name in kvp:
                # Move from KVP to data
                val = kvp.pop(old_name)
                data[new_name] = val
                renamed = True
                
            if renamed:
                try:
                    # Update row: remove old KVP and update data
                    db.delete([row.id])
                    db.write(row.toatoms(), key_value_pairs=kvp, data=data)
                    renamed_count += 1
                except Exception as e:
                    logger.error(f"Failed to rename row {row.id}: {e}")

    logger.info(f"Successfully renamed '{old_name}' to '{new_name}' in {renamed_count}/{len(rows)} rows.")
