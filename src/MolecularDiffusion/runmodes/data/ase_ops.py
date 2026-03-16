

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
    limit_print: int = 10
):
    """
    Inspects an ASE DB, printing stats and optionally plotting distributions.
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
        # Also check data
        keys.update(row.data.keys())
    logger.info(f"Available Keys (sampled {min(n_rows, limit_print)}): {keys}")

    if not output_dir:
        return

    # Full inspection with plots
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Determine keys to plot
    if not keys_to_plot:
        # Discovery
        sample_ids = random.sample(range(1, n_rows + 1), min(n_rows, sample_size))
        discovered = set()
        for rid in sample_ids:
            row = db.get(rid)
            for k, v in row.data.items():
                if isinstance(v, (int, float, np.number)): discovered.add(k)
            for k, v in row.key_value_pairs.items():
                if isinstance(v, (int, float, np.number)): discovered.add(k)
            # Check attributes
            if hasattr(row, 'natoms'): discovered.add('natoms')
            # ... check other attrs ...
        keys_to_plot = sorted(list(discovered))
        logger.info(f"Discovered numeric keys: {keys_to_plot}")
    else:
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
            
            # String handling: strip wrapping junk and split by comma
            k = k.strip("'\"()[] ")
            if ',' in k:
                final_keys.extend([ki.strip("'\"()[] ") for ki in k.split(',')])
            elif k:
                final_keys.append(k)
                
        keys_to_plot = final_keys
            
    # Always include mandatory keys
    for k in ['natoms', 'mol_weight', 'num_heteroatoms']:
        if k not in keys_to_plot:
            keys_to_plot.append(k)
            logger.info(f"Added mandatory '{k}' to keys to plot.")

    # Data Collection
    ids_to_fetch = random.sample(range(1, n_rows + 1), min(n_rows, sample_size))
    logger.info(f"Final keys to plot: {keys_to_plot}")

    data_map = defaultdict(list)
    atom_counts = Counter()
    
    for rid in tqdm(ids_to_fetch, desc="Collecting Stats"):
        row = db.get(rid)
        atoms = row.toatoms()
        atom_counts.update(atoms.get_chemical_symbols())
        
        for k in keys_to_plot:
            val = None
            if k == 'natoms': val = len(atoms)
            elif k == 'mol_weight': val = sum(atoms.get_masses())
            elif k == 'num_heteroatoms': val = sum(1 for sym in atoms.get_chemical_symbols() if sym not in ['C', 'H'])
            elif hasattr(row, k): val = getattr(row, k)
            elif k in row.data: val = row.data[k]
            elif k in row.key_value_pairs: val = row.key_value_pairs[k]
            
            if val is not None and isinstance(val, (int, float, np.number)):
                data_map[k].append(val)
                
    # Plotting
    if atom_counts:
        # Print atom type statistics
        total_atoms = sum(atom_counts.values())
        logger.info("--- Atom Type Statistics ---")
        for sym, count in atom_counts.most_common():
            logger.info(f"{sym}: {count} ({count/total_atoms*100:.2f}%)")
            
        # Plot atom types
        if output_dir:
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
        if len(vals) > 0:
             logger.info(f"{k}: Mean={np.mean(vals):.4f}, Std={np.std(vals):.4f}, Min={np.min(vals)}, Max={np.max(vals)}")



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
