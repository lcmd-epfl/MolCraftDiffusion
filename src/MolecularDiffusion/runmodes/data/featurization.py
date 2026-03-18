
"""
Featurization module for MolCraft.
Handles generating vectorial representations (Morgan fingerprints, SOAP) from 3D molecular data.
"""

import os
import sys
import logging
import random
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

from ase.io import read as ase_read
from ase.db import connect

# Internal imports
from MolecularDiffusion.runmodes.data import preparation as prep

logger = logging.getLogger(__name__)

# --- Helper: RDKit Wrapper ---
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    Chem = None
    AllChem = None

# --- Helper: Safetensors ---
try:
    from safetensors.numpy import save_file as save_safetensors
except ImportError:
    save_safetensors = None


def _get_morgan_fingerprint(mol, radius=2, nbits=2048):
    """Generates Morgan fingerprint as numpy array."""
    if mol is None:
        return np.zeros(nbits, dtype=np.float32) # Zero padding for failed molecules
    
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        arr = np.zeros((1,), dtype=np.int8)
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(fp, arr)
        return arr.astype(np.float32)
    except Exception as e:
        logger.warning(f"Morgan generation failed: {e}")
        return np.zeros(nbits, dtype=np.float32)


def generate_morgan(
    entries: List[Dict], 
    radius: int = 2, 
    nbits: int = 2048, 
    smilify_method: str = 'hybrid',
    n_jobs: int = 1
) -> Tuple[np.ndarray, List[str]]:
    """
    Generates Morgan fingerprints for a list of entries.
    Entries is a list of dicts with 'file' (path) or 'data' (atoms/coords) and 'id'.
    """
    if Chem is None:
        raise ImportError("RDKit is required for Morgan fingerprints.")
        
    features = []
    valid_ids = []
    
    for entry in tqdm(entries, desc="Generating Morgan Fingerprints"):
        uid = entry['id']
        mol = None
        
        # Get RDKit Mol
        try:
            if 'file' in entry:
                # From file (XYZ)
                _, mol = prep.smilify_structure(entry['file'], method=smilify_method, timeout=30)
            elif 'atoms' in entry:
                # From ASE Atoms (DB)
                # Need to write to temp file for smilify_structure or use specialized logic?
                # smilify_structure takes a filename. 
                # prep.smilify_xyz2mol can take atomic numbers/coords but it's internal.
                # Let's write a temp file to be safe and reuse the robust hybrid logic.
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.xyz', mode='w', delete=False) as tmp:
                    from ase.io import write
                    write(tmp.name, entry['atoms'])
                    tmp_path = tmp.name
                
                try:
                    _, mol = prep.smilify_structure(tmp_path, method=smilify_method, timeout=30)
                finally:
                    os.remove(tmp_path)
            
            fp = _get_morgan_fingerprint(mol, radius, nbits)
            features.append(fp)
            valid_ids.append(uid)
            
        except Exception as e:
            logger.warning(f"Failed to process {uid}: {e}")
            features.append(np.zeros(nbits, dtype=np.float32)) # Keep in sync or skip?
            valid_ids.append(uid) # We keep it to maintain alignment, just 0-feature
            
    return np.vstack(features), valid_ids


def generate_soap(
    entries: List[Dict],
    rcut: float = 5.0,
    nmax: int = 8,
    lmax: int = 6,
    readout: str = 'mean',
    species: List[str] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Generates SOAP descriptors.
    """
    try:
        from dscribe.descriptors import SOAP
    except ImportError:
        raise ImportError("dscribe is required for SOAP descriptors. Install with `pip install dscribe`.")

    if species is None:
        species = ["H", "B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl", "As", "Se", "Br", "I"]

    soap = SOAP(
        species=species,
        r_cut=rcut,
        n_max=nmax,
        l_max=lmax,
        sparse=False,
        average='inner' if readout == 'mean' else 'off' # dscribe has 'inner', 'outer', 'off'. 'inner' is average over center atoms? 
        # Actually dscribe 2.x changed parameters.
        # Let's stick to per-atom generation and manual averaging/summing to be safe and support 'sum'.
    )
    # Re-init with average='off' to get per-atom
    soap = SOAP(
        species=species,
        r_cut=rcut,
        n_max=nmax,
        l_max=lmax,
        sparse=False,
        average='off' 
    )

    features = []
    valid_ids = []

    for entry in tqdm(entries, desc=f"Generating SOAP ({readout})"):
        uid = entry['id']
        atoms = None
        
        try:
            if 'file' in entry:
                atoms = ase_read(entry['file'])
            elif 'atoms' in entry:
                atoms = entry['atoms']
            
            if atoms is None: continue
            
            # dscribe needs pure ASE atoms
            # Check for H only (optional, but good practice)
            # if all(a.symbol == "H" for a in atoms): features.append(...); continue
            
            desc = soap.create(atoms, n_jobs=1) # (n_atoms, n_features)
            
            if readout == 'mean':
                feat = desc.mean(axis=0)
            elif readout == 'sum':
                feat = desc.sum(axis=0)
            else:
                raise ValueError(f"Unknown readout: {readout}")
                
            features.append(feat)
            valid_ids.append(uid)
            
        except Exception as e:
            logger.warning(f"SOAP failed for {uid}: {e}")
            # Skip or padding? Let's skip for SOAP as dimension is unknown until runtime (depends on species) 
            # actually dimension is deterministic based on params.
            # But skipping is safer/easier for now.
            continue
            
    if not features:
        return np.empty((0, 0)), []
        
    return np.vstack(features), valid_ids


def save_features(features: np.ndarray, ids: List[str], output_dir: Path, format: str = 'npy'):
    """Saves features and indices."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Indices
    pd.DataFrame({'id': ids}).to_csv(output_dir / "indices.csv", index=False)
    
    # Save Features
    if format == 'npy':
        np.save(output_dir / "features.npy", features)
        logger.info(f"Saved features to {output_dir / 'features.npy'} {features.shape}")
        
    elif format == 'safetensors':
        if save_safetensors is None:
            raise ImportError("safetensors not installed.")
        # Safetensors requires dict of tensors
        save_safetensors({ "features": features }, output_dir / "features.safetensors")
        logger.info(f"Saved features to {output_dir / 'features.safetensors'} {features.shape}")
        
    else:
        raise ValueError(f"Unknown format: {format}")


def featurize(
    input_source: str,
    output_dir: str,
    method: str = 'morgan',
    format: str = 'npy',
    readout: str = 'mean',
    smilify_method: str = 'hybrid',
    # Params
    radius: int = 2,
    nbits: int = 2048,
    rcut: float = 5.0,
    nmax: int = 8,
    lmax: int = 6
):
    """
    Main featurization driver.
    """
    input_path = Path(input_source)
    output_path = Path(output_dir)
    
    # 1. Collect Entries
    entries = []
    if input_path.is_dir():
        # XYZ Directory
        files = sorted(list(input_path.glob("*.xyz")))
        for f in files:
            entries.append({'id': f.name, 'file': str(f)})
            
    elif input_path.suffix == '.db':
        # ASE Database
        db = connect(str(input_path))
        for row in db.select():
            entries.append({'id': row.id, 'atoms': row.toatoms()})
            
    else:
        raise ValueError(f"Unsupported input: {input_path}")
        
    logger.info(f"Collected {len(entries)} entries from {input_path}")
    
    # 2. Generate Features
    if method == 'morgan':
        feats, ids = generate_morgan(entries, radius=radius, nbits=nbits, smilify_method=smilify_method)
    elif method == 'soap':
        feats, ids = generate_soap(entries, rcut=rcut, nmax=nmax, lmax=lmax, readout=readout)
    else:
        raise ValueError(f"Unknown method: {method}")
        
    # 3. Save
    save_features(feats, ids, output_path, format=format)
