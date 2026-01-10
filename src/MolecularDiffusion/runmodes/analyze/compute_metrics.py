import glob
import os
import torch
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from MolecularDiffusion.utils.geom_utils import read_xyz_file, create_pyg_graph, correct_edges
from MolecularDiffusion.utils.geom_metrics import (check_validity_v1, 
                                                   check_chem_validity, 
                                                   run_postbuster, 
                                                   smilify_wrapper, 
                                                   load_molecules_from_xyz,
                                                   check_neutrality)
from MolecularDiffusion.utils import smilify_cell2mol, smilify_openbabel
from MolecularDiffusion.utils.geom_stability import compute_molecules_stability

import logging
from rdkit import RDLogger
import matplotlib.pyplot as plt

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Constants
EDGE_THRESHOLD = 4
SCALE_FACTOR = 1.2
SCORES_THRESHOLD = 3.0

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def runner(args):
    
    xyz_dir = args.input
    recheck_topo = args.recheck_topo
    # check_strain = args.check_strain # Kept as a separate flag
    # check_postbuster = args.check_postbuster # Removed, controlled by --metrics
    skip_idx = args.skip_atoms
    
    if skip_idx is None:
        skip_idx = []

    xyzs = [
    path for path in glob.glob(f"{xyz_dir}/*.xyz")
    if 'opt' not in os.path.basename(path)
]
    
    if args.portion < 1.0:
        random.shuffle(xyzs)
        xyzs = xyzs[:int(len(xyzs) * args.portion)]

    df_res_dict = {
        "file": [],
        "percent_atom_valid": [],
        "valid": [],
        "valid_connected": [],
        "num_graphs": [],
        "bad_atom_distort": [],
        "bad_atom_chem": [],
        "neutral_molecule": [],
        "smiles": [],
        "num_atoms": [],
    }
    
    if args.metrics in ["all", "core"]:
        for xyz in tqdm(xyzs, desc="Processing XYZ files", total=len(xyzs)):
            num_atoms = 0
            try:
                cartesian_coordinates_tensor, atomic_numbers_tensor = read_xyz_file(xyz)
                num_atoms = cartesian_coordinates_tensor.size(0)
                data = create_pyg_graph(cartesian_coordinates_tensor, 
                                            atomic_numbers_tensor,
                                            xyz_filename=xyz,
                                            r=EDGE_THRESHOLD)
                data = correct_edges(data, scale_factor=SCALE_FACTOR)           
                (is_valid, percent_atom_valid, num_components, bad_atom_chem, bad_atom_distort) = \
                    check_validity_v1(data, score_threshold=SCORES_THRESHOLD, 
                                      skip_indices=skip_idx,
                                      verbose=False)

            except Exception as e:
                logging.error(f"Error processing {xyz}: {e}")
                is_valid = False
                percent_atom_valid = 0
                num_components = 100
                bad_atom_chem = torch.arange(0, data.num_nodes) if 'data' in locals() and hasattr(data, 'num_nodes') else []
                bad_atom_distort = torch.arange(0, data.num_nodes) if 'data' in locals() and hasattr(data, 'num_nodes') else []
    
            try:
                smiles_list, mol_list = smilify_openbabel(xyz)
            except:
                # logging.warning(f"fail to convert xyz to mol with openbabel, retry with cell2mol")
                mol_list = None
            
            to_recheck = recheck_topo and (len(bad_atom_distort) > 0) and (len(bad_atom_chem) == 0)
            neutral_mol = check_neutrality(xyz)
            if mol_list is None and num_components < 3:
                xyz2mol_fn = smilify_cell2mol
                try:
                    _, smiles_list, mol_list, _ = smilify_wrapper([xyz], xyz2mol_fn)
                    mol_list = mol_list[0]
                except Exception as e:
                    # logging.warning(f"fail to convert xyz to mol with v0, skip and assign invalid")
                    to_recheck = False
                    is_valid = False   

            if to_recheck:
                try:
                    (natom_stability_dicts,
                        _,
                        _,
                        _,
                        bad_smiles_chem) = check_chem_validity([mol_list], skip_idx=skip_idx)
                    natom_stable = sum(natom_stability_dicts.values())
                    percent_atom_valid = natom_stable/cartesian_coordinates_tensor.size(0)
        
                    if len(bad_smiles_chem) == 0:
                    
                        is_valid = True
                    else:
                        logging.warning("Detect bad smiles in ", xyz, bad_smiles_chem)
                except Exception as e:
                    logging.error(f"Fail to check on {xyz} due to {e}, asssign invalid")
                    is_valid = False
                    percent_atom_valid = 0
                    
            if is_valid and num_components == 1:
                is_valid_connected = True   
            else:
                is_valid_connected = False         
                
            df_res_dict["smiles"].append(smiles_list[0] if len(smiles_list) == 1 else smiles_list)
            df_res_dict["file"].append(xyz)
            df_res_dict["percent_atom_valid"].append(percent_atom_valid)
            df_res_dict["valid"].append(is_valid)
            df_res_dict["valid_connected"].append(is_valid_connected)
            df_res_dict["neutral_molecule"].append(neutral_mol)
            df_res_dict["num_graphs"].append(num_components)
            df_res_dict["bad_atom_distort"].append(bad_atom_distort)
            df_res_dict["bad_atom_chem"].append(bad_atom_chem)
            df_res_dict["num_atoms"].append(num_atoms)

        df = pd.DataFrame(df_res_dict)
        df = df.sort_values(by="file")
        fully_connected = [1 if num == 1 else 0 for num in df_res_dict["num_graphs"]]

        logging.info(f"{df['percent_atom_valid'].mean() * 100:.2f}% of atoms are stable")
        logging.info(f"{df['valid'].mean() * 100:.2f}% of 3D molecules are valid")
        logging.info(f"{df['valid_connected'].mean() * 100:.2f}% of 3D molecules are valid and fully-connected")
        logging.info(f"{sum(fully_connected) / len(fully_connected) * 100:.2f}% of 3D molecules are fully connected")
        
        logging.info(f"Molecular size mean: {df['num_atoms'].mean():.2f}")
        logging.info(f"Molecular size max: {df['num_atoms'].max()}")
        logging.info(f"Molecular size std: {df['num_atoms'].std():.2f}")

        if args.check_strain: # Only run check_strain if core metrics are computed, as it relies on 'df'
            rmsd_mean = df["rmsd"].dropna().mean()
            delta_energy_mean = df["delta_energy"].dropna().mean()
            intact_topology = [1 if top else 0 for top in df_res_dict["same_topology"] if not pd.isna(top)]
            logging.info(f"RMSD mean: {rmsd_mean:.2f}")
            logging.info(f"Delta Energy mean: {delta_energy_mean:.2f}")
            logging.info(f"{sum(intact_topology) / len(intact_topology) * 100:.2f}% of 3D molecules have intact topology after the optimization")
        
        if args.output is None:
            output_path = f"{xyz_dir}/output_metrics.csv"
            hist_path = f"{xyz_dir}/molecular_size_histogram.png"
        else:
            output_path = args.output
            base, _ = os.path.splitext(output_path)
            hist_path = f"{base}_molecular_size_histogram.png"

        plt.figure()
        plt.hist(df['num_atoms'], bins='auto')
        plt.title('Histogram of Molecular Sizes')
        plt.xlabel('Number of Atoms')
        plt.ylabel('Frequency')
        plt.savefig(hist_path)
        plt.close()
        logging.info(f"Molecular size histogram saved to {hist_path}")

        df.to_csv(output_path, index=False)

    if args.metrics in ["all", "posebuster"]:
        mols, xyz_passed = load_molecules_from_xyz(xyz_dir)
        
        if args.portion < 1.0:
             xyzs_set = set(xyzs)
             filtered_data = [(m, x) for m, x in zip(mols, xyz_passed) if x in xyzs_set]
             if filtered_data:
                 mols, xyz_passed = zip(*filtered_data)
                 mols = list(mols)
                 xyz_passed = list(xyz_passed)
             else:
                 mols, xyz_passed = [], []
        
        neutral_mols = []
        for xyz in tqdm(xyz_passed, desc="Checking neutrality of molecules", total=len(xyz_passed)):
            neutral_mols.append(check_neutrality(xyz))
      
        postbuster_results = run_postbuster(mols, timeout=3000)
        if postbuster_results is not None:
            num_atoms_list = [mol.GetNumAtoms() for mol in mols]
            postbuster_results['num_atoms'] = num_atoms_list
            
            posebuster_checks = [
                'bond_lengths', 'bond_angles', 'internal_steric_clash',
                'aromatic_ring_flatness', 'non-aromatic_ring_non-flatness',
                'double_bond_flatness', 'internal_energy'
            ]
            postbuster_results['valid_posebuster'] = postbuster_results[posebuster_checks].all(axis=1)
            posebuster_checks_connected = posebuster_checks + ['all_atoms_connected']
            postbuster_results['valid_posebuster_connected'] = postbuster_results[posebuster_checks_connected].all(axis=1)
            if args.output is None:
                postbuster_output_path = f"{xyz_dir}/postbuster_metrics.csv"
                hist_path = f"{xyz_dir}/postbuster_molecular_size_histogram.png"
            else:
                base, ext = os.path.splitext(args.output)
                postbuster_output_path = f"{base}_postbuster{ext}"
                hist_path = f"{base}_postbuster_molecular_size_histogram.png"

            postbuster_results['neutral_molecule'] = neutral_mols
            postbuster_results["filename"] = [os.path.basename(xyz) for xyz in xyz_passed]
            postbuster_results.to_csv(postbuster_output_path, index=False)

            logging.info(f"Molecular size mean: {postbuster_results['num_atoms'].mean():.2f}")
            logging.info(f"Molecular size max: {postbuster_results['num_atoms'].max()}")
            logging.info(f"Molecular size std: {postbuster_results['num_atoms'].std():.2f}")

            plt.figure()
            plt.hist(postbuster_results['num_atoms'], bins='auto')
            plt.title('Histogram of Molecular Sizes (Posebuster)')
            plt.xlabel('Number of Atoms')
            plt.ylabel('Frequency')
            plt.savefig(hist_path)
            plt.close()
            logging.info(f"Molecular size histogram for posebuster saved to {hist_path}")

            logging.info(f"Sanitization: {postbuster_results['sanitization'].mean() * 100:.2f}%")
            logging.info(f"InChI Convertible: {postbuster_results['inchi_convertible'].mean() * 100:.2f}%")
            logging.info(f"All Atoms Connected: {postbuster_results['all_atoms_connected'].mean() * 100:.2f}%")
            logging.info(f"Bond Lengths: {postbuster_results['bond_lengths'].mean():.2f}")
            logging.info(f"Bond Angles: {postbuster_results['bond_angles'].mean():.2f}")
            logging.info(f"Internal Steric Clash: {postbuster_results['internal_steric_clash'].mean():.2f}")
            logging.info(f"Aromatic Ring Flatness: {postbuster_results['aromatic_ring_flatness'].mean():.2f}")
            logging.info(f"Non-Aromatic Ring Non-Flatness: {postbuster_results['non-aromatic_ring_non-flatness'].mean():.2f}")
            logging.info(f"Double Bond Flatness: {postbuster_results['double_bond_flatness'].mean():.2f}")
            logging.info(f"Internal Energy: {postbuster_results['internal_energy'].mean():.2f}")
            logging.info(f"Valid Posebuster: {postbuster_results['valid_posebuster'].mean() * 100:.2f}%")
            logging.info(f"Valid Posebuster Connected: {postbuster_results['valid_posebuster_connected'].mean() * 100:.2f}%")
            logging.info(f"Neutral Molecule: {sum(neutral_mols) / len(neutral_mols) * 100:.2f}%")

    # =========================================================================
    # GEOM_REVISED: Aromatic-aware molecule stability
    # =========================================================================
    if args.metrics in ["all", "geom_revised"]:
        from rdkit import Chem
        from MolecularDiffusion.utils.smilify import smilify_cell2mol
        
        logging.info(f"Computing geom_revised metrics (converter: {args.mol_converter})...")
        
        # Load molecules from XYZ
        mols_revised = []
        xyz_files_revised = []
        
        xyzs_to_process = [
            path for path in glob.glob(f"{xyz_dir}/*.xyz")
            if 'opt' not in os.path.basename(path)
        ]
        
        if args.portion < 1.0:
            random.shuffle(xyzs_to_process)
            xyzs_to_process = xyzs_to_process[:int(len(xyzs_to_process) * args.portion)]
        
        for xyz_file in tqdm(xyzs_to_process, desc="Loading molecules for geom_revised"):
            try:
                if args.mol_converter == "cell2mol":
                    smiles, mol = smilify_cell2mol(xyz_file, timeout=args.timeout)
                else:  # openbabel
                    from openbabel import pybel
                    mol_pb = next(pybel.readfile("xyz", xyz_file))
                    mol_sdf = mol_pb.write("sdf")
                    mol = Chem.MolFromMolBlock(mol_sdf, removeHs=False, sanitize=False)
                    if mol is not None:
                        Chem.SanitizeMol(mol)
                
                if mol is not None:
                    mols_revised.append(mol)
                    xyz_files_revised.append(xyz_file)
            except Exception as e:
                logging.debug(f"Failed to load {xyz_file}: {e}")
        
        if len(mols_revised) == 0:
            logging.warning("No molecules loaded for geom_revised metrics")
        else:
            # Compute validity metrics first (independent of aromatic mode)
            from rdkit import Chem
            
            valid_list = []  # Sanitization only
            valid_connected_list = []  # Sanitization + single fragment
            
            for mol in mols_revised:
                try:
                    Chem.SanitizeMol(mol)
                    valid_list.append(1)
                    # Check single connected component
                    if len(Chem.GetMolFrags(mol)) == 1:
                        valid_connected_list.append(1)
                    else:
                        valid_connected_list.append(0)
                except:
                    valid_list.append(0)
                    valid_connected_list.append(0)
            
            # Compute stability for both aromatic modes
            # aromatic_true = MS Arom-Dependent Valence (tuple valencies)
            # aromatic_false = MS 1.5 Arom (sum all bonds including aromatic at 1.5)
            results_dict = {
                "file": xyz_files_revised,
                "num_atoms": [mol.GetNumAtoms() for mol in mols_revised],
                "valid": valid_list,
                "valid_connected": valid_connected_list,
            }
            
            modes_to_run = [
                ("aromatic_true", True),   # Arom-Dependent Valence
                ("aromatic_false", False), # 1.5 Arom mode
            ]
            
            for mode_name, aromatic_val in modes_to_run:
                try:
                    validity, stability, n_stable_atoms, n_atoms = compute_molecules_stability(
                        mols_revised, aromatic=aromatic_val
                    )
                    results_dict[f"stable_mol_{mode_name}"] = stability.tolist()
                    results_dict[f"n_stable_atoms_{mode_name}"] = n_stable_atoms.tolist()
                    results_dict[f"n_atoms_{mode_name}"] = n_atoms.tolist()
                    results_dict[f"atom_stability_{mode_name}"] = (n_stable_atoms / n_atoms).tolist()
                except Exception as e:
                    logging.error(f"Failed to compute stability for {mode_name}: {e}")
            
            df_revised = pd.DataFrame(results_dict)
            
            # Determine output path
            if args.output is None:
                revised_output_path = f"{xyz_dir}/geom_revised_metrics.csv"
            else:
                base, ext = os.path.splitext(args.output)
                revised_output_path = f"{base}_geom_revised{ext}"
            
            df_revised.to_csv(revised_output_path, index=False)
            logging.info(f"Geom revised metrics saved to {revised_output_path}")
            
            # Print summary statistics
            n_passed = len(mols_revised)
            n_total = len(xyzs_to_process)
            conversion_rate = n_passed / n_total * 100 if n_total > 0 else 0
            
            logging.info("=" * 60)
            logging.info("GEOM_REVISED STABILITY METRICS")
            logging.info("=" * 60)
            logging.info(f"XYZ2Mol Conversion: {n_passed}/{n_total} ({conversion_rate:.2f}%)")
            
            # Helper to compute split statistics
            def get_split_stats(data_list, n_splits):
                if len(data_list) == 0: return 0.0, 0.0
                if n_splits <= 1: return np.mean(data_list) * 100, 0.0
                splits = np.array_split(data_list, n_splits)
                split_means = [np.mean(s) * 100 for s in splits if len(s) > 0]
                return np.mean(split_means), np.std(split_means)

            valid_mean, valid_std = get_split_stats(valid_list, args.n_subsets)
            conn_mean, conn_std = get_split_stats(valid_connected_list, args.n_subsets)

            logging.info(f"Valid: {sum(valid_list)}/{n_passed} ({valid_mean:.2f} ± {valid_std:.2f}%)")
            logging.info(f"Valid & Connected: {sum(valid_connected_list)}/{n_passed} ({conn_mean:.2f} ± {conn_std:.2f}%)")
            
            for mode_name in ["aromatic_true", "aromatic_false"]:
                if f"stable_mol_{mode_name}" in df_revised.columns:
                    n_stable = int(df_revised[f'stable_mol_{mode_name}'].sum())
                    
                    mol_stab_mean, mol_stab_std = get_split_stats(df_revised[f'stable_mol_{mode_name}'].values, args.n_subsets)
                    atom_stab_mean, atom_stab_std = get_split_stats(df_revised[f'atom_stability_{mode_name}'].values, args.n_subsets)
                    
                    logging.info(f"--- Mode: {mode_name} ---")
                    logging.info(f"  Molecule Stability: {n_stable}/{n_passed} ({mol_stab_mean:.2f} ± {mol_stab_std:.2f}%)")
                    logging.info(f"  Atom Stability: {atom_stab_mean:.2f} ± {atom_stab_std:.2f}%")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="input directory with xyz files")
    parser.add_argument("-o", "--output", type=str, default=None, help="output csv file")
    parser.add_argument("--recheck_topo", action="store_true", help="recheck topology")
    parser.add_argument("--check_strain", action="store_true", help="check strain")
    parser.add_argument("--metrics", type=str, default="all",
                        choices=["all", "core", "posebuster", "geom_revised"],
                        help="Specify which metrics to compute: 'all', 'core', 'posebuster', or 'geom_revised'.")
    parser.add_argument("--skip_atoms", type=int, nargs="+", default=None, help="skip atoms")
    parser.add_argument("--portion", type=float, default=1.0, help="portion of xyz files to process")
    parser.add_argument("--mol_converter", type=str, default="cell2mol",
                        choices=["cell2mol", "openbabel"],
                        help="Molecule converter for XYZ to mol: 'cell2mol' (default) or 'openbabel'")
    parser.add_argument("--n_subsets", type=int, default=5, help="number of subsets for std calculation")
    parser.add_argument("--timeout", type=int, default=10, help="timeout for xyz2mol conversion (default: 10s)")
    args = parser.parse_args()
    runner(args)
