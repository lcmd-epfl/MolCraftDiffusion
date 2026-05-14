import glob
import os
import torch
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from MolecularDiffusion.optional import OptionalDependencyError, optional_import_error, require_modules

_IMPORT_ERROR = None
try:
    from MolecularDiffusion.utils.geom_utils import read_xyz_file, create_pyg_graph, correct_edges
    from MolecularDiffusion.utils.geom_metrics import (check_validity_v1,
                                                       check_chem_validity,
                                                       run_postbuster,
                                                       smilify_wrapper,
                                                       load_molecules_from_xyz,
                                                       check_neutrality,
                                                       xyz_to_rdkit_mol,
                                                       compute_drug_likeness)
    from MolecularDiffusion.utils.shepherd_score.extract_profiles import get_electrostatic_potential
    from MolecularDiffusion.utils.shepherd_score.generate_point_cloud import (
        get_molecular_surface, get_atomic_vdw_radii)
    from MolecularDiffusion.utils.shepherd_score.pharm_utils.pharmacophore import get_pharmacophores
    from MolecularDiffusion.utils.shepherd_score.score.constants import ALPHA, LAM_SCALING
    from MolecularDiffusion.utils.shepherd_score.score.gaussian_overlap_np import get_overlap_np
    from MolecularDiffusion.utils.shepherd_score.score.electrostatic_scoring_np import get_overlap_esp_np
    from MolecularDiffusion.utils.shepherd_score.score.pharmacophore_scoring_np import get_overlap_pharm_np
    from MolecularDiffusion.utils.shepherd_score.alignment import (
        optimize_ROCS_overlay, optimize_ROCS_esp_overlay, optimize_pharm_overlay)
    from MolecularDiffusion.utils import smilify_xyz2mol, smilify_openbabel
    from MolecularDiffusion.utils.geom_stability import compute_molecules_stability
except ImportError as exc:
    _IMPORT_ERROR = optional_import_error("analyze", exc)

import logging
import matplotlib.pyplot as plt
try:
    from rdkit import RDLogger
except ImportError:
    RDLogger = None

# Suppress RDKit warnings
if RDLogger is not None:
    RDLogger.DisableLog('rdApp.*')

# Constants
EDGE_THRESHOLD = 4
SCALE_FACTOR = 1.2
SCORES_THRESHOLD = 3.0

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _get_split_stats(data_list, n_splits, scale=100.0):
    if len(data_list) == 0:
        return 0.0, 0.0
    if n_splits <= 1:
        return float(np.mean(data_list) * scale), 0.0
    splits = np.array_split(np.asarray(data_list), n_splits)
    split_means = [float(np.mean(s) * scale) for s in splits if len(s) > 0]
    if not split_means:
        return 0.0, 0.0
    return float(np.mean(split_means)), float(np.std(split_means))

def runner(args):
    split = getattr(args, "split", 1)
    if split < 1:
        raise ValueError(f"--split must be >= 1, got {split}")
    if _IMPORT_ERROR is not None:
        logging.warning(str(_IMPORT_ERROR))
        raise SystemExit(1) from _IMPORT_ERROR
    required_modules = {"torch", "torch_geometric", "ase", "rdkit"}
    if args.metrics in ["all", "core", "geom_revised"]:
        required_modules.add("cosymlib")
    if args.metrics in ["all", "posebuster", "geom_revised"]:
        required_modules.update({"openbabel", "posebusters"})
    if args.metrics in ["all", "shepherd"]:
        required_modules.add("open3d")
    try:
        require_modules("analyze", required_modules)
    except OptionalDependencyError as exc:
        logging.warning(str(exc))
        raise SystemExit(1) from exc
    
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
                # logging.warning(f"fail to convert xyz to mol with openbabel, retry with xyz2mol")
                mol_list = None
            
            to_recheck = recheck_topo and (len(bad_atom_distort) > 0) and (len(bad_atom_chem) == 0)
            neutral_mol = check_neutrality(xyz)
            if mol_list is None and num_components < 3:
                xyz2mol_fn = smilify_xyz2mol
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
        if split > 1:
            atom_mean, atom_std = _get_split_stats(df["percent_atom_valid"].values, split, scale=100.0)
            valid_mean, valid_std = _get_split_stats(df["valid"].values, split, scale=100.0)
            conn_mean, conn_std = _get_split_stats(df["valid_connected"].values, split, scale=100.0)
            full_mean, full_std = _get_split_stats(fully_connected, split, scale=100.0)
            logging.info(f"Split atoms stable mean ± std: {atom_mean:.2f} ± {atom_std:.2f}%")
            logging.info(f"Split valid mean ± std: {valid_mean:.2f} ± {valid_std:.2f}%")
            logging.info(f"Split valid&connected mean ± std: {conn_mean:.2f} ± {conn_std:.2f}%")
            logging.info(f"Split fully connected mean ± std: {full_mean:.2f} ± {full_std:.2f}%")
        
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
            if split > 1:
                sanit_mean, sanit_std = _get_split_stats(postbuster_results["sanitization"].values, split, scale=100.0)
                inchi_mean, inchi_std = _get_split_stats(postbuster_results["inchi_convertible"].values, split, scale=100.0)
                conn_mean, conn_std = _get_split_stats(postbuster_results["all_atoms_connected"].values, split, scale=100.0)
                valid_pb_mean, valid_pb_std = _get_split_stats(postbuster_results["valid_posebuster"].values, split, scale=100.0)
                valid_pb_conn_mean, valid_pb_conn_std = _get_split_stats(postbuster_results["valid_posebuster_connected"].values, split, scale=100.0)
                neutral_mean, neutral_std = _get_split_stats(neutral_mols, split, scale=100.0)
                logging.info(f"Split Sanitization mean ± std: {sanit_mean:.2f} ± {sanit_std:.2f}%")
                logging.info(f"Split InChI Convertible mean ± std: {inchi_mean:.2f} ± {inchi_std:.2f}%")
                logging.info(f"Split All Atoms Connected mean ± std: {conn_mean:.2f} ± {conn_std:.2f}%")
                logging.info(f"Split Valid Posebuster mean ± std: {valid_pb_mean:.2f} ± {valid_pb_std:.2f}%")
                logging.info(f"Split Valid Posebuster Connected mean ± std: {valid_pb_conn_mean:.2f} ± {valid_pb_conn_std:.2f}%")
                logging.info(f"Split Neutral Molecule mean ± std: {neutral_mean:.2f} ± {neutral_std:.2f}%")

    # =========================================================================
    # GEOM_REVISED: Aromatic-aware molecule stability
    # =========================================================================
    if args.metrics in ["all", "geom_revised"]:
        from rdkit import Chem
        from MolecularDiffusion.utils.smilify import smilify_xyz2mol
        from MolecularDiffusion.utils.geom_stability import (
            compute_bond_lengths_diff,
            compute_bond_angles_diff,
            compute_torsion_angles_diff,
            compute_differences,
        )
        from collections import defaultdict

        def run_bond_analysis(pairs, analysis_type="bond_length"):
            accumulated_results = defaultdict(lambda: ([], []))
            if analysis_type == "bond_length":
                results = compute_differences(pairs, compute_bond_lengths_diff)
            elif analysis_type == "bond_angle":
                results = compute_differences(pairs, compute_bond_angles_diff)
            elif analysis_type == "torsion_angle":
                results = compute_differences(pairs, compute_torsion_angles_diff)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
                
            for key, (avg_diff, std_dev, weight) in results.items():
                accumulated_results[key][0].append(avg_diff)
                accumulated_results[key][1].append(weight)
            return accumulated_results

        def summarize_bond_results(results):
            total_weighted_diffs = []
            total_weight = 0
            for key, (avg_diff_list, weight_list) in results.items():
                weighted_diffs = np.array(avg_diff_list) * np.array(weight_list)
                total_weighted_diffs.append(np.sum(weighted_diffs))
                total_weight += np.sum(weight_list)
            return np.sum(total_weighted_diffs) / total_weight if total_weight > 0 else 0

        def get_split_stats_bond(pairs, analysis_type, n_splits):
            if len(pairs) == 0: return 0.0, 0.0
            if n_splits <= 1:
                results = run_bond_analysis(pairs, analysis_type=analysis_type)
                return summarize_bond_results(results), 0.0
            fold_size = len(pairs) // n_splits
            scores = []
            for i in range(n_splits):
                if i < n_splits - 1:
                    fold_pairs = pairs[i * fold_size: (i + 1) * fold_size]
                else:
                    fold_pairs = pairs[i * fold_size:]
                if not fold_pairs: continue
                results = run_bond_analysis(fold_pairs, analysis_type=analysis_type)
                score = summarize_bond_results(results)
                scores.append(score)
            if not scores: return 0.0, 0.0
            return np.mean(scores), np.std(scores)
        
        logging.info(f"Computing geom_revised metrics (converter: {args.mol_converter})...")
        
        # Load molecules from XYZ
        mols_revised = []
        xyz_files_revised = []
        mol_pairs = []
        mol_pair_mapping = {}
        
        xyzs_to_process = [
            path for path in glob.glob(f"{xyz_dir}/*.xyz")
            if 'opt' not in os.path.basename(path)
        ]
        
        if args.portion < 1.0:
            random.shuffle(xyzs_to_process)
            xyzs_to_process = xyzs_to_process[:int(len(xyzs_to_process) * args.portion)]
        
        for xyz_file in tqdm(xyzs_to_process, desc="Loading molecules for geom_revised"):
            try:
                if args.mol_converter == "xyz2mol":
                    smiles, mol = smilify_xyz2mol(xyz_file, timeout=args.timeout)
                    if mol is not None and mol.GetNumConformers() == 0:
                        try:
                            from rdkit.Geometry import Point3D
                            cart_coords, _ = read_xyz_file(xyz_file)
                            conf = Chem.Conformer(mol.GetNumAtoms())
                            for i in range(mol.GetNumAtoms()):
                                x, y, z = cart_coords[i].tolist()
                                conf.SetAtomPosition(i, Point3D(x, y, z))
                            mol.AddConformer(conf)
                        except Exception as e:
                            logging.debug(f"Failed to add conformer to {xyz_file}: {e}")
                            mol = None
                else:  # openbabel
                    from openbabel import pybel
                    mol_pb = next(pybel.readfile("xyz", xyz_file))
                    mol_sdf = mol_pb.write("sdf")
                    mol = Chem.MolFromMolBlock(mol_sdf, removeHs=False, sanitize=False)
                    if mol is not None:
                        Chem.SanitizeMol(mol)
                
                # Generate optimized molecule using MMFF via OpenBabel
                opt_mol = None
                if mol is not None:
                    try:
                        from openbabel import pybel
                        mol_pb = next(pybel.readfile("xyz", xyz_file))
                        mol_pb.localopt(forcefield="mmff94", steps=1000)
                        
                        # Convert to RDKit mol
                        from rdkit import Chem
                        mol_sdf = mol_pb.write("sdf")
                        opt_mol = Chem.MolFromMolBlock(mol_sdf, removeHs=False, sanitize=False)
                        if opt_mol is not None:
                            Chem.SanitizeMol(opt_mol)
                    except Exception as e:
                        logging.debug(f"OpenBabel MMFF Optimization failed for {xyz_file}: {e}")
                        opt_mol = None

                if mol is not None:
                    mols_revised.append(mol)
                    xyz_files_revised.append(xyz_file)
                    if opt_mol is not None:
                        mol_pairs.append((mol, opt_mol))
                        mol_pair_mapping[xyz_file] = opt_mol

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
            
            # --- Bond Analysis per file ---
            bond_len_diffs = []
            bond_ang_diffs = []
            torsion_diffs = []
            
            for mol, xyz_file in zip(mols_revised, xyz_files_revised):
                opt_mol = mol_pair_mapping.get(xyz_file)
                if opt_mol is not None:
                    len_res = run_bond_analysis([(mol, opt_mol)], "bond_length")
                    ang_res = run_bond_analysis([(mol, opt_mol)], "bond_angle")
                    tor_res = run_bond_analysis([(mol, opt_mol)], "torsion_angle")
                    bond_len_diffs.append(summarize_bond_results(len_res))
                    bond_ang_diffs.append(summarize_bond_results(ang_res))
                    torsion_diffs.append(summarize_bond_results(tor_res))
                else:
                    bond_len_diffs.append(None)
                    bond_ang_diffs.append(None)
                    torsion_diffs.append(None)
            
            results_dict["bond_length_diff"] = bond_len_diffs
            results_dict["bond_angle_diff"] = bond_ang_diffs
            results_dict["torsion_diff"] = torsion_diffs
            
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
            
            valid_global = (sum(valid_list) / n_passed * 100) if n_passed > 0 else 0.0
            conn_global = (sum(valid_connected_list) / n_passed * 100) if n_passed > 0 else 0.0
            logging.info(f"Valid: {sum(valid_list)}/{n_passed} ({valid_global:.2f}%)")
            logging.info(f"Valid & Connected: {sum(valid_connected_list)}/{n_passed} ({conn_global:.2f}%)")
            if split > 1:
                valid_mean, valid_std = _get_split_stats(valid_list, split, scale=100.0)
                conn_mean, conn_std = _get_split_stats(valid_connected_list, split, scale=100.0)
                logging.info(f"Split Valid mean ± std: {valid_mean:.2f} ± {valid_std:.2f}%")
                logging.info(f"Split Valid & Connected mean ± std: {conn_mean:.2f} ± {conn_std:.2f}%")
            
            for mode_name in ["aromatic_true", "aromatic_false"]:
                if f"stable_mol_{mode_name}" in df_revised.columns:
                    n_stable = int(df_revised[f'stable_mol_{mode_name}'].sum())
                    
                    mol_stab_global = float(df_revised[f'stable_mol_{mode_name}'].mean() * 100)
                    atom_stab_global = float(df_revised[f'atom_stability_{mode_name}'].mean() * 100)
                    
                    logging.info(f"--- Mode: {mode_name} ---")
                    logging.info(f"  Molecule Stability: {n_stable}/{n_passed} ({mol_stab_global:.2f}%)")
                    logging.info(f"  Atom Stability: {atom_stab_global:.2f}%")
                    if split > 1:
                        mol_stab_mean, mol_stab_std = _get_split_stats(df_revised[f'stable_mol_{mode_name}'].values, split, scale=100.0)
                        atom_stab_mean, atom_stab_std = _get_split_stats(df_revised[f'atom_stability_{mode_name}'].values, split, scale=100.0)
                        logging.info(f"  Split Molecule Stability mean ± std: {mol_stab_mean:.2f} ± {mol_stab_std:.2f}%")
                        logging.info(f"  Split Atom Stability mean ± std: {atom_stab_mean:.2f} ± {atom_stab_std:.2f}%")

            if len(mol_pairs) > 0:
                bond_global, _ = get_split_stats_bond(mol_pairs, "bond_length", 1)
                ang_global, _ = get_split_stats_bond(mol_pairs, "bond_angle", 1)
                tor_global, _ = get_split_stats_bond(mol_pairs, "torsion_angle", 1)
                
                logging.info(f"--- Geometry Discrepancies (Bond Mode) ---")
                logging.info(f"  Pairs loaded: {len(mol_pairs)}")
                logging.info(f"  Bond Length: {bond_global:.4f} Å")
                logging.info(f"  Bond Angle: {ang_global:.4f}°")
                logging.info(f"  Torsion Angle: {tor_global:.4f}°")
                if split > 1:
                    bond_len_mean, bond_len_std = get_split_stats_bond(mol_pairs, "bond_length", split)
                    bond_ang_mean, bond_ang_std = get_split_stats_bond(mol_pairs, "bond_angle", split)
                    tor_mean, tor_std = get_split_stats_bond(mol_pairs, "torsion_angle", split)
                    logging.info(f"  Split Bond Length mean ± std: {bond_len_mean:.4f} ± {bond_len_std:.4f} Å")
                    logging.info(f"  Split Bond Angle mean ± std: {bond_ang_mean:.4f} ± {bond_ang_std:.4f}°")
                    logging.info(f"  Split Torsion Angle mean ± std: {tor_mean:.4f} ± {tor_std:.4f}°")

    # =========================================================================
    # SHEPHERD: Drug-likeness and conditional similarity
    # =========================================================================
    if args.metrics in ["all", "shepherd"]:
        from rdkit import Chem
        logging.info("Computing ShEPhERD-style metrics...")
        # If the standard list excluded all files (e.g. only *_opt.xyz present), use all xyz
        _shepherd_xyzs = xyzs if xyzs else glob.glob(f"{xyz_dir}/*.xyz")

        # 1. Load reference molecule if provided (for similarity metrics)
        ref_data = None
        # 1. Prepare reference data source
        data_source = None
        if hasattr(args, "reference_mol") and args.reference_mol is not None:
            import pickle
            path = str(args.reference_mol)
            mol_idx = getattr(args, "mol_idx", 0)
            try:
                if path.endswith('.pkl'):
                    with open(path, 'rb') as f:
                        data_source = pickle.load(f)
                elif path.endswith('.sdf'):
                    data_source = Chem.SDMolSupplier(path, removeHs=False)
                else:
                    raise ValueError(f"Unsupported format (expected .pkl or .sdf): {path}")
            except Exception as e:
                logging.error(f"Failed to load reference data source: {e}")
                data_source = None

        # Helper to extract profiles from a moleucle
        def extract_shepherd_profiles(mol):
            if mol is None: return None
            # Extract surface, pharmacophore and ESP using ported shepherd-score
            _pos   = mol.GetConformer().GetPositions().astype(np.float32)
            _radii = get_atomic_vdw_radii(mol)
            from rdkit.Chem import AllChem as _AllChem
            _AllChem.ComputeGasteigerCharges(mol)
            _charges = np.nan_to_num(
                np.array([float(a.GetProp('_GasteigerCharge')) for a in mol.GetAtoms()],
                         dtype=np.float32), nan=0.0)
            surf = get_molecular_surface(_pos, _radii, num_points=75,
                                             probe_radius=1.2, num_samples_per_atom=25)
            esp  = get_electrostatic_potential(mol, _charges, surf)
            p_types, p_pts, p_vecs = get_pharmacophores(
                mol, multi_vector=False, exclude=[], check_access=False, scale=1.0)
            return dict(surface=surf, pharm_pts=p_pts,
                            pharm_types=p_types, pharm_vecs=p_vecs, esp=esp, num_atoms=mol.GetNumAtoms())

        # Precompute reference if not random
        fixed_ref_data = None
        if data_source is not None and mol_idx != -1:
            try:
                if isinstance(data_source, list): # PKL
                    entry = data_source[mol_idx]
                    ref_mol = Chem.MolFromMolBlock(entry[0], removeHs=False)
                else: # SDF
                    ref_mol = data_source[mol_idx]
                
                fixed_ref_data = extract_shepherd_profiles(ref_mol)
                if fixed_ref_data:
                    logging.info(f"Loaded fixed reference mol [{mol_idx}] from {path}: "
                                 f"{fixed_ref_data['num_atoms']} atoms, {len(fixed_ref_data['pharm_types'])} pharmacophores")
            except Exception as e:
                logging.error(f"Failed to load fixed reference molecule: {e}")

        shepherd_results = []
        for xyz in tqdm(_shepherd_xyzs, desc="Computing Shepherd metrics"):
            res = {
                "file": os.path.basename(xyz),
                "valid_rdkit": False,
                "smiles": None,
                "SA_score": 0.0, "QED": 0.0, "LogP": 0.0, "fsp3": 0.0, "MW": 0.0, "HBD": 0, "HBA": 0,
                "shape_sim": 0.0, "pharm_sim": 0.0, "esp_sim": 0.0
            }
            
            # Validity & Drug-likeness
            mol = xyz_to_rdkit_mol(xyz)
            res["valid_rdkit"] = mol is not None
            if mol:
                res["smiles"] = Chem.MolToSmiles(mol)
                drug_likeness = compute_drug_likeness(mol)
                res.update(drug_likeness)
                
                # 2. Determine reference data for this entry
                ref_data = fixed_ref_data
                if mol_idx == -1 and data_source is not None:
                    try:
                        curr_idx = random.randrange(len(data_source))
                        if isinstance(data_source, list): # PKL
                            ref_mol = Chem.MolFromMolBlock(data_source[curr_idx][0], removeHs=False)
                        else: # SDF
                            ref_mol = data_source[curr_idx]
                        ref_data = extract_shepherd_profiles(ref_mol)
                        logging.info(f"Randomly selected ref mol [{curr_idx}] for {os.path.basename(xyz)}")
                    except Exception as e:
                        logging.warning(f"Failed to load random reference for {os.path.basename(xyz)}: {e}")
                        ref_data = None

                # 3. Conditional similarities
                if ref_data:
                    try:
                        # Try to load modalities from unified .npz or individual files
                        npz_path = xyz.replace(".xyz", ".npz")
                        surf_path = xyz.replace(".xyz", "_surface.npy")
                        esp_path = xyz.replace(".xyz", "_esp.npz")
                        pharm_path = xyz.replace(".xyz", "_pharm.npz")
                        
                        gen_surf, gen_esp, gen_pharm_pts, gen_pharm_types, gen_pharm_vecs = None, None, None, None, None
                        
                        if os.path.exists(npz_path):
                            try:
                                sidecar = np.load(npz_path)
                                gen_surf = sidecar.get("surf_pts", None)
                                gen_esp = sidecar.get("esp_vals", None)
                                gen_pharm_pts = sidecar.get("pharm_pts", None)
                                gen_pharm_types = sidecar.get("pharm_types", None)
                                # Note: Unified .npz doesn't currently store pharm_vecs in Generation code
                            except: pass
                        
                        # Fallback to individual files
                        if gen_surf is None and os.path.exists(surf_path):
                             try: gen_surf = np.load(surf_path)
                             except: pass
                        if gen_esp is None and os.path.exists(esp_path):
                             try: gen_esp = np.load(esp_path).get("charges", None)
                             except: pass
                        if gen_pharm_pts is None and os.path.exists(pharm_path):
                             try:
                                 p_data = np.load(pharm_path)
                                 gen_pharm_pts = p_data.get("positions", None)
                                 gen_pharm_types = p_data.get("types", None)
                                 gen_pharm_vecs = p_data.get("directions", None)
                             except: pass

                        # --- Fallback: recompute gen profiles from RDKit mol ---
                        if gen_surf is None:
                            _gen_pos   = mol.GetConformer().GetPositions().astype(np.float32)
                            _gen_radii = get_atomic_vdw_radii(mol)
                            gen_surf = get_molecular_surface(_gen_pos, _gen_radii, num_points=75,
                                                             probe_radius=1.2, num_samples_per_atom=25)
                        if gen_pharm_pts is None:
                            gen_pharm_types, gen_pharm_pts, gen_pharm_vecs = get_pharmacophores(
                                mol, multi_vector=False, exclude=[], check_access=False, scale=1.0)
                        if gen_esp is None:
                            from rdkit.Chem import AllChem as _AllChem2
                            _AllChem2.ComputeGasteigerCharges(mol)
                            _gen_charges = np.nan_to_num(
                                np.array([float(a.GetProp('_GasteigerCharge')) for a in mol.GetAtoms()],
                                         dtype=np.float32), nan=0.0)
                            gen_esp = get_electrostatic_potential(mol, _gen_charges, gen_surf)

                        # Compute similarity metrics using ported shepherd-score
                        def _alpha(n):
                            try: return float(ALPHA(np.clip(n, 50, 400)))
                            except: return 0.81

                        # Shape
                        if gen_surf is not None and len(gen_surf) > 0 and ref_data["surface"] is not None:
                            _n = len(gen_surf)
                            _a = _alpha(_n)
                            _gs = (gen_surf - gen_surf.mean(axis=0)).astype(np.float32)
                            _rs = (ref_data["surface"] - ref_data["surface"].mean(axis=0)).astype(np.float32)
                            _aligned, _, _ = optimize_ROCS_overlay(
                                torch.from_numpy(_rs), torch.from_numpy(_gs), _a, num_repeats=45)
                            res["shape_sim"] = float(get_overlap_np(_rs, _aligned.numpy(), alpha=_a))

                        # Pharmacophore
                        if gen_pharm_pts is not None and len(gen_pharm_pts) > 0 and ref_data["pharm_pts"] is not None:
                            _rpa = (ref_data["pharm_pts"] - ref_data["pharm_pts"].mean(axis=0)).astype(np.float32)
                            _gpa = (gen_pharm_pts - gen_pharm_pts.mean(axis=0)).astype(np.float32)
                            _rpv = ref_data["pharm_vecs"].astype(np.float32)
                            _gpv = (gen_pharm_vecs.astype(np.float32) if gen_pharm_vecs is not None
                                    else np.zeros_like(_gpa))
                            _rpt = ref_data["pharm_types"].astype(np.int64)
                            _gpt = gen_pharm_types.astype(np.int64)
                            _ga_t, _gv_t, _, _ = optimize_pharm_overlay(
                                torch.from_numpy(_rpt), torch.from_numpy(_gpt),
                                torch.from_numpy(_rpa), torch.from_numpy(_gpa),
                                torch.from_numpy(_rpv), torch.from_numpy(_gpv),
                                similarity='tanimoto', num_repeats=45)
                            res["pharm_sim"] = float(get_overlap_pharm_np(
                                ptype_1=_rpt, ptype_2=_gpt,
                                anchors_1=_rpa, anchors_2=_ga_t.numpy(),
                                vectors_1=_rpv, vectors_2=_gv_t.numpy(),
                                similarity='tanimoto'))

                        # ESP
                        ref_esp = ref_data["esp"]
                        if gen_esp is not None and ref_esp is not None and gen_surf is not None and ref_data["surface"] is not None:
                            _n   = len(gen_surf)
                            _a   = _alpha(_n)
                            _lam = 0.3 * LAM_SCALING
                            _gs  = (gen_surf - gen_surf.mean(axis=0)).astype(np.float32)
                            _rs  = (ref_data["surface"] - ref_data["surface"].mean(axis=0)).astype(np.float32)
                            _ge  = gen_esp.astype(np.float32)
                            _re  = ref_esp.astype(np.float32)
                            _aligned_e, _, _ = optimize_ROCS_esp_overlay(
                                torch.from_numpy(_rs), torch.from_numpy(_gs),
                                torch.from_numpy(_re), torch.from_numpy(_ge),
                                _a, _lam, num_repeats=45)
                            res["esp_sim"] = float(get_overlap_esp_np(
                                _rs, _aligned_e.numpy(), _re, _ge, alpha=_a, lam=_lam))
                        else:
                            res["esp_sim"] = 0.0

                    except Exception as e:
                        import traceback
                        logging.warning(f"Failed to compute similarity for {os.path.basename(xyz)}: {e}")
            
            shepherd_results.append(res)
            
        df_shepherd = pd.DataFrame(shepherd_results)
        
        # Save output
        if args.output is None:
            shepherd_output_path = f"{xyz_dir}/shepherd_metrics.csv"
        else:
            base, ext = os.path.splitext(args.output)
            shepherd_output_path = f"{base}_shepherd{ext}"
            
        df_shepherd.to_csv(shepherd_output_path, index=False)
        logging.info(f"ShEPhERD metrics saved to {shepherd_output_path}")
        
        # Summary
        if "SA_score" in df_shepherd.columns:
            logging.info(f"Average SA_score: {df_shepherd['SA_score'].mean():.4f}")
            logging.info(f"Average QED: {df_shepherd['QED'].mean():.4f}")
        if "shape_sim" in df_shepherd.columns:
            logging.info(f"Average Shape Similarity: {df_shepherd['shape_sim'].mean():.4f}")
        if "esp_sim" in df_shepherd.columns:
            logging.info(f"Average ESP Similarity: {df_shepherd['esp_sim'].mean():.4f}")
        if "pharm_sim" in df_shepherd.columns:
            logging.info(f"Average Pharm Similarity: {df_shepherd['pharm_sim'].mean():.4f}")
        if split > 1 and len(df_shepherd) > 0:
            valid_rdkit_mean, valid_rdkit_std = _get_split_stats(df_shepherd["valid_rdkit"].values, split, scale=100.0)
            sa_mean, sa_std = _get_split_stats(df_shepherd["SA_score"].values, split, scale=1.0)
            qed_mean, qed_std = _get_split_stats(df_shepherd["QED"].values, split, scale=1.0)
            shape_mean, shape_std = _get_split_stats(df_shepherd["shape_sim"].values, split, scale=1.0)
            esp_mean, esp_std = _get_split_stats(df_shepherd["esp_sim"].values, split, scale=1.0)
            pharm_mean, pharm_std = _get_split_stats(df_shepherd["pharm_sim"].values, split, scale=1.0)
            logging.info(f"Split valid RDKit mean ± std: {valid_rdkit_mean:.2f} ± {valid_rdkit_std:.2f}%")
            logging.info(f"Split SA_score mean ± std: {sa_mean:.4f} ± {sa_std:.4f}")
            logging.info(f"Split QED mean ± std: {qed_mean:.4f} ± {qed_std:.4f}")
            logging.info(f"Split Shape Similarity mean ± std: {shape_mean:.4f} ± {shape_std:.4f}")
            logging.info(f"Split ESP Similarity mean ± std: {esp_mean:.4f} ± {esp_std:.4f}")
            logging.info(f"Split Pharm Similarity mean ± std: {pharm_mean:.4f} ± {pharm_std:.4f}")



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
    parser.add_argument("--mol_converter", type=str, default="xyz2mol",
                        choices=["xyz2mol", "openbabel"],
                        help="Molecule converter for XYZ to mol: 'xyz2mol' (default) or 'openbabel'")
    parser.add_argument("--split", type=int, default=1, help="number of deterministic splits for summary mean/std")
    parser.add_argument("--timeout", type=int, default=10, help="timeout for xyz2mol conversion (default: 10s)")
    args = parser.parse_args()
    runner(args)
