
import glob
import logging
import multiprocessing
import os
import subprocess as sp

import networkx as nx
import numpy as np
import pandas as pd
import torch
from ase.data import covalent_radii
from openbabel import pybel
from posebusters import PoseBusters
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from tqdm import tqdm

from .geom_constant import (
    allow_n_bonds,
    allowed_shape,
    degree_angles_ref,
    valid_valencies,
    vertices_labels,
)
from .geom_utils import correct_edges, create_pyg_graph, read_xyz_file

try:
    from cosymlib import Geometry
    is_cosymlib_available = True
except ImportError:
    is_cosymlib_available = False
    Geometry = None
    
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logging.getLogger("posebusters").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)
SCORES_THRESHOLD = 3.0

EDGE_THRESHOLD = 4
SCALE_FACTOR = 1.2


#%%
def check_neutrality(filename):
    """
    Checks if a molecule (specified by an XYZ file) is neutral by running an xTB calculation.
    
    It runs 'xtb <filename> --ptb' and checks the output log for messages indicating
    a mismatch between electrons and spin multiplicity.
    
    Args:
        filename (str): Path to the XYZ file containing the molecule.
        
    Returns:
        bool: True if the molecule is neutral (no mismatch found), False otherwise.
    """
    neutral_mol = True
    execution = ["xtb", filename, "--ptb"]
    try:
        with open("xtb.log", "w") as f:
            sp.call(execution, stdout=f, stderr=sp.STDOUT, timeout=60)
    except sp.TimeoutExpired:           
        print("xTB calculation timed out.")
    
    if os.path.exists("xtb.log"):
        with open("xtb.log", "r") as f:
            lines = f.readlines()
    
    for line in lines:
        if "Number of electrons and spin multiplicity do not match" in line:
            neutral_mol = False
            break
    
    if os.path.exists("wbo"):
        os.remove("wbo")
    if os.path.exists("charges"):
        os.remove("charges")
    if os.path.exists("xtb.log"):
        os.remove("xtb.log")

    return neutral_mol


def compare_graph_topology(graph1, graph2):
    """
    Compare the topology of two PyG graphs by checking their edge indices.
    
    Args:
        graph1 (torch_geometric.data.Data): The first PyG graph.
        graph2 (torch_geometric.data.Data): The second PyG graph.
        
    Returns:
        bool: True if the graphs have the same topology, False otherwise.
    """
    edge_index1 = graph1.edge_index
    edge_index2 = graph2.edge_index
    
    # Sort the edge indices for comparison
    edge_index1_sorted = edge_index1[:, edge_index1[0].argsort()]
    edge_index2_sorted = edge_index2[:, edge_index2[0].argsort()]
    
    return torch.equal(edge_index1_sorted, edge_index2_sorted)



def is_fully_connected(edge_index, num_nodes):
    """
    Determines if the graph is fully connected.
    
    Args:
        edge_index (torch.Tensor): The edge indices of the graph.
        num_nodes (int): The number of nodes in the graph.
        
    Returns:
        tuple: (bool, int)
            - bool: True if the graph is fully connected, False otherwise.
            - int: The number of connected components in the graph.
    """
    G = to_networkx(Data(edge_index=edge_index, num_nodes=num_nodes), to_undirected=True)
    try:
        is_connected = nx.is_connected(G)
        num_components = nx.number_connected_components(G)
    except nx.NetworkXPointlessConcept:
        is_connected = False
        num_components = 100
    return is_connected, num_components


def check_validity_v0(data, 
                   angle_relax=10,
                   scale_factor=1.3,
                   verbose=False):
    
    """
    Validate a molecular structure based on atomic distances and angles (Version 0).

    Args:
        data: A dictionary containing molecular information.
            - 'atomic_numbers': List of integers representing the atomic number of each atom.
            - 'positions': List of tuples (x, y, z) representing the position of each atom.
        angle_relax (float): Tolerance allowed for bond angles in degrees. Default is 10.0.
        scale_factor (float): The scaling factor to apply to the covalent radii. Default is 1.3.
        verbose (bool): Whether to print debug messages during validation. Default is False.

    Returns:
        tuple: Contains the following elements:
            - is_valid (bool): Boolean indicating if the structure is valid.
            - percent_atom_valid (float): Percentage of atoms that meet the criteria.
            - num_components (int): Number of connected components in the molecular graph.
            - bad_atoms (list): List of indices for atoms that do not meet the criteria.
            - needs_rechecking (bool): Whether further checks are needed due to borderline cases or special conditions.

    Notes:
        This function assumes that 'data' contains valid atomic numbers and positions. 
        The validation process involves checking bond distances and angles against predefined reference values.
        Special handling is applied for atoms with certain atomic numbers, such as carbon (atomic number 6), which may require different criteria due to their bonding behavior.

    """
    atomic_numbers = data.x.view(-1).int().tolist()
    edge_index = data.edge_index
    good_atoms = []
    bad_atoms = []
    
    to_be_recheck_flag = False # soem angles are too distorted
    all_node_pass_flag = True

    is_connected, num_components = is_fully_connected(data.edge_index, data.num_nodes)
    if verbose:
        print(f"Is fully connected: {is_connected}, Number of components: {num_components}")
    
    if num_components == 100:
        return False, 0, 100, torch.arange(0,data.num_nodes), False
        
    for node in range(len(atomic_numbers)):
        is_valid_node = True
        adjacent_nodes = edge_index[1][edge_index[0] == node].tolist()
        n_degree = len(adjacent_nodes)
        if n_degree > 1:
            bond_angles = []
            for i in range(len(adjacent_nodes)):
                
                distance = torch.norm(data.pos[adjacent_nodes[i]] - data.pos[node]).item()
                r1 = covalent_radii[atomic_numbers[node]]*(1/scale_factor)
                r2 = covalent_radii[atomic_numbers[adjacent_nodes[i]]]*(1/scale_factor)
                min_dist_ref = r1 + r2
                if distance < min_dist_ref:
                    if verbose:
                        print(f"Bad distance Node {node} (Atomic Number: {atomic_numbers[node]}): Distance -> {distance:.2f}")
                    is_valid_node = False
                    all_node_pass_flag = False
                    
                    
                
                for j in range(i + 1, len(adjacent_nodes)):
                    node_i = adjacent_nodes[i]
                    node_j = adjacent_nodes[j]
                    
                    vec1 = data.pos[node_i] - data.pos[node]
                    vec2 = data.pos[node_j] - data.pos[node]
                    
                    cos_angle = torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))
                    angle = torch.acos(cos_angle).item() * (180.0 / 3.141592653589793)
                    bond_angles.append(angle)
                    # print(f"Bond angle between nodes {node}, {node_i}, and {node_j}: {angle:.2f} degrees")
            bond_angles = [angle for angle in bond_angles if not torch.isnan(torch.tensor(angle))]
            if len(bond_angles) > 0:
                average_bond_angle = sum(bond_angles) / len(bond_angles)
            else:
                average_bond_angle =0 
        else:
            bond_angles = [180.0]   
            average_bond_angle = 180.0
        ref_angle = degree_angles_ref.get(atomic_numbers[node], {}).get(n_degree, [])
        
        if ref_angle and is_valid_node:
            if any(abs(average_bond_angle - angle) <= angle_relax for angle in ref_angle):
                if any(abs(angle - ref) <= angle_relax*2 for angle in bond_angles for ref in ref_angle):                
                    is_valid_node = True
                else:
                    if verbose:
                        print(f"Bad angle Node {node} (Atomic Number: {atomic_numbers[node]}): Angle -> {average_bond_angle:.2f}")
                    is_valid_node = False
                    # Assume most false negatives are due to distorted angles (particularly n_degree = 2)
                    if n_degree > 1 and atomic_numbers[node] != 6 and all_node_pass_flag: # carbon is not influenced by electron pair
                        to_be_recheck_flag = True
            else:
                if verbose:
                    print(f"Bad angle Node {node} (Atomic Number: {atomic_numbers[node]}): Degree -> {n_degree}, Angle -> {average_bond_angle:.2f}")
                is_valid_node = False
                if n_degree > 1 and atomic_numbers[node] != 6 and all_node_pass_flag:
                    to_be_recheck_flag = True
        else:
            all_node_pass_flag = False
            if verbose:
                print(f"Bad dist or not in stat node {node} (Atomic Number: {atomic_numbers[node]}): Degree -> {n_degree}, Angle -> {average_bond_angle:.2f}, Distance -> {distance:.2f}")
            is_valid_node = False
        if is_valid_node:
            good_atoms.append(node)
        else:
            bad_atoms.append(node)
    percent_atom_valid = len(good_atoms) / len(atomic_numbers)
    if percent_atom_valid == 1:
        is_valid = True
    else:
        is_valid = False
    return is_valid, percent_atom_valid, num_components, bad_atoms, to_be_recheck_flag


def check_validity_v1(data, 
                   score_threshold=3,
                   scale_factor=1.3,
                   skip_indices=[],
                   verbose=False):
    
    """
    Validate a molecular structure based on atomic distances and angles (Version 1).

    Args:
        data: A dictionary containing molecular information.
            - 'atomic_numbers': List of integers representing the atomic number of each atom.
            - 'positions': List of tuples (x, y, z) representing the position of each atom.
        score_threshold (float): Tolerance allowed for shape values. Default is 3.0.
        scale_factor (float): The scaling factor to apply to the covalent radii. Default is 1.3.
        skip_indices (list): List of atom indices to skip during validation.
        verbose (bool): Whether to print debug messages during validation. Default is False.

    Returns:
        tuple: Contains the following elements:
            - is_valid (bool): Boolean indicating if the structure is valid.
            - percent_atom_valid (float): Percentage of atoms that meet the criteria.
            - num_components (int): Number of connected components in the molecular graph.
            - bad_atom_chem (list): List of indices for atoms that do not meet chemical valency criteria.
            - bad_atom_distort (list): List of indices for atoms that are geometrically distorted.

    Notes:
        This function assumes that 'data' contains valid atomic numbers and positions. 
        The validation process involves checking bond distances and angles against predefined reference values.
        Special handling is applied for atoms with certain atomic numbers, such as carbon (atomic number 6), which may require different criteria due to their bonding behavior.

    """
    
    if not(is_cosymlib_available):
        raise ImportError("Cosymlib is not available, do use different metrics")
    atomic_numbers = data.x.view(-1).int().tolist()
    edge_index = data.edge_index
    good_atoms = []
    bad_atom_distort = []
    bad_atom_chem = []
    
    is_connected, num_components = is_fully_connected(data.edge_index, data.num_nodes)
    if verbose:
        print(f"Is fully connected: {is_connected}, Number of components: {num_components}")
    
    if num_components == 100:
        return False, 0, 100, torch.arange(0,data.num_nodes), torch.arange(0,data.num_nodes)
       
    for node in range(len(atomic_numbers)):
        
        if node in skip_indices:
            continue
        shp_scores = {}
        is_valid_node = True
        adjacent_nodes = edge_index[1][edge_index[0] == node].tolist()
        n_degree = len(adjacent_nodes)
        
        # 1 check number of bonds
        if n_degree not in allow_n_bonds.get(atomic_numbers[node], []):
            bad_atom_chem.append(node)
            if verbose:
                print(f"Bad atom Node {node} (Atomic Number: {atomic_numbers[node]}): Degree -> {n_degree}")
            is_valid_node = False
            
            
        # 2 check bond lengths
        if is_valid_node:
            for i in range(len(adjacent_nodes)):
                
                distance = torch.norm(data.pos[adjacent_nodes[i]] - data.pos[node]).item()
                r1 = covalent_radii[atomic_numbers[node]]*(1/scale_factor)
                r2 = covalent_radii[atomic_numbers[adjacent_nodes[i]]]*(1/scale_factor)
                min_dist_ref = r1 + r2
                if distance < min_dist_ref:
                    if verbose:
                        print(f"Bad distance Node {node} (Atomic Number: {atomic_numbers[node]}): Distance -> {distance:.2f}")
                    bad_atom_distort.append(node)
                    is_valid_node = False
                
        # 3 check the shape of the molecule
        if is_valid_node:
            nodes_all = [node] + adjacent_nodes      
            symbols = [atomic_numbers[i] for i in nodes_all] 
            
            positions = data.pos[nodes_all].tolist()
            geometry = Geometry(positions=positions, symbols=symbols)
            if len(adjacent_nodes) > 1:
                shp_types = vertices_labels[len(adjacent_nodes)]
                for shp_type in shp_types:
                    shp_measure = geometry.get_shape_measure(shp_type, central_atom=1)
                    shp_scores[shp_type] = shp_measure 
                        
                shp_type = min(shp_scores, key=shp_scores.get)
                shp_score = shp_scores[shp_type]
                ref_shp = allowed_shape[atomic_numbers[node]]
                if shp_type not in ref_shp or shp_score > score_threshold:
                    bad_atom_distort.append(node)
                    is_valid = False
                    if verbose:
                        print(f"Bad shape Node {node} (Atomic Number: {atomic_numbers[node]}): Shape -> {shp_type}")
                else:
                    good_atoms.append(node)
            else:
                good_atoms.append(node)
    percent_atom_valid = len(good_atoms) / len(atomic_numbers)
    if len(bad_atom_chem) == 0  and len(bad_atom_distort) == 0:
        is_valid = True   
    else:
        is_valid = False        

    return (is_valid, percent_atom_valid, num_components, bad_atom_chem, bad_atom_distort)
    
def check_chem_validity(mol_list, 
                        skip_idx=[],
                        verbose=0):
    """
    Analyze a list of RDKit molecules for chemical validity based on atom valencies
    and identify broken or disconnected fragments.

    Parameters
    ----------
    mol_list : list of rdkit.Chem.Mol
        A list of RDKit Mol objects to be checked.
    skip_idx : list of int, optional
        Atom indices to skip when counting and checking (currently unused in logic;
        provided for future extension). Default is None.
    verbose : int, default=0
        If > 0, prints detailed information about each valency violation as:
        "<SMILES> has invalid valency: <AtomSymbol> <TotalValence> <FormalCharge>".

    Returns
    -------
    natom_stability_dicts : dict[str, int]
        Counts of atoms whose total electron count (valence minus formal charge)
        matches a known valid valency (i.e. “stable” atoms), keyed by atom symbol.
    natom_tot_dicts : dict[str, int]
        Total counts of atoms encountered, keyed by atom symbol.
    good_smiles : list of str
        Canonical SMILES strings for molecules deemed chemically valid and not
        containing disconnected fragments (".").
    bad_smiles_broken : list of str
        Canonical SMILES for molecules that are chemically valid but contain
        disconnected fragments (i.e. salts or mixtures with "." in the SMILES).
    bad_smiles_chem : list of str
        Canonical SMILES for molecules that failed the valency checks.

    Notes
    -----
    - This function relies on a pre-defined mapping `valid_valencies`:
        >>> valid_valencies = {
        ...     'H': {1}, 'C': {4}, 'N': {3, 5}, 'O': {2}, ...
        ... }
      which maps each element symbol to the set of permitted electron counts.
    - The `skip_idx` argument is currently not applied in the loop; if you wish
      to ignore certain atoms (for example, metals or explicit hydrogens), you
      could uncomment and adapt the skip logic.

    Example
    -------
    >>> from rdkit.Chem import MolFromSmiles
    >>> smiles_list = ['CCO', 'C[N+](C)(C)C', 'C.C']  # ethanol, tetramethylammonium, disconnected C and C
    >>> mols = [MolFromSmiles(s) for s in smiles_list]
    >>> valid, total, good, broken, bad = check_chem_validity(mols, verbose=1)
     # would print any valency errors if present
    """
    natom_stability_dicts = {}
    natom_tot_dicts = {}

    bad_smiles_broken = []
    bad_smiles_chem = []
    good_smiles = []

    for mol in mol_list:
        is_chem_valid = True

        for atom in mol.GetAtoms():
            if atom.GetIdx() in skip_idx:
                continue
            symbol = atom.GetSymbol()
            if symbol not in natom_tot_dicts:
                natom_tot_dicts[symbol] = 1
            else:
                natom_tot_dicts[symbol] += 1

            nelectron = atom.GetTotalValence() - atom.GetFormalCharge()
            if nelectron in valid_valencies[symbol]:
                if symbol not in natom_stability_dicts:
                    natom_stability_dicts[symbol] = 1
                else:
                    natom_stability_dicts[symbol] += 1

            else:
                is_chem_valid = False
                if verbose > 0:
                    smiles = Chem.MolToSmiles(mol,canonical=True)
                    print(
                        f"{smiles} has invalid valency: {symbol} {atom.GetTotalValence()} {atom.GetFormalCharge()}"
                    )

        smiles = Chem.MolToSmiles(mol, canonical=True)
        if is_chem_valid:
            # check for broken molecules
            if "." not in smiles:
                good_smiles.append(smiles)
            else:
                bad_smiles_broken.append(smiles)
        else:
            bad_smiles_chem.append(smiles)

    return (
        natom_stability_dicts,
        natom_tot_dicts,
        good_smiles,
        bad_smiles_broken,
        bad_smiles_chem,
    )
    
def smilify_wrapper(xyzs, xyz2mol):
    """
    Convert a list of XYZ files to SMILES strings using a provided xyz2mol function.
    
    Args:
        xyzs (list of str): List of paths to XYZ files.
        xyz2mol (callable): A function that takes an XYZ file path and returns (smiles, mol).
        
    Returns:
        tuple: (validity, smiles_list, mol_list, dicts)
            - validity (float): Fraction of successful conversions.
            - smiles_list (list of str): List of SMILES strings (None for failures).
            - mol_list (list of RDKit Mol): List of RDKit Mol objects (None for failures).
            - dicts (dict): Dictionary with 'smiles' and 'filename' lists.
    """
    smiles_list = []
    mol_list = []
    dicts = {"smiles": [], "filename": []}
    for xyz in xyzs:
        try:
            (
                smiles,
                mol,
            ) = xyz2mol(xyz)
        except Exception as e:
            print("Error in converting {xyz} to smiles due to", e)
            smiles = None
            mol = None

        if smiles is None:
            continue

        # Canonicalize the SMILES
        # smiles = Chem.MolToSmiles(mol, canonical=True)
        mol_list.append(mol)
        dicts["smiles"].append(smiles)
        dicts["filename"].append(xyz)
        smiles_list.append(smiles)

    validity = len(smiles_list) / len(xyzs)
    return validity, smiles_list, mol_list, dicts

#%% postbuster

def xyz_to_pdb(xyz_file_path, pdb_file_path):
    """
    Converts an XYZ file to a PDB file using OpenBabel (via pybel).
    
    Args:
        xyz_file_path (str): Path to input XYZ file.
        pdb_file_path (str): Path to output PDB file.
    """
    try:
        mol = next(pybel.readfile("xyz", xyz_file_path))
        mol.write("pdb", pdb_file_path, overwrite=True)
    except Exception as e:
        logger.error(f"Error converting {xyz_file_path} to PDB: {e}")


def load_molecules_from_xyz(xyz_dir):
    """
    Converts all XYZ files in a directory to RDKit Mol objects.
    
    It first converts XYZ files to PDB using OpenBabel, then loads the PDBs into RDKit.
    
    Args:
        xyz_dir (str): Directory containing XYZ files.
        
    Returns:
        tuple: (valid_molecules, pass_xyz_files)
            - valid_molecules (list of RDKit Mol): Successfully loaded molecules.
            - pass_xyz_files (list of str): Filenames of the successfully loaded molecules.
    """
    xyz_files = glob.glob(os.path.join(xyz_dir, "*.xyz"))
    valid_molecules = []
    pass_xyz_files = []

    for xyz_file in xyz_files:
        try:
            pdb_file = os.path.splitext(xyz_file)[0] + ".pdb"
            xyz_to_pdb(xyz_file, pdb_file)
            mol = Chem.MolFromPDBFile(pdb_file, removeHs=False)
            if mol is not None:
                valid_molecules.append(mol)
                pass_xyz_files.append(xyz_file)

        except Exception as e:
            logger.warning(f"Skipping {xyz_file} due to error: {e}")

    total = len(xyz_files)
    success = len(valid_molecules)
    percent = (success / total) * 100 if total > 0 else 0.0

    logger.info(f"Successfully converted {success} out of {total} XYZ files ({percent:.2f}%).")

    return valid_molecules, pass_xyz_files



def _run_buster(mols, queue):
    """
    Helper function to run PoseBusters on a list of molecules in a separate process.
    
    Args:
        mols (list of RDKit Mol): List of molecules to process.
        queue (multiprocessing.Queue): Queue to put the result (DataFrame or Exception).
    """
    try:
        buster = PoseBusters(config="mol")
        results = buster.bust(mols)
        queue.put(results)
    except Exception as e:
        queue.put(e)


def run_postbuster(mols, timeout=60, batch_size=None):
    """
    Run PoseBusters on a list of RDKit molecules, optionally in batches, with a timeout per batch.
    
    This function processes molecules using the PoseBusters library to compute various geometric checks.
    Processing happens in separate processes to enforce timeouts.
    
    Args:
        mols (list of RDKit Mol): List of molecules to evaluate.
        timeout (int, optional): Maximum time (in seconds) allowed for each batch calculation. Default is 60.
        batch_size (int, optional): Number of molecules to process in a single batch. 
                                    If None, processes all molecules in one batch. Default is None.
                                    
    Returns:
        pd.DataFrame or None: DataFrame containing PoseBusters results for all processed molecules.
                              Returns None if no results could be obtained.
    """
    if not mols:
        logger.warning("No valid molecules loaded. Exiting.")
        return None

    if batch_size is None:
        batch_size = len(mols)

    all_results = []
    num_batches = (len(mols) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Processing PoseBusters batches"):
        batch_mols = mols[i * batch_size : (i + 1) * batch_size]
        
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=_run_buster, args=(batch_mols, queue))
        process.start()
        process.join(timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            logger.warning(f"PoseBusters timed out after {timeout} seconds (Batch {i+1}/{num_batches}). Skipping batch.")
            continue
        
        try:
            result = queue.get(timeout=5)
        except Exception:
            logger.error(f"PoseBusters failed to return result (Batch {i+1}/{num_batches}). Skipping batch.")
            continue

        if isinstance(result, Exception):
            logger.error(f"PoseBusters failed with an exception: {result}. Skipping batch.")
            continue
        
        all_results.append(result)

    if not all_results:
        return None

    try:
        final_df = pd.concat(all_results, ignore_index=True)
        return final_df
    except Exception as e:
        logger.error(f"Error concatenating batch results: {e}")
        return None


#%% All
def runner(args):
    """
    Main runner function to process a directory of XYZ files, compute geometric metrics, 
    check validity, and optionally run strain and diversity checks.
    
    Args:
        args (argparse.Namespace): Arguments containing:
            - input (str): Input directory path containing .xyz files.
            - output (str, optional): Output CSV file path.
            - recheck_topo (bool): Whether to recheck topology.
            - check_strain (bool): Whether to check strain energy.
            - check_diversity (bool): Whether to compute diversity scores.
            - skip_atoms (list of int, optional): Atom indices to skip during checks.
    """
    xyz_dir = args.input
    recheck_topo = args.recheck_topo
    check_strain = args.check_strain
    check_diversity = args.check_diversity
    skip_idx = args.skip_atoms
    
    if skip_idx is None:
        skip_idx = []

    xyzs = [
    path for path in glob.glob(f"{xyz_dir}/*.xyz")
    if 'opt' not in os.path.basename(path)
]

    df_res_dict = {
        "file": [],
        "percent_atom_valid": [],
        "valid": [],
        "valid_connected": [],
        "num_graphs": [],
        "bad_atom_distort": [],
        "bad_atom_chem": [],
        "smiles": []
    }
    
    if check_diversity:
        
        similarity_3ds, = diversity_score(
            xyzs,
            type_3d="euclidean",
        )
        

        
    for xyz in tqdm(xyzs, desc="Processing XYZ files", total=len(xyzs)):

        try:
            cartesian_coordinates_tensor, atomic_numbers_tensor = read_xyz_file(xyz)
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
            print(f"Error processing {xyz}: {e}")
            is_valid = False
            percent_atom_valid = 0
            num_components = 100
            bad_atom_chem = torch.arange(0,data.num_nodes)
            bad_atom_distort = torch.arange(0,data.num_nodes)
 
        from .smilify import smilify_cell2mol, smilify_openbabel
        
        smiles_list, mol_list = smilify_openbabel(xyz)
        to_recheck = recheck_topo and (len(bad_atom_distort) > 0) and (len(bad_atom_chem) == 0)
        
        if mol_list is None and num_components < 3:
            xyz2mol_fn = smilify_cell2mol
            try:
                _, smiles_list, mol_list, _ = smilify_wrapper([xyz], xyz2mol_fn)
                mol_list = mol_list[0]
            except Exception as e:
                print(f"fail to convert xyz to mol with v0, skip and assign invalid")
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
                    print("Detect bad smiles in ", xyz, bad_smiles_chem)
            except Exception as e:
                print(f"Fail to check on {xyz} due to {e}, asssign invalid")
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
        df_res_dict["num_graphs"].append(num_components)
        df_res_dict["bad_atom_distort"].append(bad_atom_distort)
        df_res_dict["bad_atom_chem"].append(bad_atom_chem)  
    df = pd.DataFrame(df_res_dict)
    df = df.sort_values(by="file")
    fully_connected = [1 if num == 1 else 0 for num in df_res_dict["num_graphs"]]

    print(f"{df['percent_atom_valid'].mean() * 100:.2f}% of atoms are stable")
    print(f"{df['valid'].mean() * 100:.2f}% of 3D molecules are valid")
    print(f"{df['valid_connected'].mean() * 100:.2f}% of 3D molecules are valid and fully-connected")
    print(f"{sum(fully_connected) / len(fully_connected) * 100:.2f}% of 3D molecules are fully connected")
    
    if check_strain:
        rmsd_mean = df["rmsd"].dropna().mean()
        delta_energy_mean = df["delta_energy"].dropna().mean()
        intact_topology = [1 if top else 0 for top in df_res_dict["same_topology"] if not pd.isna(top)]
        print(f"RMSD mean: {rmsd_mean:.2f}")
        print(f"Delta Energy mean: {delta_energy_mean:.2f}")
        print(f"{sum(intact_topology) / len(intact_topology) * 100:.2f}% of 3D molecules have intact topology after the optimization")
    
    if args.output is None:
        output_path = f"{xyz_dir}/output_metrics.csv"
    else:
        output_path = args.output
    df.to_csv(output_path, index=False)

    if check_diversity: 
        if similarity_3ds is not None and len(similarity_3ds) > 0:
            print(f"Average 3D similarity: {np.mean(similarity_3ds):.2f}")
            print(f"Max 3D similarity: {np.max(similarity_3ds):.2f}")
            print(f"Min 3D similarity: {np.min(similarity_3ds):.2f}")


  
   
