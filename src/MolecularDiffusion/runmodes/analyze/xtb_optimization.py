import subprocess as sp
import os
import glob
import shutil
from tqdm import tqdm
import argparse
import torch
import pandas as pd

from MolecularDiffusion.utils import create_pyg_graph, correct_edges
from MolecularDiffusion.utils.geom_utils import read_xyz_file
from MolecularDiffusion.utils.geom_metrics import is_fully_connected

def check_neutrality(filename: str, 
                     charge: int = -1, 
                     timeout: int=180) -> bool:
    """
    Checks if a molecule described in an XYZ file is neutral using xTB.

    This function executes the `xtb` command with the `--ptb` (print properties)
    flag and parses its log output to detect if xTB reports a mismatch
    between the number of electrons and spin multiplicity, which indicates
    a non-neutral molecule. Temporary xTB output files are cleaned up afterwards.

    Note: This function assumes `xtb` is installed and accessible in the system's PATH.
    This functionality could potentially be integrated with other molecular property
    calculation modules if available.

    Args:
        filename (str): The path to the XYZ file of the molecule to check.
        charge (int): The molecular charge to use for the xTB calculation. Defaults to -1.
        timeout (int): The maximum time in seconds to wait for the xTB process to complete.

    Returns:
        bool: True if the molecule is inferred to be neutral based on xTB's output,
              False otherwise.
    """
    neutral_mol = True
    execution_command = ["xtb", filename, "--ptb", "-c", str(charge)]

    try:
        with open("xtb.log", "w") as f:
            sp.call(execution_command, stdout=f, stderr=sp.STDOUT, timeout=timeout)
    except sp.TimeoutExpired:
        print("xTB calculation timed out during neutrality check.")
        return False

    if os.path.exists("xtb.log"):
        with open("xtb.log", "r") as f:
            lines = f.readlines()

        for line in lines:
            if "Number of electrons and spin multiplicity do not match" in line:
                neutral_mol = False
                break
    else:
        print(f"Warning: xTB log file not found for {filename}. Cannot verify neutrality.")
        neutral_mol = False

    temp_files = ["wbo", "charges", "xtb.log"]
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return neutral_mol


def check_xyz(filename: str, connector_dicts: dict = None, scale_factor: float = 1.3) -> tuple[bool, int, bool]:
    """
    Performs a series of checks on an XYZ file to validate its molecular structure.

    This includes checking for zero coordinates, graph connectivity, and
    optionally, the degree of specific 'connector' nodes.

    Args:
        filename (str): The path to the XYZ file to be checked.
        connector_dicts (dict, optional): A dictionary where keys are node indices
            and values are lists of expected degrees for those nodes. Used to
            validate connectivity at specific points in the molecule. Defaults to None.
        scale_factor (float, optional): The scaling factor for covalent radii in edge correction. Defaults to 1.3.

    Returns:
        tuple[bool, int, bool]: A tuple containing:
            - is_connected (bool): True if the molecule's graph is fully connected.
            - num_components (int): The number of connected components in the graph.
            - match_n_degree (bool): True if all specified connector nodes have
              degrees matching their expected values in `connector_dicts`, False otherwise.
    """
    cartesian_coordinates_tensor, atomic_numbers_tensor = read_xyz_file(filename)

    if torch.all(cartesian_coordinates_tensor == 0):
        print(f"Error: All coordinates in {filename} are zero.")
        return False, 100, False

    mol_data = create_pyg_graph(cartesian_coordinates_tensor, atomic_numbers_tensor, xyz_filename=filename)

    mol_data = correct_edges(mol_data, scale_factor=scale_factor)

    num_node = mol_data.num_nodes
    edge_index = mol_data.edge_index

    is_connected, num_components = is_fully_connected(edge_index, num_node)

    match_n_degree = True
    if connector_dicts:
        for node_idx in range(num_node):
            if node_idx in connector_dicts:
                adjacent_nodes = edge_index[1][edge_index[0] == node_idx].tolist()

                node_degree = len(adjacent_nodes)

                if node_degree not in connector_dicts[node_idx]:
                    print(f"Error: Node {node_idx} has {node_degree} neighbors, expected {connector_dicts[node_idx]} in {filename}.")
                    match_n_degree = False
                    break

    return is_connected, num_components, match_n_degree


def optimize_molecule(filename: str, charge: int, level: str, timeout: int) -> str | None:
    """
    Optimizes the geometry of a molecule from an XYZ file using xTB or OpenBabel.

    This function attempts to optimize the molecule with a specified charge and
    calculation level. If the optimization is successful, it moves the
    output file to a new name based on the input filename.

    Args:
        filename (str): The path to the input XYZ file.
        charge (int): The molecular charge to use for the calculation (xTB only).
        level (str): The calculation level (e.g., "gfn1", "gfn2", "gfn-ff", "mmff94").
        timeout (int): The maximum time in seconds to wait for the process to complete.

    Returns:
        str | None: The path to the optimized XYZ file if successful,
                    otherwise None if it times out or fails to produce output.
    """
    base_filename = os.path.basename(filename).split(".")[0]
    optimized_filename = f"{base_filename}_opt.xyz"

    if level == "mmff94":
        # simple check for obminimize
        if shutil.which("obminimize") is None:
             print("Error: obminimize not found in PATH.")
             return None
        # obminimize, by default, outputs PDB format. We need to convert this to XYZ.
        # We chain obminimize -> obabel -ipdb -oxyz
        
        try:
            # 1. Run obminimize
            min_command = ["obminimize", "-ff", "MMFF94", "-n", "2500", filename]
            min_process = sp.Popen(min_command, stdout=sp.PIPE, stderr=sp.PIPE, text=True)
            min_stdout, min_stderr = min_process.communicate(timeout=timeout)
            
            if min_process.returncode != 0:
                 # Check if the failure is due to force field setup
                 if "could not setup force field" in min_stderr or "COULD NOT FIND" in min_stderr:
                     print(f"obminimize failed for {filename} (Force Field Error):")
                     print(min_stderr.strip())
                 else:
                     print(f"obminimize failed for {filename} with return code {min_process.returncode}:")
                     print(min_stderr.strip())
                 return None

            # 2. Convert output to XYZ using obabel
            # Note: obminimize output is in PDB format (HETATM...)
            obabel_command = ["obabel", "-i", "pdb", "-o", "xyz"]
            obabel_process = sp.Popen(obabel_command, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, text=True)
            xyz_stdout, xyz_stderr = obabel_process.communicate(input=min_stdout, timeout=timeout)
            
            if obabel_process.returncode != 0:
                print(f"obabel conversion failed for {filename}:")
                print(xyz_stderr)
                return None
                
            # Write converted stdout to file
            with open(optimized_filename, "w") as f:
                f.write(xyz_stdout)
            
            # Check if file is not empty and looks like XYZ
            if os.path.getsize(optimized_filename) > 0:
                 return optimized_filename
            else:
                 print(f"Optimization produced empty output for {filename}.")
                 return None

        except sp.TimeoutExpired:
            print(f"Optimization timed out for {filename}.")
            if os.path.exists(optimized_filename):
                 os.remove(optimized_filename)
            return None
        except Exception as e:
            print(f"Optimization failed for {filename}: {e}")
            if os.path.exists(optimized_filename):
                 os.remove(optimized_filename)
            return None

    else:
        # xTB optimization
        execution_command = ["xtb", filename, "--opt", "crude", "-c", str(charge), f"-{level}"]
    
        try:
            sp.call(execution_command, stdout=sp.DEVNULL, stderr=sp.STDOUT, timeout=timeout)
        except sp.TimeoutExpired:
            print(f"xTB optimization timed out for {filename}.")
            return None
    
        if os.path.exists("xtbopt.xyz"):
            shutil.move("xtbopt.xyz", optimized_filename)
            return optimized_filename
        else:
            print(f"xTB optimization failed to produce output for {filename}.")
            return None


def get_xtb_optimized_xyz(
    input_directory: str,
    output_directory: str = None,
    charge: int = -1,
    level: str = "gfn1",
    timeout: int = 240,
    scale_factor: float = 1.3,
    optimize_all: bool = True,
    csv_path: str = None,
    filter_column: str = None
) -> list[str]:
    """
    Optimizes all XYZ files in a given input directory using xTB or OpenBabel and saves them
    to an output directory.

    This function iterates through all `.xyz` files, performs initial structural
    checks (connectivity, zero coordinates, and optional degree checks), and then
    attempts to optimize valid structures using `optimize_molecule`. It skips files
    that already have an optimized counterpart in the output directory.

    Args:
        input_directory (str): The path to the directory containing the input XYZ files.
        output_directory (str, optional): The path to the directory where optimized
            XYZ files will be saved. If None, optimized files are saved in the
            `input_directory`. Defaults to None.
        charge (int, optional): The molecular charge to use for xTB optimizations. Defaults to -1.
        level (str, optional): The calculation level (e.g., "gfn1", "gfn2", "gfn-ff", "mmff94"). Defaults to "gfn1".
        timeout (int, optional): The maximum time in seconds to wait for each xTB process. Defaults to 240.
        scale_factor (float, optional): The scaling factor for covalent radii in edge correction. Defaults to 1.3.
        optimize_all (bool, optional): If True, optimizes all files regardless of existing optimized versions.
        csv_path (str, optional): Path to a CSV file to filter which XYZ files to optimize.
        filter_column (str, optional): The column name in the CSV to filter by (values must be 1).

    Returns:
        list[str]: A list of paths to the successfully optimized XYZ files.
    """
    if output_directory is None:
        output_directory = os.path.join(input_directory, "optimized_xyz")

    os.makedirs(output_directory, exist_ok=True)

    xyz_files = []
    if csv_path:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        fname_col = None
        for col in ["xyz_file", "filename", "filepath"]:
            if col in df.columns:
                fname_col = col
                break
        
        if fname_col is None:
             raise ValueError("CSV must contain 'xyz_file', 'filename', or 'filepath' column.")
        
        if filter_column:
            if filter_column not in df.columns:
                raise ValueError(f"Filter column '{filter_column}' not found in CSV.")
            # Filter rows where the value is 1 (as integer or string)
            filtered_df = df[df[filter_column].isin(['1', '1.0', True, 1])]
        else:
            filtered_df = df

        for _, row in filtered_df.iterrows():
            fname = str(row[fname_col])
            # Handle potential missing extension if it's just a name
            if not fname.lower().endswith('.xyz'):
                 fname += '.xyz'
            
            if os.path.isabs(fname):
                full_path = fname
            else:
                full_path = os.path.join(input_directory, fname)
            
            if os.path.exists(full_path):
                xyz_files.append(full_path)
            else:
                print(f"Warning: File from CSV not found: {full_path}")
                
    else:
        xyz_files = glob.glob(os.path.join(input_directory, "*.xyz"))

    optimized_files = []

    for xyz_file in tqdm(xyz_files, desc="Optimizing XYZ files", total=len(xyz_files)):
        output_file_path = os.path.join(output_directory, os.path.basename(xyz_file[:-4] + "_opt.xyz"))

        if os.path.exists(output_file_path):
            print(f"Skipping {xyz_file} as {output_file_path} already exists.")
            continue

        is_connected, num_components, match_n_degree = check_xyz(xyz_file, scale_factor=scale_factor)
        is_neutral = check_neutrality(xyz_file, charge=charge, timeout=timeout)    
        good_xyz =is_neutral and is_connected and (num_components == 1) and match_n_degree

        if good_xyz or optimize_all:
            optimized_file_basename = optimize_molecule(xyz_file, charge, level, timeout)
            if optimized_file_basename is not None:
                shutil.move(optimized_file_basename, output_file_path)
                optimized_files.append(output_file_path)
            else:
                print(f"Optimization failed for {xyz_file}.")
        else:
            print(f"Error: {xyz_file} failed initial structural checks.")

    return optimized_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize XYZ files using xTB.")
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        required=True,
        help="Input directory containing XYZ files."
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for optimized XYZ files. Defaults to input directory if not provided."
    )
    parser.add_argument(
        "--charge",
        "-c",
        type=int,
        default=-1,
        help="Molecular charge for xTB optimization. Defaults to -1."
    )
    parser.add_argument(
        "--level",
        "-l",
        type=str,
        default="gfn1",
        help="Calculation level (e.g., 'gfn1', 'gfn2', 'gfn-ff', 'mmff94'). Defaults to 'gfn1'."
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=240,
        help="Maximum time in seconds for xTB processes to complete. Defaults to 240."
    )
    parser.add_argument(
        "--scale_factor",
        "-s",
        type=float,
        default=1.3,
        help="Scaling factor for covalent radii in edge correction. Defaults to 1.3."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Path to CSV file for filtering which files to optimize."
    )
    parser.add_argument(
        "--filter_column",
        type=str,
        default=None,
        help="Column name in CSV to filter by (values must be 1 to process)."
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir if args.output_dir is not None else os.path.join(input_dir, "optimized_xyz")

    optimized_files = get_xtb_optimized_xyz(
        input_dir,
        output_dir,
        charge=args.charge,
        level=args.level,
        timeout=args.timeout,
        scale_factor=args.scale_factor,
        csv_path=args.csv_path,
        filter_column=args.filter_column
    )

    print(f"Successfully optimized {len(optimized_files)} XYZ files and saved them in '{output_dir}'.")
