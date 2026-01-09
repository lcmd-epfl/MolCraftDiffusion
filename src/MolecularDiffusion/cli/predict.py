"""Prediction command for MolCraft CLI.

Adapted from scripts/predict.py for package-level execution.
"""

import os
from glob import glob
from typing import Any, Dict, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from ase.data import atomic_numbers
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, radius_graph
from tqdm import tqdm

from MolecularDiffusion.core import Engine
from MolecularDiffusion.data.component.pointcloud import PointCloud_Mol
from MolecularDiffusion.data.component.feature import (
    onehot,
    atom_topological,
    atom_geom,
    atom_geom_compact,
    atom_geom_opt,
    atom_geom_v2,
    atom_geom_v2_trun,
)
from MolecularDiffusion.utils import RankedLogger, seed_everything
from MolecularDiffusion.utils.plot_function import (
    plot_kde_distribution,
    plot_histogram_distribution,
    plot_kde_distribution_multiple,
)

log = RankedLogger(__name__, rank_zero_only=True)


def is_rank_zero():
    """Check if current process is rank zero."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def load_model(chkpt_path):
    """Load a pre-trained model from checkpoint."""
    engine = Engine(None, None, None, None, None)
    engine = engine.load_from_checkpoint(chkpt_path, interference_mode=True)
    engine.model.eval()
    return engine


def xyz2mol(xyz_file, atom_vocab, node_feature, edge_type="fully_connected", 
            radius=4.0, n_neigh=5, device="cpu"):
    """Convert an XYZ file into a PyTorch Geometric Data object."""
    mol_obj = {}
    mol_xyz = PointCloud_Mol.from_xyz(xyz_file, with_hydrogen=True, forbidden_atoms=[])
    coords = mol_xyz.get_coord()
    n_nodes = len(mol_xyz.atoms)

    node_features = []
    for atom in mol_xyz.atoms:
        node_features.append(onehot(atom.element, atom_vocab, allow_unknown=False))
    
    charges = [
        atomic_numbers[atom.element]
        for atom in mol_xyz.atoms
        if atom.element in atomic_numbers
    ]

    if node_feature:
        if node_feature in [
            "atom_topological", "atom_geom", "atom_geom_v2",
            "atom_geom_v2_trun", "atom_geom_opt", "atom_geom_compact"
        ]:
            feature_mapping = {
                "atom_topological": atom_topological,
                "atom_geom": atom_geom,
                "atom_geom_v2": atom_geom_v2,
                "atom_geom_v2_trun": atom_geom_v2_trun,
                "atom_geom_opt": atom_geom_opt,
                "atom_geom_compact": atom_geom_compact,
            }
            feature_function = feature_mapping.get(node_feature)
            if feature_function is not None:
                node_features_extra = feature_function(charges, coords)
            node_features = torch.cat(
                [torch.tensor(node_features), node_features_extra], dim=1
            )
        else:
            raise ValueError("Unknown node feature type")
    else:
        node_features = torch.tensor(node_features, dtype=torch.float32)
    
    node_features = torch.tensor(node_features, dtype=torch.float32)
    charges = torch.as_tensor(charges, dtype=torch.long)
    node_mask = torch.ones(n_nodes, dtype=torch.int8)

    edge_mask = node_mask.unsqueeze(0) * node_mask.unsqueeze(1)
    diag_mask = ~torch.eye(n_nodes, dtype=torch.bool)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(1 * n_nodes * n_nodes, 1)
    h = node_features.view(1 * n_nodes, -1).clone()
    
    if edge_type == "distance":
        edge_index = radius_graph(coords, r=radius)
    elif edge_type == "neighbor":
        edge_index = knn_graph(coords, k=n_neigh)
    elif edge_type == "fully_connected":
        num_nodes = coords.size(0)
        row = torch.arange(num_nodes).repeat_interleave(num_nodes)
        col = torch.arange(num_nodes).repeat(num_nodes)
        edge_index = torch.stack([row, col], dim=0)
        edge_index = edge_index[:, row != col]
    else:
        raise ValueError(f"Unknown edge type {edge_type}")
    
    graph_data = Data(
        x=h,
        pos=coords,
        atomic_numbers=charges,
        natoms=torch.tensor([n_nodes]),
        edge_index=edge_index,
        times=torch.tensor([0]),
        batch=torch.zeros(n_nodes, dtype=torch.long),
    ).to(device)
    
    mol_obj["graph"] = graph_data
    return mol_obj


def count_atoms_from_xyz(path: str) -> int:
    """Fast atom counter for XYZ files."""
    try:
        with open(path, "r") as f:
            first = f.readline().strip()
            return int(first)
    except Exception:
        return 0


def _runner(solver, xyz_paths: list, max_atoms: int = 100):
    """Run predictions on XYZ files."""
    device = solver.model.device
    task_names = list(solver.model.task.keys())
    num_molecules = len(xyz_paths)

    progress_bar = tqdm(
        enumerate(xyz_paths),
        desc="Predicting molecules",
        leave=True,
        dynamic_ncols=True,
        total=num_molecules,
    )

    predictions = []
    xyz_paths_clear = []
    skipped = 0
    
    for i, xyz_path in progress_bar:
        n_atoms = count_atoms_from_xyz(xyz_path)
        if n_atoms > max_atoms:
            skipped += 1
            progress_bar.set_postfix({"batch": i + 1, "skipped": skipped})
            log.info(f"Skipping {xyz_path} (atoms={n_atoms} > max_atoms={max_atoms})")
            continue

        mol_obj = xyz2mol(
            xyz_file=xyz_path,
            atom_vocab=solver.model.atom_vocab,
            node_feature=solver.model.node_feature,
            device=device,
        )
        prediction = solver.model.predict(mol_obj, evaluate=True)[0]
        predictions.append(prediction.detach().cpu().numpy())
        current_preds_dict = {prop_name: prediction[j].item() for j, prop_name in enumerate(task_names)}
        progress_bar.set_postfix({"batch": i + 1, "skipped": skipped, **current_preds_dict})
        xyz_paths_clear.append(xyz_path)

    predictions = np.array(predictions)
    if predictions.ndim > 1 and predictions.shape[-1] == 1:
        predictions = predictions.squeeze(-1)
    
    return predictions, xyz_paths_clear


def runner(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Property prediction run."""
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating diffusion task and loading the model <{cfg.tasks._target_}>")
    solver = load_model(cfg.chkpt_directory)
   
    task_names = list(solver.model.task.keys())
    
    if not hasattr(solver.model, 'std'):
        chkpt = torch.load(cfg.chkpt_directory, weights_only=False)
        solver.model.std = chkpt["model"]["std"].to(solver.model.device)
        solver.model.weight = chkpt["model"]["weight"].to(solver.model.device)
        solver.model.mean = chkpt["model"]["mean"].to(solver.model.device)
        
    if not hasattr(solver.model, 'atom_vocab'):
        solver.model.atom_vocab = cfg.atom_vocab
    if not hasattr(solver.model, 'node_feature'):
        solver.model.node_feature = cfg.node_feature
    
    object_dict = {"cfg": cfg, "solver": solver}

    log.info("Logging hyperparameters!")
    log_hyperparameters(object_dict)
    
    os.makedirs(cfg.output_directory, exist_ok=True)

    if is_rank_zero():
        config_path = os.path.join(cfg.output_directory, "config.yaml")
        with open(config_path, "w") as f:
            OmegaConf.save(config=cfg, f=f)
        log.info(f"Configuration saved to {config_path}")
    
    log.info("Running the predictions...")
    xyz_paths = glob(f"{cfg.xyz_directory}/*.xyz")
    xyz_paths = [str(xyz_path) for xyz_path in xyz_paths]
    predictions, xyz_paths_clear = _runner(solver, xyz_paths, max_atoms=cfg.get("max_atoms", 100))

    df_dicts = {}
    for task_name, prediction in zip(task_names, predictions.T):
        df_dicts[task_name] = prediction
    df_dicts["xyz_path"] = xyz_paths_clear

    df = pd.DataFrame(df_dicts)
    df = df.sort_values(by="xyz_path")
    df.to_csv(f"{cfg.output_directory}/predictions.csv", index=False)

    log.info("Prediction statistics:")
    for task_name in task_names:
        log.info(f"--- {task_name} ---")
        log.info(f"Mean: {df[task_name].mean():.4f}")
        log.info(f"Std: {df[task_name].std():.4f}")
        log.info(f"Min: {df[task_name].min():.4f}")
        log.info(f"Max: {df[task_name].max():.4f}")

    log.info("Plotting distributions...")
    props = []
    for prop in task_names:
        plot_kde_distribution(df[prop], prop, f"{cfg.output_directory}/{prop}_kde.png")
        plot_histogram_distribution(df[prop], prop, f"{cfg.output_directory}/{prop}_hist.png")
        props.append(df[prop].values)
    
    props = np.array(props).T
    plot_kde_distribution_multiple(props, task_names, f"{cfg.output_directory}/kde_all.png")


def log_hyperparameters(object_dict: dict):
    """Log hyperparameters for debugging."""
    if not is_rank_zero():
        return

    log.info("\n========== Logging Hyperparameters ==========\n")
    for name, obj in object_dict.items():
        log.info(f"{'=' * 20} {name.upper()} {'=' * 20}")
        if name == "cfg":
            if isinstance(obj, dict):
                log.info("\n" + OmegaConf.to_yaml(OmegaConf.create(obj)))
            else:
                log.info("\n" + OmegaConf.to_yaml(obj))
        else:
            if hasattr(obj, '__dict__'):
                for k, v in vars(obj).items():
                    if not k.startswith("_"):
                        log.info(f"{k}: {v}")
        log.info(f"{'=' * (44 + len(name))}\n")

    if "task" in object_dict and hasattr(object_dict["task"], "task"):
        model = object_dict["task"].task
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"{'=' * 20} MODEL PARAMS {'=' * 20}")
        log.info(f"model/params/total: {total}")
        log.info(f"model/params/trainable: {trainable}")
        log.info("=" * 54 + "\n")
    
    log.info("========== End of Hyperparameters ==========\n")


def predict_main(cfg: DictConfig):
    """Entry point for CLI predict command."""
    runner(cfg)
