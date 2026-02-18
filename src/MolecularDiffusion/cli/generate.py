"""Generation command for MolCraft CLI.

Adapted from scripts/generate.py for package-level execution.
"""

import glob
import os
import re
import time
import copy
import pickle
from typing import Any, Dict, Optional, Tuple

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from MolecularDiffusion.core import Engine
from MolecularDiffusion.runmodes.generate.tasks_generate import GenerativeFactory
from MolecularDiffusion.utils import (
    RankedLogger,
    seed_everything,
    recursive_module_to_device,
)

log = RankedLogger(__name__, rank_zero_only=True)


def is_rank_zero():
    """Check if current process is rank zero."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True

def load_model(chkpt_directory, task_config=None, atom_vocab=None, total_step=0):
    """Load model from checkpoint directory with auto-detection."""
    ckpt_files = glob.glob(os.path.join(chkpt_directory, '*.ckpt'))
    
    if ckpt_files:
        best_metric = -1.0
        best_checkpoint = None
        
        for ckpt_file in ckpt_files:
            match = re.search(r"(?:metric|val)[_=](\d+\.?\d*)", os.path.basename(ckpt_file))
            if match:
                metric = float(match.group(1))
                if metric > best_metric:
                    best_metric = metric
                    best_checkpoint = ckpt_file
        
        if best_checkpoint is None:
            last_ckpt = os.path.join(chkpt_directory, 'last.ckpt')
            best_checkpoint = last_ckpt if os.path.exists(last_ckpt) else ckpt_files[0]
        
        try:
            with open(os.path.join(chkpt_directory, "edm_stat.pkl"), "rb") as file:
                edm_stats = pickle.load(file)
            task.node_dist_model = edm_stats.get("node")
            if "prop" in edm_stats:
                task.prop_dist_model = edm_stats["prop"]
        except (ImportError, FileNotFoundError):
            log.warning("edm_stat.pkl not found")
        
        return task
    
    # Original engine (.pkl files)
    model_path = os.path.join(chkpt_directory, "edm_chem.pkl")
    
    if not os.path.exists(model_path):
        checkpoint_files = glob.glob(os.path.join(chkpt_directory, '*.pkl'))
        checkpoint_files = [f for f in checkpoint_files if 'edm_stat.pkl' not in os.path.basename(f)]

        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoints found in {chkpt_directory}")

        best_metric = -1.0
        best_checkpoint = None
        
        for ckpt_file in checkpoint_files:
            match = re.search(r"metric=([\d.]+)\.pkl", os.path.basename(ckpt_file))
            if match:
                metric = float(match.group(1))
                if metric > best_metric:
                    best_metric = metric
                    best_checkpoint = ckpt_file
        
        model_path = best_checkpoint or checkpoint_files[0]

    log.info(f"Loading original engine checkpoint from: {model_path}")
    
    edm_stats = {"node": None, "prop": None}
    stat_path = os.path.join(chkpt_directory, "edm_stat.pkl")
    if os.path.exists(stat_path):
        try:
            with open(stat_path, "rb") as file:
                loaded_stats = pickle.load(file)
            if "node" in loaded_stats:
                edm_stats["node"] = loaded_stats["node"]
            elif "node_dist_model" in loaded_stats:
                edm_stats["node"] = loaded_stats["node_dist_model"]
            if "prop" in loaded_stats:
                edm_stats["prop"] = loaded_stats["prop"]
            elif "prop_dist_model" in loaded_stats:
                edm_stats["prop"] = loaded_stats["prop_dist_model"]
        except Exception as e:
            log.warning(f"Failed to load edm_stat.pkl: {e}")
    
    engine = Engine(None, None, None, None, None)
    engine = engine.load_from_checkpoint(model_path, interference_mode=True)
    task = engine.model
    
    if edm_stats["node"] is not None:
        task.node_dist_model = edm_stats["node"]
    if edm_stats["prop"] is not None:
        task.prop_dist_model = edm_stats["prop"]
    
    if total_step > 0:
        if hasattr(task, 'model') and hasattr(task.model, 'T'):
            task.model.T = total_step
        elif hasattr(task, 'T'):
            task.T = total_step

    task.eval()
    return task


def generate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Main generation function."""
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating diffusion task and loading the model <{cfg.tasks._target_}>")
    task = load_model(
        cfg.chkpt_directory,
        task_config=cfg.tasks,
        atom_vocab=cfg.atom_vocab,
        total_step=cfg.diffusion_steps,
    )
    
    if not hasattr(task, 'atom_vocab') or task.atom_vocab is None:
        task.atom_vocab = cfg.atom_vocab
    
    if not hasattr(task, 'device'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        recursive_module_to_device(task, device)

    # --- Reconcile extra_norm_values between training config and loaded model ---
    # The generation config may default extra_norm_values=[] even though the
    # checkpoint was trained WITH extras (weights expanded via adjust_weights).
    # This causes in_node_nf mismatch: model thinks z is 21-d but network expects 26-d.
    if hasattr(task, 'model') and hasattr(task.model, 'ndim_extra'):
        model_extras = list(getattr(task.model, 'extra_norm_values', []) or [])
        # Try to discover the training extra_norm_values from the checkpoint directory
        train_extras = None
        train_cfg_path = os.path.join(cfg.chkpt_directory, '..', '..', 'config.yaml')
        if not os.path.exists(train_cfg_path):
            train_cfg_path = os.path.join(cfg.chkpt_directory, '..', 'config.yaml')
        if not os.path.exists(train_cfg_path):
            train_cfg_path = os.path.join(cfg.chkpt_directory, 'config.yaml')
        if os.path.exists(train_cfg_path):
            from omegaconf import OmegaConf as _OC
            try:
                train_cfg = _OC.load(train_cfg_path)
                _extras = train_cfg.get('tasks', {}).get('extra_norm_values', None)
                if _extras is not None and len(_extras) > 0:
                    train_extras = list(_extras)
            except Exception:
                pass

        if train_extras and len(train_extras) > len(model_extras):
            n_new = len(train_extras) - len(model_extras)
            log.warning(
                f"Patching model: extra_norm_values mismatch. "
                f"Model has {len(model_extras)} extras, training config has {len(train_extras)}. "
                f"Adding {n_new} extra dims to in_node_nf/num_classes."
            )
            task.model.extra_norm_values = tuple(train_extras)
            task.model.ndim_extra = len(train_extras)
            task.model.in_node_nf += n_new
            task.model.num_classes += n_new

    # Inject chkpt_directory into condition_configs for feature config discovery
    if "condition_configs" in cfg.interference:
        from omegaconf import OmegaConf, open_dict
        with open_dict(cfg.interference):
            cfg.interference.condition_configs["chkpt_directory"] = cfg.chkpt_directory

    log.info(f"Instantiating generator... <{cfg.interference._target_}>")
    generator: GenerativeFactory = hydra.utils.instantiate(cfg.interference, task=task)

    object_dict = {"cfg": cfg, "task": task, "generator": generator}

    log.info("Logging hyperparameters!")
    log_hyperparameters(object_dict)
    
    os.makedirs(cfg.interference.output_path, exist_ok=True)

    if is_rank_zero():
        config_path = os.path.join(cfg.interference.output_path, "config.yaml")
        with open(config_path, "w") as f:
            OmegaConf.save(config=cfg, f=f)
        log.info(f"Configuration saved to {config_path}")
    
    generator.run()


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
    log.info("========== End of Hyperparameters ==========\n")


def generate_main(cfg: DictConfig):
    """Entry point for CLI generate command."""
    start_time = time.time()
    generate(cfg)
    total_time = time.time() - start_time
    log.warning(f"Total time of execution: {total_time:.2f} seconds")
