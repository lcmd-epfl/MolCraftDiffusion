"""Eval-Predict command for MolCraft CLI.

Adapted from scripts/eval_predict.py for package-level execution.
"""

import os
from typing import Any, Dict, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset

from MolecularDiffusion.core import Engine
from MolecularDiffusion.runmodes.train import DataModule, ModelTaskFactory_EGCL, OptimSchedulerFactory
from MolecularDiffusion.utils import RankedLogger, seed_everything
from MolecularDiffusion.utils.plot_function import (
    plot_kde_distribution,
    plot_histogram_distribution,
    plot_kde_distribution_multiple,
    plot_correlation_with_histograms,
)

log = RankedLogger(__name__, rank_zero_only=True)


def is_rank_zero():
    """Check if current process is rank zero."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def engine_wrapper(task_module, data_module, trainer_module):
    """Run evaluation with Engine."""
    trainer_module.get_optimizer()
    trainer_module.get_scheduler()

    pred_dataset = ConcatDataset([data_module.valid_set, data_module.test_set])
    solver = Engine(
        task_module.task,
        None,
        None,
        pred_dataset,
        batch_size=data_module.batch_size,
        collate_fn=data_module.collate_fn,
        logger="logging",
    )

    _, preds_test, targets_test = solver.evaluate("test")
    y_preds = torch.cat(preds_test, dim=0)
    y_trues = torch.cat(targets_test, dim=0)
    return y_preds, y_trues


def predict(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluate predictions on validation/test sets."""
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    data_module: DataModule = hydra.utils.instantiate(
        cfg.data, task_type=cfg.tasks.task_type, train_ratio=0
    )
    data_module.load()
    
    log.info(f"Instantiating task <{cfg.tasks._target_}>")
    act_fn = hydra.utils.instantiate(cfg.tasks.act_fn)
    task_module: ModelTaskFactory_EGCL = hydra.utils.instantiate(cfg.tasks, act_fn=act_fn)
    task_module.build()
    
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer_module: OptimSchedulerFactory = hydra.utils.instantiate(
        cfg.trainer, parameters=task_module.task.parameters()
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": data_module,
        "task": task_module,
        "trainer": trainer_module,
    }

    log.info("Logging hyperparameters!")
    log_hyperparameters(object_dict)

    y_preds, y_trues = engine_wrapper(task_module, data_module, trainer_module)
    
    df = pd.read_csv(cfg.data.filename)
    task_matrix = df[cfg.tasks.task_learn].to_numpy()
    filenames = df["filename"].to_numpy()
    filenames_aligned = []
    
    for row in y_trues.cpu().numpy():
        mask = np.all(np.isclose(task_matrix, row, atol=1e-4), axis=1)
        idx = np.flatnonzero(mask)
        
        if idx.size == 0:
            raise ValueError(f"No match for row {row}")
        if idx.size > 1:
            raise ValueError(f"Multiple matches for row {row}: {filenames[idx].tolist()}")

        filenames_aligned.append(filenames[idx[0]])
    
    df_compiled = pd.DataFrame({
        "filename": filenames_aligned,
        "y_true": y_trues.cpu().numpy().tolist(),
        "y_pred": y_preds.cpu().numpy().tolist(),
    })

    os.makedirs(cfg.output_directory, exist_ok=True)
    df_compiled.to_csv(f"{cfg.output_directory}/predictions.csv", index=False)
    
    log.info("Prediction statistics:")
    for task_name in cfg.tasks.task_learn:
        log.info(f"--- {task_name} ---")
        log.info(f"Mean: {df[task_name].mean():.4f}")
        log.info(f"Std: {df[task_name].std():.4f}")
        log.info(f"Min: {df[task_name].min():.4f}")
        log.info(f"Max: {df[task_name].max():.4f}")

    log.info("Plotting distributions...")
    props = []
    for i, prop in enumerate(cfg.tasks.task_learn):
        plot_kde_distribution(df[prop], prop, f"{cfg.output_directory}/{prop}_kde.png")
        plot_histogram_distribution(df[prop], prop, f"{cfg.output_directory}/{prop}_hist.png")
        plot_correlation_with_histograms(
            y_trues[:, i].cpu().numpy(),
            y_preds[:, i].cpu().numpy(),
            prop,
            "",
            f"{cfg.output_directory}/{prop}_correlation.png",
        )
        props.append(df[prop].values)

    props = np.array(props).T
    plot_kde_distribution_multiple(props, cfg.tasks.task_learn, f"{cfg.output_directory}/kde_all.png")


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


def eval_predict_main(cfg: DictConfig):
    """Entry point for CLI eval-predict command."""
    predict(cfg)
