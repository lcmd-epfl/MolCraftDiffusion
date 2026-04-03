"""Training command for MolCraft CLI.

Adapted from scripts/train.py for package-level execution.
"""

from typing import Any, Dict, Optional, Tuple
import math
import os
import pickle
import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from MolecularDiffusion.core import Engine
from MolecularDiffusion.runmodes.train import (
    evaluate,
    DataModule,
    Logger,
    OptimSchedulerFactory,
    get_versioned_output_path,
)
from MolecularDiffusion.runmodes.train.eval import _resolve_task_config
from MolecularDiffusion.utils import (
    RankedLogger,
    task_wrapper,
    seed_everything,
)

log = RankedLogger(__name__, rank_zero_only=True)



def is_rank_zero():
    """Check if current process is rank zero."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def load_weights(task, ckpt_path, task_module=None):
    """Load model weights from a checkpoint file (weights only).

    This loads the state_dict from the checkpoint into the task model,
    ignoring optimizer/scheduler states and other metadata.
    Useful for fine-tuning or starting from a pre-trained model.

    If a RuntimeError occurs due to size mismatches (e.g., from adding
    new conditions), delegates to task_module.adjust_state_dict() for
    model-specific dimension adjustment.

    Args:
        task: The task model to load weights into.
        ckpt_path: Path to the checkpoint file.
        task_module: Optional task factory with adjust_state_dict() method
                     for handling dimension mismatches.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    log.info(f"Loading weights from: {ckpt_path}")

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Validate task_type if both checkpoint and task_module expose it
    if task_module is not None and hasattr(task_module, "task_type"):
        ckpt_task_type = (
            checkpoint.get("hyperparameters", {}).get("task_type")
            or checkpoint.get("task_type")
        )
        if ckpt_task_type is not None and ckpt_task_type != task_module.task_type:
            raise ValueError(
                f"Task type mismatch: checkpoint was trained as '{ckpt_task_type}' "
                f"but current config specifies '{task_module.task_type}'. "
                f"Update your config to use tasks: {ckpt_task_type} or point to the correct checkpoint."
            )
    # Extract the model state dict — original engine uses "ema_model"/"model" keys,
    if "ema_model" in checkpoint or "model" in checkpoint:
        # Original engine format: prefer EMA weights (matches engine.load() behaviour)
        raw_state_dict = checkpoint.get("ema_model") or checkpoint.get("model")
        cleaned_state_dict = raw_state_dict
    else:
        state_dict = checkpoint.get("state_dict", {})
        cleaned_state_dict = {}
        ema_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("task.ema_model."):
                ema_state_dict[key[len("task.ema_model."):]] = value
            elif key.startswith("ema_model."):
                ema_state_dict[key[len("ema_model."):]] = value
            elif key.startswith("task."):
                cleaned_state_dict[key[5:]] = value
            else:
                cleaned_state_dict[key] = value
        # Prefer EMA weights when available
        if ema_state_dict:
            log.info(f"Found {len(ema_state_dict)} EMA parameters in Lightning checkpoint, using them")
            cleaned_state_dict = ema_state_dict

    if not cleaned_state_dict:
        raise ValueError(f"Could not extract a model state dict from checkpoint: {ckpt_path}")

    # Detect size mismatches before loading by comparing shapes
    model_state = task.state_dict()
    mismatched = [
        k for k in cleaned_state_dict
        if k in model_state and cleaned_state_dict[k].shape != model_state[k].shape
    ]
    if mismatched:
        if task_module is not None and hasattr(task_module, 'adjust_state_dict'):
            log.warning(f"{len(mismatched)} size mismatch(es) detected, delegating to task module for adjustment")
            cleaned_state_dict = task_module.adjust_state_dict(cleaned_state_dict, task)
            mismatched = [
                k for k in cleaned_state_dict
                if k in model_state and cleaned_state_dict[k].shape != model_state[k].shape
            ]
            if mismatched:
                raise RuntimeError(
                    f"Size mismatch persists after adjust_state_dict for keys: {mismatched[:5]}"
                    f"{'...' if len(mismatched) > 5 else ''}. "
                    f"Check that your model config matches the checkpoint architecture."
                )
        else:
            raise RuntimeError(
                f"{len(mismatched)} tensor(s) have mismatched shapes between the checkpoint and "
                f"the current model (e.g. {mismatched[0]}: ckpt={cleaned_state_dict[mismatched[0]].shape} "
                f"vs model={model_state[mismatched[0]].shape}). "
                f"Ensure your model config matches the checkpoint, or use chkpt_path instead of "
                f"load_weights_from to let the task factory handle dimension adjustment."
            )

    missing, unexpected = task.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        log.warning(f"Missing keys when loading weights: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        log.warning(f"Unexpected keys in checkpoint: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    log.info(f"Successfully loaded {len(cleaned_state_dict)} parameters into task.")

    # Store EMA state for later initialization by EngineLightning.on_train_start
    if locals().get("ema_state_dict"):
        task._pending_ema_state = ema_state_dict
        log.info(f"Stored {len(ema_state_dict)} EMA parameters for deferred loading")





def evaluate_and_save(i, solver, task_module, trainer_module, logger_module, versioned_ckpt_path, use_amp, **kwargs):
    """Run evaluation for any task type — routing is handled by TASK_REGISTRY inside evaluate()."""
    output_generated_dir = os.path.join(versioned_ckpt_path, "generated_molecules")
    os.makedirs(output_generated_dir, exist_ok=True)
    return evaluate(
        task_module.task_type, solver, i,
        kwargs.get("best_metrics", torch.inf), kwargs.get("best_checkpoints", []),
        logger_module.logger,
        output_path=versioned_ckpt_path,
        output_generated_dir=output_generated_dir,
        use_amp=use_amp,
        precision=trainer_module.precision,
        generative_analysis=kwargs.get("generative_analysis", False),
        n_samples=kwargs.get("n_samples", 100),
        metric=kwargs.get("metric", "Validity Relax and connected"),
        use_posebuster=kwargs.get("use_posebuster", False),
        batch_size=kwargs.get("batch_size", 1),
        save_top_k=getattr(trainer_module, "save_top_k", 3),
        save_every_val_epoch=getattr(trainer_module, "save_every_val_epoch", False),
    )


def engine_wrapper(task_module, data_module, trainer_module, logger_module,
                   resume_from_checkpoint=None, tags=None, **kwargs):
    """Training loop using original Engine."""
    trainer_module.get_optimizer()
    trainer_module.get_scheduler()
    
    solver = Engine(
        task_module.task,
        data_module.train_set,
        data_module.valid_set,
        data_module.test_set,
        batch_size=data_module.batch_size,
        collate_fn=data_module.collate_fn,
        optimizer=trainer_module.optimizer,
        ema_decay=trainer_module.ema_decay,
        scheduler=trainer_module.scheduler,
        clipping_gradient=trainer_module.gradient_clip_mode,
        clip_value=trainer_module.gradnorm_queue,
        logger=logger_module.logger,
        log_interval=logger_module.log_interval,
        name_wandb=logger_module.name_wandb,
        project_wandb=logger_module.project_wandb,
        dir_wandb=trainer_module.output_path,
        tags_wandb=tags,
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_from_checkpoint:
        start_epoch = solver.resume(resume_from_checkpoint, strict=False)
        log.info(f"Resumed from epoch {start_epoch}")
    
    use_amp = trainer_module.precision in ["bf16", 16]
    
    best_checkpoints = []
    if hasattr(task_module.task, "sample") and kwargs.get("generative_analysis"):
        models_to_save = {"node": task_module.task.node_dist_model}
        if len(getattr(task_module, "condition_names", [])) > 0:
            models_to_save["prop"] = task_module.task.prop_dist_model
        if is_rank_zero():
            with open(os.path.join(trainer_module.output_path, "edm_stat.pkl"), "wb") as f:
                pickle.dump(models_to_save, f)

    # Determine correct initial sentinel from registry + engine overrides
    _eval_cfg = _resolve_task_config(task_module.task_type)
    if kwargs.get("eval_higher_is_better") is not None:
        from dataclasses import replace as _dcreplace
        _eval_cfg = _dcreplace(_eval_cfg, higher_is_better=kwargs["eval_higher_is_better"])
    best_metrics = -torch.inf if _eval_cfg.higher_is_better else torch.inf
    
    # Create versioned checkpoint folder (like Lightning's version_X folders)
    versioned_ckpt_path = get_versioned_output_path(trainer_module.output_path)
    
    # Adjust loop to continue from start_epoch
    for i in range(start_epoch, trainer_module.num_epochs):
        metric = solver.train(num_epoch=1, num_step=trainer_module.num_steps, use_amp=use_amp, precision=trainer_module.precision)
        
        # Check if we should stop because num_steps was reached
        if trainer_module.num_steps is not None and solver.meter.batch_id >= trainer_module.num_steps:
            log.info(f"Terminating training loop after epoch {i} because num_steps={trainer_module.num_steps} was reached.")
            # Trigger final evaluation if not already done
            if i % trainer_module.validation_interval != 0:
                best_metrics, best_checkpoints = evaluate_and_save(
                    i, solver, task_module, trainer_module, logger_module, 
                    versioned_ckpt_path, use_amp, best_metrics=best_metrics, 
                    best_checkpoints=best_checkpoints, **kwargs
                )
            break

        if i % trainer_module.validation_interval == 0 or i == trainer_module.num_epochs - 1:
            best_metrics, best_checkpoints = evaluate_and_save(
                i, solver, task_module, trainer_module, logger_module, 
                versioned_ckpt_path, use_amp, best_metrics=best_metrics, 
                best_checkpoints=best_checkpoints, **kwargs
            )
    return best_metrics, solver


def _assert_train_config(cfg: DictConfig):
    """Crash early if the user passed a non-train config to the train command."""
    has_interference = "interference" in cfg
    has_chkpt_dir = "chkpt_directory" in cfg
    has_trainer = "trainer" in cfg

    if has_interference or has_chkpt_dir:
        mismatched = []
        if has_interference:
            mismatched.append("'interference'")
        if has_chkpt_dir:
            mismatched.append("'chkpt_directory'")
        raise ValueError(
            f"Config contains {', '.join(mismatched)} — this looks like a generation config, "
            f"not a training config. Did you mean: molcraft generate <config>?"
        )
    if not has_trainer:
        raise ValueError(
            "Config is missing required 'trainer' block. "
            "Please provide a training config (e.g., configs/train.yaml)."
        )


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Main training function."""
    _assert_train_config(cfg)
    output_path = cfg.trainer.output_path
    os.makedirs(output_path, exist_ok=True)

    if is_rank_zero():
        config_path = os.path.join(output_path, "config.yaml")
        with open(config_path, "w") as f:
            OmegaConf.save(config=cfg, f=f)
        log.info(f"Configuration saved to {config_path}")
    
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    data_module: DataModule = hydra.utils.instantiate(cfg.data, task_type=cfg.tasks.task_type)
    data_module.load()
    
    log.info(f"Instantiating task <{cfg.tasks._target_}>")
    data_point_chk = data_module.train_set[0]
    node_feature_0 = getattr(data_point_chk, "node_feature", None)
    if node_feature_0 is not None:
        n_dim = node_feature_0.shape[1]
    else:
        try:
            node_feature_0 = getattr(data_point_chk, "x", None)
            n_dim = node_feature_0.shape[1]
        except:
            n_dim = 0

    factory_cfg = cfg.tasks
    overrides = {}
    
    if "tasks_egt" in factory_cfg._target_ or "tasks_esen" in factory_cfg._target_ or "diffusion_tabasco" in factory_cfg._target_:
        overrides["train_set"] = data_module.train_set
        if "condition_names" in factory_cfg:
            overrides["task_names"] = factory_cfg.condition_names
            
    if "atom_vocab" in cfg.data:
        overrides["atom_vocab"] = list(cfg.data.atom_vocab)
        
    if cfg.data.get("allow_unknown", False):
        overrides["atom_vocab"].append("Suisei")
    
    if cfg.tasks.get("metrics", None) == "valid_posebuster":
        overrides["use_posebuster"] = True
        try:
            import posebusters
        except ImportError:
            log.warning("PoseBuster not installed. Falling back to 'Validity Relax and connected'.")
            overrides["use_posebuster"] = False
            overrides["metrics"] = ["Validity Relax and connected"]
        
    task_module = hydra.utils.instantiate(factory_cfg, **overrides)
    task_module.build()
    
    # Optional: Load weights from checkpoint (without resuming full state)
    if cfg.trainer.get("load_weights_from"):
        load_weights(task_module.task, cfg.trainer.load_weights_from, task_module=task_module)
    
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer_module: OptimSchedulerFactory = hydra.utils.instantiate(
        cfg.trainer, parameters=task_module.task.parameters()
    )

    name_wandb = trainer_module.output_path.split('/')[-1] if "/" in trainer_module.output_path else trainer_module.output_path
    log.info(f"Instantiating loggers... <{cfg.logger._target_}>")
    logger_module: Logger = hydra.utils.instantiate(cfg.logger, name_wandb=name_wandb)
    
    object_dict = {
        "cfg": cfg,
        "datamodule": data_module,
        "task": task_module,
        "trainer": trainer_module,
        "logger": logger_module,
    }

    log.info("Logging hyperparameters!")
    log_hyperparameters(object_dict)
    
    engine_type = cfg.get("engine", {}).get("engine_type", "original")
    log.info(f"Using engine: {engine_type}")

    # Optional eval overrides from engine config
    engine_cfg = cfg.get("engine", {})
    eval_metric_key = engine_cfg.get("eval_metric_key", None)
    eval_higher_is_better = engine_cfg.get("eval_higher_is_better", None)

    # Extract top-level tags for wandb
    tags = cfg.get("tags", None)
    if tags is not None:
        try:
            from omegaconf import OmegaConf as _OC
            tags = _OC.to_container(tags, resolve=True)
        except Exception:
            tags = list(tags)

    # Extract generative parameters with fallbacks
    tasks_cfg = cfg.get("tasks", {})
    gen_analysis = cfg.get("generative_analysis", tasks_cfg.get("generative_analysis", False))
    n_samples = cfg.get("n_samples", tasks_cfg.get("n_samples", 100))
    metric = cfg.get("metrics", cfg.get("metric", tasks_cfg.get("metrics", "Validity Relax and connected")))
    use_posebuster = cfg.get("use_posebuster", tasks_cfg.get("use_posebuster", False))
    # Preference: cli/top-level -> tasks -> data -> default
    gen_batch_size = cfg.get("batch_size", tasks_cfg.get("batch_size", cfg.data.get("batch_size", 100)))


    resume_ckpt = cfg.trainer.get("resume_from_checkpoint", None)
    metrics = engine_wrapper(
        task_module, data_module, trainer_module, logger_module,
        resume_from_checkpoint=resume_ckpt,
        generative_analysis=gen_analysis,
        n_samples=n_samples,
        metric=metric,
        use_posebuster=use_posebuster,
        batch_size=gen_batch_size,
        tags=tags,
        eval_metric_key=eval_metric_key,
        eval_higher_is_better=eval_higher_is_better,
    )

    return metrics, object_dict


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
                        if isinstance(v, torch.nn.Module):
                            log.info(f"{k}: {v.__class__.__name__}")
                        else:
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


def train_main(cfg: DictConfig):
    """Entry point for CLI train command."""
    metric, _ = train(cfg)
    return metric
