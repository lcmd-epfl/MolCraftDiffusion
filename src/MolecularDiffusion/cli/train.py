"""Training command for MolCraft CLI.

Adapted from scripts/train.py for package-level execution.
"""

from typing import Any, Dict,  Tuple
import os
import pickle

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


def load_weights(task, ckpt_path):
    """Load model weights from a checkpoint file (weights only).
    
    This loads the state_dict from the checkpoint into the task model,
    ignoring optimizer/scheduler states and other metadata.
    Useful for fine-tuning or starting from a pre-trained model.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
        
    log.info(f"Loading weights from: {ckpt_path}")
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    # Prepare state dict for loading
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("task."):
            cleaned_state_dict[key[5:]] = value
        else:
            cleaned_state_dict[key] = value
            
    # Load into task
    missing, unexpected = task.load_state_dict(cleaned_state_dict, strict=False)
    
    if len(missing) > 0:
        log.warning(f"Missing keys when loading weights: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if len(unexpected) > 0:
        log.warning(f"Unexpected keys in checkpoint: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
        
    log.info(f"Successfully loaded {len(cleaned_state_dict)} parameters into task.")




def engine_wrapper(task_module, data_module, trainer_module, logger_module, 
                   resume_from_checkpoint=None, **kwargs):
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
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_from_checkpoint:
        start_epoch = solver.resume(resume_from_checkpoint, strict=False)
        log.info(f"Resumed from epoch {start_epoch}")
    
    use_amp = trainer_module.precision in ["bf16", 16]
    
    best_checkpoints = []
    best_checkpoints = []
    if hasattr(task_module.task, "sample") and kwargs.get("generative_analysis"):
        best_metrics = -torch.inf
        models_to_save = {"node": task_module.task.node_dist_model}
        if len(getattr(task_module, "condition_names", [])) > 0:
            models_to_save["prop"] = task_module.task.prop_dist_model
        if is_rank_zero():
            with open(os.path.join(trainer_module.output_path, "edm_stat.pkl"), "wb") as f:
                pickle.dump(models_to_save, f)
    else:
        best_metrics = torch.inf
    
    # Create versioned checkpoint folder (like Lightning's version_X folders)
    versioned_ckpt_path = get_versioned_output_path(trainer_module.output_path)
    
    # Adjust loop to continue from start_epoch
    for i in range(start_epoch, trainer_module.num_epochs):
        solver.train(num_epoch=1, use_amp=use_amp, precision=trainer_module.precision)
        if i % trainer_module.validation_interval == 0 or i == trainer_module.num_epochs - 1:
            if hasattr(task_module.task, "sample"):
                output_generated_dir = os.path.join(versioned_ckpt_path, "generated_molecules")
                os.makedirs(output_generated_dir, exist_ok=True)
                best_metrics, best_checkpoints = evaluate(
                    task_module.task_type, solver, i, best_metrics, best_checkpoints,
                    logger_module.logger, output_generated_dir=output_generated_dir,
                    generative_analysis=kwargs.get("generative_analysis", False),
                    n_samples=kwargs.get("n_samples", 100),
                    metric=kwargs.get("metric", "Validity Relax and connected"),
                    output_path=versioned_ckpt_path,
                    use_amp=use_amp, precision=trainer_module.precision,
                    use_posebuster=kwargs.get("use_posebuster", False),
                    batch_size=kwargs.get("batch_size", 1),
                    save_top_k=getattr(trainer_module, "save_top_k", 3),
                    save_every_val_epoch=getattr(trainer_module, "save_every_val_epoch", False),
                )
            else:
                best_metrics, best_checkpoints = evaluate(
                    task_module.task_type, solver, i, best_metrics, best_checkpoints,
                    logger_module.logger, output_path=versioned_ckpt_path,
                    save_top_k=getattr(trainer_module, "save_top_k", 3),
                    save_every_val_epoch=getattr(trainer_module, "save_every_val_epoch", False),
                )
    return best_metrics, solver


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Main training function."""
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
        load_weights(task_module.task, cfg.trainer.load_weights_from)
    
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
    

    resume_ckpt = cfg.trainer.get("resume_from_checkpoint", None)
    if hasattr(task_module.task, "sample"):
        metrics = engine_wrapper(
            task_module, data_module, trainer_module, logger_module,
            resume_from_checkpoint=resume_ckpt,
            generative_analysis=cfg.tasks.generative_analysis,
            n_samples=cfg.tasks.n_samples,
            metric=cfg.tasks.metrics,
            use_posebuster=cfg.tasks.use_posebuster,
            batch_size=cfg.tasks.batch_size,
        )
    else:
        metrics = engine_wrapper(
            task_module, data_module, trainer_module, logger_module,
            resume_from_checkpoint=resume_ckpt,
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
