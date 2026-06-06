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


def _get_ckpt_meta(checkpoint):
    """Extract task_type and condition_names from any checkpoint format."""
    hparams = (
        checkpoint.get("hyperparameters")
        or checkpoint.get("hyper_parameters")
        or {}
    )
    task_type = hparams.get("task_type") or checkpoint.get("task_type")
    condition_names = hparams.get("condition_names") or checkpoint.get("condition_names")
    return task_type, condition_names


def _validate_task_type(checkpoint, expected_task_type):
    """Raise ValueError if checkpoint task_type doesn't match expected_task_type."""
    if expected_task_type is None:
        return
    ckpt_task_type, _ = _get_ckpt_meta(checkpoint)
    if ckpt_task_type is not None and ckpt_task_type != expected_task_type:
        raise ValueError(
            f"Task type mismatch: checkpoint was trained as '{ckpt_task_type}' "
            f"but current config specifies '{expected_task_type}'. "
            f"Update your config to use tasks: {ckpt_task_type} or point to the correct checkpoint."
        )


def _stamp_condition_names(task, checkpoint):
    """Stamp condition_names from checkpoint onto the task object."""
    _, condition_names = _get_ckpt_meta(checkpoint)
    if condition_names is not None:
        task.condition = condition_names
        log.info(f"Loaded condition_names from checkpoint: {condition_names}")


def _select_ckpt_file(path):
    """Resolve a checkpoint path that may be a .ckpt file or a directory.

    For a directory, pick the highest-metric checkpoint (by filename), else last.ckpt,
    else the first .ckpt found — mirroring load_model's selection logic.
    """
    if os.path.isfile(path):
        return path
    ckpt_files = glob.glob(os.path.join(path, '*.ckpt'))
    if not ckpt_files:
        raise FileNotFoundError(f"No .ckpt file found at {path}")
    best_metric = -1.0
    best = None
    for f in ckpt_files:
        m = re.search(r"(?:metric|val)[_=](\d+\.?\d*)", os.path.basename(f))
        if m:
            v = float(m.group(1))
            if v > best_metric:
                best_metric = v
                best = f
    if best is None:
        last = os.path.join(path, 'last.ckpt')
        best = last if os.path.exists(last) else ckpt_files[0]
    return best


def _task_config_defines_atom_vocab(task_config):
    if task_config is None:
        return False
    try:
        vocab = task_config.get("atom_vocab", None)
    except Exception:
        return False
    return vocab is not None and vocab != "???"


def _extract_clean_state_dict(ckpt, prefer_ema=True):
    """Return a {name: tensor} dict from a checkpoint, normalized to bare module names.

    Weight preference (when prefer_ema): new-format top-level `ema_model_state_dict`, then
    old-format `ema_model.*` keys embedded inside `state_dict`, then raw weights. Strips a
    leading `task.` and any `ema_model.` prefix; the raw fallback drops embedded EMA keys so
    they don't collide with the real weights.
    """
    # New-format top-level EMA dict (keys already unprefixed).
    if prefer_ema:
        ema = ckpt.get("ema_model_state_dict") or ckpt.get("ema_model")
        if ema:
            return {k[len("task."):] if k.startswith("task.") else k: v for k, v in ema.items()}

    sd = ckpt.get("state_dict") or ckpt.get("model")
    if sd is None:
        raise KeyError("Checkpoint has no loadable state dict (state_dict/ema_model_state_dict).")

    # Old-format EMA embedded as `...ema_model.<name>` keys inside state_dict.
    if prefer_ema:
        marker = "ema_model."
        ema_keys = [k for k in sd if marker in k]
        if ema_keys:
            return {k[k.index(marker) + len(marker):]: sd[k] for k in ema_keys}

    # Raw weights — drop any embedded EMA keys so they don't shadow the real ones.
    cleaned = {}
    for k, v in sd.items():
        if "ema_model." in k:
            continue
        cleaned[k[len("task."):] if k.startswith("task.") else k] = v
    return cleaned


def _apply_total_step_override(task, total_step):
    """Override the model's sampling-step count (T / fm_num_timesteps / interpolant)."""
    if total_step <= 0:
        return
    override_occured = False
    if hasattr(task, 'model'):
        m_obj = task.model
        if hasattr(m_obj, 'T'):
            log.info(f"Overriding diffusion steps (T): {m_obj.T} -> {total_step}")
            m_obj.T = total_step
            override_occured = True
        elif hasattr(m_obj, 'fm_num_timesteps'):
            log.info(f"Overriding flow-matching steps: {m_obj.fm_num_timesteps} -> {total_step}")
            m_obj.fm_num_timesteps = total_step
            override_occured = True
    if hasattr(task, 'T') and not override_occured:
        task.T = total_step
        override_occured = True
    if hasattr(task, 'interpolant') and hasattr(task.interpolant, 'num_timesteps'):
        task.interpolant.num_timesteps = total_step


def load_lightning_model(chkpt_path, task_config, atom_vocab=None, total_step=0):
    """Load model from Lightning checkpoint (.ckpt)."""
    log.info(f"Loading Lightning checkpoint from: {chkpt_path}")

    expected_task_type = task_config.get("task_type") if task_config is not None else None

    try:
        from MolecularDiffusion.core.engine_lightning import EngineLightning

        # Validate before full model reconstruction
        raw_ckpt = torch.load(chkpt_path, map_location="cpu", weights_only=False)
        _validate_task_type(raw_ckpt, expected_task_type)

        wrapper = EngineLightning.load_from_checkpoint(chkpt_path, map_location="cpu", strict=False)
        log.info("Successfully loaded model using EngineLightning.load_from_checkpoint")
        
        if atom_vocab and hasattr(wrapper.task, 'atom_vocab') and wrapper.task.atom_vocab is None:
            wrapper.task.atom_vocab = atom_vocab
        
        # Apply diffusion_steps override from config
        if total_step > 0:
            override_occured = False
            
            # 1. Override main task
            t_obj = wrapper.task
            if hasattr(t_obj, 'model'):
                m_obj = t_obj.model
                if hasattr(m_obj, 'T'):
                    log.info(f"Overriding main task diffusion steps (T): {m_obj.T} -> {total_step}")
                    m_obj.T = total_step
                    override_occured = True
                elif hasattr(m_obj, 'fm_num_timesteps'):
                    log.info(f"Overriding main task flow matching steps: {m_obj.fm_num_timesteps} -> {total_step}")
                    m_obj.fm_num_timesteps = total_step
                    override_occured = True
            
            if hasattr(t_obj, 'T') and not override_occured:
                log.info(f"Overriding main task diffusion steps (T): {t_obj.T} -> {total_step}")
                t_obj.T = total_step
                override_occured = True
                
            # 2. Handle LDMTask (Latent Diffusion)
            if hasattr(t_obj, 'interpolant') and hasattr(t_obj.interpolant, 'num_timesteps'):
                log.info(f"Overriding LDM interpolant steps: {t_obj.interpolant.num_timesteps} -> {total_step}")
                t_obj.interpolant.num_timesteps = total_step
                override_occured = True

            # 3. Override EMA model if present
            if wrapper.ema_model is not None:
                ema_t_obj = wrapper.ema_model
                ema_override = False
                if hasattr(ema_t_obj, 'model'):
                    ema_m_obj = ema_t_obj.model
                    if hasattr(ema_m_obj, 'T'):
                        log.info(f"Overriding EMA model diffusion steps (T): {ema_m_obj.T} -> {total_step}")
                        ema_m_obj.T = total_step
                        ema_override = True
                if not ema_override and hasattr(ema_t_obj, 'T'):
                    ema_t_obj.T = total_step
                if hasattr(ema_t_obj, 'interpolant') and hasattr(ema_t_obj.interpolant, 'num_timesteps'):
                    ema_t_obj.interpolant.num_timesteps = total_step

            if not override_occured:
                log.warning(f"Failed to find any attribute (T, fm_num_timesteps, interpolant.num_timesteps) to override with total_step={total_step}")
             
        _stamp_condition_names(wrapper.task, raw_ckpt)
        if 'reference_indices' in raw_ckpt:
            wrapper.task.reference_indices = raw_ckpt['reference_indices']
        if 'reference_freeze_mode' in raw_ckpt:
            wrapper.task.reference_freeze_mode = raw_ckpt['reference_freeze_mode']
        if 'reference_feature_stats' in raw_ckpt:
            wrapper.task.reference_feature_stats = raw_ckpt['reference_feature_stats']
        if 'reference_scaffold' in raw_ckpt:
            wrapper.task.reference_scaffold = raw_ckpt['reference_scaffold']
        wrapper.task.eval()
        return wrapper.task

    except Exception as e:
        log.warning(f"EngineLightning.load_from_checkpoint failed ({type(e).__name__}: {e}). Falling back to manual config reconstruction.")
    
    # Fallback: Load checkpoint manually
    checkpoint = torch.load(chkpt_path, map_location="cpu", weights_only=False)
    _validate_task_type(checkpoint, expected_task_type)

    hparams = checkpoint.get("hyper_parameters", {})
    if "model_config" in hparams and hparams["model_config"] is not None:
        task_config = OmegaConf.create(hparams["model_config"])
        log.info("Loaded task configuration from checkpoint hyperparameters")
    elif task_config is None:
        raise ValueError("task_config not provided and 'model_config' not found in checkpoint.")

    task_config = copy.deepcopy(task_config)
    OmegaConf.set_readonly(task_config, False)
    OmegaConf.set_struct(task_config, False)

    n_types = len(atom_vocab) if atom_vocab else 0
    
    if OmegaConf.is_missing(task_config, "num_atom_types") or task_config.get("num_atom_types") == "???":
        task_config.num_atom_types = n_types if n_types > 0 else 100

    # Ensure use_unknown_fallback is consistently applied during reconstruction
    ckpt_fallback = hparams.get("use_unknown_fallback")
    if ckpt_fallback is not None:
        task_config.use_unknown_fallback = ckpt_fallback
        log.info(f"Setting use_unknown_fallback={ckpt_fallback} from checkpoint hparams")
    elif task_config.get("use_unknown_fallback") is True:
        log.info("Using use_unknown_fallback=True from provided task config")

    if hasattr(task_config, "transformer_config"):
        if OmegaConf.is_missing(task_config.transformer_config, "atom_dim"):
            task_config.transformer_config.atom_dim = task_config.num_atom_types

    if hasattr(task_config, "dataset_stats"):
        if OmegaConf.is_missing(task_config.dataset_stats, "max_atoms"):
            task_config.dataset_stats.max_atoms = 150

    log.info(f"Building task from config: {task_config._target_}")
    instantiate_kwargs = {}
    if atom_vocab is not None and not _task_config_defines_atom_vocab(task_config):
        instantiate_kwargs["atom_vocab"] = atom_vocab
    task_factory = hydra.utils.instantiate(task_config, **instantiate_kwargs)
    task = task_factory.build()
    
    state_dict = checkpoint.get('state_dict', {})
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('task.'):
            cleaned_state_dict[key[5:]] = value
        else:
            cleaned_state_dict[key] = value
    
    task.load_state_dict(cleaned_state_dict, strict=False)
    log.info(f"Loaded {len(cleaned_state_dict)} parameters from checkpoint")
    _stamp_condition_names(task, checkpoint)

    if 'data_stats' in checkpoint:
        task.tabasco_model.set_data_stats(checkpoint['data_stats'])
    if 'node_dist_model' in checkpoint:
        task._node_dist_model = checkpoint['node_dist_model']
    if 'prop_dist_model' in checkpoint:
        task.prop_dist_model = checkpoint['prop_dist_model']
    if 'reference_indices' in checkpoint:
        task.reference_indices = checkpoint['reference_indices']
    if 'reference_freeze_mode' in checkpoint:
        task.reference_freeze_mode = checkpoint['reference_freeze_mode']
    if 'reference_feature_stats' in checkpoint:
        task.reference_feature_stats = checkpoint['reference_feature_stats']
    if 'reference_scaffold' in checkpoint:
        task.reference_scaffold = checkpoint['reference_scaffold']
    
    if total_step > 0:
        override_occured = False
        if hasattr(task, 'model'):
            m_obj = task.model
            if hasattr(m_obj, 'T'):
                m_obj.T = total_step
                override_occured = True
            elif hasattr(m_obj, 'fm_num_timesteps'):
                m_obj.fm_num_timesteps = total_step
                override_occured = True
        
        if hasattr(task, 'T') and not override_occured:
            task.T = total_step
            override_occured = True
            
        if hasattr(task, 'interpolant') and hasattr(task.interpolant, 'num_timesteps'):
            task.interpolant.num_timesteps = total_step
    
    task.eval()
    return task


def load_model(chkpt_directory, task_config=None, atom_vocab=None, total_step=0, base_chkpt_path=None):
    """Load model from checkpoint directory with auto-detection.

    If the selected checkpoint is a LoRA-only delta (marked `lora_only`), `base_chkpt_path`
    (the pretrained checkpoint, file or directory) is required and the model is
    reconstructed as base weights + LoRA delta.
    """
    # chkpt_directory may be a directory (globbed) or a direct checkpoint file path.
    if os.path.isfile(chkpt_directory):
        ckpt_files = [chkpt_directory]
        sidecar_dir = os.path.dirname(chkpt_directory) or "."
    else:
        ckpt_files = []
        for pattern in ("*.ckpt", "*.pt", "*.pth"):
            ckpt_files.extend(glob.glob(os.path.join(chkpt_directory, pattern)))
        sidecar_dir = chkpt_directory

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
            last_ckpt = os.path.join(sidecar_dir, 'last.ckpt')
            best_checkpoint = last_ckpt if os.path.exists(last_ckpt) else ckpt_files[0]

        # Detect checkpoint format before loading. Legacy Engine checkpoints may use
        # .pt/.pth/.ckpt extensions but store top-level "hyperparameters"; load those
        # through Engine.load_from_checkpoint so architecture comes from the checkpoint.
        peek = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
        if "hyperparameters" in peek and ("model" in peek or "ema_model" in peek):
            log.info(f"Loading legacy engine checkpoint from: {best_checkpoint}")
            expected_task_type = task_config.get("task_type") if task_config is not None else None
            _validate_task_type(peek, expected_task_type)
            engine = Engine(None, None, None, None, None)
            engine = engine.load_from_checkpoint(best_checkpoint, interference_mode=True)
            task = engine.model
            _stamp_condition_names(task, peek)
            for attr in (
                "reference_indices",
                "reference_freeze_mode",
                "reference_feature_stats",
                "reference_scaffold",
            ):
                if attr in peek:
                    setattr(task, attr, peek[attr])
            _apply_total_step_override(task, total_step)
            task.eval()
        else:
            task = load_lightning_model(best_checkpoint, task_config, atom_vocab, total_step)

        try:
            with open(os.path.join(sidecar_dir, "edm_stat.pkl"), "rb") as file:
                edm_stats = pickle.load(file)
            task.node_dist_model = edm_stats.get("node")
            if "prop" in edm_stats:
                task.prop_dist_model = edm_stats["prop"]
            if "reference_freeze_mode" in edm_stats:
                task.reference_freeze_mode = edm_stats["reference_freeze_mode"]
            if "reference_feature_stats" in edm_stats:
                task.reference_feature_stats = edm_stats["reference_feature_stats"]
            if "reference_scaffold" in edm_stats:
                task.reference_scaffold = edm_stats["reference_scaffold"]
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

    # Validate task_type for original engine checkpoints
    expected_task_type = task_config.get("task_type") if task_config is not None else None
    raw_ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    _validate_task_type(raw_ckpt, expected_task_type)

    edm_stats = {
        "node": None,
        "prop": None,
        "reference_freeze_mode": None,
        "reference_feature_stats": None,
        "reference_scaffold": None,
    }
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
            for key in (
                "reference_freeze_mode",
                "reference_feature_stats",
                "reference_scaffold",
            ):
                if key in loaded_stats:
                    edm_stats[key] = loaded_stats[key]
        except Exception as e:
            log.warning(f"Failed to load edm_stat.pkl: {e}")
    
    engine = Engine(None, None, None, None, None)
    engine = engine.load_from_checkpoint(model_path, interference_mode=True)
    task = engine.model
    _stamp_condition_names(task, raw_ckpt)
    if 'reference_indices' in raw_ckpt:
        task.reference_indices = raw_ckpt['reference_indices']
    if 'reference_freeze_mode' in raw_ckpt:
        task.reference_freeze_mode = raw_ckpt['reference_freeze_mode']
    if 'reference_feature_stats' in raw_ckpt:
        task.reference_feature_stats = raw_ckpt['reference_feature_stats']
    if 'reference_scaffold' in raw_ckpt:
        task.reference_scaffold = raw_ckpt['reference_scaffold']

    if edm_stats["node"] is not None:
        task.node_dist_model = edm_stats["node"]
    if edm_stats["prop"] is not None:
        task.prop_dist_model = edm_stats["prop"]
    if edm_stats["reference_freeze_mode"] is not None:
        task.reference_freeze_mode = edm_stats["reference_freeze_mode"]
    if edm_stats["reference_feature_stats"] is not None:
        task.reference_feature_stats = edm_stats["reference_feature_stats"]
    if edm_stats["reference_scaffold"] is not None:
        task.reference_scaffold = edm_stats["reference_scaffold"]
    
    if total_step > 0:
        override_occured = False
        if hasattr(task, 'model'):
            m_obj = task.model
            if hasattr(m_obj, 'T'):
                m_obj.T = total_step
                override_occured = True
            elif hasattr(m_obj, 'fm_num_timesteps'):
                m_obj.fm_num_timesteps = total_step
                override_occured = True
        
        if hasattr(task, 'T') and not override_occured:
            task.T = total_step
            override_occured = True
            
        if hasattr(task, 'interpolant') and hasattr(task.interpolant, 'num_timesteps'):
            task.interpolant.num_timesteps = total_step

    task.eval()
    return task


def _assert_generate_config(cfg: DictConfig):
    """Crash early if the user passed a non-generation config to the generate command."""
    has_trainer = "trainer" in cfg
    has_interference = "interference" in cfg
    has_chkpt_dir = "chkpt_directory" in cfg

    if has_trainer:
        raise ValueError(
            "Config contains 'trainer' — this looks like a training config, "
            "not a generation config. Did you mean: molcraft train <config>?"
        )
    missing = []
    if not has_interference:
        missing.append("'interference'")
    if not has_chkpt_dir:
        missing.append("'chkpt_directory'")
    if missing:
        raise ValueError(
            f"Config is missing required block(s): {', '.join(missing)}. "
            "Please provide a generation config (e.g., configs/gen_config.yaml)."
        )


def generate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Main generation function."""
    _assert_generate_config(cfg)
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    # Reconcile diffusion_steps: Prefer Root config, then Tasks config
    diffusion_steps = cfg.get("diffusion_steps", 0)
    if (diffusion_steps == 0 or diffusion_steps == 900) and "diffusion_steps" in cfg.tasks:
        # Fallback to tasks config if root is default or zero
        diffusion_steps = cfg.tasks.diffusion_steps
        log.info(f"Using diffusion_steps from tasks config: {diffusion_steps}")

    task = load_model(
        cfg.chkpt_directory,
        task_config=cfg.tasks,
        atom_vocab=cfg.atom_vocab,
        total_step=diffusion_steps,
        base_chkpt_path=cfg.get("base_chkpt_path"),
    )
    
    if not hasattr(task, 'atom_vocab') or task.atom_vocab is None:
        task.atom_vocab = cfg.atom_vocab

    # Reconcile atom_vocab with the model's one-hot width. Models trained with
    # allow_unknown=True carry an extra "unknown" class (train.py appends 'Suisei'),
    # so the one-hot head is len(vocab)+1 wide. If the gen config's atom_vocab is one
    # short, decoding falls out of the one-hot path and uses the raw atomic-number
    # channel, which is often garbage (negative/huge Z) -> broken molecules. Append the
    # unknown token so decoding matches how the model was trained.
    model_obj = getattr(task, "model", None)
    ohe_width = getattr(model_obj, "n_core_ohe", None) or getattr(model_obj, "num_classes", None)
    if (
        ohe_width is not None
        and getattr(task, "atom_vocab", None) is not None
        and len(task.atom_vocab) == ohe_width - 1
        and getattr(model_obj, "use_unknown_fallback", False)
    ):
        task.atom_vocab = list(task.atom_vocab) + ["Suisei"]
        log.warning(
            f"atom_vocab had {ohe_width - 1} entries but the model's one-hot head is "
            f"{ohe_width}-wide (trained with allow_unknown). Appended 'Suisei' unknown "
            f"token so decoding uses the one-hot path instead of the raw atomic-number channel."
        )

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
        from omegaconf import open_dict
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
