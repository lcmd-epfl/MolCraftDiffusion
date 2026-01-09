"""Hydra configuration utilities for CLI.

Provides utilities for discovering and loading bundled configs 
while allowing user configs to reference them.
"""

import os
from pathlib import Path
from typing import Optional, List
from importlib import resources


def get_package_config_path() -> Path:
    """Get the absolute path to bundled config directory.
    
    Returns:
        Path to the configs directory within the installed package.
    """
    # Use importlib.resources for Python 3.9+
    try:
        # For Python 3.9+
        pkg_files = resources.files("MolecularDiffusion")
        config_path = pkg_files / "configs"
        # Convert to real path (handles both installed and editable installs)
        if hasattr(config_path, '_path'):
            # Traversable from importlib.resources
            real_path = Path(str(config_path))
        else:
            real_path = Path(config_path)
        if real_path.is_dir():
            return real_path
    except (TypeError, AttributeError, Exception):
        pass
    
    # Fallback: relative to this module
    module_dir = Path(__file__).parent.parent
    config_path = module_dir / "configs"
    if config_path.is_dir():
        return config_path
    
    raise FileNotFoundError(
        "Could not find bundled configs. Ensure package is installed correctly."
    )


def setup_hydra_config(
    config_name: str,
    config_dir: Optional[str] = None,
    overrides: Optional[List[str]] = None,
):
    """Setup Hydra configuration with proper search paths.
    
    Configures Hydra to search:
    1. User's config_dir (if provided) or current directory
    2. Package bundled configs (via searchpath)
    
    Args:
        config_name: Name of the config file (without .yaml extension)
        config_dir: Optional user config directory
        overrides: Optional list of Hydra override strings
        
    Returns:
        DictConfig from Hydra
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    
    # Get package config path for defaults
    pkg_config_path = get_package_config_path()
    
    # Determine primary config directory
    # If config_name contains a path (e.g., "configs/train.yaml"), extract the directory
    config_name_path = Path(config_name)
    if config_name_path.parent != Path("."):
        # Config name includes directory, use that as config_dir
        if config_dir is None:
            config_dir = str(config_name_path.parent)
        config_name = config_name_path.name
    
    if config_dir:
        primary_config_dir = os.path.abspath(config_dir)
    else:
        primary_config_dir = os.getcwd()
    
    # Clear any existing Hydra state
    GlobalHydra.instance().clear()
    
    # Initialize with the primary config directory
    initialize_config_dir(
        config_dir=primary_config_dir,
        version_base="1.3",
    )
    
    # Build overrides to include searchpath for bundled configs
    all_overrides = overrides or []
    
    # Add package config path to searchpath using file:// protocol
    # This allows Hydra to find bundled defaults like data/mol_dataset.yaml
    searchpath_override = f"hydra.searchpath=[file://{pkg_config_path}]"
    all_overrides = [searchpath_override] + all_overrides
    
    # Handle config name (strip .yaml if present)
    if config_name.endswith(".yaml"):
        config_name = config_name[:-5]
    
    # Compose the configuration
    cfg = compose(config_name=config_name, overrides=all_overrides)
    
    return cfg


def run_hydra_app(
    config_name: str,
    task_function,
    config_dir: Optional[str] = None,
    overrides: Optional[List[str]] = None,
):
    """Run a Hydra-based task function with proper config setup.
    
    This is the main entry point for CLI commands that use Hydra configs.
    
    Args:
        config_name: Name of the config file
        task_function: Function to call with the composed config
        config_dir: Optional user config directory
        overrides: Optional Hydra overrides
    """
    cfg = setup_hydra_config(config_name, config_dir, overrides)
    return task_function(cfg)
