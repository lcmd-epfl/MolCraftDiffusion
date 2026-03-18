"""
MolecularDiffusion - A unified generative AI framework for 3D molecular generation.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("molcraftdiffusion")
except PackageNotFoundError:
    __version__ = "unknown"

__author__ = "Thanapat Worakul"
__email__ = "thanapat.worakul@epfl.ch"

# Submodules are loaded lazily to avoid import errors when optional
# dependencies (xyz2mol, xtb, openbabel) are not installed.
_submodules = ["core", "data", "modules", "utils", "callbacks", "runmodes"]


def __getattr__(name: str):
    if name in _submodules:
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = _submodules
