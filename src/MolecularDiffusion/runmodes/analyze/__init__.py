"""Analysis utilities for 3D molecular structures.

This module provides tools for:
- XTB geometry optimization
- XTB electronic property computation
- Validity/connectivity metrics
- Energy and RMSD computation
- Bond/angle/torsion analysis
- XYZ to SMILES conversion
"""

__all__ = [
    "optimize_molecule",
    "get_xtb_optimized_xyz",
    "compute_xtb_electronic",
    "batch_xtb_electronic",
    "run_xyz2mol"
]


def __getattr__(name: str):
    if name in {"optimize_molecule", "get_xtb_optimized_xyz"}:
        from . import xtb_optimization

        value = getattr(xtb_optimization, name)
    elif name in {"compute_xtb_electronic", "batch_xtb_electronic"}:
        from . import xtb_electronic

        value = getattr(xtb_electronic, name)
    elif name == "run_xyz2mol":
        from .xyz2mol import run_processing as value
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
