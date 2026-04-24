"""Helpers for optional dependency error messages."""

from __future__ import annotations

import importlib
from collections.abc import Iterable


class OptionalDependencyError(ImportError):
    """Raised when an optional dependency group is required but unavailable."""


def install_hint(extra: str, missing: Iterable[str] | None = None) -> str:
    """Build a concise installation hint for an optional dependency group."""
    missing_list = sorted({dep for dep in (missing or []) if dep})
    missing_text = f" Missing package(s): {', '.join(missing_list)}." if missing_list else ""
    return (
        f"Optional MolCraftDiffusion '{extra}' dependencies are required for this command."
        f"{missing_text} Install them with: pip install 'molcraftdiffusion[{extra}]'"
    )


def require_modules(extra: str, modules: Iterable[str]) -> None:
    """Import-check modules and raise a grouped optional dependency error."""
    missing: list[str] = []
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise OptionalDependencyError(install_hint(extra, missing))


def optional_import_error(extra: str, error: ImportError) -> OptionalDependencyError:
    """Convert an import error into an actionable optional-extra message."""
    missing = [error.name] if isinstance(error, ModuleNotFoundError) and error.name else []
    optional_error = OptionalDependencyError(install_hint(extra, missing))
    optional_error.__cause__ = error
    return optional_error
