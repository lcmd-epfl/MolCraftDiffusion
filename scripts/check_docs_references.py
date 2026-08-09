#!/usr/bin/env python3
"""Validate concrete YAML paths referenced by user-facing documentation."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

# Deliberately requires a concrete filename. Placeholders such as <name>, globs,
# and user-created my_*.yaml examples are not repository-owned references.
CONFIG_REFERENCE = re.compile(
    r"(?P<path>"
    r"(?:docs/cfg_examples|src/MolecularDiffusion/configs|configs)/"
    r"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.ya?ml"
    r")"
)


def documentation_sources() -> list[Path]:
    """Return maintained user-facing Markdown sources."""
    sources = [ROOT / "README.md"]
    for path in DOCS.rglob("*.md"):
        relative = path.relative_to(DOCS)
        if relative.parts[0] in {"build", "model_integrations"}:
            continue
        if relative == Path("adding_new_models.md"):
            continue
        sources.append(path)
    return sources


def main() -> int:
    """Report missing concrete YAML references and return a CI exit code."""
    missing: list[tuple[Path, int, str]] = []

    for source in documentation_sources():
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for match in CONFIG_REFERENCE.finditer(line):
                reference = match.group("path")
                if Path(reference).name.startswith("my_"):
                    continue
                if not (ROOT / reference).is_file():
                    missing.append((source.relative_to(ROOT), line_number, reference))

    if missing:
        print("Documentation config reference check failed:")
        for source, line_number, reference in missing:
            print(f"  {source}:{line_number}: missing {reference}")
        return 1

    print("All concrete YAML references in user-facing documentation resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
