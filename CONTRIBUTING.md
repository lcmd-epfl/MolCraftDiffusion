# Contributing to MolCraftDiffusion

Contributions are welcome, whether a bug report, a documentation fix, or a new
feature. Please open an issue before starting substantial work so that the
design can be agreed first.

## Development setup

```bash
conda create -n molcraft python=3.11 -y
conda activate molcraft
just dev            # pip install -e '.[dev]'
```

Python 3.10 to 3.13 are supported. Commands are collected in the `justfile`,
run `just --list` to see them all.

## Before opening a pull request

```bash
just check          # ruff check, ruff format --check, mypy, pytest, doctests
just fix            # auto-fix formatting and lint issues
```

`just check` must pass. It runs the test suite with coverage, so add a test
next to the behaviour you change under `tests/`, mirroring the package layout.

## Pull requests

- Branch from `main` and keep one logical change per pull request.
- Describe what changed and why, and link the issue it closes.
- Update the documentation under `docs/` when behaviour or CLI flags change.
- New public functions need a docstring and type annotations, since `mypy` and
  the documentation build both read them.
- Note in the description whether the change alters generated results, because
  reproducing published numbers depends on it.

## Reporting bugs

Open an issue with the bug report template and include the command you ran,
the configuration file, the full traceback, and your platform, Python version
and package version. A minimal example that reproduces the failure is the
fastest route to a fix.

## Licence

By contributing you agree that your contribution is licensed under the MIT
licence that covers this repository.
