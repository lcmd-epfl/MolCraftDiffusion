"""Zoo CLI: discover, fetch and register pretrained weights and datasets.

`MolCraftDiff zoo list` shows what exists, `fetch` pulls only what you ask
for, and `add` registers a newly integrated model so the zoo stays current
instead of turning back into a pile of scattered files.
"""

import difflib
import json
import logging
import shutil
from pathlib import Path

import click

from MolecularDiffusion import assets

logger = logging.getLogger(__name__)


def _human(size):
    value = float(size or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return (
                f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
            )
        value /= 1024
    return f"{value:.1f} TB"


def _models():
    return assets.manifest().get("models") or {}


def _asset_map():
    return assets.manifest().get("assets") or {}


def _assets_of(model_name, variant=None):
    """Every asset key a model (or one of its variants) needs."""
    model = _models().get(model_name)
    if model is None:
        close = ", ".join(sorted(_models())[:5])
        raise click.ClickException(
            f"Unknown model {model_name!r}. Try: MolCraftDiff zoo list"
            + (f"\nKnown: {close}" if close else "")
        )
    variants = model.get("variants") or {}
    if variant is not None:
        if variant not in variants:
            known = ", ".join(sorted(variants))
            raise click.ClickException(
                f"Unknown variant {variant!r} for {model_name!r}. "
                f"Known: {known}"
            )
        variants = {variant: variants[variant]}
    keys = []
    for spec in variants.values():
        for field in ("checkpoint", "data"):
            if spec.get(field):
                keys.append(spec[field])
        keys.extend((spec.get("extra") or {}).values())
    # dedupe, preserve order
    return list(dict.fromkeys(keys))


def _cached(key):
    return assets.local_path(key).exists()


@click.group()
def zoo():
    """Pretrained weights, datasets and example configs."""


@zoo.command("list")
@click.option("--tag", "-t", multiple=True, help="Filter by tag (repeatable)")
@click.option("--family", "-f", default=None, help="Filter by model family")
@click.option(
    "--fetched", is_flag=True, help="Only models already in the cache"
)
@click.option(
    "--json", "as_json", is_flag=True, help="Machine-readable output"
)
def list_cmd(tag, family, fetched, as_json):
    """List models in the zoo."""
    rows = []
    for name, model in sorted(_models().items()):
        tags = model.get("tags") or []
        if tag and not set(tag) <= set(tags):
            continue
        if family and model.get("family") != family:
            continue
        keys = _assets_of(name)
        have = all(_cached(k) for k in keys) if keys else False
        if fetched and not have:
            continue
        rows.append(
            {
                "name": name,
                "family": model.get("family", ""),
                "tags": tags,
                "variants": sorted(model.get("variants") or {}),
                "size": sum(assets.total_size(k) for k in keys),
                "cached": have,
            }
        )

    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    if not rows:
        click.echo("No models match. Cache root: " + str(assets.assets_root()))
        return

    width = max(len(r["name"]) for r in rows)
    click.echo(f"{'':2}{'MODEL':<{width}}  {'SIZE':>9}  FAMILY / TAGS")
    for row in rows:
        mark = "*" if row["cached"] else " "
        tags = ", ".join(row["tags"])
        variants = row["variants"]
        extra = f"  [{len(variants)} variants]" if len(variants) > 1 else ""
        click.echo(
            f"{mark:2}{row['name']:<{width}}  {_human(row['size']):>9}  "
            f"{row['family']}{extra}"
            + (f"\n{'':{width + 15}}{tags}" if tags else "")
        )
    click.echo(f"\n* = already in {assets.assets_root()}")


@zoo.command("info")
@click.argument("target")
def info_cmd(target):
    """Show details for MODEL or MODEL/VARIANT."""
    name, _, variant = target.partition("/")
    model = _models().get(name)
    if model is None:
        raise click.ClickException(
            f"Unknown model {name!r}. Try: MolCraftDiff zoo list"
        )

    click.echo(f"{name}  ({model.get('family', 'unknown family')})")
    if model.get("summary"):
        click.echo(f"\n  {' '.join(model['summary'].split())}\n")
    if model.get("tags"):
        click.echo(f"  tags     : {', '.join(model['tags'])}")
    if model.get("task_type"):
        click.echo(f"  task_type: {model['task_type']}")

    for vname, spec in sorted((model.get("variants") or {}).items()):
        if variant and vname != variant:
            continue
        click.echo(f"\n  variant: {vname}")
        keys = []
        for field in ("checkpoint", "data"):
            if spec.get(field):
                keys.append((field, spec[field]))
        for label, key in (spec.get("extra") or {}).items():
            keys.append((label, key))
        for label, key in keys:
            info = assets.entry(key)
            state = "cached" if _cached(key) else "not fetched"
            lic = info.get("license", "licence unstated")
            note = (
                "" if info.get("redistribute", True) else "  (build locally)"
            )
            click.echo(
                f"    {label:<16} {key:<24} "
                f"{_human(assets.total_size(key)):>9}  {state}  {lic}{note}"
            )
        if spec.get("configs"):
            click.echo(f"    {'configs':<16} {', '.join(spec['configs'])}")

    click.echo(f"\n  Fetch:  MolCraftDiff zoo fetch --model {name}")


@zoo.command("fetch")
@click.argument("names", nargs=-1)
@click.option(
    "--model",
    "-m",
    "model_name",
    default=None,
    help="Fetch every asset this model needs",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Fetch exactly what this config references",
)
@click.option("--all", "fetch_all", is_flag=True, help="Fetch everything")
@click.option("--force", is_flag=True, help="Redownload even if already valid")
@click.option("--dry-run", is_flag=True, help="Report size and licences only")
def fetch_cmd(names, model_name, config_path, fetch_all, force, dry_run):
    """Download assets into the cache.

    Nothing is downloaded unless you ask for it. Pick a granularity:
    a single asset, one model, everything one config needs, or --all.
    """
    keys = list(names)
    if model_name:
        keys += (
            _assets_of(*model_name.split("/", 1))
            if "/" in model_name
            else _assets_of(model_name)
        )
    if config_path:
        keys += _config_assets(Path(config_path))
    if fetch_all:
        keys += [k for k, v in _asset_map().items() if "alias" not in v]
    if not keys:
        raise click.ClickException(
            "Nothing selected. Give an asset name, or use --model / "
            "--config / --all."
        )

    # Resolve aliases so a shared corpus is counted and fetched once.
    resolved = []
    for key in keys:
        try:
            owner, _ = assets._resolve_key(key)  # noqa: SLF001
        except (KeyError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        if owner not in resolved:
            resolved.append(owner)

    total = sum(assets.total_size(k) for k in resolved)
    blocked = [
        k for k in resolved if assets.entry(k).get("redistribute") is False
    ]

    for key in resolved:
        info = assets.entry(key)
        state = "cached" if _cached(key) else "fetch"
        if info.get("redistribute") is False:
            state = "BUILD LOCALLY"
        click.echo(
            f"  {state:<14} {key:<26} {_human(assets.total_size(key)):>9}  "
            f"{info.get('license', 'licence unstated')}"
        )
    click.echo(f"\n  total: {_human(total)} across {len(resolved)} assets")

    if dry_run:
        click.echo("\nDry run. Re-run without --dry-run to download.")
        return
    if blocked:
        click.echo(
            "\nNot redistributable, so not downloaded: "
            + ", ".join(blocked)
            + "\nBuild each locally:  MolCraftDiff zoo recipe <asset>"
        )

    for key in resolved:
        if key in blocked:
            continue
        click.echo(f"fetching {key} ...")
        try:
            assets.fetch(key, force=force)
        except Exception as exc:
            raise click.ClickException(f"{key}: {exc}") from exc
    click.echo(f"done -> {assets.assets_root()}")


def _config_assets(path):
    """Every ${asset:...} reference in a config file."""
    import re  # noqa: PLC0415

    text = Path(path).read_text(encoding="utf-8")
    found = re.findall(r"\$\{asset:([^}]+)\}", text)
    if not found:
        raise click.ClickException(
            f"{path} contains no ${{asset:...}} references. "
            "It may be an unmigrated config using literal paths."
        )
    return found


@zoo.command("verify")
@click.argument("names", nargs=-1)
@click.option(
    "--all", "verify_all", is_flag=True, help="Verify everything cached"
)
def verify_cmd(names, verify_all):
    """Re-hash cached assets, and check the manifest is complete.

    Exits non-zero on any mismatch or any unaudited `redistribute: TODO`,
    so a half-registered model fails `just check` instead of being
    discovered a year later.
    """
    failures = 0

    for key, info in sorted(_asset_map().items()):
        if "alias" in info:
            continue
        if str(info.get("redistribute", "TODO")).upper() == "TODO":
            click.echo(f"  UNAUDITED  {key}: redistribute is still TODO")
            failures += 1
        if not info.get("license"):
            click.echo(f"  UNAUDITED  {key}: no license field")
            failures += 1

    targets = list(names)
    if verify_all or not targets:
        targets = [
            k
            for k, v in _asset_map().items()
            if "alias" not in v and assets.local_path(k).exists()
        ]

    for key in targets:
        for rel, status in assets.verify(key):
            if status == "ok":
                click.echo(f"  ok         {key}/{rel}")
            else:
                click.echo(f"  {status.upper():<10} {key}/{rel}")
                if status != "missing" or names:
                    failures += 1

    if failures:
        raise click.ClickException(f"{failures} problem(s) found")
    click.echo("all good")


@zoo.command("path")
@click.argument("name")
def path_cmd(name):
    """Print an asset's local path, for $(...) interop."""
    try:
        click.echo(assets.resolve(name))
    except (KeyError, FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@zoo.command("config")
@click.argument("name", required=False)
@click.argument("dest", type=click.Path(), default=".")
def config_cmd(name, dest):
    """Copy a bundled example config out so you can edit it.

    With no NAME, lists what is available. The bundled configs live inside
    the installed package, so copying one by hand means finding that
    directory first -- this does it for you. A config of the same name in
    your working directory takes precedence over the bundled one.
    """
    examples = Path(assets.__file__).parent / "configs" / "examples"
    if not examples.is_dir():
        raise click.ClickException(f"No bundled examples at {examples}")

    if not name:
        for f in sorted(examples.glob("*.yaml")):
            click.echo(f"  {f.name}")
        click.echo(f"\nCopy one:  MolCraftDiff zoo config <name> [dest]")
        return

    src = examples / (name if name.endswith(".yaml") else f"{name}.yaml")
    if not src.exists():
        near = difflib.get_close_matches(
            src.name, [f.name for f in examples.glob("*.yaml")], n=3
        )
        hint = f" Did you mean: {', '.join(near)}?" if near else ""
        raise click.ClickException(f"No bundled config {src.name!r}.{hint}")

    target = Path(dest)
    if target.is_dir():
        target = target / src.name
    if target.exists():
        raise click.ClickException(f"{target} already exists -- not overwriting")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)
    click.echo(f"copied -> {target}")
    click.echo(f"edit it, then:  MolCraftDiff generate {target}")


@zoo.command("recipe")
@click.argument("name")
def recipe_cmd(name):
    """Show how to build a non-redistributable asset locally."""
    try:
        info = assets.entry(name)
    except (KeyError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    up = info.get("upstream") or {}
    click.echo(f"{name}")
    if info.get("reason"):
        click.echo(f"  not bundled: {info['reason']}")
    for label, field in (
        ("upstream repo", "repo"),
        ("commit", "commit"),
        ("download", "download"),
        ("source", "source"),
        ("convert with", "recipe"),
        ("convert with", "script"),
    ):
        if up.get(field):
            click.echo(f"  {label:<14} {up[field]}")
    click.echo(f"  place result in {assets.local_path(name)}")
    for spec in info.get("files") or []:
        click.echo(f"    {spec['path']}  sha256 {spec.get('sha256', '?')}")
    click.echo(f"  then: MolCraftDiff zoo verify {name}")


@zoo.command("add")
@click.argument("name")
@click.option(
    "--ckpt",
    type=click.Path(exists=True),
    multiple=True,
    help="Checkpoint file or directory (repeatable)",
)
@click.option(
    "--data",
    "data_paths",
    type=click.Path(exists=True),
    multiple=True,
    help="Dataset file or directory (repeatable)",
)
@click.option(
    "--config",
    "config_paths",
    type=click.Path(exists=True),
    multiple=True,
    help="Example config (repeatable)",
)
@click.option("--variant", default="default", show_default=True)
@click.option("--task-type", default=None, help="The task_type string")
@click.option("--family", default=None)
@click.option("--tag", multiple=True)
def add_cmd(
    name, ckpt, data_paths, config_paths, variant, task_type, family, tag
):
    """Register a newly integrated model's assets in zoo.yaml.

    Hashes every file and writes the entry, leaving `license` and
    `redistribute` as TODO for you to fill in -- `zoo verify` then fails
    until they are, which is what stops the zoo drifting.
    """
    import yaml  # noqa: PLC0415

    if not ckpt and not data_paths:
        raise click.ClickException("Give at least one --ckpt or --data.")

    doc = assets.manifest()
    doc.setdefault("assets", {})
    doc.setdefault("models", {})

    def _entry(paths, kind, host):
        files = []
        for item in paths:
            p = Path(item)
            members = (
                sorted(f for f in p.rglob("*") if f.is_file())
                if p.is_dir()
                else [p]
            )
            for f in members:
                rel = f.relative_to(p) if p.is_dir() else Path(f.name)
                files.append(
                    {
                        "path": str(rel),
                        "sha256": assets._sha256(f),  # noqa: SLF001
                        "size": f.stat().st_size,
                    }
                )
        return {
            "kind": kind,
            "host": host,
            "rev": "main",
            "files": files,
            "license": "TODO",
            "redistribute": "TODO",
        }

    spec = {}
    if ckpt:
        key = f"{name}/pretrained"
        doc["assets"][key] = _entry(ckpt, "checkpoint", "models")
        spec["checkpoint"] = key
    if data_paths:
        key = f"{name}/data"
        doc["assets"][key] = _entry(data_paths, "dataset", "datasets")
        spec["data"] = key
    if config_paths:
        spec["configs"] = [Path(c).name for c in config_paths]

    model = doc["models"].setdefault(name, {})
    if family:
        model["family"] = family
    if task_type:
        model["task_type"] = task_type
    if tag:
        model["tags"] = list(tag)
    model.setdefault("variants", {})[variant] = spec

    # safe_dump cannot round-trip comments, so preserve the file's leading
    # header block (the "how to use this manifest" docs) by hand. Per-entry
    # inline comments are lost -- that prose belongs in the model card.
    header = ""
    if assets.MANIFEST.exists():
        lines = assets.MANIFEST.read_text(encoding="utf-8").splitlines(True)
        keep = []
        for line in lines:
            if line.strip() and not line.lstrip().startswith("#"):
                break
            keep.append(line)
        header = "".join(keep)

    body = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)
    assets.MANIFEST.write_text(header + body, encoding="utf-8")
    assets.manifest.cache_clear()

    click.echo(f"registered {name} in {assets.MANIFEST}")
    click.echo("Now fill in license / redistribute / upstream, then run:")
    click.echo("  MolCraftDiff zoo verify --all")
