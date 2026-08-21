"""Resolve zoo assets (pretrained weights, datasets) by symbolic name.

A config says ``${asset:kgdiff/pretrained}`` instead of a machine-specific
path, and this module turns that into a real directory under the local
cache. The rest of the platform never learns the zoo exists: the resolver
hands back a plain path string, so ``cli/generate.py``'s file-or-directory
handling, its ``edm_chem.pkl`` / ``edm_stat.pkl`` sidecar lookups and its
``../../config.yaml`` walk all keep working unchanged.

Three properties worth stating, because the design leans on them:

* **An asset is a directory.** Every file listed under one ``assets:`` entry
  lands in the same directory, which is what preserves the sidecar
  co-location contract that ``cli/generate.py`` depends on.
* **Literal paths still work.** ``${asset:...}`` is opt-in syntax. A config
  carrying an ordinary path never reaches this module, so every existing
  config behaves exactly as before.
* **Resolution never touches the network.** A cache hit is a
  ``Path.exists()`` call. Fetching is explicit (``MolCraftDiff zoo fetch``),
  because a config typo must not silently start a multi-gigabyte download
  inside a Hydra compose.

The manifest lives in :data:`MANIFEST` (``zoo.yaml``, shipped in the wheel).
"""

from __future__ import annotations

import difflib
import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = [
    "MANIFEST",
    "assets_root",
    "entry",
    "fetch",
    "local_path",
    "manifest",
    "register",
    "resolve",
    "verify",
]

MANIFEST = Path(__file__).parent / "zoo.yaml"

#: Default cache root. Override with ``$MOLCRAFT_ASSETS`` -- point it at a
#: big disk, a shared mount, or an unpacked Zenodo tarball (the tarball
#: layout *is* this layout, which is the whole offline story).
_DEFAULT_ROOT = Path.home() / ".cache" / "molcraft" / "zoo"

#: Set ``MOLCRAFT_ASSETS_AUTOFETCH=1`` to download on a cache miss instead
#: of raising. Off by default, on purpose -- see the module docstring.
_AUTOFETCH_ENV = "MOLCRAFT_ASSETS_AUTOFETCH"

_CHUNK = 1 << 20


@lru_cache(maxsize=1)
def manifest() -> dict[str, Any]:
    """The parsed ``zoo.yaml``, read once per process."""
    import yaml  # noqa: PLC0415 - keep package import cheap

    if not MANIFEST.exists():
        return {"schema": 1, "assets": {}, "models": {}}
    with MANIFEST.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _assets() -> dict[str, Any]:
    return manifest().get("assets") or {}


def assets_root() -> Path:
    """Where fetched assets live. ``$MOLCRAFT_ASSETS`` wins if set."""
    override = os.environ.get("MOLCRAFT_ASSETS")
    return Path(override).expanduser() if override else _DEFAULT_ROOT


def _split(name: str) -> tuple[str, str]:
    """Split ``name`` into (asset key, path remainder).

    Longest-prefix match, so both forms in the config tree resolve:
    ``kgdiff/pretrained`` (a directory) and
    ``kgdiff/pretrained/kgdiff_pretrained.ckpt`` (a file inside it).
    """
    assets = _assets()
    parts = name.strip("/").split("/")
    for i in range(len(parts), 0, -1):
        key = "/".join(parts[:i])
        if key in assets:
            return key, "/".join(parts[i:])
    close = difflib.get_close_matches(name, sorted(assets), n=3)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise KeyError(f"Unknown zoo asset {name!r}.{hint}")


def _resolve_key(name: str) -> tuple[str, str]:
    """Follow ``alias:`` chains to the key that actually owns the files."""
    key, rest = _split(name)
    assets = _assets()
    seen = [key]
    while "alias" in assets[key]:
        key = assets[key]["alias"]
        if key in seen:
            chain = " -> ".join([*seen, key])
            raise ValueError(f"Alias cycle in zoo.yaml: {chain}")
        if key not in assets:
            raise KeyError(
                f"Asset {seen[-1]!r} aliases {key!r}, which is not defined."
            )
        seen.append(key)
    return key, rest


def entry(name: str) -> dict[str, Any]:
    """The manifest entry for ``name``, with aliases already resolved."""
    key, _ = _resolve_key(name)
    return _assets()[key]


def local_path(name: str) -> Path:
    """Absolute cache path for ``name``.

    Aliases resolve *before* the path is built, so ``ipdiff/data`` and
    ``kgdiff/data`` are literally the same directory -- shared corpora are
    stored once, never copied per model.
    """
    key, rest = _resolve_key(name)
    base = assets_root() / key
    return base / rest if rest else base


def _human(size: int | None) -> str:
    if not size:
        return "unknown size"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return (
                f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
            )
        value /= 1024
    return f"{value:.1f} GB"


def total_size(name: str) -> int:
    """Declared total bytes of an asset, for ``--dry-run`` accounting."""
    info = entry(name)
    return sum(f.get("size") or 0 for f in info.get("files") or [])


def _missing_message(name: str) -> str:
    key, _ = _resolve_key(name)
    info = _assets()[key]
    where = assets_root() / key
    if info.get("redistribute") is False:
        reason = info.get("reason", "not redistributable")
        return (
            f"Asset {key!r} is not bundled ({reason}).\n"
            f"  Build it locally:  MolCraftDiff zoo recipe {key}\n"
            f"  Expected at:       {where}"
        )
    lic = info.get("license", "licence unstated")
    return (
        f"Asset {key!r} not found at\n"
        f"  {where}\n"
        f"Fetch it with:\n"
        f"  MolCraftDiff zoo fetch {key}"
        f"   ({_human(total_size(key))}, {lic})"
    )


def resolve(name: str) -> str:
    """The ``${asset:...}`` resolver body. Returns an absolute path string.

    Raises :class:`FileNotFoundError` naming the exact fetch command when
    the asset is absent, rather than auto-downloading -- unless
    ``MOLCRAFT_ASSETS_AUTOFETCH=1`` is set.
    """
    path = local_path(name)
    if path.exists():
        return str(path)
    if os.environ.get(_AUTOFETCH_ENV) == "1":
        fetch(name)
        return str(path)
    raise FileNotFoundError(_missing_message(name))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _url_for(key: str, info: dict[str, Any], rel: str) -> str:
    host = manifest().get("hosting") or {}
    target = host.get(info.get("host", "models")) or {}
    template = host.get(
        "url", "https://huggingface.co/{prefix}{repo}/resolve/{rev}/{path}"
    )
    # HuggingFace serves dataset repos under /datasets/<repo>, model repos
    # under /<repo>. Derived from `kind` rather than stored per host, so the
    # manifest cannot disagree with itself.
    prefix = "datasets/" if target.get("kind") == "dataset" else ""
    return template.format(
        prefix=prefix,
        repo=target.get("repo", ""),
        rev=info.get("rev", "main"),
        path=f"{key}/{rel}",
    )


def hf_token() -> str | None:
    """The HuggingFace token, if the user has one configured.

    Checked in the order the HF tooling itself uses. Required only because
    the zoo repos are private; a public repo needs none.
    """
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(var)
        if value:
            return value.strip()
    # $HF_HOME must be checked first: `hf auth login` writes the token
    # there, and users routinely point it off the home partition.
    candidates = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home).expanduser() / "token")
    candidates += [
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ]
    for candidate in candidates:
        if candidate.exists():
            text = candidate.read_text(encoding="utf-8").strip()
            if text:
                return text
    return None


def _install_auth_opener() -> bool:
    """Attach a bearer header to urllib's global opener, if we have a token.

    ``utils/file.py::download`` calls ``urlretrieve``, which goes through the
    global opener -- so installing one here adds auth without duplicating the
    download logic.
    """
    token = hf_token()
    if not token:
        return False
    import urllib.request  # noqa: PLC0415

    opener = urllib.request.build_opener()
    opener.addheaders = [("Authorization", f"Bearer {token}")]
    urllib.request.install_opener(opener)
    return True


def fetch(name: str, force: bool = False) -> Path:
    """Download an asset into the cache and verify every file's sha256.

    Re-fetching is a no-op: a file already present with the right hash is
    skipped. Pass ``force=True`` to redownload regardless.
    """
    from MolecularDiffusion.utils.file import download  # noqa: PLC0415

    key, _ = _resolve_key(name)
    info = _assets()[key]
    if info.get("redistribute") is False:
        raise RuntimeError(_missing_message(key))

    target = assets_root() / key
    target.mkdir(parents=True, exist_ok=True)

    private = bool((manifest().get("hosting") or {}).get("private"))
    if private and not _install_auth_opener():
        raise RuntimeError(
            "The zoo repos are private, so fetching needs a HuggingFace "
            "token.\n"
            "  Set one:  export HF_TOKEN=hf_xxx\n"
            "  or log in: huggingface-cli login\n"
            "Get a read token at https://huggingface.co/settings/tokens"
        )

    for spec in info.get("files") or []:
        rel = spec["path"]
        dest = target / rel
        want = spec.get("sha256")
        if not force and dest.exists() and (not want or _sha256(dest) == want):
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        download(
            _url_for(key, info, rel), str(dest.parent), save_file=dest.name
        )
        if want and _sha256(dest) != want:
            raise RuntimeError(
                f"sha256 mismatch for {dest}\n"
                f"  expected {want}\n"
                f"  got      {_sha256(dest)}"
            )
    return target


def verify(name: str) -> list[tuple[str, str]]:
    """Re-hash an asset's files. Returns ``[(relpath, status), ...]``.

    Status is ``ok``, ``missing``, ``sha mismatch``, or ``no sha in
    manifest``. This is what makes a locally-built, non-redistributable
    asset trustworthy: the manifest ships the hash even when it cannot
    ship the bytes.
    """
    key, _ = _resolve_key(name)
    info = _assets()[key]
    target = assets_root() / key
    results: list[tuple[str, str]] = []
    for spec in info.get("files") or []:
        rel = spec["path"]
        dest = target / rel
        want = spec.get("sha256")
        if not dest.exists():
            results.append((rel, "missing"))
        elif not want:
            results.append((rel, "no sha in manifest"))
        elif _sha256(dest) == want:
            results.append((rel, "ok"))
        else:
            results.append((rel, "sha mismatch"))
    return results


def register() -> None:
    """Register the ``${asset:...}`` OmegaConf resolver.

    Called from ``MolecularDiffusion/__init__.py`` so it covers both the
    CLI (which composes via ``cli/_hydra.py``) and plain library use
    (``OmegaConf.load`` in a notebook, or ``cli/generate.py``'s own load of
    a checkpoint's training config).
    """
    from omegaconf import OmegaConf  # noqa: PLC0415

    # replace=True keeps a reimport (pytest, module reload) a no-op
    # rather than a DuplicateResolver crash.
    OmegaConf.register_new_resolver("asset", resolve, replace=True)
