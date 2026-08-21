"""Self-check for the ${asset:...} zoo resolver.

The load-bearing assertion here is `test_literal_path_untouched`: the whole
migration is only safe because a config carrying an ordinary path never
enters the resolver. If that ever regresses, every existing config and every
user's local copy breaks at once.
"""

from __future__ import annotations

import textwrap

import pytest
from omegaconf import OmegaConf

from MolecularDiffusion import assets

STUB = textwrap.dedent(
    """
    schema: 1
    hosting:
      models: {repo: org/zoo, kind: model}
      datasets: {repo: org/zoo-data, kind: dataset}
      url: "https://example.invalid/{repo}/resolve/{rev}/{path}"
    assets:
      demo/pretrained:
        kind: checkpoint
        host: models
        license: MIT
        redistribute: yes
        files:
          - {path: model.ckpt, sha256: "%(sha)s", size: 5}
          - {path: edm_stat.pkl, sha256: "%(sha)s", size: 5}
      demo/data:
        kind: dataset
        host: datasets
        license: CC0
        redistribute: yes
        files:
          - {path: demo.db, sha256: "%(sha)s", size: 5}
      other/data:
        alias: demo/data
      gated/data:
        redistribute: no
        reason: "upstream repo is access-gated"
        files:
          - {path: gated.db, sha256: "%(sha)s", size: 5}
      cycle/a: {alias: cycle/b}
      cycle/b: {alias: cycle/a}
    models:
      demo:
        family: test
        variants:
          default: {checkpoint: demo/pretrained, data: demo/data}
    """
)

# sha256 of b"hello"
_SHA = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


@pytest.fixture
def zoo(tmp_path, monkeypatch):
    """Point the resolver at a throwaway manifest and cache root."""
    manifest = tmp_path / "zoo.yaml"
    manifest.write_text(STUB % {"sha": _SHA}, encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()

    monkeypatch.setattr(assets, "MANIFEST", manifest)
    monkeypatch.setenv("MOLCRAFT_ASSETS", str(cache))
    monkeypatch.delenv("MOLCRAFT_ASSETS_AUTOFETCH", raising=False)
    assets.manifest.cache_clear()
    assets.register()
    yield cache
    assets.manifest.cache_clear()


def _materialize(cache, key, *names):
    target = cache / key
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        (target / name).write_bytes(b"hello")
    return target


def test_resolves_directory(zoo):
    want = _materialize(zoo, "demo/pretrained", "model.ckpt")
    cfg = OmegaConf.create({"p": "${asset:demo/pretrained}"})
    assert cfg.p == str(want)


def test_resolves_file_inside_asset(zoo):
    """Longest-prefix match: both the dir and file forms must work.

    Real configs use both -- diffpharma_generate.yaml points at a
    directory, apo2mol_generate.yaml at a file.
    """
    _materialize(zoo, "demo/pretrained", "model.ckpt")
    cfg = OmegaConf.create({"p": "${asset:demo/pretrained/model.ckpt}"})
    assert cfg.p == str(zoo / "demo/pretrained/model.ckpt")


def test_sidecars_land_in_one_directory(zoo):
    """cli/generate.py hangs edm_stat.pkl off chkpt_directory itself."""
    want = _materialize(zoo, "demo/pretrained", "model.ckpt", "edm_stat.pkl")
    assert assets.resolve("demo/pretrained") == str(want)
    assert (want / "model.ckpt").exists()
    assert (want / "edm_stat.pkl").exists()


def test_literal_path_untouched(zoo):
    """The backward-compatibility guarantee the whole plan rests on."""
    literal = "docs/model_integrations/kgdiff/checkpoints/converted/x.ckpt"
    cfg = OmegaConf.create({"p": literal})
    assert cfg.p == literal


def test_alias_resolves_to_target_directory(zoo):
    """other/data aliases demo/data -- same dir, and no other/data on disk."""
    want = _materialize(zoo, "demo/data", "demo.db")
    assert assets.local_path("other/data") == want
    assert assets.resolve("other/data") == str(want)
    assert not (zoo / "other").exists()


def test_alias_cycle_raises(zoo):
    with pytest.raises(ValueError, match="Alias cycle"):
        assets.local_path("cycle/a")


def test_missing_asset_names_the_fetch_command(zoo):
    with pytest.raises(FileNotFoundError) as excinfo:
        assets.resolve("demo/pretrained")
    message = str(excinfo.value)
    assert "MolCraftDiff zoo fetch demo/pretrained" in message
    assert "MIT" in message


def test_non_redistributable_points_at_the_recipe(zoo):
    with pytest.raises(FileNotFoundError) as excinfo:
        assets.resolve("gated/data")
    message = str(excinfo.value)
    assert "zoo recipe gated/data" in message
    assert "access-gated" in message
    with pytest.raises(RuntimeError, match="not bundled"):
        assets.fetch("gated/data")


def test_unknown_asset_suggests_close_matches(zoo):
    with pytest.raises(KeyError, match="demo/data"):
        assets.local_path("demo/dat")


def test_verify_detects_corruption(zoo):
    _materialize(zoo, "demo/data", "demo.db")
    assert assets.verify("demo/data") == [("demo.db", "ok")]
    (zoo / "demo/data/demo.db").write_bytes(b"tampered")
    assert assets.verify("demo/data") == [("demo.db", "sha mismatch")]
    (zoo / "demo/data/demo.db").unlink()
    assert assets.verify("demo/data") == [("demo.db", "missing")]


def test_fetch_skips_files_already_correct(zoo, monkeypatch):
    """Re-fetch is a no-op. The URL is unreachable, so any download fails."""
    _materialize(zoo, "demo/data", "demo.db")
    calls = []

    def _boom(url, path, save_file=None, md5=None):
        calls.append(url)
        raise AssertionError("should not download an already-correct file")

    monkeypatch.setattr("MolecularDiffusion.utils.file.download", _boom)
    assets.fetch("demo/data")
    assert calls == []


def test_total_size_sums_declared_files(zoo):
    assert assets.total_size("demo/pretrained") == 10


# --- bundled-config addressing -------------------------------------------
#
# `hydra.searchpath` extends the search for config GROUPS but not for the
# PRIMARY config, so cli/_hydra.py falls back to the bundled tree. These
# pin the two properties that matter: shipped examples are reachable, and a
# user's own config of the same name still wins.


def test_bundled_config_is_reachable(tmp_path, monkeypatch):
    from MolecularDiffusion.cli import _hydra

    pkg = tmp_path / "pkg"
    (pkg / "examples").mkdir(parents=True)
    (pkg / "examples" / "demo.yaml").write_text("name: bundled\n", "utf-8")
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()

    primary, name = _hydra._fallback_to_bundled(
        str(cwd), "demo.yaml", "examples/demo.yaml", str(pkg)
    )
    assert primary == str(pkg / "examples")
    assert name == "demo.yaml"


def test_local_config_beats_bundled(tmp_path):
    """A user config in the primary dir must never be shadowed."""
    from MolecularDiffusion.cli import _hydra

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "demo.yaml").write_text("name: bundled\n", "utf-8")
    cwd = tmp_path / "mine"
    cwd.mkdir()
    (cwd / "demo.yaml").write_text("name: local\n", "utf-8")

    primary, name = _hydra._fallback_to_bundled(
        str(cwd), "demo.yaml", "demo.yaml", str(pkg)
    )
    assert primary == str(cwd)


def test_unknown_config_is_left_alone(tmp_path):
    """Absent everywhere -> unchanged, so Hydra raises its own error."""
    from MolecularDiffusion.cli import _hydra

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    cwd = tmp_path / "mine"
    cwd.mkdir()
    primary, name = _hydra._fallback_to_bundled(
        str(cwd), "nope.yaml", "nope.yaml", str(pkg)
    )
    assert (primary, name) == (str(cwd), "nope.yaml")


# --- private-repo auth ----------------------------------------------------
#
# The zoo repos are private, so an unauthenticated fetch gets a 401 and would
# otherwise write an HTML error page to disk under the asset's name.


def test_private_fetch_without_token_says_how_to_fix(zoo, monkeypatch):
    import yaml

    doc = yaml.safe_load(assets.MANIFEST.read_text(encoding="utf-8"))
    doc["hosting"]["private"] = True
    assets.MANIFEST.write_text(yaml.safe_dump(doc), encoding="utf-8")
    assets.manifest.cache_clear()

    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(assets, "hf_token", lambda: None)

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        assets.fetch("demo/data")


def test_hf_token_read_from_env(zoo, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "  hf_secret  ")
    assert assets.hf_token() == "hf_secret"


def test_private_fetch_with_token_attaches_bearer_header(zoo, monkeypatch):
    import yaml

    doc = yaml.safe_load(assets.MANIFEST.read_text(encoding="utf-8"))
    doc["hosting"]["private"] = True
    assets.MANIFEST.write_text(yaml.safe_dump(doc), encoding="utf-8")
    assets.manifest.cache_clear()
    monkeypatch.setenv("HF_TOKEN", "hf_secret")

    import urllib.request

    captured = {}
    monkeypatch.setattr(
        urllib.request,
        "install_opener",
        lambda o: captured.update(headers=o.addheaders),
    )
    # already-correct file -> no download, but auth is still installed first
    _materialize(zoo, "demo/data", "demo.db")
    assets.fetch("demo/data")
    assert ("Authorization", "Bearer hf_secret") in captured["headers"]


# --- shipped example configs ---------------------------------------------


def test_example_configs_reference_real_assets():
    """Every ${asset:...} in a bundled example must exist in zoo.yaml.

    A typo here ships a config that only fails at run time, on someone
    else's machine. Comments are skipped -- OmegaConf never parses them.
    """
    import re
    from pathlib import Path

    root = Path(assets.__file__).parent / "configs" / "examples"
    if not root.is_dir():
        pytest.skip("no bundled examples yet")

    assets.manifest.cache_clear()
    bad = []
    for cfg in sorted(root.glob("*.yaml")):
        for line in cfg.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("#"):
                continue
            for ref in re.findall(r"\$\{asset:([^}]+)\}", line):
                try:
                    assets.local_path(ref)
                except (KeyError, ValueError) as exc:
                    bad.append(f"{cfg.name}: {ref} ({exc})")
    assert not bad, "unresolvable asset references:\n  " + "\n  ".join(bad)


def test_manifest_has_no_unaudited_entries():
    """`redistribute: TODO` must never reach a commit."""
    assets.manifest.cache_clear()
    todo = [
        k
        for k, v in (assets.manifest().get("assets") or {}).items()
        if "alias" not in v
        and str(v.get("redistribute", "TODO")).upper() == "TODO"
    ]
    assert not todo, f"unaudited assets: {todo}"


def test_withheld_assets_have_no_host():
    """A non-redistributable asset must be structurally un-uploadable."""
    assets.manifest.cache_clear()
    leaky = [
        k
        for k, v in (assets.manifest().get("assets") or {}).items()
        if "alias" not in v
        and v.get("redistribute") is False
        and v.get("host")
    ]
    assert not leaky, f"withheld assets still carry a host: {leaky}"


def test_dataset_urls_carry_the_datasets_prefix(zoo, monkeypatch):
    """HF serves dataset repos under /datasets/<repo>, models under /<repo>.

    Getting this wrong 404s on the first dataset fetch only -- model
    fetches succeed, so it hides until someone pulls data.
    """
    import yaml

    doc = yaml.safe_load(assets.MANIFEST.read_text(encoding="utf-8"))
    doc["hosting"]["url"] = (
        "https://hf.test/{prefix}{repo}/resolve/{rev}/{path}"
    )
    assets.MANIFEST.write_text(yaml.safe_dump(doc), encoding="utf-8")
    assets.manifest.cache_clear()

    ck = assets._url_for(
        "demo/pretrained", assets.entry("demo/pretrained"), "model.ckpt"
    )
    ds = assets._url_for("demo/data", assets.entry("demo/data"), "demo.db")
    assert (
        ck == "https://hf.test/org/zoo/resolve/main/demo/pretrained/model.ckpt"
    )
    assert (
        ds
        == "https://hf.test/datasets/org/zoo-data/resolve/main/demo/data/demo.db"
    )


def test_hf_token_found_via_hf_home(tmp_path, monkeypatch):
    """`hf auth login` writes to $HF_HOME, which users move off $HOME."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    home = tmp_path / "hfhome"
    home.mkdir()
    (home / "token").write_text("hf_from_hf_home\n", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(home))
    assert assets.hf_token() == "hf_from_hf_home"
