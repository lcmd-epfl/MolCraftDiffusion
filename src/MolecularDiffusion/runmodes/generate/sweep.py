#!/usr/bin/env python3
"""Parameter sweep tool for MolCraftDiffusion generation experiments.

Usage:
    python sweep.py <sweep_config.yaml> [--dry-run] [--skip-gen] [--skip-eval]

Sweep YAML keys
---------------
base_config     : path to base generation YAML (required)
eval_script     : path to eval script, or list of scripts run in sequence (required)
sweep_dir       : root directory for all run outputs (required)
name            : prefix for per-run subdirectory names (default: "sweep")
parameters      : dict of {hydra.override.key: [v1, v2, ...]} — full Cartesian product (required)
extra_overrides : list of extra Hydra overrides applied to every run (optional)
search          : search settings (optional). Defaults to grid search.
collect         : list of {path, metrics} blocks describing which CSVs to read and how
                  to aggregate their columns (optional — falls back to built-in defaults)

search format
-------------
search:
  method: grid              # grid or bayesian
  include_base_config: true # try the base_config parameter values first
  objective:
    metric: hit_rate_both   # metric collected into summary.csv
    mode: max               # max or min
  bayesian:
    n_initial: 5            # Optuna warm-up trials before GP/TPE uses surrogate
    random_state: 0
    sampler: gp             # gp (default, requires optuna>=3.6) or tpe
  early_fail_batches: 3      # optional: stop generation after N failed batches before any success

collect format
--------------
collect:
  - path: optimized_xyz/merged_hits.csv   # relative to the run output dir
    metrics:
      n_total:       {agg: len}                              # total rows
      hit_rate_both: {column: both_hit,    agg: mean}
      n_both_hit:    {column: both_hit,    agg: sum}
      mean_dihedral: {column: eff_dihedral, agg: mean, exclude_value: -1.0}
  - path: postbuster_metrics.csv
    metrics:
      validity_rate: {column: valid_posebuster, agg: mean}

Supported agg values: mean, sum, count (non-null rows), len (total rows).
Values accept a list or a range string "start:stop:step" (stop inclusive).
eval_script can be a string (single script) or a list (run in sequence, each
receiving the output directory as its only argument).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import yaml

if TYPE_CHECKING:
    import optuna


def _require_optuna():
    """Import Optuna only when Bayesian search is requested."""
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "Bayesian sweep mode requires Optuna. Install it with "
            "`pip install optuna>=3.6` or use `search.method: grid`."
        ) from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    return optuna


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grid-search parameter sweep for MolCraftDiffusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("sweep_config", help="Path to sweep YAML config file")
    p.add_argument("--dry-run", action="store_true",
                   help="Print all commands without executing them")
    p.add_argument("--skip-gen", action="store_true",
                   help="Skip generation step (assumes output dirs already exist)")
    p.add_argument("--skip-eval", action="store_true",
                   help="Skip eval step (only run generation + result collection)")
    p.add_argument("--max-runs", type=int, default=None, metavar="N",
                   help="Run at most N new combinations (already-recorded runs don't count)")
    p.add_argument("--retry-failed", action="store_true",
                   help="Retry configurations already recorded with failed generation/evaluation")
    p.add_argument("--early-fail-batches", type=int, default=None, metavar="N",
                   help=("Stop a generation trial after N consecutive 'Sampling failed' "
                         "batches before any successful batch. Defaults to "
                         "search.early_fail_batches or disabled."))
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Value utilities
# ---------------------------------------------------------------------------

def _parse_values(spec: Any) -> list:
    """Expand a value spec to a flat list.

    Accepts a list, a "start:stop:step" range string, or a scalar.
    """
    if isinstance(spec, list):
        return spec
    if isinstance(spec, str) and spec.count(":") == 2:
        start, stop, step = (float(x) for x in spec.split(":"))
        vals = np.arange(start, stop + step * 1e-6, step).tolist()
        return [round(v, 10) for v in vals]
    return [spec]


def _fmt_dir(v: Any) -> str:
    """Format a value as a filesystem-safe directory fragment."""
    if isinstance(v, list):
        return "_".join(_fmt_dir(x) for x in v)
    s = f"{v:.6g}" if isinstance(v, float) else str(v)
    return s.replace(".", "p").replace("-", "m").replace("/", "_").replace(" ", "_")


def _hydra_val(v: Any) -> str:
    """Format a value as a Hydra CLI override string."""
    if isinstance(v, list):
        return "[" + ",".join(str(x) for x in v) + "]"
    return str(v)


def _jsonable(v: Any) -> Any:
    """Convert numpy/pandas scalars to stable JSON-compatible values."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    if isinstance(v, tuple):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(val) for k, val in v.items()}
    return v


_OPTUNA_COMPLEX_PREFIX = "__sweep_json__:"


def _optuna_choice(v: Any) -> Any:
    """Encode non-primitive categorical choices for Optuna persistence."""
    val = _jsonable(v)
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    return _OPTUNA_COMPLEX_PREFIX + json.dumps(
        val, sort_keys=True, separators=(",", ":")
    )


def _optuna_choice_maps(
    param_names: list[str],
    value_lists: list[list[Any]],
) -> dict[str, dict[Any, Any]]:
    """Map Optuna-safe values back to their original sweep values."""
    return {
        _short_key(p): {_optuna_choice(v): _jsonable(v) for v in values}
        for p, values in zip(param_names, value_lists)
    }


def _short_key(param: str) -> str:
    """Return the last dotted segment of a Hydra parameter path."""
    return param.split(".")[-1]


def _combo_key(combo: dict[str, Any]) -> str:
    """Stable hash for a parameter combination, independent of output path."""
    payload = json.dumps(_jsonable(combo), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _combo_tag(combo: dict[str, Any]) -> str:
    """Build a compact directory tag for a parameter combination."""
    return "_".join(f"{_short_key(p)}_{_fmt_dir(v)}" for p, v in combo.items())


def _get_nested(cfg: dict[str, Any], dotted_key: str) -> Any:
    """Return a nested value using a Hydra-style dotted key."""
    cur: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(dotted_key)
        cur = cur[part]
    return cur


def _resolve_value(cfg: dict[str, Any], value: Any) -> Any:
    """Resolve simple ${path.to.value} interpolations from a YAML dict."""
    if not (isinstance(value, str) and value.startswith("${") and value.endswith("}")):
        return value
    ref = value[2:-1].strip()
    try:
        return _resolve_value(cfg, _get_nested(cfg, ref))
    except KeyError:
        return value


def _base_combo(base_config: str, param_names: list[str]) -> dict[str, Any] | None:
    """Read the base config values for the requested parameter paths."""
    with open(base_config) as fh:
        cfg = yaml.safe_load(fh)
    combo: dict[str, Any] = {}
    missing: list[str] = []
    for param in param_names:
        if param.endswith("output_path"):
            continue
        try:
            combo[param] = _resolve_value(cfg, _get_nested(cfg, param))
        except KeyError:
            missing.append(param)
    if missing:
        print(
            "  [WARN] include_base_config skipped; missing in base_config: "
            + ", ".join(missing)
        )
        return None
    return combo


# ---------------------------------------------------------------------------
# Command execution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunResult:
    ok: bool
    early_stopped: bool = False
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


_SAMPLING_FAILED_RE = re.compile(r"\[Batch\s+\d+\]\s+Sampling failed:")
_NONZERO_SUCCESS_RE = re.compile(r"success=(?!0(?:\D|$))(\d+)")


def _run(
    cmd: list[str],
    log_path: Path,
    dry_run: bool,
    early_fail_batches: int | None = None,
) -> RunResult:
    """Run *cmd*, tee output to *log_path*, and optionally stop doomed trials."""
    print("  $ " + " ".join(cmd))
    if dry_run:
        return RunResult(ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    monitor_failures = early_fail_batches is not None and early_fail_batches > 0
    consecutive_failures = 0
    saw_success = False
    early_reason = ""

    with open(log_path, "w") as fh:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            if not monitor_failures:
                continue
            if _NONZERO_SUCCESS_RE.search(line):
                saw_success = True
                consecutive_failures = 0
            if _SAMPLING_FAILED_RE.search(line) and not saw_success:
                consecutive_failures += 1
                if consecutive_failures >= early_fail_batches:
                    early_reason = (
                        f"{consecutive_failures} consecutive generation batches failed "
                        "before any successful batch"
                    )
                    fh.write(f"\n[SWEEP] Early-stopping trial: {early_reason}\n")
                    fh.flush()
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    break
        result_code = proc.wait()

    if early_reason:
        print(f"  [WARN] early-stopped — {early_reason}; see {log_path}")
        return RunResult(ok=False, early_stopped=True, reason=early_reason)
    if result_code != 0:
        print(f"  [WARN] exit {result_code} — see {log_path}")
        return RunResult(ok=False)
    return RunResult(ok=True)


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------

_AGG_FNS: dict[str, Any] = {
    "mean":  lambda s: float(s.mean()),
    "sum":   lambda s: float(s.sum()),
    "count": lambda s: int(s.count()),   # non-null rows
    "len":   lambda s: len(s),           # total rows (ignores NaN)
}


def _apply_metric(df: pd.DataFrame, spec: dict[str, Any]) -> Any:
    """Compute one metric from a DataFrame according to a metric spec dict."""
    agg = spec.get("agg", "mean")
    fn = _AGG_FNS.get(agg)
    if fn is None:
        raise ValueError(f"Unknown agg {agg!r}. Choose from: {list(_AGG_FNS)}")

    # agg=len operates on the whole frame, column is optional
    if agg == "len":
        return len(df)

    col = spec.get("column")
    if col is None:
        raise ValueError(f"Metric spec missing 'column': {spec}")
    if col not in df.columns:
        return float("nan")

    series = df[col]
    excl = spec.get("exclude_value")
    if excl is not None:
        series = series.replace(excl, float("nan"))
    series = series.dropna()
    if len(series) == 0:
        return float("nan")

    val = fn(series)
    rnd = spec.get("round")
    return round(val, rnd) if rnd is not None else val


def _collect_from_cfg(out_dir: Path, collect_cfg: list[dict]) -> dict[str, Any]:
    """Collect metrics using user-defined collect: blocks from the sweep YAML."""
    m: dict[str, Any] = {}
    for entry in collect_cfg:
        csv_path = out_dir / entry["path"]
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        for metric_name, spec in entry.get("metrics", {}).items():
            if isinstance(spec, str):
                # shorthand: "column_name" (mean) or "column_name:agg"
                parts = spec.split(":", 1)
                spec = {"column": parts[0], "agg": parts[1] if len(parts) > 1 else "mean"}
            m[metric_name] = _apply_metric(df, spec)
    return m


def _collect_defaults(out_dir: Path) -> dict[str, Any]:
    """Built-in collection rules for the standard workflow eval script outputs."""
    m: dict[str, Any] = {}

    # `analyze metrics` writes a summary next to its CSV; prefer it over
    # re-deriving the same numbers from the per-file rows.
    for summary_path in sorted(out_dir.glob("*_summary.json")):
        try:
            with summary_path.open() as f:
                summary = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for key in ("validity", "validity_geom", "validity_connected",
                    "fully_connected", "atom_stability", "uniqueness",
                    "novelty", "diversity", "n_failed"):
            value = summary.get(key)
            if value is not None:
                m[key] = round(value, 4) if isinstance(value, float) else value

    mh = out_dir / "optimized_xyz" / "merged_hits.csv"
    if mh.exists():
        df = pd.read_csv(mh)
        m["n_total"] = len(df)
        if "success" in df:
            m["n_opt_success"] = int(df["success"].sum())
        if "both_hit" in df:
            m["n_both_hit"] = int(df["both_hit"].sum())
            m["hit_rate_both"] = round(float(df["both_hit"].mean()), 4)
        if "both_hit_mid" in df:
            m["n_both_hit_mid"] = int(df["both_hit_mid"].sum())
            m["hit_rate_both_mid"] = round(float(df["both_hit_mid"].mean()), 4)
        if "geometry_hit" in df:
            m["geom_hit_rate"] = round(float(df["geometry_hit"].mean()), 4)
        if "energy_hit" in df:
            m["energy_hit_rate"] = round(float(df["energy_hit"].mean()), 4)
        if "eff_dihedral" in df:
            valid_dih = df["eff_dihedral"].replace(-1.0, float("nan")).dropna()
            if len(valid_dih):
                m["mean_eff_dihedral"] = round(float(valid_dih.mean()), 3)

    pb = out_dir / "postbuster_metrics.csv"
    if pb.exists():
        df = pd.read_csv(pb)
        if "valid_posebuster" in df:
            m["validity_rate"] = round(float(df["valid_posebuster"].mean()), 4)

    xa = out_dir / "xyz_analysis.csv"
    if xa.exists():
        df = pd.read_csv(xa)
        if "rmsd" in df:
            m["mean_rmsd"] = round(float(df["rmsd"].dropna().mean()), 4)
        if "same_topology" in df:
            m["topology_retention"] = round(float(df["same_topology"].mean()), 4)

    sc = out_dir / "scscore_results.csv"
    if sc.exists():
        df = pd.read_csv(sc)
        if "scscore" in df:
            m["mean_scscore"] = round(float(df["scscore"].dropna().mean()), 3)

    u = out_dir / "u.csv"
    if u.exists():
        df = pd.read_csv(u)
        if "max_similarity" in df:
            m["mean_max_similarity"] = round(float(df["max_similarity"].dropna().mean()), 4)

    return m


def _collect(out_dir: Path, collect_cfg: list[dict] | None) -> dict[str, Any]:
    if collect_cfg is not None:
        return _collect_from_cfg(out_dir, collect_cfg)
    return _collect_defaults(out_dir)


def _read_summary(summary_csv: Path) -> pd.DataFrame:
    """Read the persistent trial database if it exists."""
    if not summary_csv.exists():
        return pd.DataFrame()
    try:
        summary = pd.read_csv(summary_csv)
    except Exception:
        return pd.DataFrame()
    if (
        "config_id" not in summary.columns
        and "config_json" in summary.columns
    ):
        summary["config_id"] = summary["config_json"].apply(_config_id_from_json)
    return summary


def _config_id_from_json(config_json: Any) -> str | float:
    """Recover a config_id from a config_json cell, when possible."""
    if pd.isna(config_json):
        return float("nan")
    try:
        combo = json.loads(str(config_json))
    except json.JSONDecodeError:
        return float("nan")
    if not isinstance(combo, dict):
        return float("nan")
    return _combo_key(combo)


def _upsert_result(summary_csv: Path, row: dict[str, Any], key: str = "config_id") -> None:
    """Insert or update one result row in summary_csv."""
    new_df = pd.DataFrame([row])
    existing = _read_summary(summary_csv)
    if existing.empty or key not in existing.columns or key not in new_df.columns:
        combined = new_df
        if not existing.empty:
            combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        keep = existing[key].astype(str) != str(row[key])
        combined = pd.concat([existing.loc[keep], new_df], ignore_index=True)
    combined.to_csv(summary_csv, index=False)


def _print_summary(summary_csv: Path) -> None:
    if not summary_csv.exists():
        return
    summary = pd.read_csv(summary_csv)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(f"\n{'='*60}")
    print(f"SUMMARY  ({len(summary)} run(s) total)  →  {summary_csv}")
    print("=" * 60)
    print(summary.to_string(index=False))
    try:
        summary.to_markdown(summary_csv.with_suffix(".md"), index=False)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Search selection
# ---------------------------------------------------------------------------

def _tried_ids(summary: pd.DataFrame, retry_failed: bool) -> set[str]:
    """Return configuration IDs that should not be scheduled again."""
    if summary.empty or "config_id" not in summary.columns:
        return set()
    if not retry_failed:
        return set(summary["config_id"].dropna().astype(str))
    # retry_failed=True should retry failed/stale-running rows; only keep
    # completed rows blocked.
    if "status" in summary.columns:
        mask = summary["status"].astype(str) == "complete"
        return set(summary.loc[mask, "config_id"].dropna().astype(str))

    # Backward-compatible fallback for legacy summaries without status.
    if "gen_ok" not in summary.columns and "eval_ok" not in summary.columns:
        return set(summary["config_id"].dropna().astype(str))
    mask = pd.Series(True, index=summary.index)
    if "gen_ok" in summary.columns:
        mask &= summary["gen_ok"] == True
    if "eval_ok" in summary.columns:
        mask &= summary["eval_ok"] == True
    return set(summary.loc[mask, "config_id"].dropna().astype(str))


def _optuna_distributions(
    param_names: list[str],
    value_lists: list[list[Any]],
) -> "dict[str, optuna.distributions.CategoricalDistribution]":
    """Build Optuna distributions, encoding list/dict choices as strings."""
    optuna = _require_optuna()
    from optuna.distributions import CategoricalDistribution
    return {
        _short_key(p): CategoricalDistribution([_optuna_choice(v) for v in values])
        for p, values in zip(param_names, value_lists)
    }


def _make_optuna_study(
    summary: pd.DataFrame,
    param_names: list[str],
    value_lists: list[list[Any]],
    search_cfg: dict[str, Any],
) -> "optuna.Study":
    """Create a fresh Optuna study, replaying past observations from summary."""
    optuna = _require_optuna()
    objective = search_cfg.get("objective", {})
    mode = objective.get("mode", "max")
    metric = objective.get("metric", "hit_rate_both")
    direction = "maximize" if mode == "max" else "minimize"

    bayes_cfg = search_cfg.get("bayesian", {})
    seed = int(bayes_cfg.get("random_state", 0))
    n_startup = int(bayes_cfg.get("n_initial", 5))
    sampler_name = str(bayes_cfg.get("sampler", "gp")).lower()

    if sampler_name == "tpe":
        sampler: optuna.samplers.BaseSampler = optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=n_startup
        )
    else:
        try:
            sampler = optuna.samplers.GPSampler(
                seed=seed, n_startup_trials=n_startup
            )
        except AttributeError:   # optuna < 3.6 fallback
            sampler = optuna.samplers.TPESampler(
                seed=seed, n_startup_trials=n_startup
            )

    study = optuna.create_study(direction=direction, sampler=sampler)
    dists = _optuna_distributions(param_names, value_lists)

    if summary.empty or metric not in summary.columns or "config_json" not in summary.columns:
        return study

    for _, row in summary.dropna(subset=[metric]).iterrows():
        try:
            config = json.loads(str(row["config_json"]))
        except (json.JSONDecodeError, TypeError):
            continue
        params = {_short_key(p): _optuna_choice(config[p]) for p in param_names if p in config}
        if set(params) != set(dists):
            continue
        study.add_trial(
            optuna.trial.create_trial(
                params=params,
                distributions=dists,
                value=float(row[metric]),
            )
        )

    return study


def _failed_objective_value(search_cfg: dict[str, Any]) -> tuple[str, float]:
    """Return the objective metric and value to record for failed trials."""
    objective = search_cfg.get("objective", {})
    metric = objective.get("metric", "hit_rate_both")
    if "failed_value" in objective:
        return metric, float(objective["failed_value"])
    mode = str(objective.get("mode", "max")).lower()
    return metric, 1e12 if mode == "min" else 0.0


def _select_optuna_combo(
    study: "optuna.Study",
    param_names: list[str],
    value_lists: list[list[Any]],
    candidates: list[dict[str, Any]],
    tried: set[str],
    rng: random.Random,
) -> dict[str, Any]:
    """Ask Optuna for the next combo; fall back to random if already tried."""
    untried = [c for c in candidates if _combo_key(c) not in tried]
    if not untried:
        raise StopIteration

    dists = _optuna_distributions(param_names, value_lists)
    choice_maps = _optuna_choice_maps(param_names, value_lists)
    short_to_full = {_short_key(p): p for p in param_names}
    combo_index = {_combo_key(c): c for c in candidates}

    trial = study.ask(fixed_distributions=dists)
    suggested = {
        short_to_full[sk]: choice_maps.get(sk, {}).get(sv, sv)
        for sk, sv in trial.params.items()
        if sk in short_to_full
    }
    suggested_key = _combo_key(suggested)
    if suggested_key in combo_index and suggested_key not in tried:
        return combo_index[suggested_key]

    return rng.choice(untried)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_sweep_config(cfg_path: Path) -> dict[str, Any]:
    """Load and validate the top-level sweep YAML structure."""
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if not isinstance(cfg, dict):
        loaded = type(cfg).__name__
        sys.exit(f"Sweep config must be a YAML mapping, got {loaded}: {cfg_path}")

    required = ("base_config", "sweep_dir", "eval_script")
    missing = [key for key in required if key not in cfg]
    if missing:
        keys = ", ".join(str(key) for key in cfg.keys()) or "<none>"
        sys.exit(
            f"Sweep config missing required key(s): {', '.join(missing)}\n"
            f"Loaded keys from {cfg_path}: {keys}"
        )

    return cfg


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg_path = Path(args.sweep_config)
    if not cfg_path.exists():
        sys.exit(f"Sweep config not found: {cfg_path}")

    cfg = _load_sweep_config(cfg_path)

    base_config: str = cfg["base_config"]
    sweep_dir = Path(cfg["sweep_dir"])
    name: str = cfg.get("name", "sweep")
    raw_params: dict[str, Any] = cfg.get("parameters", {})
    raw_list_params: dict[str, Any] = cfg.get("list_parameters", {})
    extra_overrides: list[str] = cfg.get("extra_overrides", [])
    collect_cfg: list[dict] | None = cfg.get("collect")
    search_cfg: dict[str, Any] = cfg.get("search", {})
    search_method = search_cfg.get("method", "grid")
    summary_csv = sweep_dir / "summary.csv"

    # eval_script: string or list of strings
    raw_eval = cfg["eval_script"]
    eval_scripts: list[str] = [raw_eval] if isinstance(raw_eval, str) else list(raw_eval)

    if not raw_params and not raw_list_params:
        sys.exit("No 'parameters' or 'list_parameters' key found in sweep config.")

    # Scalar parameters: each element is one value (int/float/string).
    param_names = list(raw_params.keys())
    value_lists = [_parse_values(raw_params[p]) for p in param_names]
    display_value_counts = [len(vl) for vl in value_lists]

    # List-valued parameters: each element is itself a list (e.g. mol_size, target_values).
    # raw_list_params values must be a list-of-lists: [[23,42], [23,55]].
    for lp, lp_vals in raw_list_params.items():
        if not isinstance(lp_vals, list) or len(lp_vals) == 0:
            sys.exit(f"list_parameters.{lp} must be a non-empty list of lists.")
        param_names.append(lp)
        value_lists.append(lp_vals)
        display_value_counts.append(len(lp_vals))

    base_first = bool(search_cfg.get("include_base_config", False))
    base_combo: dict[str, Any] | None = None
    if base_first:
        base_combo = _base_combo(base_config, param_names)
        if base_combo is not None:
            for i, param in enumerate(param_names):
                if all(_jsonable(v) != _jsonable(base_combo[param]) for v in value_lists[i]):
                    value_lists[i].append(base_combo[param])
            display_value_counts = [len(vl) for vl in value_lists]

    combos = [
        dict(zip(param_names, vals))
        for vals in itertools.product(*value_lists)
    ]
    if base_first and base_combo is not None:
        base_id = _combo_key(base_combo)
        combos = [c for c in combos if _combo_key(c) != base_id]
        combos.insert(0, base_combo)

    n_grid = len(combos)

    # Load already-recorded trials so we can skip them, including interrupted
    # rows written as status=running before the process was stopped.
    done_df = _read_summary(summary_csv)
    done_ids = _tried_ids(done_df, args.retry_failed)
    short_params = [_short_key(p) for p in param_names]
    grid_desc = " × ".join(f"{sp}({n})" for sp, n in zip(short_params, display_value_counts))
    if base_first:
        grid_desc += " + base_config"
    collect_src = "YAML collect:" if collect_cfg else "built-in defaults"
    early_fail_batches = (
        args.early_fail_batches
        if args.early_fail_batches is not None
        else search_cfg.get("early_fail_batches")
    )
    if early_fail_batches is not None:
        early_fail_batches = int(early_fail_batches)
    limit_str = f"  (--max-runs {args.max_runs})" if args.max_runs else ""
    pending_count = sum(
        1 for c in combos
        if _combo_key(c) not in done_ids
    )
    print(f"Sweep   : {cfg_path.name}")
    print(f"Search  : {search_method}")
    print(f"Grid    : {grid_desc} = {n_grid} total | {n_grid - pending_count} tried | "
          f"{pending_count} pending{limit_str}")
    print(f"Eval    : {', '.join(eval_scripts)}")
    print(f"Collect : {collect_src}")
    print(f"Output  : {sweep_dir}")
    if args.dry_run:
        print("Mode    : DRY RUN")
    if early_fail_batches is not None and early_fail_batches > 0:
        print(f"Early stop: {early_fail_batches} consecutive failed generation batch(es)")

    if not args.dry_run:
        sweep_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(cfg_path, sweep_dir / "sweep_config.yaml")

    # ---- Main loop: gen → eval → collect per run ----
    n_new = 0
    dry_run_seen: set[str] = set()
    while True:
        done_df = _read_summary(summary_csv)
        done_ids = _tried_ids(done_df, args.retry_failed)
        available = [
            combo for combo in combos
            if _combo_key(combo) not in done_ids
            and _combo_key(combo) not in dry_run_seen
        ]
        if not available:
            break
        if args.max_runs is not None and n_new >= args.max_runs:
            print(f"\n[--max-runs {args.max_runs} reached — stopping]")
            break

        if (
            base_first
            and base_combo is not None
            and _combo_key(base_combo) in {_combo_key(c) for c in available}
        ):
            combo = base_combo
        elif search_method == "grid":
            combo = available[0]
        elif search_method == "bayesian":
            available_ids = {_combo_key(c) for c in available}
            unavailable_ids = {
                config_id for c in combos
                if (config_id := _combo_key(c)) not in available_ids
            }
            unavailable_ids.update(dry_run_seen)
            _optuna_study = _make_optuna_study(
                done_df, param_names, value_lists, search_cfg
            )
            _rng = random.Random(
                int(search_cfg.get("bayesian", {}).get("random_state", 0))
            )
            combo = _select_optuna_combo(
                _optuna_study, param_names, value_lists,
                combos, unavailable_ids, _rng,
            )
        else:
            sys.exit(f"Unknown search.method {search_method!r}. Choose grid or bayesian.")

        out_dir = sweep_dir / f"{name}_{_combo_tag(combo)}"
        config_id = _combo_key(combo)
        n_new += 1
        pending = len(available)
        cap = args.max_runs if args.max_runs is not None else pending
        print(f"\n── [{n_new}/{cap}] {out_dir.name} ──")

        row_base: dict[str, Any] = {p: v for p, v in combo.items()}
        for param, value in combo.items():
            short = _short_key(param)
            if short not in row_base:
                row_base[short] = value
        row_base["config_id"] = config_id
        row_base["config_json"] = json.dumps(_jsonable(combo), sort_keys=True)
        row_base["run_dir"] = out_dir.name
        row_base["search_method"] = search_method

        if not args.dry_run:
            started_row = dict(row_base)
            started_row["status"] = "running"
            started_row["started_at"] = datetime.now().isoformat(timespec="seconds")
            _upsert_result(summary_csv, started_row)

        # Generation
        gen_result = RunResult(ok=True)
        if not args.skip_gen:
            overrides = [f"{p}={_hydra_val(v)}" for p, v in combo.items()]
            overrides += [f"interference.output_path={out_dir}"]
            overrides += extra_overrides
            gen_result = _run(
                ["MolCraftDiff", "generate", base_config] + overrides,
                out_dir / "sweep_gen.log",
                args.dry_run,
                early_fail_batches=early_fail_batches,
            )
        gen_ok = bool(gen_result)

        # Evaluation (immediately after generation)
        eval_ok = True
        if gen_ok and not args.skip_eval:
            for j, script in enumerate(eval_scripts):
                log = out_dir / (
                    "sweep_eval.log" if len(eval_scripts) == 1
                    else f"sweep_eval_{j}.log"
                )
                eval_ok = bool(_run(["bash", script, str(out_dir)], log, args.dry_run)) and eval_ok
        elif not gen_ok and not args.skip_eval:
            print("  [WARN] skipping evaluation because generation failed")
            eval_ok = False

        if args.dry_run:
            dry_run_seen.add(config_id)
            continue

        # Collect and append to persistent summary
        row = dict(row_base)
        row["status"] = "complete" if (gen_ok and eval_ok) else "failed"
        row["gen_ok"] = gen_ok
        row["eval_ok"] = eval_ok
        row["early_stopped"] = gen_result.early_stopped
        if gen_result.reason:
            row["failure_reason"] = gen_result.reason
        row["completed_at"] = datetime.now().isoformat(timespec="seconds")
        if out_dir.exists():
            row.update(_collect(out_dir, collect_cfg))
        if not (gen_ok and eval_ok):
            metric, bad_value = _failed_objective_value(search_cfg)
            if metric not in row or pd.isna(row[metric]):
                row[metric] = bad_value
        _upsert_result(summary_csv, row)
        status = "ok" if (gen_ok and eval_ok) else f"gen={'ok' if gen_ok else 'FAIL'} eval={'ok' if eval_ok else 'FAIL'}"
        print(f"  → recorded [{status}] in {summary_csv.name}")

    if args.dry_run:
        print("\n[dry-run complete — no files written]")
        return

    _print_summary(summary_csv)


if __name__ == "__main__":
    main()
