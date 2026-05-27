"""XTB Electronic Property Computation using xtb-python.

Provides functions to compute electronic properties for XYZ files using
the xtb-python interface directly, with support for batch processing and
multiple output formats (CSV, JSON, ASE database).
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

HARTREE_TO_EV = 27.211386245988
ANGSTROM_TO_BOHR = 1.8897259886
BOHR_DEBYE = 2.5417464519

# Property group definitions
PROPERTY_GROUPS = {
    "energy": ["homo", "lumo", "homo_lumo_gap"],
    "dipole": ["dipole"],
    "reactivity": ["ip", "ea"],
    "global": [
        "electrophilicity",
        "nucleophilicity",
        "electrofugality",
        "nucleofugality",
    ],
    "charges": ["charges"],
    "fukui": ["fukui_plus", "fukui_minus", "fukui_radical", "fukui_dual"],
    "bond_orders": ["bond_orders"],
}

ATOMIC_LEVEL_GROUPS = {"charges", "fukui", "bond_orders"}
MOLECULAR_LEVEL_GROUPS = {"energy", "dipole", "reactivity", "global"}

# Occupation threshold: orbitals with occupation > this are considered filled.
_OCC_THRESHOLD = 0.5


def _solvent_map() -> dict[str, Any]:
    from xtb.utils import Solvent

    return {
        "acetone": Solvent.acetone,
        "acetonitrile": Solvent.acetonitrile,
        "benzene": Solvent.benzene,
        "ch2cl2": Solvent.ch2cl2,
        "chcl3": Solvent.chcl3,
        "chloroform": Solvent.chcl3,
        "cs2": Solvent.cs2,
        "dmf": Solvent.dmf,
        "dmso": Solvent.dmso,
        "ether": Solvent.ether,
        "diethyl_ether": Solvent.ether,
        "h2o": Solvent.h2o,
        "water": Solvent.h2o,
        "methanol": Solvent.methanol,
        "nhexane": Solvent.nhexane,
        "hexane": Solvent.nhexane,
        "n_hexane": Solvent.nhexane,
        "thf": Solvent.thf,
        "toluene": Solvent.toluene,
    }


def _solvent_from_str(name: str) -> Any:
    mapping = _solvent_map()
    key = name.lower().strip()
    if key not in mapping:
        supported = ", ".join(sorted(mapping))
        msg = f"Unknown solvent '{name}'. Supported: {supported}"
        raise ValueError(msg)
    return mapping[key]


def _method_to_param(method: int | str) -> Any:
    from xtb.interface import Param

    m = str(method).lower()
    if m == "1":
        return Param.GFN1xTB
    if m == "2":
        return Param.GFN2xTB
    msg = (
        f"Unsupported method '{method}'. "
        "xtb-python supports GFN1 (1) and GFN2 (2); "
        "use method='ptb' for the binary-based PTB path."
    )
    raise ValueError(msg)


def _run_ptb_binary(
    xyz_path: str,
    charge: int,
    uhf: int,
    solvent: str | None,
    requested_props: set[str],
) -> dict[str, Any]:
    """Run a PTB single-point via the xtb binary and parse JSON output.

    PTB does not provide a total energy, so reactivity/global/fukui
    properties are not available and will be silently skipped.
    """
    if not shutil.which("xtb"):
        msg = "xtb binary not found on PATH. Install xtb via conda."
        raise RuntimeError(msg)

    abs_xyz = str(Path(xyz_path).resolve())
    cmd = ["xtb", abs_xyz, "--ptb", "--json", "--norestart"]
    if charge != 0:
        cmd += ["--chrg", str(charge)]
    if uhf != 0:
        cmd += ["--uhf", str(uhf)]
    if solvent is not None:
        cmd += ["--alpb", solvent]

    with tempfile.TemporaryDirectory() as tmpdir:
        proc = subprocess.run(
            cmd,
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=300,
        )
        json_file = Path(tmpdir) / "xtbout.json"
        if not json_file.exists():
            tail = proc.stderr[-600:] if proc.stderr else proc.stdout[-600:]
            msg = f"xtb PTB run produced no JSON output: {tail}"
            raise RuntimeError(msg)
        raw = json_file.read_text()

    # xtb 6.7.1 bug: bond-orders array starts with a spurious leading comma
    fixed = re.sub(r"(\[\s*\n)\s*,\s*\n", r"\1", raw)
    data = json.loads(fixed)

    props: dict[str, Any] = {}

    # Orbital energies (eV) + occupations → HOMO / LUMO
    if requested_props & {"homo", "lumo", "homo_lumo_gap"}:
        orb_ev = data.get("orbital energies / eV", [])
        occ = data.get("fractional occupation", [])
        if orb_ev and occ:
            eigs = np.array(orb_ev)
            occs = np.array(occ)
            occupied = np.where(occs > _OCC_THRESHOLD)[0]
            homo_idx = int(occupied[-1])
            homo_ev = float(eigs[homo_idx])
            lumo_ev = float(eigs[homo_idx + 1])
            if "homo" in requested_props:
                props["homo"] = homo_ev / HARTREE_TO_EV
                props["homo_ev"] = homo_ev
            if "lumo" in requested_props:
                props["lumo"] = lumo_ev / HARTREE_TO_EV
                props["lumo_ev"] = lumo_ev
            if "homo_lumo_gap" in requested_props:
                gap_ev = lumo_ev - homo_ev
                props["homo_lumo_gap"] = gap_ev / HARTREE_TO_EV
                props["homo_lumo_gap_ev"] = gap_ev

    # Dipole: a.u. (e·Bohr) → Debye
    if "dipole" in requested_props:
        dip_au = data.get("dipole / a.u.")
        if dip_au:
            dip_vec = np.array(dip_au) * BOHR_DEBYE
            props["dipole"] = dip_vec.tolist()
            props["dipole_magnitude"] = float(np.linalg.norm(dip_vec))

    # Partial charges (flat list, 0-indexed) → 1-indexed dict
    if "charges" in requested_props:
        qs = data.get("partial charges", [])
        props["charges"] = {i + 1: float(q) for i, q in enumerate(qs)}

    # Bond orders: list of [atom1, atom2, bo] triplets
    if "bond_orders" in requested_props:
        props["bond_orders"] = {
            f"{int(a)}-{int(b)}": float(bo)
            for a, b, bo in data.get("bond orders", [])
            if float(bo) > 0.1
        }

    unavail = requested_props & {
        "ip", "ea", "electrophilicity", "nucleophilicity",
        "electrofugality", "nucleofugality",
        "fukui_plus", "fukui_minus", "fukui_radical", "fukui_dual",
    }
    if unavail:
        logger.warning(
            "PTB provides no total energy; skipping reactivity/global/fukui: %s",
            unavail,
        )

    return props


def _make_calculator(
    param: Any,
    numbers: np.ndarray,
    positions_bohr: np.ndarray,
    charge: int,
    uhf: int,
    solvent_name: str | None,
) -> Any:
    from xtb.interface import Calculator

    calc = Calculator(
        param,
        numbers,
        positions_bohr,
        charge=charge,
        uhf=uhf,
    )
    if solvent_name is not None:
        calc.set_solvent(_solvent_from_str(solvent_name))
    return calc


def _get_homo_lumo(res: Any) -> tuple[float, float]:
    """Return (HOMO, LUMO) orbital energies in Hartree."""
    eigs = res.get_orbital_eigenvalues()
    occs = res.get_orbital_occupations()
    occupied = np.where(occs > _OCC_THRESHOLD)[0]
    homo_idx = occupied[-1]
    homo = float(eigs[homo_idx])
    lumo = float(eigs[homo_idx + 1])
    return homo, lumo


def _check_xtb_available() -> bool:
    try:
        from xtb.interface import Calculator  # noqa: F401
    except ImportError:
        logger.error(
            "xtb-python is not installed. "
            "Install with: conda install -c conda-forge xtb-python"
        )
        return False
    else:
        return True


def compute_xtb_electronic(
    xyz_path: str,
    method: int | str = 2,
    charge: int = 0,
    n_unpaired: int = 0,
    solvent: str | None = None,
    properties: list[str] | None = None,
    corrected: bool = True,
) -> dict[str, Any]:
    """Compute XTB electronic properties for a single XYZ file.

    Args:
        xyz_path: Path to XYZ file.
        method: XTB method (1=GFN1, 2=GFN2, "ptb"=PTB via xtb binary).
        charge: Molecular charge.
        n_unpaired: Number of unpaired electrons.
        solvent: Solvent name for ALPB implicit solvation (e.g. "water", "thf").
        properties: Property groups to compute. Defaults to ["energy"].
            Available: energy, dipole, reactivity, global, charges, fukui,
            bond_orders, all.  PTB supports energy/dipole/charges/bond_orders
            only (no reactivity/global/fukui — PTB gives no total energy).
        corrected: Kept for API compatibility. Previously applied morfeus
            empirical IP/EA corrections; now has no effect (raw ΔE values
            are returned).

    Returns:
        Dictionary with computed properties and metadata.

    Notes:
        IP = E(cation) - E(neutral); EA = E(neutral) - E(anion). Both in eV.
        Fukui condensed indices use the finite-difference approximation:
            f+(k) = q_k(N) - q_k(N+1)  [electrophilicity, electron addition]
            f-(k) = q_k(N-1) - q_k(N)  [nucleophilicity, electron removal]
            f0(k) = (f+ + f-) / 2
            dual(k) = f+(k) - f-(k)
        Global descriptors follow Parr/Mayr-Ofial DFT reactivity theory.
    """
    from ase.io import read as ase_read

    if corrected:
        logger.debug(
            "corrected=True has no effect; morfeus empirical corrections "
            "are not applied when using xtb-python directly."
        )

    if properties is None:
        properties = ["energy"]

    requested_props: set[str] = set()
    for group in properties:
        if group == "all":
            for g in PROPERTY_GROUPS:
                requested_props.update(PROPERTY_GROUPS[g])
        elif group in PROPERTY_GROUPS:
            requested_props.update(PROPERTY_GROUPS[group])

    needs_ion_calcs = bool(
        requested_props
        & {"ip", "ea", "electrophilicity", "nucleophilicity",
           "electrofugality", "nucleofugality",
           "fukui_plus", "fukui_minus", "fukui_radical", "fukui_dual"}
    )

    # Read XYZ
    atoms = ase_read(xyz_path)
    numbers = atoms.get_atomic_numbers()
    positions_bohr = atoms.get_positions() * ANGSTROM_TO_BOHR

    result: dict[str, Any] = {
        "filename": os.path.basename(xyz_path),
        "n_atoms": len(numbers),
        "method": method,
        "charge": charge,
        "solvent": solvent,
    }

    # PTB is not in the xtb-python bindings; delegate to the xtb binary.
    if str(method).lower() == "ptb":
        try:
            ptb_props = _run_ptb_binary(
                xyz_path, charge, n_unpaired, solvent, requested_props
            )
            result.update(ptb_props)
            result["success"] = True
        except Exception as e:
            logger.warning(f"PTB error for {xyz_path}: {e}")
            result["success"] = False
            result["error"] = str(e)
        return result

    param = _method_to_param(method)

    try:
        # Neutral singlepoint
        calc_n = _make_calculator(
            param, numbers, positions_bohr, charge, n_unpaired, solvent
        )
        res_n = calc_n.singlepoint()
        e_neutral = res_n.get_energy()

        # Energy / orbital properties
        if requested_props & {"homo", "lumo", "homo_lumo_gap"}:
            homo, lumo = _get_homo_lumo(res_n)
            if "homo" in requested_props:
                result["homo"] = homo
                result["homo_ev"] = homo * HARTREE_TO_EV
            if "lumo" in requested_props:
                result["lumo"] = lumo
                result["lumo_ev"] = lumo * HARTREE_TO_EV
            if "homo_lumo_gap" in requested_props:
                gap = lumo - homo
                result["homo_lumo_gap"] = gap
                result["homo_lumo_gap_ev"] = gap * HARTREE_TO_EV

        # Dipole (e·Bohr → Debye)
        if "dipole" in requested_props:
            dipole_vec = res_n.get_dipole() * BOHR_DEBYE
            result["dipole"] = dipole_vec.tolist()
            result["dipole_magnitude"] = float(np.linalg.norm(dipole_vec))

        # Partial charges (1-indexed dict for ASE db compat)
        if "charges" in requested_props:
            q_n = res_n.get_charges()
            result["charges"] = {
                i + 1: float(q) for i, q in enumerate(q_n)
            }

        # Bond orders (sparse upper-triangle dict)
        if "bond_orders" in requested_props:
            bo = res_n.get_bond_orders()
            n_at = bo.shape[0]
            bond_orders: dict[str, float] = {}
            for i in range(n_at):
                for j in range(i + 1, n_at):
                    if bo[i, j] > 0.1:
                        bond_orders[f"{i+1}-{j+1}"] = float(bo[i, j])
            result["bond_orders"] = bond_orders

        # Ion calculations (cation and/or anion)
        if needs_ion_calcs:
            calc_cat = _make_calculator(
                param, numbers, positions_bohr,
                charge + 1, n_unpaired, solvent,
            )
            res_cat = calc_cat.singlepoint()
            e_cation = res_cat.get_energy()

            calc_an = _make_calculator(
                param, numbers, positions_bohr,
                charge - 1, n_unpaired, solvent,
            )
            res_an = calc_an.singlepoint()
            e_anion = res_an.get_energy()

            ip = (e_cation - e_neutral) * HARTREE_TO_EV
            ea = (e_neutral - e_anion) * HARTREE_TO_EV

            if "ip" in requested_props:
                result["ip"] = ip
            if "ea" in requested_props:
                result["ea"] = ea

            # Global DFT reactivity descriptors
            global_requested = requested_props & {
                "electrophilicity", "nucleophilicity",
                "electrofugality", "nucleofugality",
            }
            if global_requested:
                mu = -(ip + ea) / 2.0
                eta = (ip - ea) / 2.0
                omega = mu**2 / (2.0 * eta) if eta != 0 else 0.0
                if "electrophilicity" in requested_props:
                    result["electrophilicity"] = omega
                if "nucleophilicity" in requested_props:
                    result["nucleophilicity"] = abs(mu)
                if "electrofugality" in requested_props:
                    result["electrofugality"] = omega + ea
                if "nucleofugality" in requested_props:
                    result["nucleofugality"] = ip - omega

            # Condensed Fukui indices
            fukui_requested = requested_props & {
                "fukui_plus", "fukui_minus", "fukui_radical", "fukui_dual"
            }
            if fukui_requested:
                q_n_arr = res_n.get_charges()
                q_cat = res_cat.get_charges()
                q_an = res_an.get_charges()
                f_plus = q_n_arr - q_an
                f_minus = q_cat - q_n_arr
                f_zero = (f_plus + f_minus) / 2.0
                f_dual = f_plus - f_minus

                def _to_dict(arr: np.ndarray) -> dict[int, float]:
                    return {i + 1: float(v) for i, v in enumerate(arr)}

                if "fukui_plus" in requested_props:
                    result["fukui_plus"] = _to_dict(f_plus)
                if "fukui_minus" in requested_props:
                    result["fukui_minus"] = _to_dict(f_minus)
                if "fukui_radical" in requested_props:
                    result["fukui_radical"] = _to_dict(f_zero)
                if "fukui_dual" in requested_props:
                    result["fukui_dual"] = _to_dict(f_dual)

        result["success"] = True

    except Exception as e:
        logger.warning(f"Error computing properties for {xyz_path}: {e}")
        result["success"] = False
        result["error"] = str(e)

    return result


def _compute_single_wrapper(args: tuple) -> dict[str, Any]:
    """Wrapper for multiprocessing."""
    return compute_xtb_electronic(*args)


def batch_xtb_electronic(
    input_dir: str,
    output_path: str | None = None,
    output_format: str = "csv",
    method: int | str = 2,
    charge: int = 0,
    n_unpaired: int = 0,
    solvent: str | None = None,
    properties: list[str] | None = None,
    corrected: bool = True,
    timeout: int = 120,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Batch compute XTB electronic properties for XYZ files in a directory.

    Args:
        input_dir: Directory containing XYZ files.
        output_path: Output file path (without extension for 'all' format).
        output_format: Output format ("csv", "json", "ase", or "all").
        method: XTB method (1=GFN1, 2=GFN2).
        charge: Molecular charge.
        n_unpaired: Number of unpaired electrons.
        solvent: Solvent name for ALPB implicit solvation.
        properties: Property groups to compute.
        corrected: Kept for API compatibility; has no effect.
        timeout: Timeout per molecule in seconds.
        n_jobs: Number of parallel jobs.

    Returns:
        DataFrame with molecular-level results.
    """
    if not _check_xtb_available():
        raise RuntimeError("xtb-python is not available")

    input_path = Path(input_dir)
    xyz_files = sorted(input_path.glob("*.xyz"))

    if not xyz_files:
        logger.warning(f"No XYZ files found in {input_dir}")
        return pd.DataFrame()

    logger.info(f"Processing {len(xyz_files)} XYZ files...")

    results: list[dict[str, Any]] = []

    if n_jobs == 1:
        for xyz_file in tqdm(xyz_files, desc="Computing XTB properties"):
            try:
                result = compute_xtb_electronic(
                    str(xyz_file),
                    method=method,
                    charge=charge,
                    n_unpaired=n_unpaired,
                    solvent=solvent,
                    properties=properties,
                    corrected=corrected,
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process {xyz_file}: {e}")
                results.append(
                    {"filename": xyz_file.name, "success": False, "error": str(e)}
                )
    else:
        args_list = [
            (str(f), method, charge, n_unpaired, solvent, properties, corrected)
            for f in xyz_files
        ]
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(_compute_single_wrapper, args): args[0]
                for args in args_list
            }
            for future in tqdm(futures, desc="Computing XTB properties"):
                try:
                    results.append(future.result(timeout=timeout))
                except FutureTimeoutError:
                    fname = Path(futures[future]).name
                    logger.warning(f"Timeout for {fname}")
                    results.append(
                        {"filename": fname, "success": False, "error": "timeout"}
                    )
                except Exception as e:
                    fname = Path(futures[future]).name
                    logger.warning(f"Error for {fname}: {e}")
                    results.append(
                        {"filename": fname, "success": False, "error": str(e)}
                    )

    has_atomic = any(p in ATOMIC_LEVEL_GROUPS for p in (properties or ["energy"]))
    if "all" in (properties or []):
        has_atomic = True

    if output_path:
        _save_outputs(results, output_path, output_format, has_atomic, input_dir)

    return _results_to_dataframe(results)


def _results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert results to DataFrame, excluding atomic-level data."""
    atomic_keys = {
        "charges", "bond_orders", "fukui_plus", "fukui_minus",
        "fukui_radical", "fukui_dual", "dipole_vector",
    }
    return pd.DataFrame(
        [{k: v for k, v in r.items() if k not in atomic_keys} for r in results]
    )


def _save_outputs(
    results: list[dict],
    output_path: str,
    output_format: str,
    has_atomic: bool,
    input_dir: str,
) -> None:
    """Save results in the specified format(s)."""
    base_path = Path(output_path).with_suffix("")
    formats = [output_format] if output_format != "all" else ["csv", "json", "ase"]

    for fmt in formats:
        if fmt == "csv":
            csv_path = base_path.with_suffix(".csv")
            _results_to_dataframe(results).to_csv(csv_path, index=False)
            logger.info(f"Saved CSV to {csv_path}")

        elif fmt == "json":
            json_path = base_path.with_suffix(".json")
            json_results = []
            for r in results:
                jr: dict[str, Any] = {}
                for k, v in r.items():
                    if isinstance(v, np.ndarray):
                        jr[k] = v.tolist()
                    elif isinstance(v, dict):
                        jr[k] = {str(kk): vv for kk, vv in v.items()}
                    else:
                        jr[k] = v
                json_results.append(jr)
            with json_path.open("w") as f:
                json.dump(json_results, f, indent=2)
            logger.info(f"Saved JSON to {json_path}")

        elif fmt == "ase":
            db_path = base_path.with_suffix(".db")
            _save_to_ase_db(results, db_path, input_dir)
            logger.info(f"Saved ASE database to {db_path}")


def _save_to_ase_db(
    results: list[dict], db_path: Path, input_dir: str
) -> None:
    """Save results to ASE database with properties in atoms.info and arrays."""
    from ase.db import connect
    from ase.io import read

    if db_path.exists():
        db_path.unlink()

    db = connect(str(db_path))
    input_path = Path(input_dir)

    atomic_array_keys = {
        "charges", "fukui_plus", "fukui_minus", "fukui_radical", "fukui_dual"
    }

    for result in results:
        filename = result.get("filename")
        if not filename or not result.get("success", False):
            continue

        xyz_file = input_path / filename
        if not xyz_file.exists():
            continue

        try:
            atoms = read(str(xyz_file))
        except Exception as e:
            logger.warning(f"Failed to read {xyz_file} for ASE db: {e}")
            continue

        mol_props: dict[str, Any] = {}
        for key, value in result.items():
            if key in {"filename", "success", "error", "n_atoms"}:
                continue

            if key in atomic_array_keys:
                # 1-indexed dict → 0-indexed array
                if isinstance(value, dict):
                    arr = np.zeros(len(atoms))
                    for idx, val in value.items():
                        arr[int(idx) - 1] = val
                    atoms.arrays[key] = arr
                elif isinstance(value, np.ndarray):
                    atoms.arrays[key] = value
            elif key == "bond_orders":
                mol_props[key] = json.dumps(
                    {str(k): v for k, v in value.items()}
                )
            elif isinstance(value, (int, float, str, bool)):
                mol_props[key] = value

        db.write(atoms, data=mol_props)
