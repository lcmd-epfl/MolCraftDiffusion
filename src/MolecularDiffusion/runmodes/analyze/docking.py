"""AutoDock Vina scoring of generated ligands against a protein pocket.

Backs ``MolCraftDiff analyze metrics --metrics sbdd``. Needs the ``[sbdd]``
extra (``vina`` + ``meeko`` + ``gemmi``); every import is deferred so the rest
of the analysis commands keep working without it.

Protocol follows the SBDD literature (targetdiff / KGDiff / IPDiff / Apo2Mol):

* the search box is centred on the ligand being scored, sized to its own
  extent plus a 5 A buffer -- generated ligands already sit in the pocket
  frame, so their own bounding box defines the site;
* three modes, increasing cost: ``score`` (in place), ``min`` (local
  minimisation), ``dock`` (full redock).

The pose matters, so molecules are perceived with OpenBabel, **not** xyz2mol:
xyz2mol returns a topology with no conformer, which silently discards the
coordinates the model generated.
"""

from __future__ import annotations

import os

BOX_BUFFER = 5.0
DOCK_MODES = ("score", "min", "dock")


def load_pose(xyz_path):
    """Read one .xyz into a sanitized RDKit mol that still has its pose.

    Returns ``(smiles, mol)``; ``mol`` is ``None`` when the structure cannot be
    perceived, is multi-fragment, or lost its conformer.
    """
    from rdkit import Chem  # noqa: PLC0415

    from MolecularDiffusion.utils.smilify import smilify_openbabel  # noqa: PLC0415

    try:
        smiles, mol = smilify_openbabel(xyz_path)
    except Exception:  # noqa: BLE001 -- an unperceivable structure is data
        return None, None
    if isinstance(mol, (list, tuple)):
        mol = mol[0] if len(mol) == 1 else None
    if isinstance(smiles, (list, tuple)):
        smiles = smiles[0] if len(smiles) == 1 else None
    if mol is None:
        return smiles, None
    if mol.GetNumConformers() == 0:
        # no 3D coordinates -> nothing to dock
        return smiles, None
    # OpenBabel hands back an unsanitized mol; AddHs and meeko both need
    # implicit valences to have been computed.
    try:
        Chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001 -- bad valences/kekulisation are data
        return smiles, None
    return smiles, mol


def ligand_pdbqt(mol):
    """Sanitized, H-added mol -> PDBQT string (meeko 0.7 API)."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy  # noqa: PLC0415

    setups = MoleculePreparation().prepare(mol)
    if not setups:
        raise ValueError("meeko produced no molecule setup")
    pdbqt, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise ValueError(f"meeko could not write PDBQT: {err}")
    return pdbqt


def box_for(mol, buffer=BOX_BUFFER):
    """Search box centred on this molecule: (center, size), both length 3."""
    pos = mol.GetConformer(0).GetPositions()
    center = ((pos.max(0) + pos.min(0)) / 2).tolist()
    size = ((pos.max(0) - pos.min(0)) + buffer).tolist()
    return center, size


def prepare_receptor(receptor_path, out_dir=None):
    """Return a receptor PDBQT path, converting from PDB only if needed.

    A ``.pdbqt`` is used as-is -- CrossDocked-style test sets ship them, and
    reusing one keeps our numbers comparable with the upstream papers.
    """
    if receptor_path.endswith(".pdbqt"):
        return receptor_path

    import subprocess  # noqa: PLC0415

    out_dir = out_dir or os.path.dirname(os.path.abspath(receptor_path))
    stem = os.path.splitext(os.path.basename(receptor_path))[0]
    out_path = os.path.join(out_dir, f"{stem}.pdbqt")
    if os.path.exists(out_path):
        return out_path

    # meeko ships this CLI; it replaces the AutoDockTools/pdb2pqr chain
    result = subprocess.run(  # noqa: S603
        ["mk_prepare_receptor.py", "--read_pdb", receptor_path, "-o", os.path.join(out_dir, stem), "-p"],
        capture_output=True,
        text=True,
        check=False,
    )
    if not os.path.exists(out_path):
        raise ValueError(
            f"could not prepare receptor {receptor_path!r}: "
            f"{result.stderr.strip() or result.stdout.strip()}. "
            "Pass an already-prepared .pdbqt instead."
        )
    return out_path


def score_pose(mol, receptor_pdbqt, mode="dock", exhaustiveness=8, seed=42, buffer=BOX_BUFFER):
    """Vina affinities for one posed molecule.

    Returns a dict with ``vina_score`` (always), plus ``vina_min`` for
    ``mode in {"min", "dock"}`` and ``vina_dock`` for ``mode == "dock"``.
    """
    from rdkit import Chem  # noqa: PLC0415
    from vina import Vina  # noqa: PLC0415

    if mode not in DOCK_MODES:
        raise ValueError(f"mode must be one of {DOCK_MODES}, got {mode!r}")

    mol_h = Chem.AddHs(mol, addCoords=True)
    center, size = box_for(mol_h, buffer)

    v = Vina(sf_name="vina", seed=seed, verbosity=0)
    v.set_receptor(receptor_pdbqt)
    v.set_ligand_from_string(ligand_pdbqt(mol_h))
    v.compute_vina_maps(center=center, box_size=size)

    out = {"vina_score": float(v.score()[0])}
    if mode in ("min", "dock"):
        out["vina_min"] = float(v.optimize()[0])
    if mode == "dock":
        v.dock(exhaustiveness=exhaustiveness, n_poses=5)
        out["vina_dock"] = float(v.energies(n_poses=1)[0][0])
    return out


def score_reference(ref_ligand_sdf, receptor_pdbqt, **kwargs):
    """Score the reference (crystal) ligand -- the bar `high_affinity` uses."""
    from rdkit import Chem  # noqa: PLC0415

    mol = Chem.MolFromMolFile(ref_ligand_sdf, sanitize=True)
    if mol is None:
        raise ValueError(f"could not read reference ligand {ref_ligand_sdf!r}")
    return score_pose(mol, receptor_pdbqt, **kwargs)
