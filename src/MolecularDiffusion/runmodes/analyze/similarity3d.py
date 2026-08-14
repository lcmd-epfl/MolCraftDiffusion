"""3D similarity of generated molecules to a reference molecule.

Backs ``MolCraftDiff analyze metrics --metrics similarity3d``, which needs
``--reference-mol``. Three Gaussian-overlap scores, each aligned before
scoring:

* ``shape_sim``  -- molecular-surface overlap (ROCS-style);
* ``esp_sim``    -- surface overlap weighted by electrostatic potential;
* ``pharm_sim``  -- pharmacophore-point overlap.

The scoring itself comes from the vendored ``utils/shepherd_score`` package;
this module only prepares the profiles and drives the alignment. Logic is
lifted unchanged from the former ``shepherd`` metric block so the numbers stay
identical across the rename.

Generated molecules may ship precomputed profiles as sidecar files next to the
``.xyz`` (``.npz``, ``_surface.npy``, ``_esp.npz``, ``_pharm.npz``); those are
reused when present and recomputed from the molecule otherwise.
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

NUM_SURFACE_POINTS = 75
PROBE_RADIUS = 1.2
SAMPLES_PER_ATOM = 25
NUM_ALIGN_REPEATS = 45
ESP_LAMBDA = 0.3


def load_reference_source(path):
    """Open a reference ``.pkl`` (list of molblocks) or ``.sdf`` supplier."""
    import pickle  # noqa: PLC0415

    from rdkit import Chem  # noqa: PLC0415

    path = str(path)
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)  # noqa: S301 -- user-supplied local cache
    if path.endswith(".sdf"):
        return Chem.SDMolSupplier(path, removeHs=False)
    raise ValueError(f"Unsupported reference format (expected .pkl or .sdf): {path}")


def reference_mol(data_source, index):
    """Pull one molecule out of a reference source by index."""
    from rdkit import Chem  # noqa: PLC0415

    if isinstance(data_source, list):  # pkl of (molblock, charges) tuples
        return Chem.MolFromMolBlock(data_source[index][0], removeHs=False)
    return data_source[index]


def extract_profiles(mol):
    """Surface points, ESP values and pharmacophores for one molecule."""
    if mol is None:
        return None

    from rdkit.Chem import AllChem  # noqa: PLC0415

    from MolecularDiffusion.utils.shepherd_score.extract_profiles import (  # noqa: PLC0415
        get_electrostatic_potential,
    )
    from MolecularDiffusion.utils.shepherd_score.generate_point_cloud import (  # noqa: PLC0415
        get_atomic_vdw_radii,
        get_molecular_surface,
    )
    from MolecularDiffusion.utils.shepherd_score.pharm_utils.pharmacophore import (  # noqa: PLC0415
        get_pharmacophores,
    )

    pos = mol.GetConformer().GetPositions().astype(np.float32)
    radii = get_atomic_vdw_radii(mol)
    AllChem.ComputeGasteigerCharges(mol)
    charges = np.nan_to_num(
        np.array(
            [float(a.GetProp("_GasteigerCharge")) for a in mol.GetAtoms()],
            dtype=np.float32,
        ),
        nan=0.0,
    )
    surface = get_molecular_surface(
        pos, radii, num_points=NUM_SURFACE_POINTS,
        probe_radius=PROBE_RADIUS, num_samples_per_atom=SAMPLES_PER_ATOM,
    )
    esp = get_electrostatic_potential(mol, charges, surface)
    p_types, p_pts, p_vecs = get_pharmacophores(
        mol, multi_vector=False, exclude=[], check_access=False, scale=1.0,
    )
    return {
        "surface": surface,
        "pharm_pts": p_pts,
        "pharm_types": p_types,
        "pharm_vecs": p_vecs,
        "esp": esp,
        "num_atoms": mol.GetNumAtoms(),
    }


def _sidecar_profiles(xyz_path):
    """Reuse precomputed profiles written next to the .xyz, when present."""
    surf = esp = pharm_pts = pharm_types = pharm_vecs = None

    npz_path = xyz_path.replace(".xyz", ".npz")
    if os.path.exists(npz_path):
        try:
            sidecar = np.load(npz_path)
            surf = sidecar.get("surf_pts", None)
            esp = sidecar.get("esp_vals", None)
            pharm_pts = sidecar.get("pharm_pts", None)
            pharm_types = sidecar.get("pharm_types", None)
        except Exception:  # noqa: BLE001
            pass

    surf_path = xyz_path.replace(".xyz", "_surface.npy")
    if surf is None and os.path.exists(surf_path):
        try:
            surf = np.load(surf_path)
        except Exception:  # noqa: BLE001
            pass

    esp_path = xyz_path.replace(".xyz", "_esp.npz")
    if esp is None and os.path.exists(esp_path):
        try:
            esp = np.load(esp_path).get("charges", None)
        except Exception:  # noqa: BLE001
            pass

    pharm_path = xyz_path.replace(".xyz", "_pharm.npz")
    if pharm_pts is None and os.path.exists(pharm_path):
        try:
            data = np.load(pharm_path)
            pharm_pts = data.get("positions", None)
            pharm_types = data.get("types", None)
            pharm_vecs = data.get("directions", None)
        except Exception:  # noqa: BLE001
            pass

    return surf, esp, pharm_pts, pharm_types, pharm_vecs


def _alpha(n_points):
    from MolecularDiffusion.utils.shepherd_score.score.constants import ALPHA  # noqa: PLC0415

    try:
        return float(ALPHA(np.clip(n_points, 50, 400)))
    except Exception:  # noqa: BLE001
        return 0.81


def compare(mol, ref_data, xyz_path=None):
    """Shape / ESP / pharmacophore similarity of ``mol`` against ``ref_data``.

    Returns a dict of the three scores; a modality that cannot be computed is
    reported as ``0.0``, matching the behaviour of the former shepherd block.
    """
    import torch  # noqa: PLC0415

    from MolecularDiffusion.utils.shepherd_score.alignment import (  # noqa: PLC0415
        optimize_pharm_overlay,
        optimize_ROCS_esp_overlay,
        optimize_ROCS_overlay,
    )
    from MolecularDiffusion.utils.shepherd_score.pharm_utils.pharmacophore import (  # noqa: PLC0415
        get_pharmacophores,
    )
    from MolecularDiffusion.utils.shepherd_score.score.constants import LAM_SCALING  # noqa: PLC0415
    from MolecularDiffusion.utils.shepherd_score.score.electrostatic_scoring_np import (  # noqa: PLC0415
        get_overlap_esp_np,
    )
    from MolecularDiffusion.utils.shepherd_score.score.gaussian_overlap_np import (  # noqa: PLC0415
        get_overlap_np,
    )
    from MolecularDiffusion.utils.shepherd_score.score.pharmacophore_scoring_np import (  # noqa: PLC0415
        get_overlap_pharm_np,
    )

    scores = {"shape_sim": 0.0, "pharm_sim": 0.0, "esp_sim": 0.0}
    if ref_data is None:
        return scores

    gen_surf = gen_esp = gen_pharm_pts = gen_pharm_types = gen_pharm_vecs = None
    if xyz_path is not None:
        gen_surf, gen_esp, gen_pharm_pts, gen_pharm_types, gen_pharm_vecs = _sidecar_profiles(xyz_path)

    # recompute whatever the sidecars did not provide
    if gen_surf is None:
        from MolecularDiffusion.utils.shepherd_score.generate_point_cloud import (  # noqa: PLC0415
            get_atomic_vdw_radii,
            get_molecular_surface,
        )

        pos = mol.GetConformer().GetPositions().astype(np.float32)
        gen_surf = get_molecular_surface(
            pos, get_atomic_vdw_radii(mol), num_points=NUM_SURFACE_POINTS,
            probe_radius=PROBE_RADIUS, num_samples_per_atom=SAMPLES_PER_ATOM,
        )
    if gen_pharm_pts is None:
        gen_pharm_types, gen_pharm_pts, gen_pharm_vecs = get_pharmacophores(
            mol, multi_vector=False, exclude=[], check_access=False, scale=1.0,
        )
    if gen_esp is None:
        from rdkit.Chem import AllChem  # noqa: PLC0415

        from MolecularDiffusion.utils.shepherd_score.extract_profiles import (  # noqa: PLC0415
            get_electrostatic_potential,
        )

        AllChem.ComputeGasteigerCharges(mol)
        charges = np.nan_to_num(
            np.array(
                [float(a.GetProp("_GasteigerCharge")) for a in mol.GetAtoms()],
                dtype=np.float32,
            ),
            nan=0.0,
        )
        gen_esp = get_electrostatic_potential(mol, charges, gen_surf)

    # --- shape -----------------------------------------------------------
    if gen_surf is not None and len(gen_surf) > 0 and ref_data["surface"] is not None:
        alpha = _alpha(len(gen_surf))
        gs = (gen_surf - gen_surf.mean(axis=0)).astype(np.float32)
        rs = (ref_data["surface"] - ref_data["surface"].mean(axis=0)).astype(np.float32)
        aligned, _, _ = optimize_ROCS_overlay(
            torch.from_numpy(rs), torch.from_numpy(gs), alpha,
            num_repeats=NUM_ALIGN_REPEATS,
        )
        scores["shape_sim"] = float(get_overlap_np(rs, aligned.numpy(), alpha=alpha))

    # --- pharmacophore ---------------------------------------------------
    if gen_pharm_pts is not None and len(gen_pharm_pts) > 0 and ref_data["pharm_pts"] is not None:
        rpa = (ref_data["pharm_pts"] - ref_data["pharm_pts"].mean(axis=0)).astype(np.float32)
        gpa = (gen_pharm_pts - gen_pharm_pts.mean(axis=0)).astype(np.float32)
        rpv = ref_data["pharm_vecs"].astype(np.float32)
        gpv = (
            gen_pharm_vecs.astype(np.float32)
            if gen_pharm_vecs is not None
            else np.zeros_like(gpa)
        )
        rpt = ref_data["pharm_types"].astype(np.int64)
        gpt = gen_pharm_types.astype(np.int64)
        ga_t, gv_t, _, _ = optimize_pharm_overlay(
            torch.from_numpy(rpt), torch.from_numpy(gpt),
            torch.from_numpy(rpa), torch.from_numpy(gpa),
            torch.from_numpy(rpv), torch.from_numpy(gpv),
            similarity="tanimoto", num_repeats=NUM_ALIGN_REPEATS,
        )
        scores["pharm_sim"] = float(get_overlap_pharm_np(
            ptype_1=rpt, ptype_2=gpt,
            anchors_1=rpa, anchors_2=ga_t.numpy(),
            vectors_1=rpv, vectors_2=gv_t.numpy(),
            similarity="tanimoto",
        ))

    # --- electrostatics --------------------------------------------------
    ref_esp = ref_data["esp"]
    if gen_esp is not None and ref_esp is not None and gen_surf is not None and ref_data["surface"] is not None:
        alpha = _alpha(len(gen_surf))
        lam = ESP_LAMBDA * LAM_SCALING
        gs = (gen_surf - gen_surf.mean(axis=0)).astype(np.float32)
        rs = (ref_data["surface"] - ref_data["surface"].mean(axis=0)).astype(np.float32)
        ge = gen_esp.astype(np.float32)
        re = ref_esp.astype(np.float32)
        aligned_e, _, _ = optimize_ROCS_esp_overlay(
            torch.from_numpy(rs), torch.from_numpy(gs),
            torch.from_numpy(re), torch.from_numpy(ge),
            alpha, lam, num_repeats=NUM_ALIGN_REPEATS,
        )
        scores["esp_sim"] = float(get_overlap_esp_np(
            rs, aligned_e.numpy(), re, ge, alpha=alpha, lam=lam,
        ))

    return scores
