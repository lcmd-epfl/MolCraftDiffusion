"""Molecule -> surface mesh -> point cloud -> equivariant shape latent.

Ported from DiffSMol ``source/utils/shape.py`` (the ``pointAE_shape`` path
only). This is the *only* module in the integration that needs optional
dependencies, and it is **precompute-only**: training and generation read a
cached ``.pt`` of tensors and never import this module.

Optional extras (``pip install '.[shape]' --no-deps``):
  * ``scikit-image``  -- marching-cubes molecular surface
  * ``trimesh``       -- mesh handling, area-uniform surface sampling, volume

Neither ``oddt`` nor ``openbabel`` is required at runtime.
``generate_surface_marching_cubes`` below is a faithful inline port of
``oddt.surface.generate_surface_marching_cubes`` (oddt 0.7), which is dead
in any modern env: it imports ``skimage.measure.marching_cubes_lewiner``,
removed in scikit-image 0.19, and silently degrades to unusable. The only
thing it needed openbabel for was a van-der-Waals radius lookup, which is
hardcoded below (``OPENBABEL_VDW_RADII``) straight from openbabel's table.
Modern ``skimage.measure.marching_cubes`` *is* the Lewiner algorithm --
it became the default when the ``_lewiner`` alias was retired -- so this is
numerically faithful to what upstream ran.

``pytorch3d`` is *not* required either: upstream used it for exactly two
calls, ``sample_points_from_meshes`` (replaced by ``trimesh.Trimesh.sample``,
the same area-uniform face sampling) and ``Meshes.get_bounding_boxes``
(which only fed the gradient guidance this integration does not port).

Frame convention -- this is the correctness-critical part. The returned
``shape_center`` is the centroid of the **sampled surface points**, not the
atom centroid and not the centre of mass. Upstream subtracts exactly this
from both the point cloud and the ligand coordinates
(``shape.py:341`` / ``shape_mol_dataset.py:115``). The training task must
subtract the same ``shape_center`` from its coordinates, or the latent and
the coordinates end up in different frames and conditioning silently breaks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

_INSTALL_HINT = (
    "DiffSMol shape precompute needs the optional '[shape]' extra.\n"
    "  pip install 'MolecularDiffusion[shape]' --no-deps\n"
    "Training and generation do NOT need it; only the one-off precompute\n"
    "of the shape cache does."
)

#: Number of points sampled from the surface mesh. Fixed by the vendored
#: shape-AE checkpoint's training config (``point_cloud_samples: 512``).
POINT_CLOUD_SAMPLES = 512

#: van-der-Waals radii in Angstrom, keyed by element symbol.
#:
#: These are **openbabel's** values (``openbabel.GetVdwRad(atomic_number)``),
#: NOT RDKit's and NOT ASE's. The vendored shape autoencoder was trained on
#: surfaces built from openbabel radii, by way of oddt's
#: ``atom_dict['radius']``. Substituting another table shifts every sphere
#: slightly, feeding the pretrained encoder out-of-distribution
#: surfaces. ``check_vdw_radii()``
#: asserts these against openbabel when it happens to be installed.
OPENBABEL_VDW_RADII: Dict[str, float] = {
    "H": 1.1,
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.8,
    "S": 1.8,
    "Cl": 1.75,
    "Br": 1.83,
    "I": 1.98,
}

_ATOMIC_NUMBER = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9,
    "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53,
}


def check_vdw_radii() -> None:
    """Assert ``OPENBABEL_VDW_RADII`` matches openbabel's table.

    A no-op (returns silently) when openbabel is not importable -- it is not
    a dependency of this module, only the provenance of the numbers.
    """
    try:
        from openbabel import openbabel as ob
    except ImportError:  # pragma: no cover - openbabel is conda-only
        return
    for sym, radius in OPENBABEL_VDW_RADII.items():
        expected = ob.GetVdwRad(_ATOMIC_NUMBER[sym])
        if abs(expected - radius) > 1e-9:
            raise AssertionError(
                f"vdW radius drift for {sym}: hardcoded {radius}, "
                f"openbabel {expected}. The vendored shape autoencoder was "
                f"trained on openbabel's radii -- do not change the table."
            )


def _require_shape_deps():
    """Import the optional surface stack, or raise with an install hint."""
    try:
        import trimesh
        from skimage.measure import marching_cubes
        # `closing` not `binary_closing`: identical on boolean input, and
        # binary_closing is deprecated in skimage 0.26 / removed in 0.28.
        from skimage.morphology import ball, closing
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(f"{_INSTALL_HINT}\n\noriginal error: {exc}") from exc
    return trimesh, marching_cubes, ball, closing


def generate_surface_marching_cubes(
    symbols: Sequence[str],
    coords: np.ndarray,
    scaling: float = 1.0,
    probe_radius: float = 1.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Marching-cubes molecular surface as ``(verts, faces)``.

    Inline port of ``oddt.surface.generate_surface_marching_cubes``. Like
    upstream it **ignores hydrogens** -- consistent with the integration's
    heavy-atoms-only decision, where the surface, the point cloud and the
    diffused atom set must describe the identical atoms.
    """
    _trimesh, marching_cubes, ball, closing = _require_shape_deps()

    if probe_radius < 0:
        raise ValueError("probe_radius needs to be a positive number")

    heavy = [i for i, s in enumerate(symbols) if str(s) != "H"]
    if not heavy:
        raise ValueError("molecule has no heavy atoms")
    coords = np.asarray(coords, dtype=np.float32)[heavy] * scaling
    # float32 to match oddt's atom_dict['radius'] dtype exactly, so the
    # `set(radii)` keys and `ball()` shapes come out bit-identical.
    radii = np.array(
        [OPENBABEL_VDW_RADII[str(symbols[i])] for i in heavy],
        dtype=np.float32,
    ) * scaling

    if radii.min() < 1:
        raise ValueError(
            "Scaling times the radius of the smallest atom must be > 1"
        )

    ball_dict = {radius: ball(radius, dtype=bool) for radius in set(radii)}
    ball_radii = np.array([ball_dict[r].shape[0] for r in radii])

    # Transform the coordinates because the grid starts at (0, 0, 0).
    min_coords = np.min(coords, axis=0)
    max_rad = np.max(ball_radii, axis=0)
    adjusted = np.round(coords - min_coords + max_rad * 5).astype(np.int64)
    offset = adjusted[0] - coords[0]

    ball_coord_min = (adjusted.T - np.floor(ball_radii / 2).astype(np.int64)).T
    ball_coord_max = (ball_coord_min.T + ball_radii).T

    grid = np.zeros(
        shape=ball_coord_max.max(axis=0) + int(8 * scaling), dtype=bool
    )
    for radius, cmin, cmax in zip(radii, ball_coord_min, ball_coord_max):
        grid[cmin[0]:cmax[0], cmin[1]:cmax[1], cmin[2]:cmax[2]] += (
            ball_dict[radius]
        )

    spacing = (1 / scaling,) * 3
    grid = closing(grid, ball(probe_radius * 2 * scaling)).astype(bool)
    verts, faces = marching_cubes(grid, level=0, spacing=spacing)[:2]
    return verts - offset / scaling, faces


def get_mesh(
    symbols: Sequence[str],
    coords: np.ndarray,
    scaling: float = 1.0,
    probe_radius: float = 1.4,
):
    """Molecular surface mesh (``trimesh.Trimesh``) via marching cubes."""
    trimesh, *_ = _require_shape_deps()
    verts, faces = generate_surface_marching_cubes(
        symbols, coords, scaling=scaling, probe_radius=probe_radius
    )
    return trimesh.Trimesh(vertices=np.asarray(verts),
                           faces=np.asarray(faces))


def shape_from_atoms(
    symbols: Sequence[str],
    coords: np.ndarray,
    shape_ae,
    num_samples: int = POINT_CLOUD_SAMPLES,
    device: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Full chain for one molecule.

    Returns ``{"shape_emb": (128, 3), "shape_center": (3,),
    "shape_volume": float}``, all CPU tensors/floats ready to cache.
    """
    mesh = get_mesh(symbols, coords)
    points = torch.as_tensor(
        np.asarray(mesh.sample(num_samples)), dtype=torch.float32
    )
    center = points.mean(dim=0)
    points = (points - center).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = shape_ae.encode(points).squeeze(0).cpu()

    return {
        "shape_emb": emb,
        "shape_center": center,
        # trimesh returns a signed volume; marching-cubes winding is not
        # guaranteed, so take the magnitude.
        "shape_volume": float(abs(mesh.volume)),
    }


def read_xyz(path: str, with_hydrogen: bool = False):
    """Minimal ``.xyz`` reader returning ``(symbols, coords)``.

    ``with_hydrogen=False`` mirrors the dataset's H filter so the surface,
    the point cloud and the diffused atom set describe the identical atoms.
    (``generate_surface_marching_cubes`` drops H a second time regardless,
    matching upstream oddt behaviour.)
    """
    symbols: List[str] = []
    coords: List[List[float]] = []
    with open(path) as fh:
        lines = fh.read().splitlines()
    n = int(lines[0].split()[0])
    for line in lines[2 : 2 + n]:
        parts = line.split()
        if not parts:
            continue
        sym = parts[0]
        if not with_hydrogen and sym == "H":
            continue
        symbols.append(sym)
        coords.append([float(p) for p in parts[1:4]])
    return symbols, np.asarray(coords, dtype=float)


if __name__ == "__main__":  # pragma: no cover - self-check
    check_vdw_radii()
    # Benzene ring, heavy atoms only.
    ang = np.arange(6) * np.pi / 3
    xyz = np.stack([1.39 * np.cos(ang), 1.39 * np.sin(ang),
                    np.zeros(6)], axis=1)
    m = get_mesh(["C"] * 6, xyz)
    # Coarse by construction: upstream runs scaling=1.0, i.e. a 1 A voxel
    # grid, so a benzene surface is a few dozen verts. Non-degenerate is the
    # bar, not smooth.
    assert len(m.vertices) > 20 and len(m.faces) > 20, m
    # Not watertight -- upstream passes level=0 on a boolean grid, so the
    # isosurface clips the grid edge. Faithful to what se.pt was trained on.
    assert 20.0 < abs(m.volume) < 500.0, m.volume
    print(f"OK: {len(m.vertices)} verts, {len(m.faces)} faces, "
          f"volume {abs(m.volume):.1f}")
