"""Conformer generation -- k 3D shapes of each molecule you already have.

This is an ADDITIVE sibling of ``GenerativeFactory``, not a modification of
it, following the same route as ``tasks_gcdm_optimize.py``:
``cli/generate.py`` does ``hydra.utils.instantiate(cfg.interference,
task=task)`` then ``generator.run()``, the ``GenerativeFactory`` annotation on
that line is cosmetic, and there is no registry -- so a new class in a new
file plugs in purely through its ``_target_``. Nothing in
``tasks_generate.py`` is touched.

WHY IT EXISTS

    A conformer generator is handed a molecule and returns 3D shapes OF THAT
    MOLECULE. Run through ``gen_unconditional`` that is mis-modelled in four
    ways:

    1. ``mol_size`` chooses a molecule SIZE, but a conformer generator's size
       is dictated by its input. For LoQI the drawn size is only used to snap
       to the nearest pool molecule, so ``mol_size`` silently selects WHICH
       molecule you get.
    2. ``num_generate`` is a flat sample count, so "20 conformers of THIS
       molecule" is inexpressible.
    3. Output is ``molecule_0.xyz``, ``molecule_1.xyz`` ... with nothing
       recording which input each conformer belongs to -- and ``.xyz`` has no
       bond channel, so that mapping is the only link back to the chemistry.
    4. De-novo metrics (uniqueness, novelty) are ~0 BY CONSTRUCTION here.

    This mode asks for ``conformers_per_molecule``, refuses ``mol_size``
    rather than accepting-and-ignoring it, groups output per input molecule,
    and scores with conformer metrics.

TASK CONTRACT (duck-typed, no registry, no base class)

    A task works with this mode if it exposes:

    * ``conditioning_pool() -> list[Data]`` -- the input molecules, as
      ``graph3d`` per-item ``Data`` (``pos``, ``z``, ``atom_idx``, ``fc``,
      ``bond_index``, ``bond_type``, ``n_nodes``);
    * ``sample(mols=[...])`` -- generate one conformer per entry of ``mols``,
      returning the platform tuple ``(one_hot, charges, coords, node_mask)``.

    Anything else raises a clear TypeError at construction.

OUTPUT LAYOUT

    output_path/
      mol_0000/conformer_000.xyz ...   <- one .xyz per conformer (what every
                                          analyze/ command already reads)
      mol_0000/conformers.sdf          <- the SAME conformers WITH bonds and
                                          formal charges, multi-conformer sdf
      mol_0000/reference.sdf           <- the input molecule + its geometry
      conformers.csv                   <- one row per conformer: which input it
                                          came from, its SMILES, its metrics

    The per-molecule directory is the mapping (3), the .sdf carries the bond
    channel .xyz cannot (also 3), and the .csv is the machine-readable index.
    Molecule building reuses ``build_rdkit_mol`` from the graph3d dataset and
    coordinate writing reuses ``save_xyz_file``; no parallel writer.

METRICS (and what is deliberately NOT computed)

    Per conformer, against the INPUT molecule's own geometry:

    * ``rmsd`` -- heavy-atom symmetry-corrected RMSD (``GetBestRMS`` after
      ``RemoveHs``), the GEOM/conformer-benchmark convention.
    * ``energy`` -- xTB single-point, opt-in (``xtb_energy: true``) and only
      if the ``xtb`` binary is on PATH; reuses ``analyze.compare_to_optimized
      .get_xtb_energy``. Otherwise the column is empty, never a made-up
      number.

    Aggregated per molecule and logged: ``rmsd_min`` (the MAT-style matching
    number) and ``coverage`` (fraction of conformers within
    ``rmsd_threshold``, precision-style).

    NOT computed: recall-side COV-R / MAT-R. Those need a reference ENSEMBLE
    of experimentally/CREST-derived conformers per molecule, and the input
    pool carries exactly one geometry per molecule. Reporting them off a
    single reference would be a meaningless number.

    HONESTY CAVEAT: ``rmsd`` is RMSD to the input geometry. That is a true
    reference only when the input carried real 3D (.sdf, or a graph3d .db).
    A SMILES input has an ETKDG conformer embedded for it, so the "reference"
    is itself a prediction -- the run logs a warning when it detects this.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any

import pandas as pd
import torch

from MolecularDiffusion.utils.geom_utils import save_xyz_file

logger = logging.getLogger(__name__)


def _to_mol(item: Any, coords) -> Any:
    """graph3d ``Data`` + coordinates -> RDKit mol. ``None`` if unbuildable."""
    from MolecularDiffusion.data.component.graph3d_dataset import (
        build_rdkit_mol,
    )

    try:
        return build_rdkit_mol(
            item.z.cpu().numpy(),
            item.bond_index.cpu().numpy(),
            item.bond_type.cpu().numpy(),
            formal_charge=item.fc.cpu().numpy(),
            coords=coords,
        )
    except Exception as exc:  # noqa: BLE001 - chemistry, not a bug
        logger.warning("Could not rebuild molecule: %s", exc)
        return None


def _rmsd(probe: Any, ref: Any) -> float | None:
    """Heavy-atom, symmetry-corrected RMSD (GEOM benchmark convention)."""
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign

    if probe is None or ref is None:
        return None
    try:
        return float(
            rdMolAlign.GetBestRMS(
                Chem.RemoveHs(Chem.Mol(probe)), Chem.RemoveHs(Chem.Mol(ref))
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("RMSD failed: %s", exc)
        return None


class ConformerFactory:
    """Generate ``conformers_per_molecule`` conformers of every input molecule.

    Deliberately does NOT accept ``mol_size`` or ``num_generate``: the input
    molecule dictates its own atom count, and the request here is per-molecule,
    not a flat total. Passing either raises rather than being ignored.
    """

    def __init__(  # noqa: PLR0913
        self,
        task: Any,
        conformers_per_molecule: int = 10,
        batch_size: int = 8,
        max_molecules: int = 0,
        rmsd_threshold: float = 0.5,
        xtb_energy: bool = False,
        seed: int | None = None,
        output_path: str = "generated_conformers",
        **kwargs: Any,
    ) -> None:
        rejected = sorted(
            set(kwargs) & {"mol_size", "num_generate", "max_mol_size"}
        )
        if rejected:
            msg = (
                f"{', '.join(rejected)} is meaningless for conformer generation "
                "and is refused rather than silently ignored: the input molecule "
                "dictates its own atom count, and the request is "
                "conformers_per_molecule, not a flat sample total. Remove it from "
                "your interference config (see configs/interference/gen_conformer.yaml)."
            )
            raise ValueError(msg)
        for unknown in sorted(set(kwargs) - {"task_type"}):
            logger.warning("Ignoring unknown interference key %r", unknown)

        missing = [
            name
            for name in ("conditioning_pool", "sample")
            if not callable(getattr(task, name, None))
        ]
        if missing:
            msg = (
                f"{type(task).__name__} is not a conformer generator: it lacks "
                f"{', '.join(missing)}. This mode needs a task exposing "
                "conditioning_pool() and sample(mols=[...]) -- see the module "
                "docstring for the contract."
            )
            raise TypeError(msg)

        self.task = task
        self.conformers_per_molecule = int(conformers_per_molecule)
        self.batch_size = max(1, int(batch_size))
        self.max_molecules = int(max_molecules)
        self.rmsd_threshold = float(rmsd_threshold)
        self.xtb_energy = bool(xtb_energy)
        self.seed = seed
        self.output_path = output_path

        if self.xtb_energy and shutil.which("xtb") is None:
            logger.warning(
                "xtb_energy=true but the 'xtb' binary is not on PATH; the energy "
                "column will be empty. Install xtb from conda to populate it."
            )
            self.xtb_energy = False

    # -- helpers -------------------------------------------------------------

    def _warn_if_reference_is_a_guess(self) -> None:
        source = getattr(self.task, "sample_input", None)
        looks_2d = not isinstance(source, str) or source.lower().endswith(
            (".smi", ".txt", ".smiles")
        )
        if looks_2d:
            logger.warning(
                "Input molecules came from SMILES, so the 'reference' geometry is "
                "itself an ETKDG guess. The rmsd column measures distance to that "
                "guess, NOT to an experimental conformer -- read it as a spread, "
                "not as accuracy. Use an .sdf/.db input for a real reference."
            )

    def _energy(self, xyz_path: str, charge: int) -> float | None:
        if not self.xtb_energy:
            return None
        from MolecularDiffusion.runmodes.analyze.compare_to_optimized import (
            get_xtb_energy,
        )

        return get_xtb_energy(xyz_path, charge=charge)

    def _write_batch(self, mol_dir: str, start: int, sampled) -> list[str]:
        """Write one sampled batch as ``conformer_XXX.xyz``; return the paths."""
        one_hot, _charges, coords, node_mask = sampled
        # ponytail: the platform writer drops atoms within 1e-4 of the origin
        # (its long-standing convention). The .sdf written alongside is built
        # from the tensors directly and keeps every atom.
        save_xyz_file(
            mol_dir,
            one_hot,
            coords,
            atom_decoder=self.task.atom_vocab,
            id_from=start,
            name="conformer",
            node_mask=node_mask,
        )
        return [
            os.path.join(mol_dir, f"conformer_{start + j:03d}.xyz")
            for j in range(one_hot.size(0))
        ]

    # -- entry point ---------------------------------------------------------

    def run(self) -> None:  # noqa: C901, PLR0912, PLR0915
        from rdkit import Chem
        from tqdm import tqdm

        if self.seed is not None:
            torch.manual_seed(int(self.seed))

        pool = self.task.conditioning_pool()
        if self.max_molecules > 0:
            pool = pool[: self.max_molecules]
        if not pool:
            msg = "The conditioning pool is empty; nothing to make conformers of."
            raise ValueError(msg)
        self._warn_if_reference_is_a_guess()

        logger.info(
            "Conformer generation: %d conformers x %d input molecules -> %s",
            self.conformers_per_molecule,
            len(pool),
            self.output_path,
        )

        rows: list[dict] = []
        failures = 0
        progress = tqdm(list(enumerate(pool)), desc="Molecules", leave=True)
        for mol_idx, item in progress:
            mol_dir = os.path.join(
                self.output_path, f"mol_{str(mol_idx).zfill(4)}"
            )
            os.makedirs(mol_dir, exist_ok=True)

            ref = _to_mol(item, item.pos.cpu().numpy())
            smiles = (
                Chem.MolToSmiles(Chem.RemoveHs(ref))
                if ref is not None
                else None
            )
            charge = int(item.fc.sum())
            if ref is not None:
                with Chem.SDWriter(
                    os.path.join(mol_dir, "reference.sdf")
                ) as w:
                    w.write(ref)

            written = 0
            sdf_path = os.path.join(mol_dir, "conformers.sdf")
            writer = Chem.SDWriter(sdf_path)
            try:
                while written < self.conformers_per_molecule:
                    n = min(
                        self.batch_size, self.conformers_per_molecule - written
                    )
                    try:
                        sampled = self.task.sample(mols=[item] * n)
                        paths = self._write_batch(mol_dir, written, sampled)
                    except Exception as exc:  # noqa: BLE001 - one bad batch != dead run
                        failures += 1
                        tqdm.write(f"[mol {mol_idx}] sampling failed: {exc}")
                        break
                    coords = sampled[2]
                    for j, path in enumerate(paths):
                        gen = _to_mol(
                            item, coords[j, : int(item.n_nodes)].cpu().numpy()
                        )
                        if gen is not None:
                            writer.write(gen)
                        rows.append(
                            {
                                "mol_index": mol_idx,
                                "conformer_index": written + j,
                                "smiles": smiles,
                                "n_atoms": int(item.n_nodes),
                                "xyz": os.path.relpath(path, self.output_path),
                                "rmsd": _rmsd(gen, ref),
                                "energy_hartree": self._energy(path, charge),
                            }
                        )
                    written += len(paths)
            finally:
                writer.close()

            done = [r for r in rows if r["mol_index"] == mol_idx]
            got = [r["rmsd"] for r in done if r["rmsd"] is not None]
            progress.set_postfix(
                {
                    "conformers": len(done),
                    "rmsd_min": f"{min(got):.2f}" if got else "n/a",
                }
            )

        if not rows:
            msg = (
                "No conformers were written: every input molecule failed to sample. "
                "The first failure is above."
            )
            raise RuntimeError(msg)

        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.output_path, "conformers.csv")
        df.to_csv(csv_path, index=False)

        valid = df["rmsd"].notna()
        logger.info(
            "Wrote %d conformers for %d molecules -> %s",
            len(df),
            df["mol_index"].nunique(),
            csv_path,
        )
        if valid.any():
            per_mol = df[valid].groupby("mol_index")["rmsd"]
            logger.info(
                "RMSD to input geometry (heavy-atom, symmetry-corrected): "
                "median %.3f A, best-per-molecule (MAT-style) mean %.3f A, "
                "coverage within %.2f A %.1f%% "
                "(precision-side only -- COV-R/MAT-R need a reference ensemble)",
                float(df.loc[valid, "rmsd"].median()),
                float(per_mol.min().mean()),
                self.rmsd_threshold,
                100
                * float((df.loc[valid, "rmsd"] <= self.rmsd_threshold).mean()),
            )
        else:
            logger.warning(
                "RMSD unavailable for every conformer; no metric reported."
            )
        if failures:
            logger.warning(
                "%d molecule(s) failed to sample completely.", failures
            )
