"""DiTMC conformer-generation task (``task_type: diffusion_ditmc``).

Wraps ``modules/models/ditmc`` in the platform's duck-typed ``Task`` contract
(docs/adding_new_models.md 2.1) and adapts the ``graph3d`` PyG ``Batch``
(``bond_collate: raw``) into the three graphs DiTMC consumes.

**This model does not invent molecules.** You hand it a molecule you already
have -- its atoms and its bonds -- and it returns 3D conformers of exactly that
molecule. Nothing about the composition is generated, so:

* :meth:`DiTMCTask.sample` **raises**. There is no unconditional mode: without a
  molecular graph there is nothing to place in 3D, and the platform's
  ``(one_hot, charges, coords, node_mask)`` return contract has no meaning here.
  Generation goes through :class:`DiTMCConformerGenerator`, pointed at by
  ``configs/interference/gen_ditmc_conformers.yaml`` -- the same route
  ``gen_diffdec_scaffold.yaml`` and ``gen_apo2mol_pocket.yaml`` take.
* ``node_dist_model`` / ``n_node_dist`` are deliberately absent. They exist to
  let ``GenerativeFactory`` choose a molecule size; DiTMC's size is dictated by
  the input molecule.

Bond mapping (canonical -> DiTMC): ``ditmc_class = canonical_class - 1``, so
``1=SINGLE -> 0``, ``2=DOUBLE -> 1``, ``3=TRIPLE -> 2``, ``4=AROMATIC -> 3``.
Canonical class 0 ("no bond") is simply never an edge of the conditioner graph,
which is exactly the platform's storage rule -- a pass-through. **Do not set
``kekulize: true``**: upstream trains on RDKit's aromatic perception and
aromatic is a real class in its 4-wide vocabulary.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch
from torch import nn

from MolecularDiffusion.modules.models.ditmc import (
    CondGraph,
    FlowMatching,
    LatentGraph,
    MoleculeFeatureCache,
    PriorGraph,
    build_prior,
    build_variant,
    center_data,
    node_attr_dim,
    rotation_augmentation,
)

logger = logging.getLogger(__name__)

#: canonical bond class -> DiTMC one-hot column. Class 0 is never materialized.
N_BOND_CLASSES = 4


class Graph3DBatchToDiTMCAdapter(nn.Module):
    """One ``graph3d`` PyG ``Batch`` -> ``(LatentGraph, CondGraph, PriorGraph)``.

    Everything derived per molecule (``node_attr``, shortest hops, the Laplacian
    eigendecomposition) is cached by SMILES, exactly as upstream caches it at
    dataset-build time.

    The one featurization column that is *derived* rather than copied is the
    chiral tag: upstream reads it off the GEOM mol, and here it is recovered by
    RDKit from the stored coordinates. It is the first thing to check if a
    converted pretrained checkpoint underperforms.
    """

    def __init__(self, dataset: str = "qm9") -> None:
        super().__init__()
        self.dataset = dataset
        self.cache = MoleculeFeatureCache(dataset=dataset)

    def forward(self, batch: Any):
        pyg = batch["graph"] if isinstance(batch, dict) else batch
        return self.build(pyg.to_data_list(), device=pyg.pos.device)

    def build(self, items: list, device=None):  # noqa: PLR0914
        node_attr, atomic_numbers, positions, batch_segments = [], [], [], []
        lat_rec, lat_sen, hops = [], [], []
        cond_sen, cond_rec, cond_attr = [], [], []
        prior_d, prior_sen, prior_rec, prior_attr = [], [], [], []
        offset = 0

        for gid, item in enumerate(items):
            n = int(item.pos.shape[0])
            attr, hop, d, p = self.cache.get(item)

            node_attr.append(torch.as_tensor(attr, dtype=torch.float32))
            atomic_numbers.append(item.z.long())
            positions.append(item.pos.float())
            batch_segments.append(torch.full((n,), gid, dtype=torch.long))

            # All ordered pairs, C-ORDER over (i, j), receivers=i, senders=j.
            # Same order as `hop`, which is the only reason they line up.
            idx = np.arange(n)
            ii, jj = np.meshgrid(idx, idx, indexing="ij")
            mask = ii != jj
            lat_rec.append(torch.as_tensor(ii[mask] + offset, dtype=torch.long))
            lat_sen.append(torch.as_tensor(jj[mask] + offset, dtype=torch.long))
            hops.append(torch.as_tensor(hop, dtype=torch.long))

            # Bond edges, mirrored to both directions; class -> one-hot col - 1.
            bi = item.bond_index.long().reshape(2, -1)
            bt = item.bond_type.long().reshape(-1)
            if bt.numel() and int(bt.max()) > N_BOND_CLASSES:
                msg = (
                    f"bond class {int(bt.max())} exceeds DiTMC's 4-class "
                    f"vocabulary (1=SINGLE 2=DOUBLE 3=TRIPLE 4=AROMATIC)"
                )
                raise ValueError(msg)
            both = torch.cat([bi, bi.flip(0)], dim=1)
            bt2 = torch.cat([bt, bt])
            cond_sen.append(both[0] + offset)
            cond_rec.append(both[1] + offset)
            cond_attr.append(
                torch.nn.functional.one_hot(
                    (bt2 - 1).clamp(min=0), N_BOND_CLASSES
                ).float()
            )

            # Harmonic prior: complete index grid, senders = eigen index.
            prior_d.append(torch.as_tensor(d, dtype=torch.float32))
            prior_rec.append(torch.as_tensor(ii.reshape(-1) + offset, dtype=torch.long))
            prior_sen.append(torch.as_tensor(jj.reshape(-1) + offset, dtype=torch.long))
            prior_attr.append(torch.as_tensor(p.reshape(-1), dtype=torch.float32))

            offset += n

        def cat(xs):
            return torch.cat(xs).to(device) if xs else torch.zeros(0)

        node_attr_t = torch.cat(node_attr).to(device)
        senders = cat(lat_sen)
        num_nodes = offset
        latent = LatentGraph(
            atomic_numbers=cat(atomic_numbers),
            node_attr=node_attr_t,
            positions=cat(positions),
            senders=senders,
            receivers=cat(lat_rec),
            shortest_hops=cat(hops),
            batch_segments=cat(batch_segments),
            num_graphs=len(items),
            cond_scaling_nodes=torch.ones(num_nodes, device=device),
            cond_scaling_edges=torch.ones(senders.shape[0], device=device),
            x1=cat(positions),
        )
        cond = CondGraph(
            node_attr=node_attr_t,
            senders=cat(cond_sen),
            receivers=cat(cond_rec),
            edge_attr=torch.cat(cond_attr).to(device)
            if cond_attr
            else torch.zeros(0, N_BOND_CLASSES, device=device),
        )
        prior = PriorGraph(
            node_attr=cat(prior_d),
            senders=cat(prior_sen),
            receivers=cat(prior_rec),
            edge_attr=cat(prior_attr),
        )
        return latent, cond, prior


class DiTMCTask(nn.Module):
    """DiTMC in the platform's Task contract."""

    def __init__(  # noqa: PLR0913
        self,
        variant: str,
        dataset: str,
        model_kwargs: dict,
        flow_kwargs: dict,
        prior: str = "harmonic",
        augmentation_bool: bool = True,
        atom_vocab: list | None = None,
        task_type: str = "diffusion_ditmc",
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.variant = variant
        self.dataset = dataset
        self.atom_vocab = list(atom_vocab) if atom_vocab else None
        self.augmentation_bool = augmentation_bool

        self.adapter = Graph3DBatchToDiTMCAdapter(dataset=dataset)
        self.net = build_variant(
            variant, node_attr_dim=node_attr_dim(dataset), **model_kwargs
        )
        self.process = FlowMatching(prior=build_prior(prior), **flow_kwargs)

    # -- contract ----------------------------------------------------------

    @property
    def model(self) -> "DiTMCTask":
        return self

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, batch: Any):
        latent, cond, prior = self.adapter(batch)
        if self.augmentation_bool and self.training:
            # One Haar rotation per molecule, applied to positions and x1 --
            # ON by default upstream (globals.augmentation_bool: True). Without
            # it dit_ape/dit_rpe, which are not equivariant by construction,
            # train on a different distribution than the published weights.
            rotated, _ = rotation_augmentation(
                {"positions": latent.positions, "x1": latent.x1},
                latent.batch_segments,
                latent.num_graphs,
            )
            latent = latent.replace(**rotated)
        return self.process.loss(self.net, latent, prior, cond)

    def predict_and_target(self, batch: Any, all_loss=None, metric=None):  # noqa: ARG002
        loss, _ = self.forward(batch)
        if loss.dim() == 0:
            loss = loss.unsqueeze(0)
        return loss.detach(), torch.zeros_like(loss)

    def evaluate(self, pred: torch.Tensor, target: torch.Tensor):  # noqa: ARG002
        return {"val_loss": pred.mean()}

    def sample(self, *args, **kwargs):  # noqa: ARG002
        msg = (
            "DiTMC has no unconditional generation mode -- it places an EXISTING "
            "molecule in 3D, so there is nothing to sample without one. Use "
            "configs/interference/gen_ditmc_conformers.yaml, which points at "
            "MolecularDiffusion.modules.tasks.diffusion_ditmc."
            "DiTMCConformerGenerator."
        )
        raise NotImplementedError(msg)

    # -- generation --------------------------------------------------------

    @torch.no_grad()
    def generate_conformers(  # noqa: PLR0913
        self,
        items: list,
        *,
        num_steps: int = 50,
        free_guidance_scale: float = 1.0,
        logarithmic_time_bool: bool = False,
        return_trajectory: bool = False,
        generator: torch.Generator | None = None,
    ):
        latent, cond, prior = self.adapter.build(items, device=self.device)
        out = self.process.sample(
            self.net,
            latent,
            prior,
            cond,
            num_steps=num_steps,
            free_guidance_scale=free_guidance_scale,
            logarithmic_time_bool=logarithmic_time_bool,
            return_trajectory=return_trajectory,
            generator=generator,
        )
        if return_trajectory:
            graph, traj = out
            return graph.positions, latent.batch_segments, traj
        return out.positions, latent.batch_segments, None


class DiTMCTaskFactory:
    """Factory instantiated by ``cli/train.py``.

    ``train_set`` is deliberately **not** declared: DiTMC needs no dataset
    statistics at construction (no marginals, no valency table, no size
    histogram), so the declarative injection seam simply does not fire.
    """

    def __init__(  # noqa: PLR0913
        self,
        task_type: str = "diffusion_ditmc",
        variant: str = "so3",
        dataset: str = "qm9",
        model: dict | None = None,
        flow: dict | None = None,
        prior: str = "harmonic",
        augmentation_bool: bool = True,
        atom_vocab: list | None = None,
        **kwargs: Any,
    ) -> None:
        self.task_type = task_type
        self.variant = variant
        self.dataset = dataset
        self.model_kwargs = _plain(model or {})
        self.flow_kwargs = _plain(flow or {})
        self.prior = prior
        self.augmentation_bool = augmentation_bool
        self.atom_vocab = list(atom_vocab) if atom_vocab else None
        self.kwargs = kwargs
        self.task: DiTMCTask | None = None

    def build(self) -> DiTMCTask:
        self.task = DiTMCTask(
            variant=self.variant,
            dataset=self.dataset,
            model_kwargs=self.model_kwargs,
            flow_kwargs=self.flow_kwargs,
            prior=self.prior,
            augmentation_bool=self.augmentation_bool,
            atom_vocab=self.atom_vocab,
            task_type=self.task_type,
        )
        n_params = sum(p.numel() for p in self.task.parameters())
        logger.info(
            "Built DiTMC variant=%s dataset=%s node_attr_dim=%d params=%d",
            self.variant,
            self.dataset,
            node_attr_dim(self.dataset),
            n_params,
        )
        return self.task


def _plain(cfg: Any) -> dict:
    """OmegaConf node -> plain dict (Hydra hands these over as DictConfig)."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass
    return dict(cfg)


class DiTMCConformerGenerator:
    """Generate conformers of molecules you supply.

    ``cli/generate.py`` does ``instantiate(cfg.interference, task=task)`` then
    ``.run()``, so every key in the interference YAML lands in ``__init__``.

    Inputs, in priority order: ``smiles`` (a string, a list, or a path to a file
    with one SMILES per line) or ``db_path`` (a graph3d ASE db) + ``indices``.
    SMILES are embedded with RDKit **only to get atoms and bonds** -- never
    coordinates; the coordinates are what the model produces.
    """

    def __init__(  # noqa: PLR0913
        self,
        task: Any,
        smiles: Any = None,
        db_path: str | None = None,
        indices: Any = None,
        num_generate: int = 5,
        batch_size: int = 8,
        num_steps: int = 50,
        free_guidance_scale: float = 1.0,
        logarithmic_time_bool: bool = False,
        use_chirality_correction: bool = False,
        n_frames: int = 0,
        max_atom: int = 200,
        seed: int = 42,
        device: str | None = None,
        output_path: str = "generated_ditmc",
    ) -> None:
        self.task = getattr(task, "task", task)
        self.smiles = smiles
        self.db_path = db_path
        self.indices = indices
        self.num_generate = num_generate
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.free_guidance_scale = free_guidance_scale
        self.logarithmic_time_bool = logarithmic_time_bool
        self.use_chirality_correction = use_chirality_correction
        self.n_frames = n_frames
        self.max_atom = max_atom
        self.seed = seed
        self.output_path = output_path
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    # -- inputs ------------------------------------------------------------

    def _items_from_smiles(self) -> list:
        from rdkit import Chem
        from torch_geometric.data import Data

        # Pre-import: MolecularDiffusion.data.component.graph3d_dataset has a
        # latent import cycle with MolecularDiffusion.data.dataset that only
        # bites when graph3d_dataset is the FIRST of the two imported. This is
        # pre-existing platform behaviour, not something to fix from here.
        import MolecularDiffusion.data.dataset  # noqa: F401

        from MolecularDiffusion.data.component.graph3d_dataset import (
            rdkit_bond_types,
        )

        smis = self.smiles
        if isinstance(smis, str) and os.path.exists(smis):
            with open(smis) as fh:
                smis = [ln.strip() for ln in fh if ln.strip()]
        elif isinstance(smis, str):
            smis = [smis]

        inv = {bt: i for i, bt in enumerate(rdkit_bond_types()) if bt is not None}
        items = []
        for smi in smis:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                logger.warning("RDKit could not parse SMILES %r, skipping", smi)
                continue
            mol = Chem.AddHs(mol)
            n = mol.GetNumAtoms()
            bi, bt = [], []
            for bond in mol.GetBonds():
                bi.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                bt.append(inv[bond.GetBondType()])
            items.append(
                Data(
                    pos=torch.zeros(n, 3),
                    z=torch.tensor(
                        [a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long
                    ),
                    fc=torch.tensor(
                        [a.GetFormalCharge() for a in mol.GetAtoms()], dtype=torch.long
                    ),
                    bond_index=torch.tensor(bi, dtype=torch.long).reshape(-1, 2).T
                    if bi
                    else torch.zeros(2, 0, dtype=torch.long),
                    bond_type=torch.tensor(bt, dtype=torch.long)
                    if bt
                    else torch.zeros(0, dtype=torch.long),
                    n_nodes=n,
                    smiles=Chem.MolToSmiles(mol),
                )
            )
        return items

    def _items_from_db(self) -> list:
        # Pre-import: MolecularDiffusion.data.component.graph3d_dataset has a
        # latent import cycle with MolecularDiffusion.data.dataset that only
        # bites when graph3d_dataset is the FIRST of the two imported. This is
        # pre-existing platform behaviour, not something to fix from here.
        import MolecularDiffusion.data.dataset  # noqa: F401

        from MolecularDiffusion.data.component.graph3d_dataset import Graph3DDataset

        ds = Graph3DDataset(
            root=os.path.dirname(self.db_path) or ".",
            ase_db_path=self.db_path,
            dataset_name=f"ditmc_gen_{os.path.basename(self.db_path)}",
            max_atom=self.max_atom,
            atom_vocab=self.task.atom_vocab,
            allow_unknown=True,
            kekulize=False,
            center_coords=True,
            compute_stats=False,
        )
        idx = self.indices
        if idx is None:
            idx = list(range(min(len(ds), 10)))
        elif isinstance(idx, int):
            idx = [idx]
        return [ds[int(i)]["graph"] for i in idx]

    def _load_items(self) -> list:
        if self.smiles is not None:
            return self._items_from_smiles()
        if self.db_path is not None:
            return self._items_from_db()
        msg = (
            "DiTMC needs a molecule to place in 3D. Set `smiles` (a string, a "
            "list, or a path to a file of SMILES) or `db_path` (+ `indices`) in "
            "the interference config."
        )
        raise ValueError(msg)

    # -- output ------------------------------------------------------------

    @staticmethod
    def _write_xyz(path: str, symbols, coords) -> None:
        with open(path, "w") as fh:
            fh.write(f"{len(symbols)}\n\n")
            for s, (x, y, z) in zip(symbols, coords, strict=True):
                fh.write(f"{s} {x:.6f} {y:.6f} {z:.6f}\n")

    def run(self) -> str:
        from rdkit import Chem
        from rdkit.Chem import rdMolTransforms  # noqa: F401  (import parity)

        # Pre-import: MolecularDiffusion.data.component.graph3d_dataset has a
        # latent import cycle with MolecularDiffusion.data.dataset that only
        # bites when graph3d_dataset is the FIRST of the two imported. This is
        # pre-existing platform behaviour, not something to fix from here.
        import MolecularDiffusion.data.dataset  # noqa: F401

        from MolecularDiffusion.data.component.graph3d_dataset import build_rdkit_mol

        os.makedirs(self.output_path, exist_ok=True)
        torch.manual_seed(self.seed)
        # The generator device must match the tensors it fills, or torch.randn
        # raises. Bind it to wherever the task actually landed.
        gen = torch.Generator(device=self.device).manual_seed(self.seed)

        self.task = self.task.to(self.device).eval()
        items = self._load_items()
        if not items:
            msg = "No input molecules could be built; nothing to generate."
            raise ValueError(msg)
        logger.info("DiTMC: %d input molecules", len(items))

        written = 0
        for mol_i, item in enumerate(items):
            replicas = [item] * self.num_generate
            all_coords = []
            for start in range(0, len(replicas), self.batch_size):
                chunk = replicas[start : start + self.batch_size]
                coords, segments, _traj = self.task.generate_conformers(
                    chunk,
                    num_steps=self.num_steps,
                    free_guidance_scale=self.free_guidance_scale,
                    logarithmic_time_bool=self.logarithmic_time_bool,
                    return_trajectory=self.n_frames > 0,
                    generator=gen,
                )
                coords = center_data(coords, segments, len(chunk)).cpu()
                for k in range(len(chunk)):
                    all_coords.append(coords[segments.cpu() == k].numpy())

            z = item.z.numpy()
            symbols = [Chem.Atom(int(v)).GetSymbol() for v in z]
            for k, xyz in enumerate(all_coords):
                self._write_xyz(
                    os.path.join(self.output_path, f"mol_{mol_i:04d}_conf_{k:03d}.xyz"),
                    symbols,
                    xyz,
                )
                written += 1

            # One SDF per input molecule, carrying the INPUT bonds so that
            # `MolCraftDiff analyze` has a real molecule to work with.
            sdf_path = os.path.join(self.output_path, f"mol_{mol_i:04d}.sdf")
            writer = Chem.SDWriter(sdf_path)
            for xyz in all_coords:
                try:
                    mol = build_rdkit_mol(
                        z,
                        item.bond_index.numpy(),
                        item.bond_type.numpy(),
                        formal_charge=item.fc.numpy(),
                        coords=xyz,
                    )
                    writer.write(mol)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("SDF write failed for molecule %d: %s", mol_i, exc)
            writer.close()

        logger.info("DiTMC wrote %d conformers to %s", written, self.output_path)
        return self.output_path


ModelTaskFactory = DiTMCTaskFactory
