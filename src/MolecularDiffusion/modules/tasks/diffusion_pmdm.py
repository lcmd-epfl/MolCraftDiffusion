"""PMDM task: pocket-conditioned 3D ligand diffusion.

Three objects, mirroring the platform's usual layout:

* :class:`PMDMDiffusionTask` -- the duck-typed Task (docs/adding_new_models.md
  Section 2.1) wrapping :class:`~...models.pmdm.PMDMEpsNet`.
* :class:`ModelTaskFactory` -- the ``_target_`` of
  ``configs/tasks/diffusion_pmdm.yaml``.
* :class:`PMDMPocketGenerator` -- the ``_target_`` of
  ``configs/interference/gen_pmdm_pocket.yaml``. ``GenerativeFactory``'s
  ``sample(batch_size, nodesxsample, ...)`` has no channel for "which
  pocket", so pocket-conditioned models get their own generator behind their
  own ``_target_``; that is the established in-tree pattern
  (``DiffPharmaPocketGenerator``, the DiffSMol shape generator) and needs no
  core change -- ``cli/generate.py`` only does
  ``instantiate(cfg.interference, task=task)`` then ``.run()``.

The batch is NOT a PointCloud dict: PMDM needs a diffused ligand cloud plus a
fixed pocket cloud, flat-concatenated with scatter indices. The collate in
``data/component/pmdm_data.py`` already emits PMDM's own attribute names, so
the adapter here is a one-line ``SimpleNamespace(**batch)``.

Out of scope this pass (see the integration plan): ``inpainting_sample`` /
``linker_sample``, the VAE-latent and property-context branches, and CFG /
gradient guidance.
"""

from __future__ import annotations

import random
from collections import Counter
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn

from MolecularDiffusion.data.component.pmdm_data import (
    PMDM_ATOM_VOCAB,
    PMDMDataset,
    fully_connected_edges,
)
from MolecularDiffusion.modules.models.pmdm import PMDMEpsNet
from MolecularDiffusion.modules.tasks.pocket_generator import PocketGenerator

INT_TYPE = torch.int64

#: Widest ligand the size histogram can represent. Fixed so the buffer's
#: shape never depends on the training set -- a checkpoint trained on one
#: dataset must still load into a task built with ``train_set=None``
#: (which is exactly what ``cli/generate.py`` does).
MAX_LIGAND_SIZE = 128

#: Closest two atoms may sit before the geometry is unphysical. The shortest
#: real bond here is H-H at ~0.74 A; every one of the 31 reference CrossDocked
#: ligands has a minimum interatomic distance of 1.39 A.
MIN_ATOM_SEPARATION = 0.9

#: Slack on the sum of covalent radii when deciding two atoms are bonded --
#: the usual convention, and what ``utils/smilify.py`` assumes.
BOND_TOLERANCE = 0.4

#: Symbol -> Z for the ligand vocabulary. ``others`` is upstream's Z=119
#: placeholder; it maps to 0, for which ASE returns a 0.2 A stub radius, so
#: such an atom only ever bonds at implausibly short range and usually leaves
#: the molecule fragmented -- which is the outcome we want for a placeholder.
ATOMIC_NUMBERS = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9,
    "P": 15, "S": 16, "Cl": 17, "Se": 34, "others": 0,
}

#: Keys the epsilon network reads off the batch.
_BATCH_KEYS = (
    "ligand_pos",
    "ligand_atom_feature",
    "ligand_bond_index",
    "ligand_bond_type",
    "ligand_element_batch",
    "protein_pos",
    "protein_atom_feature_full",
    "protein_element_batch",
)


class LigandSizeDistribution:
    """Ligand-atom-count prior, sampled at generation time.

    Same shape of interface as the other in-tree node distributions
    (``TabascoNodeDistribution``, ``DiffSMolShapeNodeDistribution``):
    ``.histogram`` / ``.n_node_dist`` / ``.sample(n)``. Written here rather
    than imported from ``diffusion_tabasco`` because that module hard-imports
    ``tensordict`` and the whole TABASCO flow stack at module scope.
    """

    def __init__(self, histogram: Dict[int, float]) -> None:
        self.histogram = {int(k): float(v) for k, v in histogram.items() if v > 0}
        self.n_node_dist = self.histogram

    def sample(self, n_samples: int) -> torch.Tensor:
        if not self.histogram:
            raise ValueError(
                "The ligand-size histogram is empty: this task was built without "
                "a train_set (e.g. loaded from a checkpoint trained before the "
                "histogram buffer existed). Set interference.mol_size to an "
                "explicit list of ligand sizes."
            )
        sizes = random.choices(
            list(self.histogram.keys()), weights=list(self.histogram.values()), k=n_samples
        )
        return torch.tensor(sizes, dtype=INT_TYPE)


class PMDMDiffusionTask(nn.Module):
    """Task contract around :class:`PMDMEpsNet`."""

    def __init__(
        self,
        model: PMDMEpsNet,
        atom_vocab: Optional[List[str]] = None,
        size_histogram: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        # attribute name must stay `model`: cli/generate.py reaches for
        # task.model when reconciling diffusion_steps.
        self.model = model
        self.atom_vocab = list(atom_vocab) if atom_vocab else list(PMDM_ATOM_VOCAB)
        self.prop_dist_model = None
        self.split = "train"
        self._node_dist_override = None

        hist = torch.zeros(MAX_LIGAND_SIZE)
        if size_histogram is not None:
            n = min(len(size_histogram), MAX_LIGAND_SIZE)
            hist[:n] = size_histogram[:n]
        # a buffer, so the prior round-trips through the checkpoint: the
        # factory that rebuilds this task at generation time has no train_set.
        self.register_buffer("size_histogram", hist)

    # -- contract properties -------------------------------------------- #
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def node_dist_model(self) -> LigandSizeDistribution:
        if self._node_dist_override is not None:
            return self._node_dist_override
        return LigandSizeDistribution(
            {i: float(c) for i, c in enumerate(self.size_histogram.tolist())}
        )

    @node_dist_model.setter
    def node_dist_model(self, value) -> None:
        # cli/generate.py assigns this when an edm_stat.pkl sidecar exists.
        self._node_dist_override = value

    @property
    def n_node_dist(self) -> Dict[int, float]:
        return self.node_dist_model.n_node_dist

    # -- training -------------------------------------------------------- #
    def _adapt(self, batch: Dict[str, Any]) -> SimpleNamespace:
        device = self.device
        return SimpleNamespace(
            **{
                k: batch[k].to(device) if torch.is_tensor(batch[k]) else batch[k]
                for k in _BATCH_KEYS
            }
        )

    def forward(self, batch):
        loss, loss_pos, loss_node = self.model(
            self._adapt(batch), return_unreduced_loss=True
        )
        loss = loss.mean()
        return loss, {
            "loss": loss,
            "loss_pos": loss_pos.mean(),
            "loss_node": loss_node.mean(),
        }

    def predict_and_target(self, batch):
        loss, _ = self.forward(batch)
        pred = loss.detach().reshape(1)
        return pred, torch.zeros_like(pred)

    def evaluate(self, pred, target):  # noqa: ARG002 - target is a zeros stub
        return {"val_loss": pred.mean()}

    # -- generation ------------------------------------------------------ #
    @torch.no_grad()
    def sample(
        self,
        batch_size=None,
        nodesxsample=None,
        num_steps=None,
        batch=None,
        **kwargs,
    ):
        """Sample ligands inside the pocket carried by ``batch``.

        **The signature deliberately deviates from Section 2.1**: it takes a
        pocket. ``batch`` must hold the ``protein_*`` keys of a collated PMDM
        batch (:class:`PMDMPocketGenerator` builds it). There is no
        unconditional mode -- the network has no path that runs without a
        pocket.

        Returns ``(one_hot, charges, coords, node_mask)`` padded to
        ``(B, N, .)``, in the ORIGINAL pocket frame. ``charges`` is zeros:
        PMDM's pocket configuration has no charge channel.
        """
        if batch is None:
            raise ValueError(
                "PMDMDiffusionTask.sample() is pocket-conditioned: pass "
                "batch=<dict with protein_pos / protein_atom_feature_full / "
                "protein_element_batch>. Use interference: gen_pmdm_pocket, "
                "not gen_unconditional."
            )
        device = self.device
        protein_batch = batch["protein_element_batch"].to(device, INT_TYPE)
        n_samples = int(protein_batch.max().item()) + 1

        if nodesxsample is None:
            nodesxsample = self.node_dist_model.sample(n_samples)
        sizes = torch.as_tensor(nodesxsample).long().tolist()
        if len(sizes) != n_samples:
            raise ValueError(
                f"nodesxsample has {len(sizes)} entries but the batch carries "
                f"{n_samples} pockets."
            )

        total = int(sum(sizes))
        num_atom = self.model.num_atom
        ligand_batch = torch.repeat_interleave(
            torch.arange(n_samples, device=device), torch.tensor(sizes, device=device)
        )
        edge_index = fully_connected_edges(sizes).to(device)

        ligand_pos, _pos_traj, ligand_atom_type, _atom_traj = (
            self.model.langevin_dynamics_sample(
                ligand_atom_type=torch.randn(total, num_atom, device=device),
                ligand_pos_init=torch.randn(total, 3, device=device),
                ligand_bond_index=edge_index,
                ligand_bond_type=torch.full(
                    (edge_index.size(1),), 2, dtype=INT_TYPE, device=device
                ),
                ligand_batch=ligand_batch,
                protein_atom_feature_full=batch["protein_atom_feature_full"].to(
                    device, torch.float32
                ),
                protein_pos=batch["protein_pos"].to(device, torch.float32),
                protein_batch=protein_batch,
                num_graphs=n_samples,
                n_steps=int(num_steps or self.model.T),
                **kwargs,
            )
        )
        return self._pad(ligand_pos, ligand_atom_type, sizes)

    @staticmethod
    def _pad(coords_flat, feats_flat, sizes):
        """flat ``(n_total, .)`` -> padded ``(B, N, .)`` contract shape."""
        n_samples = len(sizes)
        n_max = max(sizes)
        device = coords_flat.device
        one_hot = torch.zeros(
            (n_samples, n_max, feats_flat.size(-1)), device=device, dtype=feats_flat.dtype
        )
        coords = torch.zeros((n_samples, n_max, 3), device=device)
        node_mask = torch.zeros((n_samples, n_max), device=device)

        offset = 0
        for i, n in enumerate(sizes):
            coords[i, :n] = coords_flat[offset : offset + n]
            one_hot[i, :n] = feats_flat[offset : offset + n]
            node_mask[i, :n] = 1
            offset += n
        charges = torch.zeros((n_samples, n_max), device=device)
        return one_hot, charges, coords, node_mask


class ModelTaskFactory:
    """Hydra entry point for ``configs/tasks/diffusion_pmdm.yaml``."""

    def __init__(
        self,
        task_type: str = "diffusion_pmdm",
        num_atom: int = 10,
        protein_feature_dim: int = 31,
        hidden_dim: int = 128,
        protein_hidden_dim: int = 128,
        num_convs: int = 3,
        num_convs_local: int = 3,
        protein_num_convs: int = 2,
        cutoff: float = 3.0,
        g_cutoff: float = 6.0,
        encoder_cutoff: float = 6.0,
        edge_order: int = 3,
        mlp_act: str = "relu",
        edge_encoder: str = "mlp",
        soft_edge: bool = True,
        norm_coors: bool = True,
        beta_schedule: str = "sigmoid",
        beta_start: float = 1.0e-7,
        beta_end: float = 2.0e-3,
        num_diffusion_timesteps: int = 1000,
        atom_vocab: Optional[List[str]] = None,
        train_set: Any = None,
        **kwargs: Any,
    ) -> None:
        # `train_set` is declared so cli/train.py's declarative seam injects
        # the training set (for the ligand-size prior). **kwargs absorbs the
        # overrides cli/train.py passes unconditionally (node_feature,
        # node_feature_choice, node_feature_dim, ...).
        self.task_type = task_type
        self.atom_vocab = list(atom_vocab) if atom_vocab else list(PMDM_ATOM_VOCAB)
        self.condition_names: List[str] = []
        self.train_set = train_set
        self.model_kwargs = dict(
            num_atom=num_atom,
            protein_feature_dim=protein_feature_dim,
            hidden_dim=hidden_dim,
            protein_hidden_dim=protein_hidden_dim,
            num_convs=num_convs,
            num_convs_local=num_convs_local,
            protein_num_convs=protein_num_convs,
            cutoff=cutoff,
            g_cutoff=g_cutoff,
            encoder_cutoff=encoder_cutoff,
            edge_order=edge_order,
            mlp_act=mlp_act,
            edge_encoder=edge_encoder,
            soft_edge=soft_edge,
            norm_coors=norm_coors,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
        self.task: Optional[PMDMDiffusionTask] = None

    def _size_histogram(self) -> Optional[torch.Tensor]:
        if self.train_set is None:
            return None
        hist = torch.zeros(MAX_LIGAND_SIZE)
        for item in self.train_set:
            n = len(item["ligand_pos"])
            if n < MAX_LIGAND_SIZE:
                hist[n] += 1
        return hist

    def build(self) -> PMDMDiffusionTask:
        self.task = PMDMDiffusionTask(
            PMDMEpsNet(**self.model_kwargs),
            atom_vocab=self.atom_vocab,
            size_histogram=self._size_histogram(),
        )
        return self.task


class PMDMPocketGenerator(PocketGenerator):
    """Pocket-conditioned generation behind ``interference/gen_pmdm_pocket``.

    The pocket comes from one row of a converted ASE db
    (``docs/model_integrations/pmdm/scripts/convert_dataset.py``), read with
    ``center=False`` so the sampled ligand comes back in that pocket's own
    frame.

    The only generator that may write FEWER than ``num_generate``: unusable
    samples are filtered out and resampled, bounded by ``max_retries``.

    The sampling loop itself lives in :class:`PocketGenerator`.
    """

    tag = "pmdm"
    # never seeded numpy; seeding it would change what this model generates.
    seed_numpy = False
    db_required_msg = (
        "interference.pocket_db is required: PMDM has no unconditional "
        "mode. Point it at a converted ASE db "
        "(scripts/convert_dataset.py)."
    )

    def __init__(
        self,
        task,
        pocket_db: Optional[str] = None,
        pocket_index: int = 0,
        num_generate: int = 20,
        batch_size: int = 4,
        num_steps: Optional[int] = None,
        mol_size: Optional[list] = None,
        sampling_type: str = "generalized",
        step_lr: float = 1.0e-6,
        clip: float = 1000.0,
        clip_pos: Optional[float] = None,
        global_start_sigma: Optional[float] = None,
        w_global_pos: float = 1.0,
        w_global_node: float = 1.0,
        w_local_pos: float = 1.0,
        w_local_node: float = 1.0,
        output_path: str = "generated_pmdm",
        seed: int = 42,
        device: Optional[str] = None,
        validity_filter: bool = True,
        max_retries: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            task,
            pocket_db=pocket_db,
            pocket_index=pocket_index,
            num_generate=num_generate,
            batch_size=batch_size,
            num_steps=num_steps,
            mol_size=mol_size,
            output_path=output_path,
            seed=seed,
            device=device,
            **kwargs,
        )
        self.validity_filter = validity_filter
        self.max_retries = max_retries
        self._rejected: list = []
        self.sampler_kwargs = dict(
            sampling_type=sampling_type,
            step_lr=step_lr,
            clip=clip,
            clip_pos=clip_pos,
            global_start_sigma=(
                float("inf") if global_start_sigma is None else global_start_sigma
            ),
            w_global_pos=w_global_pos,
            w_global_node=w_global_node,
            w_local_pos=w_local_pos,
            w_local_node=w_local_node,
        )

    def _pocket(self) -> Dict[str, torch.Tensor]:
        dataset = PMDMDataset(self.pocket_db, center=False)
        return dataset[self.pocket_index]

    @staticmethod
    def _repeat(item: Dict[str, torch.Tensor], n: int) -> Dict[str, Any]:
        """Tile one pocket ``n`` times, with fresh scatter indices."""
        size = len(item["protein_pos"])
        return {
            "protein_pos": item["protein_pos"].repeat(n, 1),
            "protein_atom_feature_full": item["protein_atom_feature_full"].repeat(n, 1),
            "protein_element_batch": torch.repeat_interleave(
                torch.arange(n, dtype=INT_TYPE), size
            ),
        }

    def _sizes(self, n: int) -> torch.Tensor:
        if self.mol_size:
            return torch.tensor(random.choices(self.mol_size, k=n), dtype=INT_TYPE)
        return self.task.node_dist_model.sample(n).clamp(min=2)

    def _sample_kwargs(self, item: Dict[str, Any], n: int) -> Dict[str, Any]:
        return self.sampler_kwargs

    def _reject_reason(self, symbols: list, coords: torch.Tensor) -> Optional[str]:
        """``None`` if the molecule is usable, else why it was rejected.

        Upstream's criterion is "reconstructs to a sanitisable single-fragment
        molecule" (``sample.py:390-394``): fragmented SMILES are skipped and a
        failed reconstruction raises. Reconstruction needs the optional
        xyz2mol/openbabel stack, so this uses the two geometric conditions that
        make reconstruction fail in the first place -- an interatomic distance
        no real bond can explain, or a disconnected graph. Cheaper and with no
        optional dependency, but it is a proxy: it will pass a connected
        molecule with an impossible valence that upstream's reconstruction
        would reject.

        # ponytail: geometric proxy, not real reconstruction. Swap in
        # utils/smilify.smilify_xyz2mol if the [analyze] extra is a given.
        """
        from MolecularDiffusion.utils.smilify import check_connected, get_cutoffs

        if len(symbols) < 2:
            return "too_small"

        z = np.array([ATOMIC_NUMBERS.get(s, 0) for s in symbols])
        pos = coords.numpy()
        dist = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        np.fill_diagonal(dist, np.inf)
        if dist.min() < MIN_ATOM_SEPARATION:
            return "clash"

        radii = np.array(get_cutoffs(z))
        adjacency = (dist < radii[:, None] + radii[None] + BOND_TOLERANCE).astype(float)
        if not check_connected(adjacency):
            return "fragmented"
        return None

    def _accept(self, one_hot, coords, node_mask) -> list:
        """Indices of the batch's molecules that pass the filter."""
        if not self.validity_filter:
            return list(range(one_hot.size(0)))
        vocab = self.task.atom_vocab
        keep, rejects = [], []
        mask = node_mask.squeeze(-1) if node_mask.dim() == 3 else node_mask
        for i in range(one_hot.size(0)):
            live = mask[i].bool()
            symbols = [vocab[k] for k in one_hot[i][live].argmax(-1).tolist()]
            reason = self._reject_reason(symbols, coords[i][live])
            (rejects if reason else keep).append(reason if reason else i)
        self._rejected.extend(r for r in rejects if r)
        return keep

    def _start(self, item: Dict[str, Any]) -> None:
        self._rejected = []
        print(
            f"[{self.tag}] pocket '{item['name']}': {len(item['protein_pos'])} atoms, "
            f"generating {self.num_generate} ligands"
        )

    def _summary(self, written: int, attempts: int) -> None:
        if self.validity_filter:
            counts = Counter(self._rejected)
            total = written + sum(counts.values())
            print(
                f"[{self.tag}] kept {written}/{total} sampled "
                f"({100 * written / total if total else 0:.0f}%); "
                f"rejected: {dict(counts) or 'none'}"
            )
        if written < self.num_generate:
            # Never report a short run as a full one.
            print(
                f"[{self.tag}] WARNING: wrote {written} of {self.num_generate} requested "
                f"after {attempts} sampling rounds. Raise interference.max_retries, "
                "or set interference.validity_filter=false to keep raw samples."
            )
        print(f"[{self.tag}] wrote {written} ligands to {self.output_path}")
