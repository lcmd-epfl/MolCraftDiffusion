"""MiDi task: joint diffusion over coordinates, atom types, bonds and charges.

MiDi (Vignac et al., ECML 2023, arXiv:2302.09048) is the first model in this
platform that generates the *molecular graph itself* -- bond orders and formal
charges are diffused jointly with the 3D coordinates, so a sample arrives with
an explicit bond table instead of needing post-hoc perception.

Data path: ``data_type: graph3d`` with ``bond_collate: dense``.
``graph3d_dense_collate`` already produces MiDi's exact dense shapes, so the
adapter below is a handful of ``F.one_hot`` calls and upstream's
``utils.to_dense`` (the PyG -> dense bridge) is not needed at all.

Bond classes are the platform's canonical five (``0=none, 1=SINGLE, 2=DOUBLE,
3=TRIPLE, 4=AROMATIC``) and MiDi uses exactly those, in the same order -- the
mapping is the identity.

Formal charges are stored raw and signed; the offset and class count are
applied here (``charge_offset``/``n_charge_classes``: QM9 -> +1/3,
GEOM -> +2/6), never baked into the dataset.

Out of scope this pass (see the integration plan): ``ExtraFeatures`` (every
released config sets ``extra_features: null``), the variational-NLL validation
path, MiDi's own molecular metrics, the size-aware loader, and all
conditioning/guidance modes -- MiDi is unconditional-only.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from MolecularDiffusion.data.component.graph3d_dataset import (
    N_BOND_CLASSES,
    build_rdkit_mol,
)
from MolecularDiffusion.modules.models.midi import (
    Dims,
    DiscreteUniformTransition,
    GraphTransformer,
    MarginalUniformTransition,
    PlaceHolder,
)

# Histogram-backed size sampler already implemented for TABASCO and reused by
# FlowMol; MiDi's own DistributionNodes adds only log_prob, which serves the
# out-of-scope variational NLL.
from MolecularDiffusion.modules.tasks.diffusion_tabasco import (
    TabascoNodeDistribution,
)

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
except ImportError:  # only the optional .sdf sidecar needs RDKit
    Chem = None


def _charge_marginals(
    stats: Any, n_atom_types: int, charge_offset: int, n_charge_classes: int
) -> torch.Tensor:
    """Rebuild MiDi's ``charges_marginals`` from ``graph3d_stats``.

    Mirrors ``AbstractDatasetInfos.complete_infos`` (upstream
    ``abstract_dataset.py:145``): a per-atom-type charge distribution
    ``(K, C)``, row-normalized, weighted by the atom-type marginal.
    """
    charge_types = np.zeros((n_atom_types, n_charge_classes), dtype=np.float64)
    for atom_type, per_charge in stats.charge_counts.items():
        if atom_type >= n_atom_types:
            continue
        for raw_charge, count in per_charge.items():
            cls = int(raw_charge) + charge_offset
            if not 0 <= cls < n_charge_classes:
                msg = (
                    f"formal charge {raw_charge} on atom type {atom_type} maps "
                    f"to class {cls}, outside [0, {n_charge_classes}). Raise "
                    "n_charge_classes / adjust charge_offset (QM9: +1/3, "
                    "GEOM: +2/6)."
                )
                raise ValueError(msg)
            charge_types[atom_type, cls] += count

    row_sums = charge_types.sum(axis=1, keepdims=True)
    charge_types = np.divide(
        charge_types, row_sums, out=np.zeros_like(charge_types), where=row_sums > 0
    )
    atom_marginal = np.asarray(stats.atom_type_marginal(), dtype=np.float64)
    atom_marginal = atom_marginal[:n_atom_types]
    marginals = (charge_types * atom_marginal[:, None]).sum(axis=0)
    total = marginals.sum()
    if total <= 0:
        msg = "charge marginals are all zero -- graph3d_stats has no charges"
        raise ValueError(msg)
    return torch.from_numpy(marginals / total).float()


def _or_uniform(
    marginal: Optional[torch.Tensor], n_classes: int
) -> torch.Tensor:
    """Return ``marginal``, or a uniform placeholder of ``n_classes``."""
    if marginal is None:
        return torch.ones(n_classes) / n_classes
    return marginal.detach().clone().float()


class ModelTaskFactory:
    """Hydra entry point for the MiDi task (``configs/tasks/diffusion_midi``).

    Declares ``train_set`` so ``cli/train.py``'s declarative seam injects the
    training dataset: MiDi's noise model needs the atom/bond/charge marginals
    and the size histogram *at construction time*, and neither is an
    ``nn.Module`` buffer, so neither survives in a checkpoint.

    ``sdf_output_path`` is declared generation-time (docs §2.5b): the task is
    rebuilt from the checkpoint's training-time config, where it is ``null``,
    so without this declaration the generate config's value never arrives and
    the bond sidecar -- the whole 2D half of the model -- is silently dropped.
    """

    generation_time_keys = ("sdf_output_path",)

    def __init__(  # noqa: PLR0913
        self,
        task_type: str = "diffusion_midi",
        n_layers: int = 12,
        hidden_mlp_dims: Optional[dict] = None,
        hidden_dims: Optional[dict] = None,
        diffusion_steps: int = 500,
        diffusion_noise_schedule: str = "cosine",
        transition: str = "marginal",
        nu: Optional[dict] = None,
        lambda_train: Optional[list] = None,
        charge_offset: int = 1,
        n_charge_classes: int = 3,
        sdf_output_path: Optional[str] = None,
        atom_vocab: Optional[list] = None,
        train_set: Optional[torch.utils.data.Dataset] = None,
        **kwargs: Any,
    ) -> None:
        self.task_type = task_type
        self.n_layers = n_layers
        self.hidden_mlp_dims = dict(
            hidden_mlp_dims or {"X": 256, "E": 64, "y": 256, "pos": 64}
        )
        self.hidden_dims = dict(
            hidden_dims
            or {
                "dx": 256,
                "de": 64,
                "dy": 128,
                "n_head": 8,
                "dim_ffX": 256,
                "dim_ffE": 64,
                "dim_ffy": 256,
            }
        )
        self.diffusion_steps = diffusion_steps
        self.diffusion_noise_schedule = diffusion_noise_schedule
        self.transition = transition
        self.nu = dict(nu or {"p": 2.5, "x": 1, "c": 1, "e": 1.5, "y": 1})
        self.lambda_train = list(lambda_train or [3, 0.4, 1, 2, 0])
        self.charge_offset = charge_offset
        self.n_charge_classes = n_charge_classes
        self.sdf_output_path = sdf_output_path
        self.atom_vocab = list(atom_vocab) if atom_vocab else None
        self.train_set = train_set
        self.kwargs = kwargs
        self.task: Optional[MidiDiffusionTask] = None

    def build(self) -> MidiDiffusionTask:
        """Construct the task, deriving marginals/size histogram from data."""
        if not self.atom_vocab:
            msg = (
                "MiDi needs atom_vocab (it sizes the atom-type head). Set "
                "data.atom_vocab, or tasks.atom_vocab explicitly."
            )
            raise ValueError(msg)
        n_atom_types = len(self.atom_vocab)

        stats = getattr(self.train_set, "graph3d_stats", None)
        x_marginals = e_marginals = charges_marginals = None
        n_atoms_hist: dict[int, int] = {}

        if stats is not None:
            atom_marginal = np.asarray(stats.atom_type_marginal())
            if atom_marginal.shape[0] < n_atom_types:
                atom_marginal = np.pad(
                    atom_marginal, (0, n_atom_types - atom_marginal.shape[0])
                )
            x_marginals = torch.from_numpy(
                np.asarray(atom_marginal[:n_atom_types], dtype=np.float64)
            ).float()
            e_marginals = torch.from_numpy(
                np.asarray(stats.bond_type_marginal(), dtype=np.float64)
            ).float()
            charges_marginals = _charge_marginals(
                stats,
                n_atom_types,
                self.charge_offset,
                self.n_charge_classes,
            )
            n_atoms_hist = {int(k): int(v) for k, v in stats.n_atoms_hist.items()}
            logger.info(
                "MiDi marginals from graph3d_stats over %d molecules: "
                "atoms=%s bonds=%s charges=%s",
                stats.n_molecules,
                np.round(x_marginals.numpy(), 4).tolist(),
                np.round(e_marginals.numpy(), 4).tolist(),
                np.round(charges_marginals.numpy(), 4).tolist(),
            )
        else:
            # No train_set: this is the generation path (cli/generate.py never
            # builds a DataModule). The marginals are registered buffers, so
            # the real ones arrive with the checkpoint's state_dict a moment
            # later and overwrite these placeholders in place.
            logger.warning(
                "No train_set.graph3d_stats available -- building MiDi with "
                "UNIFORM placeholder marginals and an empty size histogram. "
                "Expected when loading a checkpoint for generation (both are "
                "restored from it); if this appears during TRAINING, set "
                "data.data_type=graph3d with graph3d_stats: true."
            )

        self.task = MidiDiffusionTask(
            atom_vocab=self.atom_vocab,
            n_layers=self.n_layers,
            hidden_mlp_dims=self.hidden_mlp_dims,
            hidden_dims=self.hidden_dims,
            diffusion_steps=self.diffusion_steps,
            diffusion_noise_schedule=self.diffusion_noise_schedule,
            transition=self.transition,
            nu=self.nu,
            lambda_train=self.lambda_train,
            charge_offset=self.charge_offset,
            n_charge_classes=self.n_charge_classes,
            sdf_output_path=self.sdf_output_path,
            x_marginals=x_marginals,
            e_marginals=e_marginals,
            charges_marginals=charges_marginals,
            n_atoms_hist=n_atoms_hist,
        )
        return self.task


class MidiDiffusionTask(nn.Module):
    """MiDi wrapped in the platform's duck-typed Task contract (§2.1)."""

    def __init__(  # noqa: PLR0913
        self,
        atom_vocab: list,
        n_layers: int,
        hidden_mlp_dims: dict,
        hidden_dims: dict,
        diffusion_steps: int,
        diffusion_noise_schedule: str,
        transition: str,
        nu: dict,
        lambda_train: list,
        charge_offset: int,
        n_charge_classes: int,
        sdf_output_path: Optional[str],
        x_marginals: Optional[torch.Tensor],
        e_marginals: Optional[torch.Tensor],
        charges_marginals: Optional[torch.Tensor],
        n_atoms_hist: dict,
    ) -> None:
        super().__init__()
        self.task_type = "diffusion_midi"
        self.atom_vocab = list(atom_vocab)
        self.n_atom_types = len(self.atom_vocab)
        self.charge_offset = charge_offset
        self.n_charge_classes = n_charge_classes
        self.lambda_train = list(lambda_train)
        self.sdf_output_path = sdf_output_path

        self.input_dims = Dims(
            X=self.n_atom_types,
            charges=n_charge_classes,
            E=N_BOND_CLASSES,
            y=1,  # the timestep; extra_features is null
            pos=3,
        )
        self.output_dims = Dims(
            X=self.n_atom_types,
            charges=n_charge_classes,
            E=N_BOND_CLASSES,
            y=0,  # unconditional
            pos=3,
        )

        self.backbone = GraphTransformer(
            input_dims=self.input_dims,
            n_layers=n_layers,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=self.output_dims,
        )

        # The class marginals are dataset statistics, not learned weights --
        # but they are registered as BUFFERS on purpose. cli/generate.py never
        # builds a DataModule, so the only way they can reach a generation run
        # is through the checkpoint's state_dict. NoiseModel is a plain Python
        # object holding tensors, so `_sync_marginals()` re-derives its
        # transition matrices from these buffers before every use -- covering
        # both `load_state_dict` and `.to(device)`, either of which can swap
        # the underlying tensor out from under it.
        self.register_buffer(
            "x_marginals",
            _or_uniform(x_marginals, self.n_atom_types),
        )
        self.register_buffer(
            "e_marginals",
            _or_uniform(e_marginals, N_BOND_CLASSES),
        )
        self.register_buffer(
            "charges_marginals",
            _or_uniform(charges_marginals, n_charge_classes),
        )

        if transition == "uniform":
            self.noise_model = DiscreteUniformTransition(
                output_dims=self.output_dims,
                nu=nu,
                diffusion_steps=diffusion_steps,
                noise_schedule=diffusion_noise_schedule,
            )
        elif transition == "marginal":
            self.noise_model = MarginalUniformTransition(
                x_marginals=self.x_marginals,
                e_marginals=self.e_marginals,
                charges_marginals=self.charges_marginals,
                y_classes=self.output_dims.y,
                nu=nu,
                diffusion_steps=diffusion_steps,
                noise_schedule=diffusion_noise_schedule,
            )
        else:
            msg = f"Unknown transition type '{transition}'"
            raise ValueError(msg)

        # Class index -> signed formal charge, MiDi's `collapse_charges`.
        self.register_buffer(
            "collapse_charges",
            torch.arange(n_charge_classes, dtype=torch.long) - charge_offset,
        )

        # Number of reverse steps to take. Distinct from noise_model.T (the
        # schedule length, 500): a smaller value strides the schedule, which is
        # MiDi's `general.faster_sampling`. cli/generate.py's `total_step`
        # override writes here via the `model` property.
        self.T = diffusion_steps

        self.node_dist_model = TabascoNodeDistribution(
            {"num_atoms_histogram": n_atoms_hist}
        )
        self.prop_dist_model = None  # unconditional-only
        self.last_bond_types: Optional[torch.Tensor] = None

    # -- properties required by the contract --------------------------------

    @property
    def model(self) -> MidiDiffusionTask:
        """``tasks_generate.py`` reads ``task.model.T``; self is the model."""
        return self

    @property
    def device(self) -> torch.device:
        """Device of the backbone parameters."""
        return next(self.parameters()).device

    @property
    def n_node_dist(self) -> dict:
        """``{n_atoms: count}`` histogram used to clamp ``mol_size``."""
        return self.node_dist_model.n_node_dist

    def _sync_marginals(self) -> None:
        """Point the noise model at the current marginal buffers.

        Cheap (``expand`` allocates no data) and idempotent, so it just runs
        before every forward/sample rather than trying to hook every event
        that could replace a buffer tensor.
        """
        nm = self.noise_model
        if not isinstance(nm, MarginalUniformTransition):
            return
        nm.X_marginals = self.x_marginals
        nm.E_marginals = self.e_marginals
        nm.charges_marginals = self.charges_marginals
        nm.Px = (
            self.x_marginals.unsqueeze(0).expand(nm.X_classes, -1).unsqueeze(0)
        )
        nm.Pe = (
            self.e_marginals.unsqueeze(0).expand(nm.E_classes, -1).unsqueeze(0)
        )
        nm.Pcharges = (
            self.charges_marginals.unsqueeze(0)
            .expand(nm.charges_classes, -1)
            .unsqueeze(0)
        )

    # -- adapters -----------------------------------------------------------

    def _to_placeholder(self, batch: dict) -> PlaceHolder:
        """``graph3d_dense_collate`` dict -> MiDi's dense ``PlaceHolder``.

        This replaces upstream ``utils.to_dense`` outright: the collate
        already emits ``pos (B,N,3)``, ``atom_idx (B,N)``, ``charges (B,N)``
        (raw signed), ``bond_type (B,N,N)`` (symmetric integer class ids with a
        zero diagonal) and ``node_mask (B,N)``.
        """
        pos = batch["pos"].float()
        node_mask = batch["node_mask"].bool()
        atom_idx = batch["atom_idx"].long()
        raw_charges = batch["charges"].long()
        bond = batch["bond_type"].long()

        shifted = raw_charges + self.charge_offset
        # Padded rows carry charge 0 -> class == charge_offset, in range.
        if int(shifted.min()) < 0 or int(shifted.max()) >= self.n_charge_classes:
            observed = (int(raw_charges.min()), int(raw_charges.max()))
            msg = (
                f"formal charges {observed} do not fit "
                f"{self.n_charge_classes} classes at offset "
                f"{self.charge_offset} (QM9: +1/3, GEOM: +2/6)"
            )
            raise ValueError(msg)

        x = F.one_hot(atom_idx, self.n_atom_types).float()
        charges = F.one_hot(shifted, self.n_charge_classes).float()
        e = F.one_hot(bond, N_BOND_CLASSES).float()
        y = pos.new_zeros((pos.size(0), 0))

        return PlaceHolder(
            pos=pos, X=x, charges=charges, E=e, y=y, node_mask=node_mask
        ).mask()

    def _denoise(self, z_t: PlaceHolder) -> PlaceHolder:
        """Run the backbone on a noised batch, appending ``t`` to ``y``."""
        model_input = z_t.copy()
        model_input.X = z_t.X.float()
        model_input.charges = z_t.charges.float()
        model_input.E = z_t.E.float()
        model_input.y = torch.hstack((z_t.y, z_t.t)).float()
        return self.backbone(model_input)

    # -- training -----------------------------------------------------------

    def forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """One training step: noise the batch, denoise it, weight the losses."""
        self._sync_marginals()
        dense_data = self._to_placeholder(batch)
        z_t = self.noise_model.apply_noise(dense_data)
        pred = self._denoise(z_t)
        return self._loss(pred, dense_data)

    def _loss(
        self, pred: PlaceHolder, true: PlaceHolder
    ) -> tuple[torch.Tensor, dict]:
        """MiDi's ``TrainLoss`` as plain tensor ops.

        Upstream accumulates through torchmetrics/wandb; the platform's logger
        already does that, so only the weighted scalar and a stats dict are
        produced here. The reductions match: ``MeanSquaredError`` == mean-MSE,
        ``CrossEntropyMetric`` == sum CE / n == mean CE.
        """
        node_mask = true.node_mask
        bs, n = node_mask.shape

        pos_mse = F.mse_loss(pred.pos[node_mask], true.pos[node_mask])
        x_ce = F.cross_entropy(
            pred.X[node_mask], true.X[node_mask].argmax(dim=-1)
        )
        charges_ce = F.cross_entropy(
            pred.charges[node_mask], true.charges[node_mask].argmax(dim=-1)
        )

        diag_mask = ~torch.eye(
            n, device=node_mask.device, dtype=torch.bool
        ).unsqueeze(0).expand(bs, -1, -1)
        edge_mask = diag_mask & node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        e_ce = F.cross_entropy(
            pred.E[edge_mask], true.E[edge_mask].argmax(dim=-1)
        )

        loss = (
            self.lambda_train[0] * pos_mse
            + self.lambda_train[1] * x_ce
            + self.lambda_train[2] * charges_ce
            + self.lambda_train[3] * e_ce
        )
        stats = {
            "pos_mse": pos_mse.detach(),
            "x_ce": x_ce.detach(),
            "charges_ce": charges_ce.detach(),
            "e_ce": e_ce.detach(),
        }
        return loss, stats

    def predict_and_target(
        self, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pure-generative stub: the loss is both prediction and target."""
        loss, _ = self.forward(batch)
        loss = loss.detach().reshape(1)
        return loss, torch.zeros_like(loss)

    def evaluate(
        self, pred: torch.Tensor, target: torch.Tensor  # noqa: ARG002
    ) -> dict:
        """Reduce accumulated losses into the logged validation metric."""
        return {"val_loss": pred.mean()}

    # -- generation ---------------------------------------------------------

    @torch.no_grad()
    def sample(  # noqa: PLR0913
        self,
        batch_size: Optional[int] = None,
        nodesxsample: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        batch: Optional[dict] = None,  # noqa: ARG002
        mode: Optional[str] = None,  # noqa: ARG002 - DDIM modes out of scope
        n_frames: int = 0,  # noqa: ARG002 - trajectories out of scope
        **kwargs: Any,  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unconditional sampling.

        Returns the platform's ``(one_hot, charges, coords, node_mask)``
        tuple. ``charges`` carries **signed formal charges** (FlowMol's
        precedent for this slot) -- the element identity is in ``one_hot``.
        The generated bond matrix has no channel in that tuple, so it is
        stashed on ``self.last_bond_types`` and, when ``sdf_output_path`` is
        set, written as an ``.sdf`` sidecar alongside the platform's ``.xyz``.
        """
        self._sync_marginals()
        device = self.device
        if nodesxsample is None:
            if batch_size is None:
                msg = "sample() needs either nodesxsample or batch_size"
                raise ValueError(msg)
            nodesxsample = self.node_dist_model.sample(batch_size)
        n_nodes = torch.as_tensor(nodesxsample, dtype=torch.long, device=device)
        bs = int(n_nodes.numel())
        n_max = int(n_nodes.max().item())

        arange = torch.arange(n_max, device=device).unsqueeze(0).expand(bs, -1)
        node_mask = arange < n_nodes.unsqueeze(1)

        z_t = self.noise_model.sample_limit_dist(node_mask=node_mask)

        schedule_t = self.noise_model.T
        requested = int(num_steps or self.T)
        stride = max(1, schedule_t // max(requested, 1))
        for s_int in reversed(range(0, schedule_t, stride)):
            s_array = s_int * torch.ones(
                (bs, 1), dtype=torch.long, device=device
            )
            pred = self._denoise(z_t)
            z_t = self.noise_model.sample_zs_from_zt_and_pred(
                z_t=z_t, pred=pred, s_int=s_array
            )

        final = z_t.collapse(self.collapse_charges)
        # collapse marks padding out of range (X=-1, charges=1000, E=-1);
        # clamp before anything downstream indexes with it.
        atom_idx = final.X.clamp(min=0)
        one_hot = F.one_hot(atom_idx, self.n_atom_types).float()
        one_hot = one_hot * node_mask.unsqueeze(-1)
        charges = torch.where(
            node_mask, final.charges, torch.zeros_like(final.charges)
        )
        bond_types = final.E.clamp(min=0)
        coords = final.pos * node_mask.unsqueeze(-1)

        self.last_bond_types = bond_types
        if self.sdf_output_path is not None:
            self._write_sdf(atom_idx, charges, bond_types, coords, node_mask)

        return one_hot, charges, coords, node_mask.long()

    def _write_sdf(  # noqa: PLR0913
        self,
        atom_idx: torch.Tensor,
        charges: torch.Tensor,
        bond_types: torch.Tensor,
        coords: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> None:
        """Append the sampled molecules to ``sdf_output_path``.

        Opt-in (``sdf_output_path: null`` by default) because the platform's
        writer emits ``.xyz``, which has no bond channel. Molecule building
        reuses ``build_rdkit_mol`` from the graph3d dataset -- that file is
        read from, never modified. Append mode so multi-batch generation
        accumulates into one file; a molecule RDKit refuses to sanitize is
        warned about and skipped, never raised: one bad sample must not kill a
        generation run.
        """
        if Chem is None:
            logger.warning("RDKit unavailable, skipping .sdf sidecar")
            return
        from ase.data import atomic_numbers as _atomic_numbers

        parent = os.path.dirname(os.path.abspath(self.sdf_output_path))
        os.makedirs(parent, exist_ok=True)

        z_of_vocab = [_atomic_numbers[s] for s in self.atom_vocab]
        atom_idx = atom_idx.cpu()
        charges = charges.cpu()
        bond_types = bond_types.cpu()
        coords = coords.cpu()
        node_mask = node_mask.cpu()

        with open(self.sdf_output_path, "a") as handle:  # noqa: PTH123
            writer = Chem.SDWriter(handle)
            for b in range(atom_idx.size(0)):
                n = int(node_mask[b].sum())
                if n == 0:
                    continue
                zs = [z_of_vocab[int(i)] for i in atom_idx[b, :n]]
                sub = bond_types[b, :n, :n]
                rows, cols = torch.triu_indices(n, n, offset=1)
                keep = sub[rows, cols] > 0
                bond_index = torch.stack((rows[keep], cols[keep])).numpy()
                bond_type = sub[rows, cols][keep].numpy()
                try:
                    mol = build_rdkit_mol(
                        zs,
                        bond_index,
                        bond_type,
                        formal_charge=charges[b, :n].numpy(),
                        coords=coords[b, :n].numpy(),
                    )
                    writer.write(mol)
                except Exception as exc:  # noqa: BLE001 - chemistry, not a bug
                    logger.warning("Skipping unsanitizable sample %d: %s", b, exc)
            writer.close()
