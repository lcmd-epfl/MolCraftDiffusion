"""Task-owned data module for SynCoGen (integration plan, Option A).

SynCoGen's nodes are *reaction building blocks* and its edges are *reaction
templates*: there is no atom-type channel and no chemical-bond channel, so none
of the platform's three ``data_type`` shapes (``pointcloud``, ``pyg``,
``graph3d``) describes it. Rather than add a fourth platform representation, the
approved plan routes ``configs/data/syncogen_dataset.yaml`` straight at this
class by ``_target_``. Nothing in ``MolecularDiffusion/data/`` is touched.

What ``cli/train.py`` actually needs off a data module is duck-typed and small
(``cli/train.py:273-277, 365, 600-604``)::

    .load()  .train_set  .valid_set  .test_set  .batch_size  .collate_fn

Everything else it wants -- ``atom_vocab``, ``use_ohe_feature``,
``allow_unknown``, ``node_feature_choice`` -- it reads off ``cfg.data``, never
off the object, which is why those keys live in the YAML and are swallowed here.

The batch handed to the task is upstream's own PyG ``Data``/``Batch`` throughout;
no adapter runs in the data layer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from MolecularDiffusion.modules.models.syncogen.vocab import ensure_vocabulary

logger = logging.getLogger(__name__)


class SyncogenDataModule:
    """Wraps upstream's ``SyncogenDataManager`` in the platform's data-module surface.

    Parameters mirror ``syncogen/data/dataloader.py::SyncogenDataManager`` one for
    one, plus ``vocab_dir`` (which upstream passed on the command line and read
    into process globals before importing anything).
    """

    def __init__(  # noqa: PLR0913
        self,
        graphs_path: str | Path,
        conformers_path: str | Path,
        vocab_dir: str | Path,
        pharmacophore_path: str | Path | None = None,
        max_bbs: int = 5,
        batch_size: int = 4,
        eval_batch_size: int = 4,
        train_size: float = 0.9,
        validation_size: float = 0.1,
        test_size: float = 0.0,
        sample_conformer: bool = False,
        load_bonds: bool = True,
        load_pharmacophores: bool = False,
        shuffle_train: bool = True,
        coord_mask_value: float = 0.0,
        valid_seed: int | None = None,
        num_workers: int = 0,
        task_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        # MUST come before any other import from this package -- see vocab.py.
        ensure_vocabulary(vocab_dir)

        from MolecularDiffusion.modules.models.syncogen.data.dataloader import (
            SyncogenDataManager,
        )

        self.vocab_dir = Path(vocab_dir)
        self.task_type = task_type
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.manager = SyncogenDataManager(
            graphs_path=graphs_path,
            conformers_path=conformers_path,
            pharmacophore_path=pharmacophore_path,
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
            batch_size=int(batch_size),
            eval_batch_size=int(eval_batch_size),
            num_workers=int(num_workers),
            pin_memory=False,
            shuffle_train=shuffle_train,
            sample_conformer=sample_conformer,
            load_pharmacophores=load_pharmacophores,
            load_bonds=load_bonds,
            coord_mask_value=coord_mask_value,
            valid_seed=valid_seed,
            max_bbs=max_bbs,
        )

        self.train_set: Any = None
        self.valid_set: Any = None
        self.test_set: Any = None

        if kwargs:
            # atom_vocab / use_ohe_feature / allow_unknown / node_feature_choice
            # are declared in the YAML for cli/train.py's benefit, not ours.
            logger.debug(
                "SyncogenDataModule ignoring config-only keys: %s",
                sorted(kwargs),
            )

    # ------------------------------------------------------------------ loading

    def load(self) -> None:
        """Build the train/validation/test splits and their datasets."""
        from MolecularDiffusion.modules.models.syncogen.data.dataloader import (
            SyncogenDataset,
        )

        splits = self.manager.get_graph_data_splits()
        common = {
            "conformers_path": str(self.manager.conformers_path),
            "sample_conformer": self.manager.sample_conformer,
            "coord_mask_value": self.manager.coord_mask_value,
            "pharmacophore_path": self.manager.pharmacophore_path,
            "load_pharmacophores": self.manager.load_pharmacophores,
            "load_bonds": self.manager.load_bonds,
        }
        self.train_set = SyncogenDataset(data_list=splits["train"], **common)
        self.valid_set = SyncogenDataset(
            data_list=splits["validation"], **common
        )
        self.test_set = SyncogenDataset(data_list=splits["test"], **common)
        logger.info(
            "SynCoGen splits: train=%d valid=%d test=%d (fragment counts %s)",
            len(self.train_set),
            len(self.valid_set),
            len(self.test_set),
            self.num_fragments_probs(),
        )

    def num_fragments_probs(self) -> dict[int, float]:
        """The train-split fragment-count histogram upstream caches on disk.

        Not consumed by the task (which takes its own ``num_fragments_probs``
        from the task config, so generation works with no dataset present), but
        printed at load time so a config whose prior disagrees with the data it
        is training on is visible in the log.
        """
        values = self.manager.train_length_values
        probs = self.manager.train_length_probs
        if values is None or probs is None:
            return {}
        return {
            int(v): round(float(p), 6)
            for v, p in zip(values.tolist(), probs.tolist())
        }

    # ------------------------------------------------------------- collation

    @property
    def collate_fn(self):
        """PyG's own collater -- the batch stays a ``torch_geometric.data.Batch``."""
        from torch_geometric.loader.dataloader import Collater

        try:
            return Collater(
                dataset=self.train_set, follow_batch=None, exclude_keys=None
            )
        except (
            TypeError
        ):  # older PyG signature: Collater(follow_batch, exclude_keys)
            return Collater(None, None)

    # ------------------------------------------------- convenience accessors

    @property
    def atom_vocab(self) -> Sequence[str]:
        from MolecularDiffusion.modules.models.syncogen.constants import (
            constants as C,
        )

        return list(C.ATOM_TYPES)
