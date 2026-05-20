"""
PyTorch Lightning DataModule wrapper for MolecularDiffusion.

Wraps the existing DataModule to conform to Lightning's LightningDataModule interface.
"""

import logging
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from MolecularDiffusion.data.dataloader import graph_collate

logger = logging.getLogger(__name__)


class MolecularDiffusionDataModule(pl.LightningDataModule):
    """
    Lightning DataModule wrapper for MolecularDiffusion.

    This wraps the existing DataModule class and provides train/val/test dataloaders
    that work with Lightning's training loop.

    Parameters:
        data_module: The existing MolecularDiffusion DataModule instance
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for data loading
        pin_memory (bool): Whether to pin memory for faster GPU transfer
    """

    def __init__(
        self,
        data_module,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.data_module = data_module
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        # persistent_workers only valid when num_workers > 0
        self.persistent_workers = persistent_workers and num_workers > 0
        self._loaded = False

    def setup(self, stage: Optional[str] = None):
        """
        Load datasets if not already loaded.

        This is called on every GPU in distributed training.
        Note: In the current workflow, data is already loaded in train.py before
        creating this DataModule, so we check if datasets exist first.
        """
        # Check if datasets are already loaded (they should be from train.py)
        if hasattr(self.data_module, 'train_set') and self.data_module.train_set is not None:
            logger.info("Datasets already loaded, skipping redundant load()")
            self._loaded = True
            return

        # Load if not already loaded (shouldn't happen but handle gracefully)
        if not self._loaded:
            logger.info("Loading datasets via DataModule.load()")
            self.data_module.load()
            self._loaded = True

    def train_dataloader(self):
        """Return training dataloader."""
        # Auto-adjust batch size to avoid batch size of 1 (causes issues with BatchNorm)
        batch_size = self.batch_size
        while len(self.data_module.train_set) % batch_size == 1:
            batch_size += 1

        if batch_size != self.batch_size:
            logger.warning(f"Adjusted batch size from {self.batch_size} to {batch_size} for training")

        return DataLoader(
            self.data_module.train_set,
            batch_size=batch_size,
            collate_fn=self.data_module.collate_fn or graph_collate,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,  # Lightning handles shuffling with DDP via DistributedSampler
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        if self.data_module.valid_set is None or len(self.data_module.valid_set) == 0:
            logger.warning("No validation set available, skipping val_dataloader")
            return None

        batch_size = self.batch_size
        while len(self.data_module.valid_set) % batch_size == 1:
            batch_size += 1

        if batch_size != self.batch_size:
            logger.warning(f"Adjusted batch size from {self.batch_size} to {batch_size} for validation")

        return DataLoader(
            self.data_module.valid_set,
            batch_size=batch_size,
            collate_fn=self.data_module.collate_fn or graph_collate,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        """Return test dataloader."""
        batch_size = self.batch_size
        while len(self.data_module.test_set) % batch_size == 1:
            batch_size += 1

        if batch_size != self.batch_size:
            logger.warning(f"Adjusted batch size from {self.batch_size} to {batch_size} for testing")

        return DataLoader(
            self.data_module.test_set,
            batch_size=batch_size,
            collate_fn=self.data_module.collate_fn or graph_collate,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )
