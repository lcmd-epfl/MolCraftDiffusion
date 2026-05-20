"""
EMA (Exponential Moving Average) callback for PyTorch Lightning.

Maintains an EMA copy of the model and swaps it in for validation/testing.
"""

import copy
import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from MolecularDiffusion.callbacks import EMA

logger = logging.getLogger(__name__)


class EMACallback(Callback):
    """
    Exponential Moving Average callback.

    Maintains an EMA copy of the model weights and uses it during validation/testing.
    This is commonly used in diffusion models to stabilize generation.

    Parameters:
        decay (float): EMA decay rate (default: 0.9999)
    """

    def __init__(self, decay: float = 0.9999):
        super().__init__()
        self.decay = decay
        self.ema = EMA(beta=decay)
        self.ema_model = None
        self._original_task = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Create EMA model at the start of training."""
        logger.info(f"Initializing EMA model with decay={self.decay}")
        self.ema_model = copy.deepcopy(pl_module.task)
        self.ema_model.eval()

        # Freeze EMA model
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ):
        """Update EMA model after each training batch."""
        if self.ema_model is not None:
            self.ema.update_model_average(self.ema_model, pl_module.task)

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Swap to EMA model before validation."""
        if self.ema_model is not None:
            self._original_task = pl_module.task
            pl_module.task = self.ema_model
            logger.debug("Swapped to EMA model for validation")

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Swap back to original model after validation."""
        if self._original_task is not None:
            pl_module.task = self._original_task
            self._original_task = None
            logger.debug("Swapped back to original model after validation")

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Swap to EMA model before testing."""
        if self.ema_model is not None:
            self._original_task = pl_module.task
            pl_module.task = self.ema_model
            logger.debug("Swapped to EMA model for testing")

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Swap back to original model after testing."""
        if self._original_task is not None:
            pl_module.task = self._original_task
            self._original_task = None
            logger.debug("Swapped back to original model after testing")

    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict):
        """Save EMA model state in checkpoint."""
        if self.ema_model is not None:
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()
            logger.debug("Saved EMA model state to checkpoint")

    def on_load_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict):
        """Load EMA model state from checkpoint."""
        if 'ema_model_state_dict' in checkpoint:
            if self.ema_model is None:
                self.ema_model = copy.deepcopy(pl_module.task)
                self.ema_model.eval()
                for param in self.ema_model.parameters():
                    param.requires_grad = False

            self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
            logger.info("Loaded EMA model state from checkpoint")
