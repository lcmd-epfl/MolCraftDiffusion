import pytest
from unittest.mock import Mock

from MolecularDiffusion.runmodes.train.logger import Logger


def test_logger_init():
    """Test Logger initialization."""
    mock_logger_obj = Mock()
    logger = Logger(
        logger=mock_logger_obj,
        log_interval=10,
        name_wandb="test_run",
        project_wandb="test_project",
        dir_wandb="/tmp/wandb",
    )

    assert logger.logger == mock_logger_obj
    assert logger.log_interval == 10
    assert logger.name_wandb == "test_run"
    assert logger.project_wandb == "test_project"
    assert logger.dir_wandb == "/tmp/wandb" 