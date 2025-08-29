"""Makes the 'config' directory a package and exposes configuration classes."""

from .dataset_config import DatasetConfig, ProcessingConfig
from .training_config import ModelConfig, TrainingConfig, get_device

__all__ = [
    'DatasetConfig',
    'ProcessingConfig',
    'ModelConfig',
    'TrainingConfig',
    'get_device'
]