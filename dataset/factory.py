"""
This module decouples the main script from the specific implementation
details of the dataset classes.
"""

from config.dataset_config import DatasetConfig, ProcessingConfig
from .multimodal_dataset import MultiModalDataset

class DatasetFactory:
    """A factory class to create instances of the MultiModalDataset."""
    
    @staticmethod
    def create_dataset(
        partition: str,
        dataset_config: DatasetConfig,
        processing_config: ProcessingConfig,
        preload_workers: int,
    ) -> MultiModalDataset:
        """
        Creates and returns a configured MultiModalDataset instance.

        Args:
            partition (str): The dataset split ('train', 'val', 'test').
            dataset_config (DatasetConfig): Configuration for dataset paths and flags.
            processing_config (ProcessingConfig): Configuration for data preprocessing.
            preload_workers (int): Number of workers for parallel data preloading.

        Returns:
            MultiModalDataset: An initialized dataset object.
        """
        return MultiModalDataset(
            dataset_config=dataset_config,
            processing_config=processing_config,
            partition=partition,
            preload_workers=preload_workers
        )