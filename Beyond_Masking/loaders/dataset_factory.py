"""
Defines a factory class for creating and managing dataset loaders.
"""

from typing import List

# Import configurations and loader classes
from config.dataset_config import DatasetConfig
from .dataset_loaders import (
    DeepSpeakLoader, KoDFLoader, FakeAVCelebLoader, 
    DFDCLoader, DeepfakeTIMITLoader, FaceForensicsLoader
)
from .base_loader import BaseDatasetLoader

class DatasetLoaderFactory:
    """
    A factory responsible for creating specific dataset loader instances based on
    the provided DatasetConfig.
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initializes the factory with a dataset configuration.

        Args:
            config (DatasetConfig): The configuration object specifying which
                                    datasets to load and their paths.
        """
        self.config = config
        self._loaders: List[BaseDatasetLoader] = []
        self._initialize_loaders()
    
    def _initialize_loaders(self):
        """
        Instantiates and collects all active loaders based on the config flags.
        A loader is considered active only if its corresponding load flag is True
        and its base directory path is valid.
        """
        # DeepSpeak
        if (self.config.load_deepspeak_real or self.config.load_deepspeak_fake) and self.config.deepspeak_base_dir:
            loader = DeepSpeakLoader(
                self.config.deepspeak_base_dir,
                load_real=self.config.load_deepspeak_real,
                load_fake=self.config.load_deepspeak_fake
            )
            if loader.validate_paths():
                self._loaders.append(loader)
        
        # KoDF
        if (self.config.load_kodf_real or self.config.load_kodf_fake) and self.config.kodf_features_root_dir:
            loader = KoDFLoader(
                self.config.kodf_features_root_dir,
                load_real=self.config.load_kodf_real,
                load_fake=self.config.load_kodf_fake
            )
            if loader.validate_paths():
                self._loaders.append(loader)
        
        # FakeAVCeleb
        if self.config.load_fakeavceleb and self.config.fakeavceleb_dir:
            loader = FakeAVCelebLoader(self.config.fakeavceleb_dir)
            if loader.validate_paths():
                self._loaders.append(loader)
        
        # DFDC
        if (self.config.load_dfdc_real or self.config.load_dfdc_fake) and self.config.dfdc_dir:
            loader = DFDCLoader(
                self.config.dfdc_dir,
                load_real=self.config.load_dfdc_real,
                load_fake=self.config.load_dfdc_fake
            )
            if loader.validate_paths():
                self._loaders.append(loader)
        
        # DeepfakeTIMIT
        if (self.config.load_deepfaketimit_real or self.config.load_deepfaketimit_fake) and self.config.deepfaketimit_dir:
            loader = DeepfakeTIMITLoader(
                self.config.deepfaketimit_dir,
                load_real=self.config.load_deepfaketimit_real,
                load_fake=self.config.load_deepfaketimit_fake
            )
            if loader.validate_paths():
                self._loaders.append(loader)
        
        # FaceForensics++
        if (self.config.load_faceforensics_real or self.config.load_faceforensics_fake) and self.config.faceforensics_dir:
            loader = FaceForensicsLoader(
                self.config.faceforensics_dir,
                load_real=self.config.load_faceforensics_real,
                load_fake=self.config.load_faceforensics_fake
            )
            if loader.validate_paths():
                self._loaders.append(loader)

    def get_active_loaders(self) -> List[BaseDatasetLoader]:
        """
        Returns the list of initialized and validated loader instances.

        Returns:
            List[BaseDatasetLoader]: A list of active loader objects.
        """
        return self._loaders
    
    def get_loader_names(self) -> List[str]:
        """
        Returns the names of all active loaders.

        Returns:
            List[str]: A list of names for the active datasets.
        """
        return [loader.get_dataset_name() for loader in self._loaders]