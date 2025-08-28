"""
Defines the base interface for all dataset loaders.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import os

class BaseDatasetLoader(ABC):
    """
    Abstract Base Class for dataset loaders.

    Each specific dataset loader (e.g., for DeepSpeak, KoDF) should inherit
    from this class and implement the `load_files` method.
    """
    
    def __init__(self, base_dir: str):
        """
        Initializes the loader with the base directory of the dataset.

        Args:
            base_dir (str): The root directory path for the dataset.
        """
        self.base_dir = base_dir
        
    @abstractmethod
    def load_files(self, partition: str) -> List[Dict[str, Any]]:
        """
        Loads the file list for a specific data partition (train/val/test).

        This method must be implemented by all subclasses.

        Args:
            partition (str): The dataset split, one of 'train', 'val', or 'test'.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                  represents a data sample and must contain the keys:
                                  - 'path': Path to the video file.
                                  - 'audio_path': Path to the audio file.
                                  - 'landmarks': Path to the landmarks file.
                                  - 'label': The binary label (0 for real, 1 for fake).
                                  - 'data_label': A unique integer identifier for the dataset.
        """
        pass
    
    def validate_paths(self) -> bool:
        """Checks if the base directory for the dataset exists."""
        return os.path.isdir(self.base_dir) if self.base_dir else False
    
    def _check_files_exist(self, video_path: str, audio_path: str, landmark_path: str) -> bool:
        """
        Helper method to check for the existence of all required files for a sample.

        Args:
            video_path (str): Path to the video file.
            audio_path (str): Path to the audio file.
            landmark_path (str): Path to the landmark file.

        Returns:
            bool: True if all three files exist, False otherwise.
        """
        return os.path.exists(video_path) and os.path.exists(audio_path) and os.path.exists(landmark_path)
    
    def get_dataset_name(self) -> str:
        """
        Returns the name of the dataset based on the loader's class name.
        e.g., 'DeepSpeakLoader' -> 'DeepSpeak'
        """
        return self.__class__.__name__.replace('Loader', '')