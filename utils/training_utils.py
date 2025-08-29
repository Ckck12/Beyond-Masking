"""
General utility functions for the training pipeline, such as setting seeds,
managing GPU memory, and handling model checkpoints.
"""

import torch
import numpy as np
import random
import os
from pathlib import Path
from typing import Optional

def set_all_seeds(seed: int = 42):
    """Sets the random seeds for all relevant libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For full reproducibility, these are often needed, but can slow down training.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"🌱 All random seeds set to {seed}.")

def print_system_info():
    """Prints information about the current system and GPU environment."""
    print("\n--- System Information ---")
    print(f"Python version: {os.sys.version.split(' ')[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    print("------------------------\n")

def check_gpu_memory(device_id: int = 0) -> str:
    """Returns a string with the current GPU memory usage."""
    if not torch.cuda.is_available():
        return "CUDA not available."
    
    allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 3)
    total = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)
    
    return (
        f"GPU Memory (GB): Allocated={allocated:.2f}, "
        f"Reserved={reserved:.2f}, Total={total:.2f}"
    )

class EarlyStopping:
    """
    A helper class to stop training when a monitored metric has stopped improving.
    """
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = float('inf')
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Updates the state and returns True if training should stop.
        
        Args:
            val_loss (float): The validation loss of the current epoch.
        
        Returns:
            bool: True if early stopping is triggered, False otherwise.
        """
        if val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop