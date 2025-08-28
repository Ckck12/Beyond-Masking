import torch
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

@dataclass
class ModelConfig:
    """Configuration for the Landmark-Guided Multimodal Framework architecture."""
    # General
    input_frames: int = 60
    feature_dim: int = 256  # Common feature dimension for video, audio, and landmark features
    dropout_rate: float = 0.3

    # Video Extractor (3D CNN)
    video_conv_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    video_kernel_size: Tuple[int, int, int] = (3, 3, 3)
    video_padding: Tuple[int, int, int] = (1, 1, 1)

    # Audio Extractor (1D CNN for MFCC)
    audio_conv_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    audio_kernel_size: int = 3
    audio_padding: int = 1
    
    # Landmark Projector & Predictor
    landmark_input_dim: int = 478 * 2  # 956

    # Transformer Encoder
    transformer_nhead: int = 4
    transformer_num_layers: int = 2
    transformer_dim_feedforward: int = 512

    # Final Classifier
    classifier_hidden_dims: List[int] = field(default_factory=lambda: [256])

@dataclass
class TrainingConfig:
    """Configuration for the training process and hyperparameters."""
    # Basic Training Params
    num_epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_workers: int = 4
    device: str = "auto"
    seed: int = 42
    
    # Loss Function Weights (from paper)
    final_loss_alpha: float = 0.8  # alpha: Weight between auxiliary losses and BCE loss
    lambda_beta: float = 0.5       # beta: Weight for the audio part in KL loss
    lambda_gamma: float = 0.5      # gamma: Weight for the contrastive loss within auxiliary losses
    
    # Temperature Parameters for Losses
    kl_loss_temp: float = 0.1         # T: Temperature for KL Divergence
    contrastive_loss_temp: float = 0.07 # tau: Temperature for Contrastive Loss

    # Model Saving
    save_best_model: bool = True
    model_save_dir: str = "saved_models"
    
    # Early Stopping
    early_stopping_patience: Optional[int] = 10
    early_stopping_min_delta: float = 1e-4
    
    # Logging
    log_every_n_batches: int = 10
    verbose: bool = True

def get_device(device_str: str = "auto") -> torch.device:
    """
    Checks the requested device and returns the optimal torch.device object.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✅ Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            print("⚠️ CUDA not available, using CPU")
    else:
        device = torch.device(device_str)
        if device.type == "cuda" and not torch.cuda.is_available():
            print(f"🚨 Warning: CUDA '{device_str}' requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
    return device