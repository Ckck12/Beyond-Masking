from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class TrainingConfig:
    """학습 관련 설정"""
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-3
    recon_loss_weight: float = 0.5
    num_workers: int = 4
    device: str = "cuda"
    seed: int = 42
    
    # 모델 저장 관련
    save_best_model: bool = True
    model_save_dir: str = "saved_models"
    save_every_n_epochs: Optional[int] = None
    
    # 조기 종료 설정
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 1e-4
    
    # 로깅 설정
    log_every_n_batches: int = 10
    verbose: bool = True


@dataclass  
class ModelConfig:
    """모델 구조 관련 설정"""
    input_frames: int = 60
    video_feature_dim: int = 256
    recon_target_dim: int = 956  # 478 * 2
    dropout_rate: float = 0.3
    
    # 비디오 인코더 설정
    video_conv_channels: tuple = (16, 32, 64)
    video_kernel_size: tuple = (3, 3, 3)
    video_padding: tuple = (1, 1, 1)
    
    # 분류기 설정
    classifier_hidden_dims: tuple = (256, 64)
    
    def __post_init__(self):
        """설정 검증"""
        if self.recon_target_dim != 478 * 2:
            print(f"Warning: recon_target_dim ({self.recon_target_dim}) != 478*2. "
                  f"Make sure this matches your landmark dimension.")
            
        if self.input_frames <= 0:
            raise ValueError("input_frames must be positive")
            
        if self.dropout_rate < 0 or self.dropout_rate >= 1:
            raise ValueError("dropout_rate must be in [0, 1)")


def get_device(device_str: str = "auto") -> torch.device:
    """최적 디바이스 선택"""
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            print("CUDA not available, using CPU")
    else:
        device = torch.device(device_str)
        if device.type == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")
    
    return device