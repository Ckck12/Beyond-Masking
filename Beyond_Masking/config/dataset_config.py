from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ProcessingConfig:
    """Configuration for the data preprocessing stage."""
    # Video
    desired_num_frames: int = 60
    video_size: Tuple[int, int] = (224, 224)
    
    # Landmarks
    landmark_dim: int = 478 * 2
    
    # Audio (MFCC)
    n_mfcc: int = 40
    sample_rate: int = 48000
    hop_length: int = 512
    n_fft: int = 2048

@dataclass
class DatasetConfig:
    """Configuration for dataset source paths and loading options."""
    # Dataset Base Directories
    deepspeak_base_dir: Optional[str] = None
    kodf_features_root_dir: Optional[str] = None
    fakeavceleb_dir: Optional[str] = None
    dfdc_dir: Optional[str] = None
    deepfaketimit_dir: Optional[str] = None
    faceforensics_dir: Optional[str] = None
    
    # Flags to enable/disable loading for each dataset component
    load_deepspeak_real: bool = False
    load_deepspeak_fake: bool = False
    load_kodf_real: bool = False
    load_kodf_fake: bool = False
    load_fakeavceleb: bool = False
    load_dfdc_real: bool = False
    load_dfdc_fake: bool = False
    load_deepfaketimit_fake: bool = False
    load_deepfaketimit_real: bool = False
    load_faceforensics_real: bool = False
    load_faceforensics_fake: bool = False
