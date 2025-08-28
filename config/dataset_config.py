from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ProcessingConfig:
    """데이터 처리 관련 설정"""
    desired_num_frames: int = 60
    video_size: Tuple[int, int] = (224, 224)
    landmark_dim: int = 478 * 2  # MediaPipe 얼굴 랜드마크 차원
    
    
@dataclass
class DatasetConfig:
    """데이터셋 경로 및 로딩 옵션 설정"""
    # 데이터셋 경로들
    deepspeak_base_dir: Optional[str] = None
    kodf_features_root_dir: Optional[str] = None
    fakeavceleb_dir: Optional[str] = None
    dfdc_dir: Optional[str] = None
    deepfaketimit_dir: Optional[str] = None
    faceforensics_dir: Optional[str] = None
    
    # 로딩 플래그들
    load_deepspeak_real: bool = False
    load_deepspeak_fake: bool = False
    load_kodf_real: bool = False
    load_kodf_fake: bool = False
    load_fakeavceleb: bool = False
    load_dfdc_real: bool = False
    load_dfdc_fake: bool = False
    load_deepfaketimit_fake: bool = False
    load_faceforensics_real: bool = False
    load_faceforensics_fake: bool = False