"""
데이터셋 로더 모듈

각 데이터셋별 로더 클래스들과 팩토리를 제공합니다.
"""

from .base_loader import BaseDatasetLoader
from .dataset_loaders import (
    DeepSpeakLoader,
    KoDFLoader, 
    FakeAVCelebLoader,
    DFDCLoader,
    DeepfakeTIMITLoader,
    FaceForensicsLoader
)
from .dataset_factory import DatasetLoaderFactory

# 편의 함수들
def create_loader_by_name(dataset_name, base_dir, **kwargs):
    """데이터셋 이름으로 로더 생성"""
    loader_map = {
        'deepspeak': DeepSpeakLoader,
        'kodf': KoDFLoader,
        'fakeavceleb': FakeAVCelebLoader,
        'dfdc': DFDCLoader,
        'deepfaketimit': DeepfakeTIMITLoader,
        'faceforensics': FaceForensicsLoader
    }
    
    dataset_name = dataset_name.lower()
    if dataset_name not in loader_map:
        available = ', '.join(loader_map.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    
    loader_class = loader_map[dataset_name]
    return loader_class(base_dir, **kwargs)

def get_available_datasets():
    """사용 가능한 데이터셋 목록 반환"""
    return [
        'deepspeak',
        'kodf', 
        'fakeavceleb',
        'dfdc',
        'deepfaketimit',
        'faceforensics'
    ]

def validate_dataset_paths(config):
    """데이터셋 경로 유효성 검사"""
    from pathlib import Path
    
    path_mapping = {
        'deepspeak': config.deepspeak_base_dir,
        'kodf': config.kodf_features_root_dir,
        'fakeavceleb': config.fakeavceleb_dir,
        'dfdc': config.dfdc_dir,
        'deepfaketimit': config.deepfaketimit_dir,
        'faceforensics': config.faceforensics_dir
    }
    
    valid_paths = {}
    invalid_paths = {}
    
    for name, path in path_mapping.items():
        if path and Path(path).exists():
            valid_paths[name] = path
        elif path:  # 경로가 설정되었지만 존재하지 않음
            invalid_paths[name] = path
    
    return valid_paths, invalid_paths

# 버전 정보
__version__ = "1.0.0"

# 공개 API
__all__ = [
    # 기본 클래스
    'BaseDatasetLoader',
    
    # 개별 로더들
    'DeepSpeakLoader',
    'KoDFLoader',
    'FakeAVCelebLoader', 
    'DFDCLoader',
    'DeepfakeTIMITLoader',
    'FaceForensicsLoader',
    
    # 팩토리
    'DatasetLoaderFactory',
    
    # 편의 함수들
    'create_loader_by_name',
    'get_available_datasets',
    'validate_dataset_paths'
]