"""
리팩토링된 멀티모달 딥페이크 탐지 데이터셋
- 모듈화된 구조로 각 데이터셋 로더 분리
- 설정 기반 관리
- 효율적인 병렬 처리
- 에러 처리 개선
"""

# 기존 코드와의 호환성을 위한 임포트들
from dataset.multimodal_dataset import (
    MultiModalDataset as SAMMD_dataset,  # 기존 클래스명과 호환
    DatasetFactory,
    seed_worker
)

from config.dataset_config import DatasetConfig, ProcessingConfig
from processors.data_processor import DataProcessor
from loaders.dataset_factory import DatasetLoaderFactory

# 기존 함수들의 호환성 wrapper
def pad_frames(data_tensor, desired_frames):
    """기존 코드 호환성을 위한 wrapper"""
    return DataProcessor.pad_frames(data_tensor, desired_frames)

def read_video(path: str, desired_num_frames: int):
    """기존 코드 호환성을 위한 wrapper"""
    processor = DataProcessor()
    video_tensor = processor.process_video(path, desired_num_frames)
    info = {"video_fps": 30}  # 기본값
    return video_tensor, info

def normalize_landmarks_standardize(landmarks_seq):
    """기존 코드 호환성을 위한 wrapper"""
    return DataProcessor.normalize_landmarks(landmarks_seq)

def read_landmark(landmark_path: str, desired_num_frames: int):
    """기존 코드 호환성을 위한 wrapper"""
    processor = DataProcessor()
    return processor.process_landmarks(landmark_path, desired_num_frames)

def process_single_item_for_cache(item_data, desired_num_frames_global):
    """기존 코드 호환성을 위한 wrapper"""
    from dataset.multimodal_dataset import process_single_item_worker
    return process_single_item_worker(item_data, desired_num_frames_global)


# 편의 함수들
def create_dataset_from_args(args, partition: str):
    """argparse 객체로부터 데이터셋 생성"""
    
    # 설정 객체 생성
    dataset_config = DatasetConfig(
        deepspeak_base_dir=getattr(args, 'deepspeak_data_dir', None),
        kodf_features_root_dir=getattr(args, 'kodf_data_dir', None),
        fakeavceleb_dir=getattr(args, 'fakeavceleb_dir', None),
        dfdc_dir=getattr(args, 'dfdc_dir', None),
        deepfaketimit_dir=getattr(args, 'deepfaketimit_dir', None),
        faceforensics_dir=getattr(args, 'faceforensics_dir', None),
        load_deepspeak_real=getattr(args, 'load_deepspeak_real', False),
        load_deepspeak_fake=getattr(args, 'load_deepspeak_fake', False),
        load_kodf_real=getattr(args, 'load_kodf_real', False),
        load_kodf_fake=getattr(args, 'load_kodf_fake', False),
        load_fakeavceleb=getattr(args, 'load_fakeavceleb', False),
        load_dfdc_real=getattr(args, 'load_dfdc_real', False),
        load_dfdc_fake=getattr(args, 'load_dfdc_fake', False),
        load_deepfaketimit_fake=getattr(args, 'load_deepfaketimit_fake', False),
        load_faceforensics_real=getattr(args, 'load_faceforensics_real', False),
        load_faceforensics_fake=getattr(args, 'load_faceforensics_fake', False),
    )
    
    processing_config = ProcessingConfig(
        desired_num_frames=getattr(args, 'input_frames', 60),
        video_size=(224, 224),
        landmark_dim=478 * 2
    )
    
    # 데이터셋 생성
    dataset = DatasetFactory.create_dataset(
        partition=partition,
        dataset_config=dataset_config,
        processing_config=processing_config,
        preload_workers=getattr(args, 'num_workers_preload', None)
    )
    
    return dataset


# 사용 예시를 위한 설정 생성 함수
def create_sample_config():
    """샘플 설정 생성 (테스트/데모용)"""
    dataset_config = DatasetConfig(
        load_deepspeak_real=True,
        load_deepspeak_fake=True,
        deepspeak_base_dir="/path/to/deepspeak"
    )
    
    processing_config = ProcessingConfig(
        desired_num_frames=60,
        video_size=(224, 224)
    )
    
    return dataset_config, processing_config


# 모듈 정보
__version__ = "2.0.0"
__author__ = "Refactored Dataset Module"

# 주요 클래스와 함수들 export
__all__ = [
    'SAMMD_dataset',
    'DatasetFactory', 
    'DatasetConfig',
    'ProcessingConfig',
    'DataProcessor',
    'DatasetLoaderFactory',
    'seed_worker',
    'create_dataset_from_args',
    'create_sample_config',
    # 호환성 함수들
    'pad_frames',
    'read_video', 
    'normalize_landmarks_standardize',
    'read_landmark',
    'process_single_item_for_cache'
]


if __name__ == "__main__":
    # 간단한 테스트 코드
    print("리팩토링된 데이터셋 모듈")
    print(f"버전: {__version__}")
    print(f"사용 가능한 클래스들: {[cls for cls in __all__ if cls[0].isupper()]}")
    
    # 설정 예시
    dataset_config, processing_config = create_sample_config()
    print(f"샘플 설정 - 프레임 수: {processing_config.desired_num_frames}")
    print(f"샘플 설정 - 비디오 크기: {processing_config.video_size}")