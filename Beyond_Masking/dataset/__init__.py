"""
데이터셋 모듈

멀티모달 딥페이크 탐지를 위한 통합 데이터셋 클래스를 제공합니다.
"""

from .multimodal_dataset import (
    MultiModalDataset,
    DatasetFactory,
    process_single_item_worker,
    seed_worker
)

# 편의 함수들
def create_dataset(
    partition,
    dataset_config, 
    processing_config,
    preload_workers=None,
    **kwargs
):
    """데이터셋 생성 편의 함수"""
    return DatasetFactory.create_dataset(
        partition=partition,
        dataset_config=dataset_config,
        processing_config=processing_config,
        preload_workers=preload_workers,
        **kwargs
    )

def create_quick_dataset(partition="train", **dataset_flags):
    """빠른 데이터셋 생성 (테스트용)"""
    from config import DatasetConfig, ProcessingConfig
    
    # 플래그에서 True인 것들만 활성화
    dataset_config = DatasetConfig(**{k: v for k, v in dataset_flags.items() if v})
    processing_config = ProcessingConfig()
    
    return create_dataset(partition, dataset_config, processing_config)

def get_dataset_info(dataset):
    """데이터셋 정보 반환"""
    info = {
        'length': len(dataset),
        'partition': getattr(dataset, 'partition', 'unknown'),
        'num_preprocessed': sum(1 for item in dataset.preprocessed_data if item is not None) 
                           if hasattr(dataset, 'preprocessed_data') else 0
    }
    
    # 로더 정보
    if hasattr(dataset, 'loader_factory'):
        info['active_loaders'] = dataset.loader_factory.get_loader_names()
    
    return info

def benchmark_dataset_loading(
    dataset, 
    num_samples=100,
    batch_size=16,
    num_workers=4
):
    """데이터셋 로딩 성능 벤치마크"""
    import time
    from torch.utils.data import DataLoader
    
    # 샘플 수 조정
    actual_samples = min(num_samples, len(dataset))
    
    # DataLoader 생성
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    print(f"Benchmarking dataset loading...")
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {batch_size}, Workers: {num_workers}")
    
    start_time = time.time()
    total_batches = 0
    
    for i, batch in enumerate(loader):
        total_batches += 1
        if i * batch_size >= actual_samples:
            break
    
    elapsed_time = time.time() - start_time
    samples_per_sec = actual_samples / elapsed_time
    
    print(f"Processed {actual_samples} samples in {elapsed_time:.2f}s")
    print(f"Throughput: {samples_per_sec:.1f} samples/sec")
    
    return {
        'samples_processed': actual_samples,
        'elapsed_time': elapsed_time,
        'throughput': samples_per_sec,
        'batches_processed': total_batches
    }

# 버전 정보
__version__ = "1.0.0"

# 공개 API
__all__ = [
    # 메인 클래스들
    'MultiModalDataset',
    'DatasetFactory',
    
    # 워커 함수들
    'process_single_item_worker',
    'seed_worker',
    
    # 편의 함수들
    'create_dataset',
    'create_quick_dataset', 
    'get_dataset_info',
    'benchmark_dataset_loading'
]