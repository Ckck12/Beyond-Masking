import torch
import time
import multiprocessing as mp
from torch.utils.data import Dataset
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import List, Dict, Any, Optional, Tuple

from config.dataset_config import DatasetConfig, ProcessingConfig
from loaders.dataset_factory import DatasetLoaderFactory
from processors.data_processor import DataProcessor


def process_single_item_worker(item_data: Dict[str, Any], desired_frames: int) -> Tuple:
    """
    단일 아이템 처리 함수 (프로세스 풀에서 실행)
    멀티프로세싱을 위해 최상위 레벨에 정의
    """
    processor = DataProcessor()
    
    try:
        video_path = item_data.get('path')
        landmark_path = item_data.get('landmarks') 
        label = item_data.get('label', 0)
        original_path = item_data.get('path', 'unknown_path')
        
        if not all([video_path, landmark_path, label is not None]):
            raise ValueError(f"Missing required data for {original_path}")
        
        # 데이터 처리
        video_tensor = processor.process_video(video_path, desired_frames)
        landmark_tensor = processor.process_landmarks(landmark_path, desired_frames)
        
        return (
            video_tensor,
            "text_placeholder",  # 텍스트 데이터 자리
            landmark_tensor,
            torch.tensor(int(label), dtype=torch.long),
            original_path
        )
        
    except Exception as e:
        print(f"Processing error for {item_data.get('path', 'unknown')}: {e}")
        return processor.create_dummy_data(
            item_data.get('path', 'unknown'),
            desired_frames,
            str(e)
        )


class MultiModalDataset(Dataset):
    """멀티모달 딥페이크 탐지 데이터셋"""
    
    def __init__(
        self,
        dataset_config: DatasetConfig,
        processing_config: ProcessingConfig,
        partition: str = "train",
        preload_workers: Optional[int] = None
    ):
        assert partition in ["train", "val", "test"], f"Invalid partition: {partition}"
        
        self.partition = partition
        self.processing_config = processing_config
        
        # 컴포넌트 초기화
        self.loader_factory = DatasetLoaderFactory(dataset_config)
        self.data_processor = DataProcessor(processing_config)
        
        # 파일 목록 로드
        self.file_list = self._load_all_files()
        self.preprocessed_data = [None] * len(self.file_list)
        
        # 워커 수 설정
        cpu_cores = mp.cpu_count() or 4
        self.preload_workers = preload_workers or max(1, cpu_cores // 4)
        
        print(f"Initialized {self.__class__.__name__} for {partition}: "
              f"{len(self.file_list)} files, {self.preload_workers} workers")
    
    def _load_all_files(self) -> List[Dict[str, Any]]:
        """모든 활성화된 데이터셋에서 파일 로드"""
        all_files = []
        active_loaders = self.loader_factory.get_active_loaders()
        
        if not active_loaders:
            print(f"Warning: No active loaders for partition '{self.partition}'")
            return all_files
        
        for loader in active_loaders:
            try:
                files = loader.load_files(self.partition)
                all_files.extend(files)
                print(f"Loaded {len(files)} files from {loader.get_dataset_name()}")
            except Exception as e:
                print(f"Error loading from {loader.get_dataset_name()}: {e}")
        
        print(f"Total files loaded for {self.partition}: {len(all_files)}")
        return all_files
    
    def preload_data(self, batch_size: int = 100) -> None:
        """데이터 병렬 전처리 및 메모리 로드"""
        if not self.file_list:
            print(f"No files to preload for {self.partition}")
            return
        
        print(f"Starting preload for {self.partition}: {len(self.file_list)} items, "
              f"{self.preload_workers} workers, batch_size={batch_size}")
        
        total_items = len(self.file_list)
        num_batches = (total_items + batch_size - 1) // batch_size
        
        start_time = time.time()
        total_processed = 0
        total_failed = 0
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_items)
            batch_indices = list(range(batch_start, batch_end))
            
            batch_processed, batch_failed = self._process_batch(batch_indices, batch_idx + 1, num_batches)
            total_processed += batch_processed
            total_failed += batch_failed
            
            # 진행상황 출력
            elapsed = time.time() - start_time
            items_done = total_processed + total_failed
            if items_done > 0:
                eta = (total_items - items_done) * (elapsed / items_done)
                print(f"Progress: {items_done}/{total_items} "
                      f"(Success: {total_processed}, Failed: {total_failed}) "
                      f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
        
        total_time = time.time() - start_time
        print(f"Preload completed for {self.partition}: "
              f"Success: {total_processed}, Failed: {total_failed}, "
              f"Total time: {total_time:.1f}s")
    
    def _process_batch(self, indices: List[int], batch_num: int, total_batches: int) -> Tuple[int, int]:
        """배치 단위 데이터 처리"""
        processed_count = 0
        failed_count = 0
        
        print(f"Processing batch {batch_num}/{total_batches} "
              f"(items {indices[0]+1}-{indices[-1]+1}) for {self.partition}")
        
        try:
            current_workers = min(len(indices), self.preload_workers)
            
            with ProcessPoolExecutor(
                max_workers=current_workers,
                mp_context=mp.get_context('spawn')
            ) as executor:
                # 작업 제출
                future_to_idx = {
                    executor.submit(
                        process_single_item_worker,
                        self.file_list[idx],
                        self.processing_config.desired_num_frames
                    ): idx
                    for idx in indices
                }
                
                # 결과 수집
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result(timeout=60)  # 60초 타임아웃
                        
                        # 에러 플레이스홀더 확인
                        if len(result) >= 2 and "error_placeholder" in str(result[1]):
                            failed_count += 1
                        else:
                            processed_count += 1
                            
                        self.preprocessed_data[idx] = result
                        
                    except Exception as e:
                        failed_count += 1
                        item_path = self.file_list[idx].get('path', f'index_{idx}')
                        print(f"Batch processing error for {item_path}: {e}")
                        
                        self.preprocessed_data[idx] = self.data_processor.create_dummy_data(
                            item_path,
                            self.processing_config.desired_num_frames,
                            str(e)
                        )
                        
        except Exception as e:
            print(f"Batch executor error for batch {batch_num}: {e}")
            # 실패한 배치의 모든 아이템을 더미 데이터로 채움
            for idx in indices:
                if self.preprocessed_data[idx] is None:
                    failed_count += 1
                    self.preprocessed_data[idx] = self.data_processor.create_dummy_data(
                        self.file_list[idx].get('path', f'index_{idx}'),
                        self.processing_config.desired_num_frames,
                        "Batch processing failed"
                    )
        
        return processed_count, failed_count
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, index: int) -> Tuple:
        """데이터 아이템 반환"""
        # 인덱스 유효성 검사
        if index >= len(self.file_list):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.file_list)}")
        
        # 전처리된 데이터 반환
        if (self.preprocessed_data and 
            index < len(self.preprocessed_data) and 
            self.preprocessed_data[index] is not None):
            return self.preprocessed_data[index]
        
        # 실시간 처리
        try:
            item_data = self.file_list[index]
            return process_single_item_worker(
                item_data, 
                self.processing_config.desired_num_frames
            )
        except Exception as e:
            print(f"Runtime processing error for index {index}: {e}")
            return self.data_processor.create_dummy_data(
                self.file_list[index].get('path', f'index_{index}'),
                self.processing_config.desired_num_frames,
                str(e)
            )
    
    # 기존 코드와의 호환성을 위한 메서드들
    def preload_and_cache_all(self, num_workers: Optional[int] = None) -> None:
        """기존 코드 호환성을 위한 메서드"""
        if num_workers is not None:
            self.preload_workers = num_workers
        self.preload_data()
    
    def preload_all_data_to_memory(self, num_workers: Optional[int] = None) -> None:
        """기존 코드 호환성을 위한 메서드"""  
        self.preload_and_cache_all(num_workers)


class DatasetFactory:
    """데이터셋 생성 팩토리"""
    
    @staticmethod
    def create_dataset(
        partition: str,
        dataset_config: DatasetConfig,
        processing_config: ProcessingConfig,
        **kwargs
    ) -> MultiModalDataset:
        """데이터셋 인스턴스 생성"""
        return MultiModalDataset(
            dataset_config=dataset_config,
            processing_config=processing_config,
            partition=partition,
            **kwargs
        )


# 시드 설정 유틸리티 (기존 코드 호환성)
def seed_worker(worker_id):
    """DataLoader 워커 시드 설정"""
    import torch
    import numpy as np
    import random
    
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)