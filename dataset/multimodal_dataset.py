import torch
import time
import multiprocessing as mp
from torch.utils.data import Dataset
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple

from config.dataset_config import DatasetConfig, ProcessingConfig
from loaders.dataset_factory import DatasetLoaderFactory
from processors.data_processor import DataProcessor

def process_single_item_worker(item_data: Dict[str, Any], processing_config: ProcessingConfig) -> Tuple:
    """
    Worker function to process a single data item.
    It now processes video, audio, and landmarks.
    """
    processor = DataProcessor(processing_config)
    
    try:
        # Extract paths for all three modalities
        video_path = item_data.get('path')
        audio_path = item_data.get('audio_path')
        landmark_path = item_data.get('landmarks')
        label = item_data.get('label', 0)
        
        if not all([video_path, audio_path, landmark_path]):
            raise ValueError(f"Missing required data paths for item: {video_path}")
        
        # Process each modality using the DataProcessor
        video_tensor = processor.process_video(video_path)
        audio_mfcc_tensor = processor.process_audio(audio_path)
        landmark_tensor = processor.process_landmarks(landmark_path)
        
        # Return the full multimodal tuple
        return (video_tensor, audio_mfcc_tensor, landmark_tensor, 
                torch.tensor(int(label), dtype=torch.long), video_path, "")

        
    except Exception as e:
        # 실패 시 에러 메시지와 함께 더미 데이터를 확실하게 반환하도록 수정
        print(f"\nWorker error on file {item_data.get('path', 'unknown')}: {e}")
        # 새로운 DataProcessor를 생성하여 config가 확실히 있도록 보장
        return DataProcessor(processing_config).create_dummy_data(
            item_data.get('path', 'unknown'),
            str(e)
        )

class MultiModalDataset(Dataset):
    """
    A PyTorch Dataset for loading and processing multimodal deepfake data,
    including video, audio (as MFCCs), and facial landmarks.
    """
    
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
        self.loader_factory = DatasetLoaderFactory(dataset_config)
        self.file_list = self._load_all_files()
        self.preprocessed_data = [None] * len(self.file_list)
        
        cpu_cores = mp.cpu_count() or 4
        self.preload_workers = preload_workers if preload_workers is not None else max(1, cpu_cores // 2)
        
        print(f"✅ Initialized {self.__class__.__name__} for '{partition}': "
              f"{len(self.file_list)} files found, using {self.preload_workers} workers for preloading.")
    
    def _load_all_files(self) -> List[Dict[str, Any]]:
        """Loads file paths from all active dataset loaders."""
        all_files = []
        active_loaders = self.loader_factory.get_active_loaders()
        
        if not active_loaders:
            print(f"⚠️ Warning: No active dataset loaders for partition '{self.partition}'.")
            return all_files
        
        print(f"📂 Loading file lists for partition '{self.partition}'...")
        for loader in active_loaders:
            try:
                files = loader.load_files(self.partition)
                all_files.extend(files)
                print(f"  - Loaded {len(files)} files from {loader.get_dataset_name()}")
            except Exception as e:
                print(f"🚨 Error loading from {loader.get_dataset_name()}: {e}")
        
        print(f"Total files loaded for '{self.partition}': {len(all_files)}")
        return all_files
    
    def preload_data(self, max_items: Optional[int] = None) -> None:
        """Preprocess part (or all) of the data in parallel and store it in memory.
        
        Args:
            max_items (int, optional): Limit the number of items to preload.
                                    If None, preload everything.
        """
        if not self.file_list: 
            return
        
        n_items = len(self.file_list) if max_items is None else min(max_items, len(self.file_list))
        print(f"\n🔄 Starting data preloading for '{self.partition}' ({n_items}/{len(self.file_list)} items)...")
        start_time = time.time()
        total_processed, total_failed = 0, 0

        with ProcessPoolExecutor(max_workers=self.preload_workers, mp_context=mp.get_context('spawn')) as executor:
            future_to_idx = {
                    # CORRECTED: Pass the entire self.processing_config object
                    executor.submit(
                        process_single_item_worker, 
                        self.file_list[i], 
                        self.processing_config
                    ): i
                    for i in range(n_items)
                }
            
            progress_bar = tqdm(as_completed(future_to_idx), total=n_items, desc=f"Preloading {self.partition}")
            for future in progress_bar:
                idx = future_to_idx[future]
                try:
                    result = future.result(timeout=120)
                    self.preprocessed_data[idx] = result
                    if len(result) >= 6 and result[5] is not None: 
                        total_failed += 1
                    else:
                        total_processed += 1
                except Exception as e:
                    total_failed += 1
                    self.preprocessed_data[idx] = DataProcessor(self.processing_config).create_dummy_data(
                        self.file_list[idx].get('path', ''), str(e))
                
                progress_bar.set_postfix({'success': total_processed, 'failed': total_failed})

        print(f"\n✅ Preload completed in {time.time() - start_time:.2f}s. (Success: {total_processed}, Failed: {total_failed})")

    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, index: int) -> Tuple:
        """Retrieves a single data item."""
        if self.preprocessed_data[index] is not None:
            return self.preprocessed_data[index]
        
        return process_single_item_worker(self.file_list[index], self.processing_config)

def seed_worker(worker_id):
    """Seed function for DataLoader workers to ensure reproducibility."""
    import numpy as np
    import random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)