from abc import ABC, abstractmethod
from typing import List, Dict, Any
import os


class BaseDatasetLoader(ABC):
    """데이터셋 로더의 기본 인터페이스"""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        
    @abstractmethod
    def load_files(self, partition: str) -> List[Dict[str, Any]]:
        """
        파일 목록 로드
        Returns:
            List of dicts with keys: 'path', 'landmarks', 'label', 'data_label'
        """
        pass
    
    def validate_paths(self) -> bool:
        """경로 유효성 검사"""
        return os.path.exists(self.base_dir) if self.base_dir else False
    
    def _check_file_exists(self, video_path: str, landmark_path: str) -> bool:
        """비디오와 랜드마크 파일 존재 확인"""
        return os.path.exists(video_path) and os.path.exists(landmark_path)
    
    def get_dataset_name(self) -> str:
        """데이터셋 이름 반환"""
        return self.__class__.__name__.replace('Loader', '')