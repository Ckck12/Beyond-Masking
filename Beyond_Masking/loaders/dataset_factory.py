from typing import List
from config.dataset_config import DatasetConfig
from .dataset_loaders import (
    DeepSpeakLoader, KoDFLoader, FakeAVCelebLoader, 
    DFDCLoader, DeepfakeTIMITLoader, FaceForensicsLoader
)
from .base_loader import BaseDatasetLoader


class DatasetLoaderFactory:
    """데이터셋 로더 팩토리 클래스"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self._loaders = []
        self._initialize_loaders()
    
    def _initialize_loaders(self):
        """설정에 따라 활성 로더들 초기화"""
        # DeepSpeak
        if (self.config.load_deepspeak_real or self.config.load_deepspeak_fake) and self.config.deepspeak_base_dir:
            loader = DeepSpeakLoader(
                self.config.deepspeak_base_dir,
                load_real=self.config.load_deepspeak_real,
                load_fake=self.config.load_deepspeak_fake
            )
            if loader.validate_paths():
                self._loaders.append(loader)
        
        # KoDF
        if (self.config.load_kodf_real or self.config.load_kodf_fake) and self.config.kodf_features_root_dir:
            loader = KoDFLoader(
                self.config.kodf_features_root_dir,
                load_real=self.config.load_kodf_real,
                load_fake=self.config.load_kodf_fake
            )
            if loader.validate_paths():
                self._loaders.append(loader)
        
        # FakeAVCeleb
        if self.config.load_fakeavceleb and self.config.fakeavceleb_dir:
            loader = FakeAVCelebLoader(self.config.fakeavceleb_dir)
            if loader.validate_paths():
                self._loaders.append(loader)
        
        # DFDC
        if (self.config.load_dfdc_real or self.config.load_dfdc_fake) and self.config.dfdc_dir:
            loader = DFDCLoader(
                self.config.dfdc_dir,
                load_real=self.config.load_dfdc_real,
                load_fake=self.config.load_dfdc_fake
            )
            if loader.validate_paths():
                self._loaders.append(loader)
        
        # DeepfakeTIMIT
        if self.config.load_deepfaketimit_fake and self.config.deepfaketimit_dir:
            loader = DeepfakeTIMITLoader(self.config.deepfaketimit_dir)
            if loader.validate_paths():
                self._loaders.append(loader)
        
        # FaceForensics++
        if (self.config.load_faceforensics_real or self.config.load_faceforensics_fake) and self.config.faceforensics_dir:
            loader = FaceForensicsLoader(
                self.config.faceforensics_dir,
                load_real=self.config.load_faceforensics_real,
                load_fake=self.config.load_faceforensics_fake
            )
            if loader.validate_paths():
                self._loaders.append(loader)
    
    def get_active_loaders(self) -> List[BaseDatasetLoader]:
        """활성화된 로더 목록 반환"""
        return self._loaders
    
    def get_loader_names(self) -> List[str]:
        """활성화된 로더 이름 목록 반환"""
        return [loader.get_dataset_name() for loader in self._loaders]