import os
import json
import pandas as pd
from typing import List, Dict, Any
from .base_loader import BaseDatasetLoader


class DeepSpeakLoader(BaseDatasetLoader):
    """DeepSpeak 데이터셋 로더"""
    
    def __init__(self, base_dir: str, load_real: bool = False, load_fake: bool = False):
        super().__init__(base_dir)
        self.load_real = load_real
        self.load_fake = load_fake
        self.features_root = os.path.join(base_dir, "features", "features_mtcnn")
    
    def load_files(self, partition: str) -> List[Dict[str, Any]]:
        files = []
        
        if self.load_real:
            files.extend(self._load_deepspeak_files(partition, is_fake=False))
        if self.load_fake:
            files.extend(self._load_deepspeak_files(partition, is_fake=True))
            
        return files
    
    def _load_deepspeak_files(self, partition: str, is_fake: bool) -> List[Dict[str, Any]]:
        """DeepSpeak 파일 로드"""
        files = []
        label = 1 if is_fake else 0
        data_label = 2 if is_fake else 1
        
        annotation_file = "annotations-fake.csv" if is_fake else "annotations-real.csv"
        annotation_path = os.path.join(self.base_dir, annotation_file)
        
        if not os.path.exists(annotation_path):
            return files
            
        annotations_df = pd.read_csv(annotation_path)
        valid_identities = self._get_valid_identities(partition)
        
        for _, row in annotations_df.iterrows():
            identity = str(row.get("identity-source" if is_fake else "identity", ""))
            if not identity or (valid_identities and identity not in valid_identities):
                continue
                
            video_filename = str(row.get("video-file", ""))
            if not video_filename:
                continue
                
            feature_dir = self._find_feature_directory(video_filename.replace(".mp4", ""))
            if feature_dir:
                video_path = os.path.join(feature_dir, "cropped_video.mp4")
                landmark_path = os.path.join(feature_dir, "landmarks.npy")
                
                if self._check_file_exists(video_path, landmark_path):
                    files.append({
                        "path": video_path,
                        "landmarks": landmark_path,
                        "label": label,
                        "data_label": data_label
                    })
        
        return files
    
    def _get_valid_identities(self, partition: str) -> set:
        """파티션별 유효한 identity 목록 반환"""
        split_file = os.path.join(self.base_dir, "annotations-split-def.csv")
        if not os.path.exists(split_file):
            return set()
            
        split_df = pd.read_csv(split_file)
        actual_partition = "test" if partition == "val" else partition
        
        return set(split_df[split_df['split'] == actual_partition]["identity"].astype(str).unique())
    
    def _find_feature_directory(self, feature_name: str) -> str:
        """특징 디렉토리 탐색"""
        potential_dirs = [os.path.join(self.features_root, feature_name)]
        
        if os.path.exists(self.features_root):
            for subdir in os.listdir(self.features_root):
                subdir_path = os.path.join(self.features_root, subdir)
                if os.path.isdir(subdir_path):
                    potential_dirs.append(os.path.join(subdir_path, feature_name))
        
        for dir_path in potential_dirs:
            if os.path.isdir(dir_path):
                video_path = os.path.join(dir_path, "cropped_video.mp4")
                landmark_path = os.path.join(dir_path, "landmarks.npy")
                if self._check_file_exists(video_path, landmark_path):
                    return dir_path
        
        return None


class KoDFLoader(BaseDatasetLoader):
    """KoDF 데이터셋 로더"""
    
    def __init__(self, base_dir: str, load_real: bool = False, load_fake: bool = False):
        super().__init__(base_dir)
        self.load_real = load_real
        self.load_fake = load_fake
    
    def load_files(self, partition: str) -> List[Dict[str, Any]]:
        files = []
        
        if self.load_real:
            files.extend(self._load_kodf_files(partition, is_fake=False))
        if self.load_fake:
            files.extend(self._load_kodf_files(partition, is_fake=True))
            
        return files
    
    def _load_kodf_files(self, partition: str, is_fake: bool) -> List[Dict[str, Any]]:
        """KoDF 파일 로드"""
        files = []
        label = 1 if is_fake else 0
        data_label = 4 if is_fake else 3
        
        meta_file = "fake.csv" if is_fake else "real.csv"
        meta_path = os.path.join(self.base_dir, "validate_meta_data", meta_file)
        
        if not os.path.exists(meta_path):
            return files
            
        try:
            df = pd.read_csv(meta_path, encoding='cp949')
        except:
            df = pd.read_csv(meta_path, encoding='utf-8')
        
        # 데이터 분할 (재현성을 위해 random_state 고정)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        total_len = len(df)
        train_end = int(total_len * 0.7)
        val_end = int(total_len * 0.9)
        
        if partition == "train":
            current_df = df.iloc[:train_end]
        elif partition == "val":
            current_df = df.iloc[train_end:val_end]
        elif partition == "test":
            current_df = df.iloc[val_end:]
        else:
            return files
        
        if is_fake:
            files.extend(self._process_kodf_fake(current_df, label, data_label))
        else:
            files.extend(self._process_kodf_real(current_df, label, data_label))
        
        return files
    
    def _process_kodf_real(self, df: pd.DataFrame, label: int, data_label: int) -> List[Dict[str, Any]]:
        """KoDF Real 파일 처리"""
        files = []
        real_folders = ["원본1", "원본2"]
        
        for _, row in df.iterrows():
            video_id = str(row.get("영상ID", ""))
            if not video_id:
                continue
                
            folder_name = video_id.replace(".mp4", "")
            prefix = folder_name.split('_')[0]
            
            for parent_folder in real_folders:
                base_path = os.path.join(self.base_dir, parent_folder, prefix, folder_name)
                video_path = os.path.join(base_path, "cropped_video.mp4")
                landmark_path = os.path.join(base_path, "landmarks.npy")
                
                if self._check_file_exists(video_path, landmark_path):
                    files.append({
                        "path": video_path,
                        "landmarks": landmark_path,
                        "label": label,
                        "data_label": data_label
                    })
                    break
        
        return files
    
    def _process_kodf_fake(self, df: pd.DataFrame, label: int, data_label: int) -> List[Dict[str, Any]]:
        """KoDF Fake 파일 처리"""
        files = []
        
        for _, row in df.iterrows():
            folder1 = str(row.get("folder", ""))
            folder2_raw = row.get("folder2", "")
            filename = str(row.get("filename", ""))
            
            if not (folder1 and filename):
                continue
                
            folder2 = str(int(float(folder2_raw))) if folder2_raw != "" and not pd.isna(folder2_raw) else "0"
            folder_name = filename.replace(".mp4", "")
            
            base_path = os.path.join(self.base_dir, folder1, folder2, folder_name)
            video_path = os.path.join(base_path, "cropped_video.mp4")
            landmark_path = os.path.join(base_path, "landmarks.npy")
            
            if self._check_file_exists(video_path, landmark_path):
                files.append({
                    "path": video_path,
                    "landmarks": landmark_path,
                    "label": label,
                    "data_label": data_label
                })
        
        return files


class FakeAVCelebLoader(BaseDatasetLoader):
    """FakeAVCeleb 데이터셋 로더"""
    
    def load_files(self, partition: str) -> List[Dict[str, Any]]:
        files = []
        csv_path = os.path.join(self.base_dir, 'FakeAVCeleb_metadata.csv')
        
        if not os.path.exists(csv_path):
            return files
            
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading FakeAVCeleb CSV: {e}")
            return files
        
        required_cols = ['type', 'video_label', 'file_path']
        if not all(col in df.columns for col in required_cols):
            print(f"Missing required columns in FakeAVCeleb metadata")
            return files
        
        partition_df = df[df['type'] == partition].copy()
        if partition_df.empty:
            return files
        
        base_feature_dir = os.path.join(self.base_dir, "landmark_features", "features_mediapipe")
        
        for _, row in partition_df.iterrows():
            if pd.isna(row['file_path']):
                continue
                
            label = int(row["video_label"])
            file_path = str(row["file_path"])
            
            relative_dir = os.path.dirname(file_path)
            filename_no_ext = os.path.splitext(os.path.basename(file_path))[0]
            
            feature_folder = os.path.join(base_feature_dir, relative_dir, filename_no_ext)
            video_path = os.path.join(feature_folder, "cropped_video.mp4")
            landmark_path = os.path.join(feature_folder, "landmarks.npy")
            
            if self._check_file_exists(video_path, landmark_path):
                files.append({
                    "path": video_path,
                    "landmarks": landmark_path,
                    "label": label,
                    "data_label": 5  # FakeAVCeleb
                })
        
        return files


class DFDCLoader(BaseDatasetLoader):
    """DFDC 데이터셋 로더"""
    
    def __init__(self, base_dir: str, load_real: bool = False, load_fake: bool = False):
        super().__init__(base_dir)
        self.load_real = load_real
        self.load_fake = load_fake
    
    def load_files(self, partition: str) -> List[Dict[str, Any]]:
        files = []
        json_path = os.path.join(self.base_dir, "dfdc_preview_set", "dataset.json")
        
        if not os.path.exists(json_path):
            return files
            
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Error reading DFDC JSON: {e}")
            return files
        
        # DFDC는 train/test만 있으므로 val -> test 매핑
        current_partition = "test" if partition == "val" else partition
        
        for video_key, video_meta in metadata.items():
            if video_meta.get("set") != current_partition:
                continue
                
            label_str = video_meta.get("label", "fake")
            label = 0 if label_str.lower() == "real" else 1
            
            if (not self.load_real and label == 0) or (not self.load_fake and label == 1):
                continue
            
            feature_path = os.path.splitext(video_key)[0]
            base_path = os.path.join(self.base_dir, "dfdc_preview_set_features", feature_path)
            
            video_path = os.path.join(base_path, "cropped_video.mp4")
            landmark_path = os.path.join(base_path, "landmarks.npy")
            
            if self._check_file_exists(video_path, landmark_path):
                files.append({
                    "path": video_path,
                    "landmarks": landmark_path,
                    "label": label,
                    "data_label": 6  # DFDC
                })
        
        return files


class DeepfakeTIMITLoader(BaseDatasetLoader):
    """DeepfakeTIMIT 데이터셋 로더 (모두 fake)"""
    
    def load_files(self, partition: str) -> List[Dict[str, Any]]:
        files = []
        all_samples = []
        
        for quality in ["higher_quality", "low_quality"]:
            quality_dir = os.path.join(self.base_dir, "features_mtcnn", quality)
            if not os.path.isdir(quality_dir):
                continue
                
            for root, _, filenames in os.walk(quality_dir):
                if "cropped_video.mp4" in filenames:
                    video_path = os.path.join(root, "cropped_video.mp4")
                    landmark_path = os.path.join(root, "landmarks.npy")
                    
                    if self._check_file_exists(video_path, landmark_path):
                        all_samples.append({
                            "path": video_path,
                            "landmarks": landmark_path,
                            "label": 1,  # 모두 fake
                            "data_label": 7  # DeepfakeTIMIT
                        })
        
        if not all_samples:
            return files
        
        # 데이터 분할 (재현성을 위해 random_state 고정)
        df = pd.DataFrame(all_samples).sample(frac=1, random_state=456).reset_index(drop=True)
        total_len = len(df)
        train_end = int(total_len * 0.7)
        val_end = int(total_len * 0.9)
        
        if partition == "train":
            selected_df = df.iloc[:train_end]
        elif partition == "val":
            selected_df = df.iloc[train_end:val_end]
        elif partition == "test":
            selected_df = df.iloc[val_end:]
        else:
            return files
        
        return selected_df.to_dict('records')


class FaceForensicsLoader(BaseDatasetLoader):
    """FaceForensics++ 데이터셋 로더"""
    
    def __init__(self, base_dir: str, load_real: bool = False, load_fake: bool = False):
        super().__init__(base_dir)
        self.load_real = load_real
        self.load_fake = load_fake
    
    def load_files(self, partition: str) -> List[Dict[str, Any]]:
        files = []
        
        if self.load_real:
            files.extend(self._load_faceforensics_real(partition))
        if self.load_fake:
            files.extend(self._load_faceforensics_fake(partition))
            
        return files
    
    def _get_valid_pairs(self, partition: str) -> set:
        """파티션별 유효한 비디오 페어 반환"""
        split_file = os.path.join(self.base_dir, "FaceForensics++", "splits", f"{partition}.json")
        valid_pairs = set()
        
        if os.path.exists(split_file):
            try:
                with open(split_file, 'r') as f:
                    split_data = json.load(f)
                for pair in split_data:
                    if len(pair) == 2:
                        valid_pairs.add(f"{pair[0]}_{pair[1]}")
                        valid_pairs.add(f"{pair[1]}_{pair[0]}")
            except Exception as e:
                print(f"Error reading FaceForensics split file: {e}")
        
        return valid_pairs
    
    def _load_faceforensics_real(self, partition: str) -> List[Dict[str, Any]]:
        """FaceForensics++ Real 파일 로드"""
        files = []
        valid_pairs = self._get_valid_pairs(partition)
        
        real_root = os.path.join(
            self.base_dir, "FaceForensics++", "video_data", 
            "c40_features", "original_sequences", "youtube"
        )
        
        if not os.path.isdir(real_root):
            return files
        
        for video_id in os.listdir(real_root):
            # split 파일이 있으면 해당 ID가 유효한 페어에 포함되는지 확인
            if valid_pairs:
                is_valid = any(video_id in pair_str.split('_') for pair_str in valid_pairs)
                if not is_valid:
                    continue
            
            video_path = os.path.join(real_root, video_id, "cropped_video.mp4")
            landmark_path = os.path.join(real_root, video_id, "landmarks.npy")
            
            if self._check_file_exists(video_path, landmark_path):
                files.append({
                    "path": video_path,
                    "landmarks": landmark_path,
                    "label": 0,  # Real
                    "data_label": 8  # FaceForensics++ Real
                })
        
        return files
    
    def _load_faceforensics_fake(self, partition: str) -> List[Dict[str, Any]]:
        """FaceForensics++ Fake 파일 로드"""
        files = []
        valid_pairs = self._get_valid_pairs(partition)
        
        fake_root = os.path.join(
            self.base_dir, "FaceForensics++", "video_data", 
            "c40_features", "manipulated_sequences"
        )
        
        if not os.path.isdir(fake_root):
            return files
        
        for manip_type in os.listdir(fake_root):  # Deepfakes, FaceSwap 등
            manip_path = os.path.join(fake_root, manip_type, "c40", "videos")
            if not os.path.isdir(manip_path):
                continue
            
            for video_file in os.listdir(manip_path):
                if not video_file.endswith(".mp4"):
                    continue
                
                id_pair = video_file.replace(".mp4", "")
                
                # split 파일이 있으면 해당 페어가 유효한지 확인
                if valid_pairs:
                    reverse_pair = f"{id_pair.split('_')[1]}_{id_pair.split('_')[0]}"
                    if id_pair not in valid_pairs and reverse_pair not in valid_pairs:
                        continue
                
                video_path = os.path.join(manip_path, id_pair, "cropped_video.mp4")
                landmark_path = os.path.join(manip_path, id_pair, "landmarks.npy")
                
                if self._check_file_exists(video_path, landmark_path):
                    files.append({
                        "path": video_path,
                        "landmarks": landmark_path,
                        "label": 1,  # Fake
                        "data_label": 9  # FaceForensics++ Fake
                    })
        
        return files