"""
Defines concrete loader classes for each specific dataset.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from .base_loader import BaseDatasetLoader

# [DEBUG] Set to True to print every path the loader is checking.
DEBUG = False

class DeepSpeakLoader(BaseDatasetLoader):
    """Loads file paths from the DeepSpeak dataset."""
    
    def __init__(self, base_dir: str, load_real: bool = False, load_fake: bool = False):
        super().__init__(base_dir)
        self.load_real = load_real
        self.load_fake = load_fake
        self.features_root = os.path.join(base_dir, "features", "features_mtcnn")
    
    def load_files(self, partition: str) -> List[Dict[str, Any]]:
        files = []
        if self.load_real:
            files.extend(self._load_partition_files(partition, is_fake=False))
        if self.load_fake:
            files.extend(self._load_partition_files(partition, is_fake=True))
        return files
    
    def _load_partition_files(self, partition: str, is_fake: bool) -> List[Dict[str, Any]]:
        files = []
        label = 1 if is_fake else 0
        data_label = 2 if is_fake else 1
        
        annotation_file = "annotations-fake.csv" if is_fake else "annotations-real.csv"
        annotation_path = os.path.join(self.base_dir, annotation_file)
        
        if not os.path.exists(annotation_path): return files
            
        annotations_df = pd.read_csv(annotation_path)
        valid_identities = self._get_valid_identities(partition)
        
        for _, row in annotations_df.iterrows():
            identity_col = "identity-source" if is_fake else "identity"
            identity = str(row.get(identity_col, ""))
            if not identity or (valid_identities and identity not in valid_identities):
                continue
                
            video_filename = str(row.get("video-file", "")).replace(".mp4", "")
            if not video_filename: continue
                
            feature_dir = self._find_feature_directory(video_filename)
            if feature_dir:
                video_path = os.path.join(feature_dir, "cropped_video.mp4")
                audio_path = os.path.join(feature_dir, "audio.wav")
                landmark_path = os.path.join(feature_dir, "landmarks.npy")
                
                # [DEBUG] Print the paths being checked
                if DEBUG:
                    print(f"Checking:\n  V: {video_path}\n  A: {audio_path}\n  L: {landmark_path}\n")

                if self._check_files_exist(video_path, audio_path, landmark_path):
                    files.append({
                        "path": video_path, "audio_path": audio_path, "landmarks": landmark_path,
                        "label": label, "data_label": data_label
                    })
        return files
    
    def _get_valid_identities(self, partition: str) -> set:
        split_file = os.path.join(self.base_dir, "annotations-split-def.csv")
        if not os.path.exists(split_file): return set()
        split_df = pd.read_csv(split_file)
        actual_partition = "test" if partition == "val" else partition
        valid_ids = split_df[split_df['split'] == actual_partition]["identity"].astype(str).unique()
        return set(valid_ids)
    
    def _find_feature_directory(self, feature_name: str) -> str:
        full_path = os.path.join(self.features_root, feature_name)
        if os.path.isdir(full_path):
            return full_path
        return None

class KoDFLoader(BaseDatasetLoader):
    """Loads file paths from the KoDF dataset."""
    
    def __init__(self, base_dir: str, load_real: bool = False, load_fake: bool = False):
        super().__init__(base_dir)
        self.load_real = load_real
        self.load_fake = load_fake
    
    def load_files(self, partition: str) -> List[Dict[str, Any]]:
        files = []
        if self.load_real:
            files.extend(self._load_partition_files(partition, is_fake=False))
        if self.load_fake:
            files.extend(self._load_partition_files(partition, is_fake=True))
        return files
    
    def _load_partition_files(self, partition: str, is_fake: bool) -> List[Dict[str, Any]]:
        label = 1 if is_fake else 0
        data_label = 4 if is_fake else 3
        meta_file = "fake.csv" if is_fake else "real.csv"
        meta_path = os.path.join(self.base_dir, "validate_meta_data", meta_file)
        
        if not os.path.exists(meta_path): return []
        try:
            df = pd.read_csv(meta_path, encoding='cp949')
        except UnicodeDecodeError:
            df = pd.read_csv(meta_path, encoding='utf-8')
        
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        train_df, val_df, test_df = np.split(df_shuffled, [int(.7*len(df_shuffled)), int(.9*len(df_shuffled))])
        
        if partition == "train": current_df = train_df
        elif partition == "val": current_df = val_df
        else: current_df = test_df
        
        return self._process_df(current_df, is_fake, label, data_label)

    def _process_df(self, df: pd.DataFrame, is_fake: bool, label: int, data_label: int) -> List[Dict[str, Any]]:
        files = []
        for _, row in df.iterrows():
            if is_fake:
                folder1 = str(row.get("folder", ""))
                folder2_raw = row.get("folder2", "")
                filename = str(row.get("filename", ""))
                if not (folder1 and filename): continue
                folder2 = str(int(float(folder2_raw))) if folder2_raw != "" and not pd.isna(folder2_raw) else "0"
                folder_name = filename.replace(".mp4", "")
                base_path = os.path.join(self.base_dir, folder1, folder2, folder_name)
            else:
                video_id = str(row.get("영상ID", ""))
                if not video_id: continue
                folder_name = video_id.replace(".mp4", "")
                prefix = folder_name.split('_')[0]
                base_path = None
                for parent_folder in ["원본1", "원본2"]:
                    path_candidate = os.path.join(self.base_dir, parent_folder, prefix, folder_name)
                    if os.path.isdir(path_candidate):
                        base_path = path_candidate
                        break
                if not base_path: continue

            video_path = os.path.join(base_path, "cropped_video.mp4")
            audio_path = os.path.join(base_path, "audio.wav")
            landmark_path = os.path.join(base_path, "landmarks.npy")
            
            # [DEBUG] Print the paths being checked
            if DEBUG:
                print(f"Checking:\n  V: {video_path}\n  A: {audio_path}\n  L: {landmark_path}\n")

            if self._check_files_exist(video_path, audio_path, landmark_path):
                files.append({
                    "path": video_path, "audio_path": audio_path, "landmarks": landmark_path,
                    "label": label, "data_label": data_label
                })
        return files

class FakeAVCelebLoader(BaseDatasetLoader):
    """Loads file paths from the FakeAVCeleb dataset."""
    
    def load_files(self, partition: str) -> List[Dict[str, Any]]:
        files = []
        # Let's use the corrected CSV name
        csv_path = os.path.join(self.base_dir, 'FakeAVCeleb_metadata_corrected.csv')
        
        if not os.path.exists(csv_path):
            print(f"❌ Error: Metadata file not found at {csv_path}. Make sure you have run correct_csv.py.")
            return []
        
        df = pd.read_csv(csv_path)
        partition_df = df[df['type'] == partition]
        base_feature_dir = os.path.join(self.base_dir, "landmark_features", "features_mediapipe")
        
        for _, row in partition_df.iterrows():
            # Get the relative path from the CSV, e.g., "FakeVideo-FakeAudio/xyz/abc.mp4"
            relative_path_from_csv = row.get("file_path")
            if pd.isna(relative_path_from_csv): continue

            # --- START OF MODIFIED SECTION ---
            # This is the most robust and simple way to get the feature folder path.
            # We simply remove the '.mp4' suffix from the path in the CSV.
            
            feature_folder_relative = relative_path_from_csv.replace(".mp4", "")
            feature_folder_absolute = os.path.join(base_feature_dir, feature_folder_relative)
            
            # --- END OF MODIFIED SECTION ---

            video_path = os.path.join(feature_folder_absolute, "cropped_video.mp4")
            audio_path = os.path.join(feature_folder_absolute, "cropped_video.wav")
            landmark_path = os.path.join(feature_folder_absolute, "landmarks.npy")
            
            if self._check_files_exist(video_path, audio_path, landmark_path):
                files.append({
                    "path": video_path, "audio_path": audio_path, "landmarks": landmark_path,
                    "label": int(row["video_label"]), "data_label": 5
                })
            # Optional: Add an else block to see which specific paths are failing
            # else:
            #     print(f"-> Files not found at: {feature_folder_absolute}")

        return files


class DFDCLoader(BaseDatasetLoader):
    """Loads file paths from the DFDC dataset."""

    def __init__(self, base_dir: str, load_real: bool = False, load_fake: bool = False):
        super().__init__(base_dir)
        self.load_real = load_real
        self.load_fake = load_fake

    def load_files(self, partition: str) -> List[Dict[str, Any]]:
        files = []
        json_path = os.path.join(self.base_dir, "dfdc_preview_set", "dataset.json")
        if not os.path.exists(json_path): return []

        with open(json_path, 'r') as f:
            metadata = json.load(f)

        current_partition = "test" if partition == "val" else partition
        
        for video_key, video_meta in metadata.items():
            if video_meta.get("set") != current_partition: continue
            
            label = 0 if video_meta.get("label", "fake").lower() == "real" else 1
            if (not self.load_real and label == 0) or (not self.load_fake and label == 1): continue

            feature_path = os.path.splitext(video_key)[0]
            base_path = os.path.join(self.base_dir, "dfdc_preview_set_features", feature_path)
            
            video_path = os.path.join(base_path, "cropped_video.mp4")
            audio_path = os.path.join(base_path, "audio.wav")
            landmark_path = os.path.join(base_path, "landmarks.npy")
            
            # [DEBUG] Print the paths being checked
            if DEBUG:
                print(f"Checking:\n  V: {video_path}\n  A: {audio_path}\n  L: {landmark_path}\n")

            if self._check_files_exist(video_path, audio_path, landmark_path):
                files.append({
                    "path": video_path, "audio_path": audio_path, "landmarks": landmark_path,
                    "label": label, "data_label": 6
                })
        return files

class DeepfakeTIMITLoader(BaseDatasetLoader):
    """Loads file paths from the DeepfakeTIMIT dataset."""

    def __init__(self, base_dir: str, load_real: bool = False, load_fake: bool = False):
        super().__init__(base_dir)
        self.load_real = load_real
        self.load_fake = load_fake

    def load_files(self, partition: str) -> List[Dict[str, Any]]:
        files = []
        if self.load_real:
            files.extend(self._load_partition_files(partition, is_fake=False))
        if self.load_fake:
            files.extend(self._load_partition_files(partition, is_fake=True))
        return files

    def _load_partition_files(self, partition: str, is_fake: bool) -> List[Dict[str, Any]]:
        all_samples = []
        label = 1 if is_fake else 0
        data_label = 7

        if is_fake:
            for quality in ["higher_quality", "low_quality"]:
                quality_dir = os.path.join(self.base_dir, "features_mtcnn", quality)
                if os.path.isdir(quality_dir):
                    all_samples.extend(self._find_samples_in_dir(quality_dir, label, data_label))
        else:
            real_dir = os.path.join(self.base_dir, "features_mtcnn", "real")
            if os.path.isdir(real_dir):
                all_samples.extend(self._find_samples_in_dir(real_dir, label, data_label))
        
        if not all_samples: return []
        
        df = pd.DataFrame(all_samples).sample(frac=1, random_state=456).reset_index(drop=True)
        train_df, val_df, test_df = np.split(df, [int(.7*len(df)), int(.9*len(df))])
        
        if partition == "train": return train_df.to_dict('records')
        if partition == "val": return val_df.to_dict('records')
        return test_df.to_dict('records')

    def _find_samples_in_dir(self, directory: str, label: int, data_label: int) -> List[Dict[str, Any]]:
        samples = []
        for root, _, filenames in os.walk(directory):
            if "cropped_video.mp4" in filenames:
                video_path = os.path.join(root, "cropped_video.mp4")
                audio_path = os.path.join(root, "audio.wav")
                landmark_path = os.path.join(root, "landmarks.npy")

                # [DEBUG] Print the paths being checked
                if DEBUG:
                    print(f"Checking:\n  V: {video_path}\n  A: {audio_path}\n  L: {landmark_path}\n")
                
                if self._check_files_exist(video_path, audio_path, landmark_path):
                    samples.append({
                        "path": video_path, "audio_path": audio_path, "landmarks": landmark_path,
                        "label": label, "data_label": data_label
                    })
        return samples

class FaceForensicsLoader(BaseDatasetLoader):
    """Loads file paths from the FaceForensics++ dataset."""
    
    def __init__(self, base_dir: str, load_real: bool = False, load_fake: bool = False):
        super().__init__(base_dir)
        self.load_real = load_real
        self.load_fake = load_fake
    
    def load_files(self, partition: str) -> List[Dict[str, Any]]:
        files = []
        if self.load_real:
            files.extend(self._load_real_files(partition))
        if self.load_fake:
            files.extend(self._load_fake_files(partition))
        return files
        
    def _get_valid_ids(self, partition: str) -> set:
        split_file = os.path.join(self.base_dir, "FaceForensics++", "splits", f"{partition}.json")
        if not os.path.exists(split_file): return None
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        return set(id for pair in split_data for id in pair)

    def _load_real_files(self, partition: str) -> List[Dict[str, Any]]:
        files = []
        valid_ids = self._get_valid_ids(partition)
        real_root = os.path.join(self.base_dir, "FaceForensics++", "video_data", "c40_features", "original_sequences", "youtube")
        if not os.path.isdir(real_root): return []
        
        for video_id in os.listdir(real_root):
            if valid_ids and video_id not in valid_ids: continue
            
            base_path = os.path.join(real_root, video_id)
            video_path = os.path.join(base_path, "cropped_video.mp4")
            audio_path = os.path.join(base_path, "audio.wav")
            landmark_path = os.path.join(base_path, "landmarks.npy")
            
            # [DEBUG] Print the paths being checked
            if DEBUG:
                print(f"Checking:\n  V: {video_path}\n  A: {audio_path}\n  L: {landmark_path}\n")

            if self._check_files_exist(video_path, audio_path, landmark_path):
                files.append({
                    "path": video_path, "audio_path": audio_path, "landmarks": landmark_path,
                    "label": 0, "data_label": 8
                })
        return files
    
    def _load_fake_files(self, partition: str) -> List[Dict[str, Any]]:
        files = []
        valid_ids_set = self._get_valid_ids(partition)
        fake_root = os.path.join(self.base_dir, "FaceForensics++", "video_data", "c40_features", "manipulated_sequences")
        if not os.path.isdir(fake_root): return []
        
        for manip_type in os.listdir(fake_root):
            manip_path = os.path.join(fake_root, manip_type, "c40", "videos")
            if not os.path.isdir(manip_path): continue
            
            video_filenames = [f for f in os.listdir(manip_path) if f.endswith('.mp4')]
            for video_file in video_filenames:
                id_pair = video_file.replace(".mp4", "")
                
                if '_' not in id_pair: continue
                source_id, target_id = id_pair.split('_')
                
                if valid_ids_set and (source_id not in valid_ids_set and target_id not in valid_ids_set): continue

                base_path = os.path.join(manip_path, id_pair)
                video_path = os.path.join(base_path, "cropped_video.mp4")
                audio_path = os.path.join(base_path, "audio.wav")
                landmark_path = os.path.join(base_path, "landmarks.npy")
                
                # [DEBUG] Print the paths being checked
                if DEBUG:
                    print(f"Checking:\n  V: {video_path}\n  A: {audio_path}\n  L: {landmark_path}\n")

                if self._check_files_exist(video_path, audio_path, landmark_path):
                    files.append({
                        "path": video_path, "audio_path": audio_path, "landmarks": landmark_path,
                        "label": 1, "data_label": 9
                    })
        return files