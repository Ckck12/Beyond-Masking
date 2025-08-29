"""
DataProcessor for handling video, audio, and landmark data.
"""

import torch
import numpy as np
import cv2
import librosa
from decord import VideoReader, cpu
import torchvision.transforms as T
from typing import Tuple
import torch.nn.functional as F
from config.dataset_config import ProcessingConfig

class DataProcessor:
    """Handles the loading and preprocessing of video, audio, and landmark data."""
    
    def __init__(self, processing_config: ProcessingConfig):
        """
        Initializes the processor with a processing configuration.

        Args:
            processing_config (ProcessingConfig): Configuration object with parameters
                                                  for data processing (e.g., frame count, size, mfcc).
        """
        self.config = processing_config

    def process_video(self, video_path: str) -> torch.Tensor:
        """
        Loads a video, processes it, and returns a tensor.
        Processing includes frame sampling, histogram equalization, resizing, and normalization.
        """
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            num_frames = len(vr)
            
            if num_frames == 0:
                raise ValueError("Video is empty or cannot be read.")
                
            frames_array = vr.get_batch(range(num_frames)).asnumpy()
            
            processed_frames = []
            for frame in frames_array:
                # Apply histogram equalization to the Y channel for better contrast
                frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
                frame_ycrcb[:, :, 0] = cv2.equalizeHist(frame_ycrcb[:, :, 0])
                frame_eq = cv2.cvtColor(frame_ycrcb, cv2.COLOR_YCrCb2RGB)
                processed_frames.append(frame_eq)
            
            # Convert to tensor, normalize, and change layout to (T, C, H, W)
            frames_tensor = torch.tensor(np.stack(processed_frames), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
            
            # Pad or crop frames to the desired length
            padded_frames = self.pad_frames(frames_tensor, self.config.desired_num_frames)

            # Resize frames to the target size
            resize_transform = T.Resize(self.config.video_size, antialias=True)
            resized_frames = resize_transform(padded_frames)
            
            return resized_frames # Shape: (T, C, H, W)

        except Exception as e:
            # print(f"Video processing error for {video_path}: {e}")
            raise

    def process_audio(self, audio_path: str) -> torch.Tensor:
        """
        Loads an audio file, computes its MFCC features, and pads/crops to a fixed length.
        """
        try:
            waveform, sr = librosa.load(audio_path, sr=self.config.sample_rate)
            
            mfccs = librosa.feature.mfcc(
                y=waveform, sr=sr, n_mfcc=self.config.n_mfcc,
                n_fft=self.config.n_fft, hop_length=self.config.hop_length
            )
            
            mfccs_tensor = torch.tensor(mfccs, dtype=torch.float32) # Shape: (n_mfcc, T_audio)
            target_len = self.config.desired_num_frames * 10 
            n_frames = mfccs_tensor.shape[1]
            if n_frames > target_len:
                start = (n_frames - target_len) // 2
                mfccs_tensor = mfccs_tensor[:, start:start + target_len]
            elif n_frames < target_len:
                # Pad with zeros
                padding = target_len - n_frames
                mfccs_tensor = F.pad(mfccs_tensor, (0, padding), 'constant', 0)
            return mfccs_tensor # Shape: (n_mfcc, target_len)

        except Exception as e:
            raise

    def process_landmarks(self, landmark_path: str) -> torch.Tensor:
        """
        Loads landmark data, normalizes it, and returns a tensor.
        """
        try:
            landmarks = np.load(landmark_path)
            
            if landmarks.size == 0:
                raise ValueError("Landmark file is empty.")
            
            # Reshape from (T, 478, 2) to (T, 956) if needed
            if landmarks.ndim == 3:
                landmarks = landmarks.reshape(landmarks.shape[0], -1)
            
            if landmarks.shape[1] != self.config.landmark_dim:
                 raise ValueError(f"Unexpected landmark dimension: {landmarks.shape[1]}")

            landmarks_normalized = self.normalize_landmarks(landmarks)
            landmarks_tensor = torch.tensor(landmarks_normalized, dtype=torch.float32)
            
            # Pad or crop frames to the desired length
            padded_landmarks = self.pad_frames(landmarks_tensor, self.config.desired_num_frames)
            
            return padded_landmarks # Shape: (T, landmark_dim)
            
        except Exception as e:
            # print(f"Landmarks processing error for {landmark_path}: {e}")
            raise

    def create_dummy_data(self, path: str, error_msg: str = "Processing failed") -> Tuple:
        """Creates a tuple of zero-filled tensors to be used as a fallback on error."""
        # --- START OF MODIFIED SECTION ---
        
        # Get target dimensions from the config
        T_video = self.config.desired_num_frames
        C, H, W = 3, self.config.video_size[0], self.config.video_size[1]
        T_audio = T_video * 10 

        dummy_video = torch.zeros((T_video, C, H, W), dtype=torch.float32)
        dummy_audio = torch.zeros((self.config.n_mfcc, T_audio), dtype=torch.float32)
        dummy_landmarks = torch.zeros((T_video, self.config.landmark_dim), dtype=torch.float32)
        dummy_label = torch.tensor(0, dtype=torch.long)
        error_info = f"error_placeholder: {error_msg}"[:100]

        return (dummy_video, dummy_audio, dummy_landmarks, dummy_label, path, error_info)
    
    @staticmethod
    def pad_frames(data_tensor: torch.Tensor, desired_frames: int) -> torch.Tensor:
        """
        Pads or crops a tensor along the first dimension (time/frames) to a desired length.
        - Crops from the center if too long.
        - Repeats the last frame if too short.
        """
        current_frames = data_tensor.shape[0]
        
        if current_frames > desired_frames:
            start_idx = (current_frames - desired_frames) // 2
            return data_tensor[start_idx:start_idx + desired_frames]
            
        elif current_frames < desired_frames:
            padding_size = desired_frames - current_frames
            last_frame = data_tensor[-1:].clone()
            padding = last_frame.repeat([padding_size] + [1] * (data_tensor.ndim - 1))
            return torch.cat((data_tensor, padding), dim=0)
            
        return data_tensor

    @staticmethod
    def normalize_landmarks(landmarks_seq: np.ndarray) -> np.ndarray:
        """Standardizes a sequence of landmarks (zero mean, unit variance)."""
        mean = np.mean(landmarks_seq, axis=0)
        std = np.std(landmarks_seq, axis=0)
        std[std == 0] = 1e-6 # Avoid division by zero
        
        return (landmarks_seq - mean) / std