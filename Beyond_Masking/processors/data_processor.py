import torch
import numpy as np
import cv2
from decord import VideoReader, cpu
import torchvision.transforms as T
from typing import Tuple, Dict, Any


class DataProcessor:
    """비디오 및 랜드마크 데이터 처리 클래스"""
    
    def __init__(self, processing_config=None):
        self.processing_config = processing_config
        
    @staticmethod
    def pad_frames(data_tensor: torch.Tensor, desired_frames: int) -> torch.Tensor:
        """프레임 패딩 처리 - 중앙 크롭 또는 마지막 프레임 반복"""
        current_frames = data_tensor.shape[0]
        
        if current_frames == 0:
            return torch.zeros((desired_frames, *data_tensor.shape[1:]), dtype=data_tensor.dtype)
            
        if current_frames > desired_frames:
            # 중앙에서 desired_frames만큼 크롭
            start_idx = (current_frames - desired_frames) // 2
            return data_tensor[start_idx:start_idx + desired_frames]
            
        elif current_frames < desired_frames:
            # 마지막 프레임 반복으로 패딩
            padding_size = desired_frames - current_frames
            if data_tensor.ndim == 1:
                padding = data_tensor[-1:].repeat(padding_size)
            else:
                repeat_dims = [padding_size] + [1] * (data_tensor.ndim - 1)
                padding = data_tensor[-1:].repeat(*repeat_dims)
            return torch.cat((data_tensor, padding), dim=0)
            
        return data_tensor
    
    def process_video(self, video_path: str, desired_frames: int) -> torch.Tensor:
        """비디오 파일 처리 - shape: (frames, 3, 224, 224)"""
        video_tensor = torch.zeros((desired_frames, 3, 224, 224), dtype=torch.float32)
        
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            num_frames = len(vr)
            
            if num_frames == 0:
                return video_tensor
                
            frames_array = vr.get_batch(range(num_frames)).asnumpy()
            
            # 히스토그램 균등화 적용
            processed_frames = []
            for frame in frames_array:
                if frame.shape[-1] == 3:  # RGB 채널 확인
                    try:
                        # YCrCb 변환 후 Y 채널 히스토그램 균등화
                        frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
                        frame_ycrcb[..., 0] = cv2.equalizeHist(frame_ycrcb[..., 0])
                        frame_eq = cv2.cvtColor(frame_ycrcb, cv2.COLOR_YCrCb2RGB)
                        processed_frames.append(frame_eq)
                    except cv2.error:
                        processed_frames.append(frame)
                else:
                    processed_frames.append(frame)
            
            if processed_frames:
                # numpy -> tensor 변환 및 정규화
                frames_tensor = torch.tensor(
                    np.stack(processed_frames, axis=0), dtype=torch.float32
                ).permute(0, 3, 1, 2) / 255.0
                
                # 프레임 수 조정
                frames_tensor = self.pad_frames(frames_tensor, desired_frames)
                
                # 리사이즈 (224x224)
                if frames_tensor.shape[0] > 0:
                    _, _, h, w = frames_tensor.shape
                    if h != 224 or w != 224:
                        resize_transform = T.Resize((224, 224), antialias=True)
                        resized_frames = []
                        for i in range(frames_tensor.shape[0]):
                            if frames_tensor[i].numel() > 0:
                                resized_frames.append(resize_transform(frames_tensor[i]))
                            else:
                                resized_frames.append(torch.zeros((3, 224, 224), dtype=torch.float32))
                        
                        if resized_frames:
                            video_tensor = torch.stack(resized_frames)
                    else:
                        video_tensor = frames_tensor
                        
        except Exception as e:
            print(f"Video processing error for {video_path}: {e}")
            
        return video_tensor
    
    @staticmethod
    def normalize_landmarks(landmarks_seq: np.ndarray) -> np.ndarray:
        """랜드마크 시퀀스 표준화"""
        if not isinstance(landmarks_seq, np.ndarray):
            landmarks_seq = np.array(landmarks_seq)
            
        if landmarks_seq.size == 0:
            return landmarks_seq
            
        mean = np.mean(landmarks_seq, axis=0)
        std = np.std(landmarks_seq, axis=0)
        std[std == 0] = 1e-6  # 0으로 나누기 방지
        
        return (landmarks_seq - mean) / std
    
    def process_landmarks(self, landmark_path: str, desired_frames: int) -> torch.Tensor:
        """랜드마크 파일 처리 - shape: (frames, 956)"""
        fallback_dim = 478 * 2
        landmarks_tensor = torch.zeros((desired_frames, fallback_dim), dtype=torch.float32)
        
        try:
            landmarks = np.load(landmark_path)
            
            # 3D -> 2D reshape
            if landmarks.ndim == 3:
                landmarks = landmarks.reshape(landmarks.shape[0], -1)
                
            if landmarks.size == 0:
                return landmarks_tensor
                
            # 차원 확인 및 조정
            if landmarks.ndim > 1 and landmarks.shape[1] > 0:
                fallback_dim = landmarks.shape[1]
            
            landmarks_tensor = torch.zeros((desired_frames, fallback_dim), dtype=torch.float32)
            
            # 정규화
            landmarks_normalized = self.normalize_landmarks(landmarks)
            temp_tensor = torch.tensor(landmarks_normalized, dtype=torch.float32)
            
            # 프레임 수 조정
            landmarks_tensor = self.pad_frames(temp_tensor, desired_frames)
            
            # 차원 불일치 시 기본값 반환
            if landmarks_tensor.shape[1] != fallback_dim:
                return torch.zeros((desired_frames, fallback_dim), dtype=torch.float32)
                
        except Exception as e:
            print(f"Landmarks processing error for {landmark_path}: {e}")
            
        return landmarks_tensor
    
    def create_dummy_data(self, path: str, desired_frames: int, error_msg: str = "") -> Tuple:
        """에러 시 더미 데이터 생성"""
        dummy_video = torch.zeros((desired_frames, 3, 224, 224), dtype=torch.float32)
        dummy_landmarks = torch.zeros((desired_frames, 478 * 2), dtype=torch.float32)
        dummy_label = torch.tensor(0, dtype=torch.long)
        
        error_text = f"error_placeholder: {error_msg}"[:100]  # 길이 제한
        
        return (dummy_video, error_text, dummy_landmarks, dummy_label, path)