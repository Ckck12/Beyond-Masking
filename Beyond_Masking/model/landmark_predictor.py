import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from config.training_config import ModelConfig


class LambdaLayer(nn.Module):
    """람다 함수를 모듈로 래핑"""
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def forward(self, x):
        return self.func(x)


class VideoFeatureExtractor(nn.Module):
    """3D CNN 기반 비디오 특징 추출기"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_frames = config.input_frames
        
        # 3D CNN 레이어들 동적 생성
        layers = []
        in_channels = 3
        
        for out_channels in config.video_conv_channels:
            layers.extend([
                nn.Conv3d(in_channels, out_channels, 
                         kernel_size=config.video_kernel_size,
                         padding=config.video_padding),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 2, 2))
            ])
            in_channels = out_channels
        
        # 적응형 풀링으로 고정 크기 출력
        layers.append(nn.AdaptiveAvgPool3d((self.input_frames, 1, 1)))
        
        self.feature_extractor = nn.Sequential(*layers)
        self.final_channels = config.video_conv_channels[-1]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) 형태의 비디오 텐서
        Returns:
            (B, final_channels, T, 1, 1) 형태의 특징
        """
        return self.feature_extractor(x)


class VideoFeatureProjector(nn.Module):
    """비디오 특징을 시간축 특징으로 투영"""
    
    def __init__(self, input_channels: int, output_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Flatten(start_dim=3),  # (B, C, T, 1*1) -> (B, C, T, 1)
            LambdaLayer(lambda x: x.permute(0, 2, 1, 3)),  # (B, T, C, 1)
            nn.Flatten(start_dim=2),  # (B, T, C)
            nn.Linear(input_channels, output_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, 1, 1) 형태의 특징
        Returns:
            (B, T, output_dim) 형태의 시간축 특징
        """
        return self.projection(x)


class LandmarkPredictor(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 비디오 특징 추출기
        self.video_extractor = VideoFeatureExtractor(config)
        
        # 비디오 특징 투영기
        self.video_projector = VideoFeatureProjector(
            input_channels=self.video_extractor.final_channels,
            output_dim=config.video_feature_dim
        )
        
        # 랜드마크 예측기 (재구성)
        self.landmark_predictor = self._build_predictor(config)
        
        # 이진 분류기
        self.classifier = self._build_classifier(config)
        
    def _build_predictor(self, config: ModelConfig) -> nn.Module:
        return nn.Sequential(
            nn.Linear(config.video_feature_dim, config.video_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.video_feature_dim // 2, config.recon_target_dim)
        )
    
    def _build_classifier(self, config: ModelConfig) -> nn.Module:
        layers = []
        input_dim = config.video_feature_dim
        
        for hidden_dim in config.classifier_hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout_rate)
            ])
            input_dim = hidden_dim
        
        # 최종 출력층
        layers.append(nn.Linear(input_dim, 1))
        
        return nn.Sequential(*layers)
    
    def forward(self, video: torch.Tensor, landmarks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            video: (B, C, T, H, W) 형태의 비디오
            landmarks: (B, T, landmark_dim) 형태의 실제 랜드마크
        """
        # 비디오 특징 추출 및 투영
        video_features_raw = self.video_extractor(video)  # (B, C, T, 1, 1)
        video_features_temporal = self.video_projector(video_features_raw)  # (B, T, feature_dim)
        
        # 랜드마크 예측
        predicted_landmarks = self.landmark_predictor(video_features_temporal)  # (B, T, recon_target_dim)
        
        # 재구성 손실 계산
        reconstruction_loss = self._compute_reconstruction_loss(predicted_landmarks, landmarks)
        
        # 시간축 평균으로 집계하여 분류
        aggregated_features = torch.mean(video_features_temporal, dim=1)  # (B, feature_dim)
        logits = self.classifier(aggregated_features)  # (B, 1)
        
        return logits, reconstruction_loss
    
    def _compute_reconstruction_loss(self, predicted: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        """재구성 손실 계산 (shape 자동 조정)"""
        # Shape 일치 확인 및 조정
        if predicted.shape != actual.shape:
            # actual landmarks가 (B, T, 478, 2) 형태인 경우 flatten
            if actual.dim() == 4 and actual.shape[-2:] == (478, 2):
                actual = actual.view(actual.size(0), actual.size(1), -1)
            
            # 여전히 불일치하면 에러
            if predicted.shape != actual.shape:
                raise ValueError(
                    f"Shape mismatch in reconstruction loss: "
                    f"predicted {predicted.shape}, actual {actual.shape}. "
                    f"Expected recon_target_dim={self.config.recon_target_dim}"
                )
        
        return F.mse_loss(predicted, actual, reduction='mean')
    
    def get_video_features(self, video: torch.Tensor) -> torch.Tensor:
        """비디오 특징만 추출 (추론용)"""
        with torch.no_grad():
            video_features_raw = self.video_extractor(video)
            video_features_temporal = self.video_projector(video_features_raw)
            return torch.mean(video_features_temporal, dim=1)
    
    def predict_landmarks(self, video: torch.Tensor) -> torch.Tensor:
        """비디오로부터 랜드마크 예측 (추론용)"""
        with torch.no_grad():
            video_features_raw = self.video_extractor(video)
            video_features_temporal = self.video_projector(video_features_raw)
            predicted_landmarks = self.landmark_predictor(video_features_temporal)
            return predicted_landmarks


# 팩토리 함수
def create_model(config: ModelConfig) -> LandmarkPredictor:
    """모델 생성 팩토리 함수"""
    return LandmarkPredictor(config)


# 기존 코드와의 호환성을 위한 wrapper
class LandmarkPredictorLegacy(LandmarkPredictor):
    """기존 코드 호환성을 위한 래퍼 클래스"""
    
    def __init__(self, input_frames=60, video_feature_dim=256, recon_target_dim=956, dropout_rate=0.3):
        config = ModelConfig(
            input_frames=input_frames,
            video_feature_dim=video_feature_dim,
            recon_target_dim=recon_target_dim,
            dropout_rate=dropout_rate
        )
        super().__init__(config)


# 기존 import와의 호환성
LandmarkPredictor = LandmarkPredictorLegacy