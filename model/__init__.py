"""
모델 모듈

멀티모달 딥페이크 탐지를 위한 모델 클래스들을 제공합니다.
"""

from .landmark_predictor import (
    LandmarkPredictor,
    LandmarkPredictorLegacy,
    VideoFeatureExtractor,
    VideoFeatureProjector,
    LambdaLayer,
    create_model
)

from config.training_config import ModelConfig

def create_landmark_predictor(
    input_frames=60,
    video_feature_dim=256, 
    recon_target_dim=956,
    dropout_rate=0.3,
    **kwargs
):
    config = ModelConfig(
        input_frames=input_frames,
        video_feature_dim=video_feature_dim,
        recon_target_dim=recon_target_dim,
        dropout_rate=dropout_rate,
        **kwargs
    )
    return create_model(config)

def get_model_info(model):
    import torch
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info = [
        f"Model: {model.__class__.__name__}",
        f"Total parameters: {total_params:,}",
        f"Trainable parameters: {trainable_params:,}",
        f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)"
    ]
    if hasattr(model, 'config'):
        config = model.config
        info.extend([
            f"Input frames: {config.input_frames}",
            f"Video feature dim: {config.video_feature_dim}",
            f"Reconstruction target dim: {config.recon_target_dim}",
            f"Dropout rate: {config.dropout_rate}"
        ])
    return "\n".join(info)

from ..model import (
    create_landmark_predictor,
    get_model_info,
    create_lightweight_model,
    create_robust_model,
    test_model_forward,
    compare_model_outputs,
    benchmark_model,
    validate_model_config
)