"""
리팩토링된 LandmarkPredictor 모델
- 모듈화된 구조로 각 컴포넌트 분리
- 설정 기반 모델 구성
- 개선된 순전파 및 손실 계산
- 기존 코드와의 호환성 유지
"""

# 메인 모델 클래스 임포트
from model.landmark_predictor import (
    LandmarkPredictor,  # 기존 이름으로 임포트 (호환성)
    LandmarkPredictorLegacy,
    VideoFeatureExtractor,
    VideoFeatureProjector,
    LambdaLayer,
    create_model
)

from config.training_config import ModelConfig

# 편의 함수들
def create_landmark_predictor(
    input_frames=60,
    video_feature_dim=256, 
    recon_target_dim=956,
    dropout_rate=0.3,
    **kwargs
):
    """LandmarkPredictor 생성 편의 함수"""
    config = ModelConfig(
        input_frames=input_frames,
        video_feature_dim=video_feature_dim,
        recon_target_dim=recon_target_dim,
        dropout_rate=dropout_rate,
        **kwargs
    )
    return create_model(config)


def create_lightweight_model(input_frames=30, feature_dim=128):
    """경량화된 모델 생성"""
    config = ModelConfig(
        input_frames=input_frames,
        video_feature_dim=feature_dim,
        recon_target_dim=478 * 2,
        dropout_rate=0.2,
        video_conv_channels=(8, 16, 32),  # 채널 수 줄임
        classifier_hidden_dims=(128, 32)   # 은닉층 크기 줄임
    )
    return create_model(config)


def create_robust_model(input_frames=90, feature_dim=512):
    """고성능 모델 생성"""
    config = ModelConfig(
        input_frames=input_frames,
        video_feature_dim=feature_dim,
        recon_target_dim=478 * 2,
        dropout_rate=0.4,
        video_conv_channels=(32, 64, 128, 256),  # 더 깊은 네트워크
        classifier_hidden_dims=(512, 256, 128)   # 더 큰 분류기
    )
    return create_model(config)


def get_model_info(model):
    """모델 정보 출력"""
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


def test_model_forward(model, batch_size=2, input_frames=60, device="cpu"):
    """모델 순전파 테스트"""
    import torch
    
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    
    # 더미 입력 생성
    video_input = torch.randn(batch_size, 3, input_frames, 224, 224).to(device)
    landmark_input = torch.randn(batch_size, input_frames, 478 * 2).to(device)
    
    print(f"Testing model forward pass...")
    print(f"Video input shape: {video_input.shape}")
    print(f"Landmark input shape: {landmark_input.shape}")
    
    try:
        with torch.no_grad():
            logits, recon_loss = model(video_input, landmark_input)
        
        print(f"✓ Forward pass successful!")
        print(f"Logits shape: {logits.shape}")
        print(f"Reconstruction loss: {recon_loss.item():.6f}")
        
        # 예측 확률 계산
        probs = torch.sigmoid(logits)
        print(f"Prediction probabilities: {probs.squeeze().tolist()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False


def compare_model_outputs(model1, model2, batch_size=1, input_frames=60):
    """두 모델의 출력 비교"""
    import torch
    
    model1.eval()
    model2.eval()
    
    # 동일한 입력으로 테스트
    video_input = torch.randn(batch_size, 3, input_frames, 224, 224)
    landmark_input = torch.randn(batch_size, input_frames, 478 * 2)
    
    with torch.no_grad():
        logits1, recon_loss1 = model1(video_input, landmark_input)
        logits2, recon_loss2 = model2(video_input, landmark_input)
    
    print("Model Comparison:")
    print(f"Model 1 - Logits: {logits1.item():.6f}, Recon Loss: {recon_loss1.item():.6f}")
    print(f"Model 2 - Logits: {logits2.item():.6f}, Recon Loss: {recon_loss2.item():.6f}")
    print(f"Logits difference: {abs(logits1.item() - logits2.item()):.6f}")
    print(f"Recon loss difference: {abs(recon_loss1.item() - recon_loss2.item()):.6f}")


def benchmark_model(model, batch_sizes=[1, 4, 8, 16], input_frames=60, device="cuda"):
    """모델 성능 벤치마킹"""
    import torch
    import time
    
    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"
        print("CUDA not available, using CPU")
    
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    
    print(f"Benchmarking model on {device}")
    print("=" * 50)
    
    results = []
    
    for batch_size in batch_sizes:
        video_input = torch.randn(batch_size, 3, input_frames, 224, 224).to(device)
        landmark_input = torch.randn(batch_size, input_frames, 478 * 2).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(video_input, landmark_input)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # 실제 측정
        times = []
        with torch.no_grad():
            for _ in range(20):
                start_time = time.time()
                _ = model(video_input, landmark_input)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.time() - start_time)
        
        avg_time = np.mean(times) * 1000  # ms
        throughput = batch_size / (avg_time / 1000)  # samples/sec
        
        results.append({
            'batch_size': batch_size,
            'avg_time_ms': avg_time,
            'throughput': throughput
        })
        
        print(f"Batch Size: {batch_size:2d} | "
              f"Avg Time: {avg_time:6.2f} ms | "
              f"Throughput: {throughput:6.1f} samples/sec")
    
    return results


# 사용 예시
def example_usage():
    """사용 예시"""
    print("=== 리팩토링된 Model 사용 예시 ===")
    
    # 1. 기본 사용법 (기존 코드와 호환)
    print("1. 기존 호환 방식:")
    print("""
    from model import LandmarkPredictor
    
    model = LandmarkPredictor(
        input_frames=60,
        video_feature_dim=256,
        recon_target_dim=956,
        dropout_rate=0.3
    )
    """)
    
    # 2. 새로운 설정 기반 방식
    print("2. 새로운 설정 기반 방식:")
    print("""
    from model import ModelConfig, create_model
    
    config = ModelConfig(
        input_frames=60,
        video_feature_dim=256,
        recon_target_dim=956,
        dropout_rate=0.3
    )
    model = create_model(config)
    """)
    
    # 3. 편의 함수 사용
    print("3. 편의 함수 사용:")
    print("""
    from model import create_landmark_predictor, create_lightweight_model
    
    # 기본 모델
    model = create_landmark_predictor()
    
    # 경량화 모델
    light_model = create_lightweight_model(input_frames=30)
    """)


# 모델 검증 함수
def validate_model_config(config: ModelConfig) -> bool:
    """모델 설정 검증"""
    checks = []
    
    if config.input_frames <= 0:
        checks.append("input_frames must be positive")
    
    if config.video_feature_dim <= 0:
        checks.append("video_feature_dim must be positive")
    
    if config.recon_target_dim <= 0:
        checks.append("recon_target_dim must be positive")
    
    if not (0 <= config.dropout_rate < 1):
        checks.append("dropout_rate must be in [0, 1)")
    
    if len(config.video_conv_channels) == 0:
        checks.append("video_conv_channels cannot be empty")
    
    if len(config.classifier_hidden_dims) == 0:
        checks.append("classifier_hidden_dims cannot be empty")
    
    if checks:
        print("Model configuration validation errors:")
        for error in checks:
            print(f"  - {error}")
        return False
    
    return True


# 모듈 정보
__version__ = "2.0.0"
__author__ = "Refactored Model Module"

# 주요 클래스와 함수들 export
__all__ = [
    # 메인 모델 클래스
    'LandmarkPredictor',
    'VideoFeatureExtractor',
    'VideoFeatureProjector',
    'LambdaLayer',
    
    # 설정 클래스
    'ModelConfig',
    
    # 모델 생성 함수들
    'create_model',
    'create_landmark_predictor',
    'create_lightweight_model', 
    'create_robust_model',
    
    # 유틸리티 함수들
    'get_model_info',
    'test_model_forward',
    'compare_model_outputs',
    'benchmark_model',
    'validate_model_config'
]


if __name__ == "__main__":
    print("리팩토링된 모델 모듈")
    print(f"버전: {__version__}")
    example_usage()
    
    # 간단한 테스트
    print("\n=== 모델 테스트 ===")
    model = create_landmark_predictor()
    print(get_model_info(model))
    
    success = test_model_forward(model, batch_size=2, device="cpu")
    if success:
        print("✓ 모델 테스트 완료!")
    else:
        print("✗ 모델 테스트 실패!")