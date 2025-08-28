"""
리팩토링된 학습 파이프라인
- 모듈화된 구조로 학습/검증/테스트 분리
- 개선된 에러 처리 및 메트릭 계산
- 설정 기반 관리
- 조기 종료 및 모델 체크포인트
- 기존 코드와의 호환성 유지
"""

# 기존 코드와의 호환성을 위한 메인 임포트
from trainer.training_pipeline import (
    ModelTrainer,
    TrainingPipeline,
    run_training_pipeline,  # 기존 함수명 유지
    create_trainer_from_config,
    quick_train
)

from config.training_config import TrainingConfig, ModelConfig, get_device
from utils.metrics import MetricsCalculator, format_metrics, calculate_simple_metrics
from utils.training_utils import (
    set_all_seeds,
    get_model_summary,
    save_training_config,
    count_parameters,
    check_gpu_memory,
    cleanup_gpu_memory,
    EarlyStopping,
    ModelCheckpoint,
    TrainingLogger,
    print_system_info,
    validate_config
)

# 편의 함수들
def create_training_pipeline(
    model,
    train_loader,
    config_dict=None,
    device="auto"
):
    """간편한 학습 파이프라인 생성"""
    import torch.nn as nn
    import torch.optim as optim
    
    # 설정 생성
    if config_dict is None:
        config_dict = {}
    
    config = TrainingConfig(
        num_epochs=config_dict.get('num_epochs', 10),
        batch_size=config_dict.get('batch_size', 16),
        learning_rate=config_dict.get('learning_rate', 1e-3),
        recon_loss_weight=config_dict.get('recon_loss_weight', 0.5),
        device=device
    )
    
    # 디바이스 설정
    device_obj = get_device(device)
    model = model.to(device_obj)
    
    # 손실함수와 옵티마이저
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    return TrainingPipeline(model, criterion, optimizer, config)


def train_with_defaults(
    model,
    train_loader,
    val_loader=None,
    test_loader=None,
    epochs=10,
    lr=1e-3,
    device="auto"
):
    """기본 설정으로 빠른 학습"""
    pipeline = create_training_pipeline(
        model=model,
        train_loader=train_loader,
        config_dict={
            'num_epochs': epochs,
            'learning_rate': lr
        },
        device=device
    )
    
    pipeline.train(train_loader, val_loader, test_loader)
    return pipeline


# 사용 예시 및 테스트용 코드
def example_usage():
    """사용 예시"""
    print("=== 리팩토링된 Trainer 사용 예시 ===")
    
    # 1. 기본 사용법 (기존 코드와 호환)
    print("1. 기존 호환 방식:")
    print("""
    from trainer import run_training_pipeline
    
    run_training_pipeline(
        model, train_loader, val_loader, test_loader,
        criterion_bce, optimizer, config, device
    )
    """)
    
    # 2. 새로운 객체지향 방식
    print("2. 새로운 객체지향 방식:")
    print("""
    from trainer import TrainingConfig, ModelTrainer
    
    config = TrainingConfig(num_epochs=20, learning_rate=1e-4)
    trainer = ModelTrainer(model, criterion, optimizer, config, device)
    trainer.train(train_loader, val_loader, test_loader)
    """)
    
    # 3. 간편 사용법
    print("3. 간편 사용법:")
    print("""
    from trainer import train_with_defaults
    
    pipeline = train_with_defaults(
        model, train_loader, val_loader, test_loader,
        epochs=15, lr=2e-4
    )
    """)


# 모듈 정보
__version__ = "2.0.0"
__author__ = "Refactored Training Module"

# 주요 클래스와 함수들 export
__all__ = [
    # 메인 클래스들
    'ModelTrainer',
    'TrainingPipeline',
    'TrainingConfig',
    'ModelConfig',
    
    # 기존 호환 함수
    'run_training_pipeline',
    
    # 유틸리티
    'MetricsCalculator',
    'EarlyStopping',
    'ModelCheckpoint',
    'TrainingLogger',
    
    # 편의 함수들
    'create_training_pipeline',
    'train_with_defaults',
    'create_trainer_from_config',
    'quick_train',
    
    # 도구 함수들
    'set_all_seeds',
    'get_device',
    'get_model_summary',
    'check_gpu_memory',
    'cleanup_gpu_memory',
    'print_system_info',
    'validate_config'
]


if __name__ == "__main__":
    print("리팩토링된 트레이너 모듈")
    print(f"버전: {__version__}")
    example_usage()
    print_system_info()