import torch
import numpy as np
import random
import os
from pathlib import Path
from typing import Optional


def set_all_seeds(seed: int = 42):
    """모든 라이브러리의 시드 설정"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # CUDA 연산 재현성을 위한 설정 (성능 저하 가능)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 환경 변수 설정
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"All seeds set to {seed}")


def get_model_summary(model: torch.nn.Module, input_shape: tuple = None) -> str:
    """모델 요약 정보 생성"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = [
        f"Model: {model.__class__.__name__}",
        f"Total parameters: {total_params:,}",
        f"Trainable parameters: {trainable_params:,}",
        f"Non-trainable parameters: {total_params - trainable_params:,}"
    ]
    
    if input_shape:
        summary.append(f"Expected input shape: {input_shape}")
    
    return "\n".join(summary)


def save_training_config(config, save_path: str):
    """학습 설정을 파일로 저장"""
    import json
    from dataclasses import asdict
    
    config_dict = asdict(config) if hasattr(config, '__dataclass_fields__') else vars(config)
    
    # torch.device 객체를 문자열로 변환
    for key, value in config_dict.items():
        if isinstance(value, torch.device):
            config_dict[key] = str(value)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    print(f"Training config saved to {save_path}")


def count_parameters(model: torch.nn.Module) -> dict:
    """모델 파라미터 수 상세 정보"""
    param_counts = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf module
            num_params = sum(p.numel() for p in module.parameters())
            if num_params > 0:
                param_counts[name] = {
                    'total': num_params,
                    'trainable': sum(p.numel() for p in module.parameters() if p.requires_grad)
                }
    
    return param_counts


def check_gpu_memory():
    """GPU 메모리 사용량 확인"""
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    device = torch.cuda.current_device()
    memory_stats = []
    
    allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
    cached = torch.cuda.memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    
    memory_stats.append(f"GPU {device}: {torch.cuda.get_device_name(device)}")
    memory_stats.append(f"Allocated: {allocated:.2f} GB")
    memory_stats.append(f"Cached: {cached:.2f} GB") 
    memory_stats.append(f"Total: {total:.2f} GB")
    memory_stats.append(f"Free: {total - allocated:.2f} GB")
    
    return "\n".join(memory_stats)


def cleanup_gpu_memory():
    """GPU 메모리 캐시 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cache cleared")


class EarlyStopping:
    """조기 종료 헬퍼 클래스"""
    
    def __init__(
        self, 
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        self.monitor_op = np.less if mode == 'min' else np.greater
        self.min_delta *= 1 if mode == 'min' else -1
    
    def __call__(self, score: float) -> bool:
        """점수 업데이트 및 조기 종료 판단"""
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.monitor_op(score - self.min_delta, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop


class ModelCheckpoint:
    """모델 자동 저장 헬퍼 클래스"""
    
    def __init__(
        self,
        save_dir: str = "checkpoints",
        filename: str = "model_{epoch:02d}_{score:.4f}.pth",
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        
        self.best_score = None
        self.monitor_op = np.less if mode == 'min' else np.greater
    
    def save(self, model: torch.nn.Module, epoch: int, metrics: dict):
        """모델 저장 여부 결정 및 저장"""
        score = metrics.get(self.monitor)
        if score is None:
            print(f"Warning: {self.monitor} not found in metrics")
            return None
        
        should_save = True
        if self.save_best_only:
            if self.best_score is None or self.monitor_op(score, self.best_score):
                self.best_score = score
            else:
                should_save = False
        
        if should_save:
            filename = self.filename.format(epoch=epoch, score=score)
            filepath = self.save_dir / filename
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': metrics
            }, filepath)
            
            print(f"Model saved: {filepath}")
            return str(filepath)
        
        return None


# 학습 상태 로깅을 위한 간단한 로거
class TrainingLogger:
    """학습 과정 로깅"""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.history = []
        
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, epoch: int, train_metrics: dict, val_metrics: dict = None):
        """에포크 결과 로깅"""
        log_entry = {
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics or {}
        }
        
        self.history.append(log_entry)
        
        if self.log_file:
            self._write_to_file(log_entry)
    
    def _write_to_file(self, entry: dict):
        """파일에 로그 기록"""
        import json
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry, default=str) + '\n')
    
    def get_history(self) -> list:
        """학습 히스토리 반환"""
        return self.history
    
    def save_history(self, filepath: str):
        """학습 히스토리를 파일로 저장"""
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
        
        print(f"Training history saved to {filepath}")


# 편의 함수들
def print_system_info():
    """시스템 정보 출력"""
    print("=== System Information ===")
    print(f"Python version: {os.sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print(check_gpu_memory())
    print("=" * 30)


def validate_config(config) -> bool:
    """설정 유효성 검사"""
    checks = []
    
    # 기본 설정 검사
    if hasattr(config, 'num_epochs') and config.num_epochs <= 0:
        checks.append("num_epochs must be positive")
    
    if hasattr(config, 'batch_size') and config.batch_size <= 0:
        checks.append("batch_size must be positive")
    
    if hasattr(config, 'learning_rate') and config.learning_rate <= 0:
        checks.append("learning_rate must be positive")
    
    # 디바이스 설정 검사
    if hasattr(config, 'device'):
        device_str = str(config.device)
        if 'cuda' in device_str and not torch.cuda.is_available():
            checks.append("CUDA device requested but not available")
    
    if checks:
        print("Configuration validation errors:")
        for error in checks:
            print(f"  - {error}")
        return False
    
    return True