import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Any, Tuple

from config.training_config import TrainingConfig
from utils.metrics import MetricsCalculator, format_metrics, print_label_distribution


class ModelTrainer:
    """모델 학습 관리 클래스"""
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        device: torch.device
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
        # 모델 저장 디렉토리 생성
        if config.save_best_model:
            Path(config.model_save_dir).mkdir(parents=True, exist_ok=True)
        
        # 학습 상태 추적
        self.best_val_loss = float('inf')
        self.best_model_path = None
        self.current_epoch = 0
        
        # 조기 종료 관련
        self.patience_counter = 0
        self.should_stop = False
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """한 에포크 학습"""
        self.model.train()
        metrics_calc = MetricsCalculator()
        
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {self.current_epoch+1}/{self.config.num_epochs} Training"
        )
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # 배치 데이터 처리
            try:
                loss_dict, pred_dict = self._process_batch(batch_data, is_training=True)
                
                # 메트릭 업데이트
                metrics_calc.update(
                    predictions=pred_dict['predictions'],
                    labels=pred_dict['labels'],
                    probabilities=pred_dict['probabilities'],
                    total_loss=loss_dict['total_loss'],
                    bce_loss=loss_dict['bce_loss'],
                    recon_loss=loss_dict['recon_loss']
                )
                
                # 진행상황 업데이트
                if batch_idx % self.config.log_every_n_batches == 0:
                    current_metrics = metrics_calc.compute_metrics()
                    progress_bar.set_postfix({
                        'loss': f"{current_metrics['loss_total']:.4f}",
                        'acc': f"{current_metrics['accuracy']:.3f}"
                    })
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        return metrics_calc.compute_metrics()
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """한 에포크 검증"""
        if not val_loader or len(val_loader) == 0:
            return {}
        
        self.model.eval()
        metrics_calc = MetricsCalculator()
        
        with torch.no_grad():
            progress_bar = tqdm(
                val_loader,
                desc=f"Epoch {self.current_epoch+1}/{self.config.num_epochs} Validation"
            )
            
            for batch_data in progress_bar:
                try:
                    loss_dict, pred_dict = self._process_batch(batch_data, is_training=False)
                    
                    metrics_calc.update(
                        predictions=pred_dict['predictions'],
                        labels=pred_dict['labels'],
                        probabilities=pred_dict['probabilities'],
                        total_loss=loss_dict['total_loss'],
                        bce_loss=loss_dict['bce_loss'],
                        recon_loss=loss_dict['recon_loss']
                    )
                    
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        return metrics_calc.compute_metrics()
    
    def test_model(self, test_loader: DataLoader, model_path: Optional[str] = None) -> Dict[str, float]:
        """모델 테스트"""
        if not test_loader or len(test_loader) == 0:
            return {}
        
        # 지정된 모델 로드
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()
        metrics_calc = MetricsCalculator()
        
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc="Testing")
            
            for batch_data in progress_bar:
                try:
                    loss_dict, pred_dict = self._process_batch(batch_data, is_training=False)
                    
                    metrics_calc.update(
                        predictions=pred_dict['predictions'],
                        labels=pred_dict['labels'],
                        probabilities=pred_dict['probabilities'],
                        total_loss=loss_dict['total_loss'],
                        bce_loss=loss_dict['bce_loss'],
                        recon_loss=loss_dict['recon_loss']
                    )
                    
                except Exception as e:
                    print(f"Error in test batch: {e}")
                    continue
        
        return metrics_calc.compute_metrics()
    
    def _process_batch(self, batch_data, is_training: bool = True) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """배치 데이터 처리"""
        # 배치 데이터 언패킹 (datasets.py __getitem__ 순서에 맞춤)
        videos_batch, text_batch, landmarks_batch, labels_batch, filename_batch = batch_data
        
        # 디바이스로 이동 및 형변환
        labels = labels_batch.to(self.device).float().unsqueeze(1)  # (B, 1)
        videos = videos_batch.to(self.device).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        landmarks = landmarks_batch.to(self.device)  # (B, T, landmark_dim)
        
        # 순전파
        if is_training:
            self.optimizer.zero_grad()
        
        logits, recon_loss = self.model(videos, landmarks)
        
        # 손실 계산
        bce_loss = self.criterion(logits, labels)
        total_loss = bce_loss + self.config.recon_loss_weight * recon_loss
        
        # 역전파 (학습 시에만)
        if is_training:
            total_loss.backward()
            self.optimizer.step()
        
        # 예측 결과 계산
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).long()
        
        # 결과 딕셔너리 구성
        loss_dict = {
            'total_loss': total_loss.item(),
            'bce_loss': bce_loss.item(),
            'recon_loss': recon_loss.item()
        }
        
        pred_dict = {
            'predictions': predictions,
            'labels': labels,
            'probabilities': probabilities
        }
        
        return loss_dict, pred_dict
    
    def _save_model(self, val_loss: float, epoch: int):
        """모델 저장"""
        if not self.config.save_best_model:
            return
        
        # 최고 성능 모델 저장
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            
            # 이전 모델 파일 삭제
            if self.best_model_path and os.path.exists(self.best_model_path):
                os.remove(self.best_model_path)
            
            # 새 모델 저장
            self.best_model_path = os.path.join(
                self.config.model_save_dir,
                f"best_model_epoch_{epoch+1}_loss_{val_loss:.4f}.pth"
            )
            torch.save(self.model.state_dict(), self.best_model_path)
            print(f"Best model saved to {self.best_model_path}")
        
        # 정기적 모델 저장 (옵션)
        if (self.config.save_every_n_epochs and 
            (epoch + 1) % self.config.save_every_n_epochs == 0):
            regular_save_path = os.path.join(
                self.config.model_save_dir,
                f"model_epoch_{epoch+1}.pth"
            )
            torch.save(self.model.state_dict(), regular_save_path)
            print(f"Regular checkpoint saved to {regular_save_path}")
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """조기 종료 확인"""
        if not self.config.early_stopping_patience:
            return False
        
        if val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.config.early_stopping_patience:
            print(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
            return True
        
        return False
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None
    ):
        """전체 학습 파이프라인 실행"""
        print(f"Starting training for {self.config.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # 학습
            train_metrics = self.train_epoch(train_loader)
            
            # 검증
            val_metrics = {}
            if val_loader:
                val_metrics = self.validate_epoch(val_loader)
            
            # 결과 출력
            self._print_epoch_results(epoch, train_metrics, val_metrics)
            
            # 모델 저장
            if val_metrics:
                val_loss = val_metrics.get('loss_total', float('inf'))
                self._save_model(val_loss, epoch)
                
                # 조기 종료 확인
                if self._check_early_stopping(val_loss):
                    break
            
        # 테스트 평가
        if test_loader and self.best_model_path:
            print(f"\n--- Final Test Evaluation ---")
            test_metrics = self.test_model(test_loader, self.best_model_path)
            if test_metrics:
                print("Test Results:")
                print(format_metrics(test_metrics, "  "))
            else:
                print("Test evaluation failed or no test data")
        
        print(f"\nTraining completed!")
        if self.best_model_path:
            print(f"Best model saved at: {self.best_model_path}")
    
    def _print_epoch_results(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """에포크 결과 출력"""
        print(f"\nEpoch [{epoch+1}/{self.config.num_epochs}]:")
        
        # 학습 결과
        if train_metrics:
            print(format_metrics(train_metrics, "  Train "))
        
        # 검증 결과  
        if val_metrics:
            print(format_metrics(val_metrics, "  Val   "))
        else:
            print("  Val   Loss: N/A (No validation data)")


class TrainingPipeline:
    """학습 파이프라인 래퍼 클래스"""
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module, 
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        
        # 기존 코드 호환성을 위한 속성들
        self.device = torch.device(config.device)
        
    def train(
        self, 
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None, 
        test_loader: Optional[DataLoader] = None
    ):
        """학습 실행"""
        trainer = ModelTrainer(
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            config=self.config,
            device=self.device
        )
        
        trainer.train(train_loader, val_loader, test_loader)


# 기존 코드와의 호환성을 위한 함수
def run_training_pipeline(
    model, train_loader, val_loader, test_loader,
    criterion_bce, optimizer, config, device
):
    """기존 인터페이스와 호환되는 학습 함수"""
    
    # config 객체가 TrainingConfig가 아닌 경우 변환
    if not isinstance(config, TrainingConfig):
        training_config = TrainingConfig(
            num_epochs=getattr(config, 'num_epochs', 10),
            batch_size=getattr(config, 'batch_size', 16),
            learning_rate=getattr(config, 'learning_rate', 1e-3),
            recon_loss_weight=getattr(config, 'recon_loss_weight', 0.5),
            device=str(device),
            save_best_model=True,
            model_save_dir="saved_models"
        )
    else:
        training_config = config
    
    # 트레이너 생성 및 학습 실행
    trainer = ModelTrainer(
        model=model,
        criterion=criterion_bce,
        optimizer=optimizer,
        config=training_config,
        device=device
    )
    
    trainer.train(train_loader, val_loader, test_loader)


# 사용 예시를 위한 헬퍼 함수들
def create_trainer_from_config(
    model: nn.Module,
    train_config: TrainingConfig,
    device: torch.device
) -> ModelTrainer:
    """설정으로부터 트레이너 생성"""
    
    # 손실 함수와 옵티마이저 생성
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    
    return ModelTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        config=train_config,
        device=device
    )


def quick_train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    device: str = "cuda"
) -> str:
    """빠른 학습을 위한 간편 함수"""
    
    config = TrainingConfig(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        save_best_model=True
    )
    
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device_obj)
    
    trainer = create_trainer_from_config(model, config, device_obj)
    trainer.train(train_loader, val_loader, test_loader)
    
    return trainer.best_model_path or "No model saved"


# 모듈 exports
__all__ = [
    'ModelTrainer',
    'TrainingPipeline', 
    'run_training_pipeline',
    'create_trainer_from_config',
    'quick_train'
]