import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from typing import List, Dict, Any, Union
import warnings


class MetricsCalculator:
    """학습/검증/테스트 메트릭 계산 클래스"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """메트릭 초기화"""
        self.predictions = []
        self.labels = []
        self.probabilities = []
        self.losses = {
            'total': [],
            'bce': [],
            'reconstruction': []
        }
    
    def update(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor, 
        probabilities: torch.Tensor,
        total_loss: float,
        bce_loss: float,
        recon_loss: float
    ):
        """배치 결과로 메트릭 업데이트"""
        # 텐서를 numpy로 변환하고 차원 처리
        preds_np = predictions.detach().cpu().numpy().squeeze()
        labels_np = labels.detach().cpu().numpy().squeeze()
        probs_np = probabilities.detach().cpu().numpy().squeeze()
        
        # 스칼라 값 처리 (배치 크기가 1인 경우)
        if preds_np.ndim == 0:
            self.predictions.append(preds_np.item())
            self.labels.append(labels_np.item())
            self.probabilities.append(probs_np.item())
        else:
            self.predictions.extend(preds_np.tolist())
            self.labels.extend(labels_np.tolist())
            self.probabilities.extend(probs_np.tolist())
        
        # 손실 값 추가
        self.losses['total'].append(total_loss)
        self.losses['bce'].append(bce_loss)
        self.losses['reconstruction'].append(recon_loss)
    
    def compute_metrics(self) -> Dict[str, float]:
        """현재까지 누적된 데이터로 메트릭 계산"""
        if not self.predictions or not self.labels:
            return self._empty_metrics()
        
        # 평균 손실 계산
        avg_losses = {
            key: np.mean(values) if values else 0.0 
            for key, values in self.losses.items()
        }
        
        # 분류 성능 메트릭 계산
        try:
            accuracy = accuracy_score(self.labels, self.predictions)
        except Exception as e:
            print(f"Accuracy calculation error: {e}")
            accuracy = 0.0
        
        try:
            f1 = f1_score(
                self.labels, self.predictions, 
                average='binary', pos_label=1, zero_division=0
            )
        except Exception as e:
            print(f"F1 score calculation error: {e}")
            f1 = 0.0
        
        try:
            # AUC는 양/음성 클래스가 모두 있을 때만 계산 가능
            unique_labels = np.unique(self.labels)
            if len(unique_labels) > 1 and len(self.probabilities) == len(self.labels):
                auc = roc_auc_score(self.labels, self.probabilities)
            else:
                auc = float('nan')
        except Exception as e:
            print(f"AUC calculation error: {e}")
            auc = float('nan')
        
        return {
            'loss_total': avg_losses['total'],
            'loss_bce': avg_losses['bce'],
            'loss_reconstruction': avg_losses['reconstruction'],
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc,
            'sample_count': len(self.predictions)
        }
    
    def _empty_metrics(self) -> Dict[str, float]:
        """빈 메트릭 반환"""
        return {
            'loss_total': 0.0,
            'loss_bce': 0.0,
            'loss_reconstruction': 0.0,
            'accuracy': 0.0,
            'f1_score': 0.0,
            'auc': float('nan'),
            'sample_count': 0
        }
    
    def get_label_distribution(self) -> Dict[str, int]:
        """라벨 분포 반환"""
        if not self.labels:
            return {}
        
        unique, counts = np.unique(self.labels, return_counts=True)
        return {f'class_{int(label)}': int(count) for label, count in zip(unique, counts)}


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """메트릭을 보기 좋은 문자열로 포맷팅"""
    if not metrics or metrics['sample_count'] == 0:
        return f"{prefix}No data"
    
    auc_str = f"{metrics['auc']:.4f}" if not np.isnan(metrics['auc']) else "N/A"
    
    return (
        f"{prefix}Loss: {metrics['loss_total']:.4f} "
        f"(BCE: {metrics['loss_bce']:.4f}, Recon: {metrics['loss_reconstruction']:.4f}) | "
        f"Acc: {metrics['accuracy']:.4f} | "
        f"F1: {metrics['f1_score']:.4f} | "
        f"AUC: {auc_str}"
    )


def print_label_distribution(calculator: MetricsCalculator, phase: str):
    """라벨 분포 출력"""
    distribution = calculator.get_label_distribution()
    if distribution:
        dist_str = ", ".join([f"{k}: {v}" for k, v in distribution.items()])
        print(f"{phase} label distribution - {dist_str}")


# 단순 메트릭 계산 함수들 (기존 코드 호환성)
def calculate_simple_metrics(predictions: List, labels: List, probabilities: List) -> Dict[str, float]:
    """간단한 메트릭 계산 (기존 코드 호환성)"""
    calculator = MetricsCalculator()
    
    # 리스트를 텐서로 변환하여 업데이트
    if predictions and labels and probabilities:
        preds_tensor = torch.tensor(predictions)
        labels_tensor = torch.tensor(labels) 
        probs_tensor = torch.tensor(probabilities)
        
        calculator.update(
            preds_tensor, labels_tensor, probs_tensor,
            0.0, 0.0, 0.0  # 손실 값은 0으로 설정
        )
    
    metrics = calculator.compute_metrics()
    # 손실 관련 메트릭 제거
    return {k: v for k, v in metrics.items() if not k.startswith('loss_')}