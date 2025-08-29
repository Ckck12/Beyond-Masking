# FILE: utils/metrics.py

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from typing import Dict

class MetricsCalculator:
    """A helper class to accumulate and calculate metrics over an epoch."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Resets all stored metrics to their initial state."""
        self.predictions = []
        self.labels = []
        self.probabilities = []
        self.losses = {
            'total': [], 'bce': [], 'kl': [], 'contrastive': []
        }
    
    def update(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor, 
        probabilities: torch.Tensor,
        loss_total: float,
        loss_bce: float,
        loss_kl: float,
        loss_contrastive: float
    ):
        """Updates the metrics with the results from a new batch."""
        self.predictions.extend(predictions.detach().cpu().numpy().flatten().tolist())
        self.labels.extend(labels.detach().cpu().numpy().flatten().tolist())
        self.probabilities.extend(probabilities.detach().cpu().numpy().flatten().tolist())
        
        self.losses['total'].append(loss_total)
        self.losses['bce'].append(loss_bce)
        self.losses['kl'].append(loss_kl)
        self.losses['contrastive'].append(loss_contrastive)
    
    def compute_metrics(self) -> Dict[str, float]:
        """Calculates the final metrics from all accumulated batch results."""
        if not self.labels:
            return self._empty_metrics()
        
        avg_losses = {key: np.mean(val) if val else 0.0 for key, val in self.losses.items()}
        
        try:
            accuracy = accuracy_score(self.labels, self.predictions)
            f1 = f1_score(self.labels, self.predictions, average='binary', pos_label=1, zero_division=0)
            auc = roc_auc_score(self.labels, self.probabilities) if len(np.unique(self.labels)) > 1 else float('nan')
        except Exception:
            accuracy, f1, auc = 0.0, 0.0, float('nan')
        
        return {
            'loss_total': avg_losses['total'],
            'loss_bce': avg_losses['bce'],
            'loss_kl': avg_losses['kl'],
            'loss_contrastive': avg_losses['contrastive'],
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc,
            'sample_count': len(self.labels)
        }
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Returns a dictionary of zeroed-out metrics."""
        return {
            'loss_total': 0.0, 'loss_bce': 0.0, 'loss_kl': 0.0, 'loss_contrastive': 0.0,
            'accuracy': 0.0, 'f1_score': 0.0, 'auc': float('nan'), 'sample_count': 0
        }

def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """Formats the metrics dictionary into a human-readable string for logging."""
    if not metrics or metrics.get('sample_count', 0) == 0:
        return f"{prefix:<5} | No data"
    
    auc_str = f"{metrics['auc']:.4f}" if not np.isnan(metrics['auc']) else "N/A"
    
    # Updated to display the new loss components
    loss_str = (
        f"Loss: {metrics['loss_total']:.4f} "
        f"(BCE: {metrics['loss_bce']:.4f}, "
        f"KL: {metrics['loss_kl']:.4f}, "
        f"Contr: {metrics['loss_contrastive']:.4f})"
    )
    
    perf_str = f"Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | AUC: {auc_str}"
    
    return f"{prefix:<5} | {loss_str} | {perf_str}"