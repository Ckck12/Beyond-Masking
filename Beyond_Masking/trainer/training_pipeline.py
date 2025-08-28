"""
Defines the core training pipeline, including the ModelTrainer class.

The ModelTrainer handles the main training, validation, and testing loops,
including the complex loss calculation for the multimodal framework.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Tuple

from config.training_config import TrainingConfig
from utils.metrics import MetricsCalculator, format_metrics

class ModelTrainer:
    """Manages the training and evaluation of the deepfake detection model."""
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        device: torch.device
    ):
        self.model = model
        self.criterion = criterion  # BCEWithLogitsLoss for the main task
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
        # Create model save directory if it doesn't exist
        if config.save_best_model:
            Path(config.model_save_dir).mkdir(parents=True, exist_ok=True)
            
        # State tracking
        self.best_val_loss = float('inf')
        self.best_model_path = None
        self.current_epoch = 0
        self.patience_counter = 0

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Runs a single training epoch."""
        self.model.train()
        metrics_calc = MetricsCalculator()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}/{self.config.num_epochs} Training", leave=False)
        
        for batch_data in progress_bar:
            try:
                loss_dict, pred_dict = self._process_batch(batch_data, is_training=True)
                
                # Update metrics with all loss components
                metrics_calc.update(
                    predictions=pred_dict['predictions'], labels=pred_dict['labels'], probabilities=pred_dict['probabilities'],
                    loss_total=loss_dict['total'], loss_bce=loss_dict['bce'],
                    loss_kl=loss_dict['kl'], loss_contrastive=loss_dict['contrastive']
                )

                if progress_bar.n % self.config.log_every_n_batches == 0:
                    current_metrics = metrics_calc.compute_metrics()
                    progress_bar.set_postfix({
                        'Loss': f"{current_metrics['loss_total']:.4f}",
                        'Acc': f"{current_metrics['accuracy']:.3f}"
                    })
            except Exception as e:
                print(f"\nError during training batch: {e}")
                continue
            
        return metrics_calc.compute_metrics()

    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Runs a single validation epoch."""
        if not val_loader: return {}
        
        self.model.eval()
        metrics_calc = MetricsCalculator()
        
        progress_bar = tqdm(val_loader, desc=f"Epoch {self.current_epoch+1}/{self.config.num_epochs} Validation", leave=False)
        
        with torch.no_grad():
            for batch_data in progress_bar:
                try:
                    loss_dict, pred_dict = self._process_batch(batch_data, is_training=False)
                    metrics_calc.update(
                        predictions=pred_dict['predictions'], labels=pred_dict['labels'], probabilities=pred_dict['probabilities'],
                        loss_total=loss_dict['total'], loss_bce=loss_dict['bce'],
                        loss_kl=loss_dict['kl'], loss_contrastive=loss_dict['contrastive']
                    )
                except Exception as e:
                    print(f"\nError during validation batch: {e}")
                    continue

        return metrics_calc.compute_metrics()

    def _process_batch(self, batch_data: tuple, is_training: bool) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Processes a single batch of data for either training or evaluation."""
        # Unpack data from our multimodal dataset
        videos, audios, landmarks, labels, _, _ = batch_data
        
        # Move data to the target device and adjust shapes
        labels = labels.to(self.device).float().unsqueeze(1)    # Shape: (B, 1)
        videos = videos.to(self.device).permute(0, 2, 1, 3, 4)  # Shape: (B, C, T, H, W)
        audios = audios.to(self.device)                         # Shape: (B, n_mfcc, T_audio)
        landmarks = landmarks.to(self.device)                   # Shape: (B, T, landmark_dim)
        
        if is_training:
            self.optimizer.zero_grad()
        
        # Forward pass: model returns logits and a dictionary of auxiliary losses
        logits, aux_losses = self.model(videos, audios, landmarks)
        
        # --- Loss Calculation (as per paper's final equation) ---
        bce_loss = self.criterion(logits, labels)
        kl_loss = aux_losses['kl']
        contrastive_loss = aux_losses['contrastive']
        
        alpha = self.config.final_loss_alpha
        gamma = self.config.lambda_gamma

        # Combine losses
        aux_loss_combined = kl_loss + gamma * contrastive_loss
        total_loss = (1 - alpha) * bce_loss + alpha * aux_loss_combined
        
        if is_training:
            total_loss.backward()
            self.optimizer.step()
        
        # Prepare outputs for metric calculation
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).long()
        
        loss_dict = {
            'total': total_loss.item(), 'bce': bce_loss.item(),
            'kl': kl_loss.item(), 'contrastive': contrastive_loss.item()
        }
        pred_dict = {'predictions': predictions, 'labels': labels, 'probabilities': probabilities}
        
        return loss_dict, pred_dict

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, test_loader: Optional[DataLoader] = None):
        """Executes the full training and evaluation pipeline."""
        print(f"🚀 Starting training for {self.config.num_epochs} epochs on device '{self.device}'...")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)
            
            self._print_epoch_results(epoch, train_metrics, val_metrics)
            
            if val_metrics:
                val_loss = val_metrics.get('loss_total', float('inf'))
                self._save_model_checkpoint(val_loss, epoch)
                if self._check_early_stopping(val_loss):
                    break
        
        print("\n🎉 Training finished!")
        if self.best_model_path:
            print(f"🏆 Best model saved at: {self.best_model_path}")
            if test_loader:
                self.test(test_loader)

    def test(self, test_loader: DataLoader):
        """Runs final evaluation on the test set with the best model."""
        if not self.best_model_path:
            print("No best model was saved. Skipping test phase.")
            return

        print(f"\n🧪 Running final evaluation on the test set with model: {os.path.basename(self.best_model_path)}")
        self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
        
        self.model.eval()
        metrics_calc = MetricsCalculator()
        
        progress_bar = tqdm(test_loader, desc="Testing", leave=False)
        with torch.no_grad():
            for batch_data in progress_bar:
                loss_dict, pred_dict = self._process_batch(batch_data, is_training=False)
                metrics_calc.update(
                    predictions=pred_dict['predictions'], labels=pred_dict['labels'], probabilities=pred_dict['probabilities'],
                    loss_total=loss_dict['total'], loss_bce=loss_dict['bce'],
                    loss_kl=loss_dict['kl'], loss_contrastive=loss_dict['contrastive']
                )
        
        test_metrics = metrics_calc.compute_metrics()
        print("\n--- Test Results ---")
        print(format_metrics(test_metrics, "  "))
        print("--------------------")

    def _save_model_checkpoint(self, val_loss: float, epoch: int):
        if not self.config.save_best_model: return

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            
            if self.best_model_path and os.path.exists(self.best_model_path):
                try:
                    os.remove(self.best_model_path) # Remove old best model
                except OSError as e:
                    print(f"Error removing old model file: {e}")
            
            self.best_model_path = os.path.join(
                self.config.model_save_dir,
                f"best_model_epoch_{epoch+1}_loss_{val_loss:.4f}.pth"
            )
            torch.save(self.model.state_dict(), self.best_model_path)
            print(f"   => ✨ New best model saved: {os.path.basename(self.best_model_path)}")

    def _check_early_stopping(self, val_loss: float) -> bool:
        if not self.config.early_stopping_patience: return False
        
        if val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            print(f"   => ⏳ Early stopping counter: {self.patience_counter}/{self.config.early_stopping_patience}")
        
        if self.patience_counter >= self.config.early_stopping_patience:
            print("   => 🛑 Early stopping triggered.")
            return True
        return False
        
    def _print_epoch_results(self, epoch, train_metrics, val_metrics):
        print(f"\n--- Epoch {epoch+1}/{self.config.num_epochs} Summary ---")
        if train_metrics:
            print(format_metrics(train_metrics, "Train"))
        if val_metrics:
            print(format_metrics(val_metrics, "Val  "))
        print("--------------------------")