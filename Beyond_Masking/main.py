#!/usr/bin/env python3


import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import refactored modules from their new locations
from config.base_configs import (
    DatasetConfig, ProcessingConfig, ModelConfig, TrainingConfig, get_device
)
from dataset.factory import DatasetFactory
from dataset.dataset import seed_worker
from models.landmark_predictor import create_model
from models.utils import get_model_info # Assuming model utils are in models/utils.py
from training.trainer import ModelTrainer
from utils.common import set_all_seeds, print_system_info

def create_argument_parser():
    """Creates and configures the argument parser."""
    parser = argparse.ArgumentParser(
        description="Train a Landmark-Guided Multimodal Deepfake Detection model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Training Settings ---
    g_train = parser.add_argument_group('🏃 Training Settings')
    g_train.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs.')
    g_train.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    g_train.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate.')
    g_train.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping. None to disable.')

    # --- Loss Function Weights ---
    g_loss = parser.add_argument_group('💸 Loss Function Weights')
    g_loss.add_argument('--final_loss_alpha', type=float, default=0.8, help='Alpha: Weight between auxiliary and BCE losses.')
    g_loss.add_argument('--lambda_beta', type=float, default=0.5, help='Beta: Weight for the audio part in KL loss.')
    g_loss.add_argument('--lambda_gamma', type=float, default=0.5, help='Gamma: Weight for the contrastive loss.')

    # --- Model Architecture ---
    g_model = parser.add_argument_group('🧠 Model Architecture')
    g_model.add_argument('--input_frames', type=int, default=60, help='Number of input video frames.')
    g_model.add_argument('--feature_dim', type=int, default=256, help='Common feature dimension for video, audio, and landmarks.')
    g_model.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate in MLP layers.')
    g_model.add_argument('--transformer_nhead', type=int, default=4, help='Number of heads in Transformer layers.')

    # --- System & I/O ---
    g_sys = parser.add_argument_group('💻 System and I/O')
    g_sys.add_argument('--device', type=str, default='auto', help='Device to use (e.g., "cpu", "cuda", "cuda:0").')
    g_sys.add_argument('--num_workers', type=int, default=8, help='Number of workers for DataLoader.')
    g_sys.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    g_sys.add_argument('--model_save_dir', type=str, default="saved_models", help='Directory to save model checkpoints.')
    g_sys.add_argument('--verbose', action='store_true', help='Enable verbose output.')

    # --- Data Settings ---
    g_data = parser.add_argument_group('📂 Data Settings')
    g_data.add_argument('--preload_data', action='store_true', help='Enable preloading data into memory for faster training.')
    g_data.add_argument('--num_workers_preload', type=int, default=16, help='Number of workers for data preloading.')

    # --- Dataset Paths ---
    g_path = parser.add_argument_group('📁 Dataset Paths')
    g_path.add_argument('--deepspeak_data_dir', type=str, default=None, help='Path to DeepSpeak dataset.')
    g_path.add_argument('--kodf_data_dir', type=str, default=None, help='Path to KoDF dataset.')
    g_path.add_argument('--fakeavceleb_dir', type=str, default=None, help='Path to FakeAVCeleb dataset.')
    g_path.add_argument('--dfdc_dir', type=str, default=None, help='Path to DFDC dataset.')
    g_path.add_argument('--deepfaketimit_dir', type=str, default=None, help='Path to DeepfakeTIMIT dataset.')
    g_path.add_argument('--faceforensics_dir', type=str, default=None, help='Path to FaceForensics++ dataset.')

    # --- Dataset Loading Flags ---
    g_load = parser.add_argument_group('📋 Dataset Loading Flags')
    g_load.add_argument('--load_deepspeak_real', action='store_true')
    g_load.add_argument('--load_deepspeak_fake', action='store_true')
    g_load.add_argument('--load_kodf_real', action='store_true')
    g_load.add_argument('--load_kodf_fake', action='store_true')
    g_load.add_argument('--load_fakeavceleb', action='store_true')
    g_load.add_argument('--load_dfdc_real', action='store_true')
    g_load.add_argument('--load_dfdc_fake', action='store_true')
    g_load.add_argument('--load_deepfaketimit_real', action='store_true')
    g_load.add_argument('--load_deepfaketimit_fake', action='store_true')
    g_load.add_argument('--load_faceforensics_real', action='store_true')
    g_load.add_argument('--load_faceforensics_fake', action='store_true')

    return parser

def main():
    """Main function to run the training pipeline."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # --- 1. Setup Environment ---
    set_all_seeds(args.seed)
    device = get_device(args.device)
    if args.verbose:
        print_system_info()

    # --- 2. Create Configurations from Args ---
    dataset_cfg = DatasetConfig(
        deepspeak_base_dir=args.deepspeak_data_dir, kodf_features_root_dir=args.kodf_data_dir,
        fakeavceleb_dir=args.fakeavceleb_dir, dfdc_dir=args.dfdc_dir,
        deepfaketimit_dir=args.deepfaketimit_dir, faceforensics_dir=args.faceforensics_dir,
        load_deepspeak_real=args.load_deepspeak_real, load_deepspeak_fake=args.load_deepspeak_fake,
        load_kodf_real=args.load_kodf_real, load_kodf_fake=args.load_kodf_fake,
        load_fakeavceleb=args.load_fakeavceleb, load_dfdc_real=args.load_dfdc_real,
        load_dfdc_fake=args.load_dfdc_fake, load_deepfaketimit_real=args.load_deepfaketimit_real,
        load_deepfaketimit_fake=args.load_deepfaketimit_fake,
        load_faceforensics_real=args.load_faceforensics_real, load_faceforensics_fake=args.load_faceforensics_fake,
    )
    processing_cfg = ProcessingConfig(desired_num_frames=args.input_frames)
    model_cfg = ModelConfig(
        input_frames=args.input_frames, feature_dim=args.feature_dim,
        dropout_rate=args.dropout_rate, transformer_nhead=args.transformer_nhead
    )
    training_cfg = TrainingConfig(
        num_epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        num_workers=args.num_workers, device=str(device), seed=args.seed,
        model_save_dir=args.model_save_dir, early_stopping_patience=args.early_stopping_patience,
        final_loss_alpha=args.final_loss_alpha, lambda_beta=args.lambda_beta, lambda_gamma=args.lambda_gamma
    )
    
    # --- 3. Prepare Data Pipeline ---
    print("\n[Phase 1/4] Preparing Data Pipeline...")
    dataloaders = {}
    for partition in ['train', 'val', 'test']:
        dataset = DatasetFactory.create_dataset(
            partition=partition, dataset_config=dataset_cfg,
            processing_config=processing_cfg, preload_workers=args.num_workers_preload
        )
        if not dataset.file_list:
            print(f"  - ⚠️ {partition.capitalize()} dataset is empty. Skipping.")
            dataloaders[partition] = None
            continue
            
        if args.preload_data:
            dataset.preload_data()

        dataloaders[partition] = DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=(partition == 'train'), num_workers=args.num_workers,
            pin_memory=True, worker_init_fn=seed_worker,
            drop_last=(partition == 'train')
        )
    
    if not dataloaders['train']:
        print("\n❌ Error: Training dataloader is empty. Please check dataset paths and flags.")
        sys.exit(1)

    # --- 4. Prepare Model and Trainer ---
    print("\n[Phase 2/4] Preparing Model...")
    model = create_model(model_cfg).to(device)
    print(get_model_info(model))
    
    print("\n[Phase 3/4] Preparing Trainer...")
    optimizer = optim.Adam(model.parameters(), lr=training_cfg.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    trainer = ModelTrainer(
        model=model, criterion=criterion, optimizer=optimizer,
        config=training_cfg, device=device
    )
    
    # --- 5. Run Training ---
    print("\n[Phase 4/4] Starting Training...")
    try:
        trainer.train(
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            test_loader=dataloaders['test']
        )
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()