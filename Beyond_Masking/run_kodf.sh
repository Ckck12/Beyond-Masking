#!/bin/bash

# Script to train the model on the KoDF dataset
# Usage: bash run_kodf.sh

echo "🇰🇷 Starting training with the KoDF dataset..."

# --- Basic Settings (preserved from original script) ---
DEVICE="cuda:0"
EPOCHS=25
BATCH_SIZE=12
LEARNING_RATE=5e-4
SEED=42

# --- New Model & Loss Hyperparameters ---
FEATURE_DIM=256       # Using default feature dimension
DROPOUT_RATE=0.4      # Preserved from original
TRANSFORMER_NHEAD=4
ALPHA=0.3             # Set based on original recon_loss_weight
BETA=0.5
GAMMA=0.5

# --- Dataset Path ---
KODF_DIR="/media/NAS/DATASET/KoDF/features_mtcnn/"

# --- Model Save Path ---
MODEL_SAVE_DIR="saved_models/kodf_$(date +%Y%m%d_%H%M%S)"
mkdir -p $MODEL_SAVE_DIR

# --- Execute the Python Training Script ---
python main.py \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --device $DEVICE \
    --seed $SEED \
    --model_save_dir $MODEL_SAVE_DIR \
    \
    --feature_dim $FEATURE_DIM \
    --dropout_rate $DROPOUT_RATE \
    --transformer_nhead $TRANSFORMER_NHEAD \
    \
    --final_loss_alpha $ALPHA \
    --lambda_beta $BETA \
    --lambda_gamma $GAMMA \
    \
    --kodf_data_dir $KODF_DIR \
    --load_kodf_real \
    --load_kodf_fake \
    \
    --preload_data \
    --num_workers 6 \
    --num_workers_preload 12 \
    --early_stopping_patience 7 \
    --verbose

echo "✅ KoDF training finished!"
echo "💾 Model saved in: $MODEL_SAVE_DIR"