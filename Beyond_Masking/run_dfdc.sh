#!/bin/bash

# Script to train the model on the DFDC dataset
# Usage: bash run_dfdc.sh

echo "🏆 Starting training with the DFDC dataset..."

# --- Basic Settings (preserved from original script) ---
DEVICE="cuda:0"
EPOCHS=35
BATCH_SIZE=10
LEARNING_RATE=1e-4
SEED=42

# --- New Model & Loss Hyperparameters ---
INPUT_FRAMES=90       # Preserved from original
FEATURE_DIM=384       # Preserved from original (video_feature_dim)
TRANSFORMER_NHEAD=4
ALPHA=0.6             # Set based on original recon_loss_weight (1 - 0.4)
BETA=0.5
GAMMA=0.5

# --- Dataset Path ---
DFDC_DIR="/media/NAS/DATASET/DFDC-Official/"

# --- Model Save Path ---
MODEL_SAVE_DIR="saved_models/dfdc_$(date +%Y%m%d_%H%M%S)"
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
    --input_frames $INPUT_FRAMES \
    --feature_dim $FEATURE_DIM \
    --transformer_nhead $TRANSFORMER_NHEAD \
    \
    --final_loss_alpha $ALPHA \
    --lambda_beta $BETA \
    --lambda_gamma $GAMMA \
    \
    --dfdc_dir $DFDC_DIR \
    --load_dfdc_real \
    --load_dfdc_fake \
    \
    --preload_data \
    --num_workers 6 \
    --num_workers_preload 12 \
    --early_stopping_patience 8 \
    --verbose

echo "✅ DFDC training finished!"
echo "💾 Model saved in: $MODEL_SAVE_DIR"