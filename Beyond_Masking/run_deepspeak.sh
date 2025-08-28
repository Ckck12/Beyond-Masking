#!/bin/bash

# Script to train the model on the FakeAVCeleb dataset
# Usage: bash run_fakeavceleb.sh

echo "🎭 Starting training with the FakeAVCeleb dataset..."

# --- Basic Settings (preserved from original script) ---
DEVICE="cuda:0"
EPOCHS=30
BATCH_SIZE=8        # Adjusted for higher memory usage
LEARNING_RATE=2e-4
SEED=42

# --- New Model & Loss Hyperparameters ---
FEATURE_DIM=512       # Preserved from original (video_feature_dim)
TRANSFORMER_NHEAD=8   # Increased heads for larger feature dim
ALPHA=0.6             # Set based on original recon_loss_weight
BETA=0.5
GAMMA=0.5

# --- Dataset Path ---
FAKEAVCELEB_DIR="/media/NAS/DATASET/FakeAVCeleb_v1.2/"

# --- Model Save Path ---
MODEL_SAVE_DIR="saved_models/fakeavceleb_$(date +%Y%m%d_%H%M%S)"
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
    --transformer_nhead $TRANSFORMER_NHEAD \
    \
    --final_loss_alpha $ALPHA \
    --lambda_beta $BETA \
    --lambda_gamma $GAMMA \
    \
    --fakeavceleb_dir $FAKEAVCELEB_DIR \
    --load_fakeavceleb \
    \
    --preload_data \
    --num_workers 4 \
    --num_workers_preload 8 \
    --early_stopping_patience 10 \
    --verbose

echo "✅ FakeAVCeleb training finished!"
echo "💾 Model saved in: $MODEL_SAVE_DIR"