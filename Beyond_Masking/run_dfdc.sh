#!/bin/bash

# DFDC 데이터셋 학습 스크립트
# 사용법: bash scripts/run_dfdc.sh

echo "🏆 DFDC 데이터셋으로 학습 시작..."

# 기본 설정
DEVICE="cuda:0"
EPOCHS=35
BATCH_SIZE=10
LEARNING_RATE=1e-4
SEED=42

# 데이터셋 경로
DFDC_DIR="/media/NAS/DATASET/DFDC-Official/"

# 모델 저장 경로
MODEL_SAVE_DIR="saved_models/dfdc_$(date +%Y%m%d_%H%M%S)"

# Python 스크립트 실행
python main.py \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --device $DEVICE \
    --seed $SEED \
    --model_save_dir $MODEL_SAVE_DIR \
    --dfdc_dir $DFDC_DIR \
    --load_dfdc_real \
    --load_dfdc_fake \
    --preload_data \
    --num_workers 6 \
    --num_workers_preload 12 \
    --early_stopping_patience 8 \
    --recon_loss_weight 0.4 \
    --input_frames 90 \
    --video_feature_dim 384 \
    --verbose

echo "✅ DFDC 학습 완료!"
echo "💾 모델 저장 경로: $MODEL_SAVE_DIR"