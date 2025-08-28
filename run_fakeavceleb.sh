#!/bin/bash

# FakeAVCeleb 데이터셋 학습 스크립트
# 사용법: bash scripts/run_fakeavceleb.sh

echo "🎭 FakeAVCeleb 데이터셋으로 학습 시작..."

# 기본 설정
DEVICE="cuda:0"
EPOCHS=30
BATCH_SIZE=8  # 메모리 사용량 고려
LEARNING_RATE=2e-4
SEED=42

# 데이터셋 경로
FAKEAVCELEB_DIR="/media/NAS/DATASET/FakeAVCeleb_v1.2/"

# 모델 저장 경로
MODEL_SAVE_DIR="saved_models/fakeavceleb_$(date +%Y%m%d_%H%M%S)"

# Python 스크립트 실행
python main.py \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --device $DEVICE \
    --seed $SEED \
    --model_save_dir $MODEL_SAVE_DIR \
    --fakeavceleb_dir $FAKEAVCELEB_DIR \
    --load_fakeavceleb \
    --preload_data \
    --num_workers 4 \
    --num_workers_preload 8 \
    --early_stopping_patience 10 \
    --recon_loss_weight 0.6 \
    --video_feature_dim 512 \
    --verbose

echo "✅ FakeAVCeleb 학습 완료!"
echo "💾 모델 저장 경로: $MODEL_SAVE_DIR"