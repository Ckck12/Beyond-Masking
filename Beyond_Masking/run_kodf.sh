#!/bin/bash

# KoDF 데이터셋 학습 스크립트
# 사용법: bash scripts/run_kodf.sh

echo "🇰🇷 KoDF 데이터셋으로 학습 시작..."

# 기본 설정
DEVICE="cuda:0"
EPOCHS=25
BATCH_SIZE=12  # KoDF는 더 큰 데이터이므로 배치 크기 조정
LEARNING_RATE=5e-4
SEED=42

# 데이터셋 경로
KODF_DIR="/media/NAS/DATASET/KoDF/features_mtcnn/"

# 모델 저장 경로
MODEL_SAVE_DIR="saved_models/kodf_$(date +%Y%m%d_%H%M%S)"

# Python 스크립트 실행
python main.py \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --device $DEVICE \
    --seed $SEED \
    --model_save_dir $MODEL_SAVE_DIR \
    --kodf_data_dir $KODF_DIR \
    --load_kodf_real \
    --load_kodf_fake \
    --preload_data \
    --num_workers 6 \
    --num_workers_preload 12 \
    --early_stopping_patience 7 \
    --recon_loss_weight 0.3 \
    --dropout_rate 0.4 \
    --verbose

echo "✅ KoDF 학습 완료!"
echo "💾 모델 저장 경로: $MODEL_SAVE_DIR"