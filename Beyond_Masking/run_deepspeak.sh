#!/bin/bash

# DeepSpeak 데이터셋 학습 스크립트
# 사용법: bash scripts/run_deepspeak.sh

echo "🎬 DeepSpeak 데이터셋으로 학습 시작..."

# 기본 설정
DEVICE="cuda:0"
EPOCHS=20
BATCH_SIZE=16
LEARNING_RATE=1e-3
SEED=42

# 데이터셋 경로 (필요에 따라 수정)
DEEPSPEAK_DIR="/media/NAS/DATASET/deepspeak/"

# 모델 저장 경로
MODEL_SAVE_DIR="saved_models/deepspeak_$(date +%Y%m%d_%H%M%S)"

# Python 스크립트 실행
python main.py \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --device $DEVICE \
    --seed $SEED \
    --model_save_dir $MODEL_SAVE_DIR \
    --deepspeak_data_dir $DEEPSPEAK_DIR \
    --load_deepspeak_real \
    --load_deepspeak_fake \
    --preload_data \
    --num_workers 8 \
    --num_workers_preload 16 \
    --early_stopping_patience 5 \
    --verbose

echo "✅ DeepSpeak 학습 완료!"
echo "💾 모델 저장 경로: $MODEL_SAVE_DIR"