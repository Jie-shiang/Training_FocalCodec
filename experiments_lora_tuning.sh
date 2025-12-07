#!/bin/bash
################################################################################
# FocalCodec LoRA Fine-tuning 實驗腳本
#
# 重要說明:
# 1. 完全凍結: 所有組件都用 --freeze_* (baseline, 預期 dWER 0.03~0.04, dCER 0.30~0.35)
# 2. LoRA tune: 使用 --train_* + --lora_* (不要用 --freeze_*)
#
# 可調整的主要參數:
# - learning_rate: 學習率 (1e-3 ~ 1e-6)
# - lora_rank: LoRA rank (4, 8, 16, 32, 64)
# - lora_alpha: LoRA alpha (通常是 rank 的 2 倍)
# - batch_size: batch size (8, 16, 32, 64)
# - num_epochs: 訓練輪數
# - weight_feature: feature loss 權重 (0.5 ~ 2.0)
# - weight_mel: mel loss 權重 (0.1 ~ 0.5)
# - patience: early stopping patience
# - overlap: 訓練資料 chunk overlap (0.0 ~ 0.5)
# - gradient_clip: gradient clipping (0.5 ~ 5.0)
# - weight_decay: weight decay (1e-6 ~ 1e-4)
################################################################################

# 公用參數
GPU_ID=3
TRAIN_ENV="focalcodec"
EVAL_ENV="codec_eval"
CODEC_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison"
NUM_EPOCHS=300
BATCH_SIZE=32
OVERLAP=0.5
PATIENCE=20

# 基礎參數 (baseline)
BASE_LR=1e-4
BASE_LORA_RANK=16
BASE_LORA_ALPHA=32
BASE_WEIGHT_FEATURE=1.0
BASE_WEIGHT_MEL=0.15
BASE_GRADIENT_CLIP=1.0
BASE_WEIGHT_DECAY=1e-5

################################################################################
# 實驗組 1: 調整 Learning Rate (LoRA tune encoder)
################################################################################

# 實驗 1a: lr = 1e-3 (較大)
echo "=========================================="
echo "實驗 1a: LoRA Encoder - LR 1e-3"
echo "=========================================="

python run_full_pipeline.py \
    --freeze_encoder \
    --freeze_compressor \
    --freeze_decompressor \
    --train_decoder \
    --use_lora \
    --lora_decoder \
    --lora_rank ${BASE_LORA_RANK} \
    --lora_alpha ${BASE_LORA_ALPHA} \
    --use_feature_loss \
    --use_mel_loss \
    --weight_feature ${BASE_WEIGHT_FEATURE} \
    --weight_mel ${BASE_WEIGHT_MEL} \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate 1e-3 \
    --gradient_clip ${BASE_GRADIENT_CLIP} \
    --weight_decay ${BASE_WEIGHT_DECAY} \
    --early_stopping \
    --patience ${PATIENCE} \
    --overlap ${OVERLAP} \
    --gpu_id ${GPU_ID} \
    --train_env ${TRAIN_ENV} \
    --eval_env ${EVAL_ENV} \
    --codec_comparison_dir ${CODEC_DIR} \
    --cleanup_inference \
    --cleanup_codec_comparison \
    --cleanup_best_model

sleep 5

# 實驗 1b: lr = 1e-4
echo "=========================================="
echo "實驗 1b: LoRA Encoder - LR 1e-4"
echo "=========================================="

python run_full_pipeline.py \
    --freeze_encoder \
    --freeze_compressor \
    --freeze_decompressor \
    --train_decoder \
    --use_lora \
    --lora_decoder \
    --lora_rank ${BASE_LORA_RANK} \
    --lora_alpha ${BASE_LORA_ALPHA} \
    --use_feature_loss \
    --use_mel_loss \
    --weight_feature ${BASE_WEIGHT_FEATURE} \
    --weight_mel ${BASE_WEIGHT_MEL} \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate 1e-4 \
    --gradient_clip ${BASE_GRADIENT_CLIP} \
    --weight_decay ${BASE_WEIGHT_DECAY} \
    --early_stopping \
    --patience ${PATIENCE} \
    --overlap ${OVERLAP} \
    --gpu_id ${GPU_ID} \
    --train_env ${TRAIN_ENV} \
    --eval_env ${EVAL_ENV} \
    --codec_comparison_dir ${CODEC_DIR} \
    --cleanup_inference \
    --cleanup_codec_comparison \
    --cleanup_best_model

sleep 5

# 實驗 1c: lr = 1e-5 (baseline)
echo "=========================================="
echo "實驗 1c: LoRA Encoder - LR 1e-5 (baseline)"
echo "=========================================="

python run_full_pipeline.py \
    --freeze_encoder \
    --freeze_compressor \
    --freeze_decompressor \
    --train_decoder \
    --use_lora \
    --lora_decoder \
    --lora_rank ${BASE_LORA_RANK} \
    --lora_alpha ${BASE_LORA_ALPHA} \
    --use_feature_loss \
    --use_mel_loss \
    --weight_feature ${BASE_WEIGHT_FEATURE} \
    --weight_mel ${BASE_WEIGHT_MEL} \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${BASE_LR} \
    --gradient_clip ${BASE_GRADIENT_CLIP} \
    --weight_decay ${BASE_WEIGHT_DECAY} \
    --early_stopping \
    --patience ${PATIENCE} \
    --overlap ${OVERLAP} \
    --gpu_id ${GPU_ID} \
    --train_env ${TRAIN_ENV} \
    --eval_env ${EVAL_ENV} \
    --codec_comparison_dir ${CODEC_DIR} \
    --cleanup_inference \
    --cleanup_codec_comparison \
    --cleanup_best_model

sleep 5

# 實驗 1d: lr = 1e-6
echo "=========================================="
echo "實驗 1d: LoRA Encoder - LR 1e-6"
echo "=========================================="

python run_full_pipeline.py \
    --freeze_encoder \
    --freeze_compressor \
    --freeze_decompressor \
    --train_decoder \
    --use_lora \
    --lora_decoder \
    --lora_rank ${BASE_LORA_RANK} \
    --lora_alpha ${BASE_LORA_ALPHA} \
    --use_feature_loss \
    --use_mel_loss \
    --weight_feature ${BASE_WEIGHT_FEATURE} \
    --weight_mel ${BASE_WEIGHT_MEL} \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate 1e-6 \
    --gradient_clip ${BASE_GRADIENT_CLIP} \
    --weight_decay ${BASE_WEIGHT_DECAY} \
    --early_stopping \
    --patience ${PATIENCE} \
    --overlap ${OVERLAP} \
    --gpu_id ${GPU_ID} \
    --train_env ${TRAIN_ENV} \
    --eval_env ${EVAL_ENV} \
    --codec_comparison_dir ${CODEC_DIR} \
    --cleanup_inference \
    --cleanup_codec_comparison \
    --cleanup_best_model

sleep 5

# 實驗 1e: lr = 1e-7
echo "=========================================="
echo "實驗 1e: LoRA Encoder - LR 1e-7"
echo "=========================================="

python run_full_pipeline.py \
    --freeze_encoder \
    --freeze_compressor \
    --freeze_decompressor \
    --train_decoder \
    --use_lora \
    --lora_decoder \
    --lora_rank ${BASE_LORA_RANK} \
    --lora_alpha ${BASE_LORA_ALPHA} \
    --use_feature_loss \
    --use_mel_loss \
    --weight_feature ${BASE_WEIGHT_FEATURE} \
    --weight_mel ${BASE_WEIGHT_MEL} \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate 1e-7 \
    --gradient_clip ${BASE_GRADIENT_CLIP} \
    --weight_decay ${BASE_WEIGHT_DECAY} \
    --early_stopping \
    --patience ${PATIENCE} \
    --overlap ${OVERLAP} \
    --gpu_id ${GPU_ID} \
    --train_env ${TRAIN_ENV} \
    --eval_env ${EVAL_ENV} \
    --codec_comparison_dir ${CODEC_DIR} \
    --cleanup_inference \
    --cleanup_codec_comparison \
    --cleanup_best_model

sleep 5

# 實驗 1f: lr = 1e-8 (較小)
echo "=========================================="
echo "實驗 1f: LoRA Encoder - LR 1e-8"
echo "=========================================="

python run_full_pipeline.py \
    --freeze_encoder \
    --freeze_compressor \
    --freeze_decompressor \
    --train_decoder \
    --use_lora \
    --lora_decoder \
    --lora_rank ${BASE_LORA_RANK} \
    --lora_alpha ${BASE_LORA_ALPHA} \
    --use_feature_loss \
    --use_mel_loss \
    --weight_feature ${BASE_WEIGHT_FEATURE} \
    --weight_mel ${BASE_WEIGHT_MEL} \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate 1e-8 \
    --gradient_clip ${BASE_GRADIENT_CLIP} \
    --weight_decay ${BASE_WEIGHT_DECAY} \
    --early_stopping \
    --patience ${PATIENCE} \
    --overlap ${OVERLAP} \
    --gpu_id ${GPU_ID} \
    --train_env ${TRAIN_ENV} \
    --eval_env ${EVAL_ENV} \
    --codec_comparison_dir ${CODEC_DIR} \
    --cleanup_inference \
    --cleanup_codec_comparison \
    --cleanup_best_model

sleep 5

# 實驗 1g: lr = 1e-9 (較小)
echo "=========================================="
echo "實驗 1f: LoRA Encoder - LR 1e-8"
echo "=========================================="

python run_full_pipeline.py \
    --freeze_encoder \
    --freeze_compressor \
    --freeze_decompressor \
    --train_decoder \
    --use_lora \
    --lora_decoder \
    --lora_rank ${BASE_LORA_RANK} \
    --lora_alpha ${BASE_LORA_ALPHA} \
    --use_feature_loss \
    --use_mel_loss \
    --weight_feature ${BASE_WEIGHT_FEATURE} \
    --weight_mel ${BASE_WEIGHT_MEL} \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate 1e-8 \
    --gradient_clip ${BASE_GRADIENT_CLIP} \
    --weight_decay ${BASE_WEIGHT_DECAY} \
    --early_stopping \
    --patience ${PATIENCE} \
    --overlap ${OVERLAP} \
    --gpu_id ${GPU_ID} \
    --train_env ${TRAIN_ENV} \
    --eval_env ${EVAL_ENV} \
    --codec_comparison_dir ${CODEC_DIR} \
    --cleanup_inference \
    --cleanup_codec_comparison \
    --cleanup_best_model

sleep 5

# 實驗 1h: lr = 1e-10 (較小)
echo "=========================================="
echo "實驗 1f: LoRA Encoder - LR 1e-10"
echo "=========================================="

python run_full_pipeline.py \
    --freeze_encoder \
    --freeze_compressor \
    --freeze_decompressor \
    --train_decoder \
    --use_lora \
    --lora_decoder \
    --lora_rank ${BASE_LORA_RANK} \
    --lora_alpha ${BASE_LORA_ALPHA} \
    --use_feature_loss \
    --use_mel_loss \
    --weight_feature ${BASE_WEIGHT_FEATURE} \
    --weight_mel ${BASE_WEIGHT_MEL} \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate 1e-10 \
    --gradient_clip ${BASE_GRADIENT_CLIP} \
    --weight_decay ${BASE_WEIGHT_DECAY} \
    --early_stopping \
    --patience ${PATIENCE} \
    --overlap ${OVERLAP} \
    --gpu_id ${GPU_ID} \
    --train_env ${TRAIN_ENV} \
    --eval_env ${EVAL_ENV} \
    --codec_comparison_dir ${CODEC_DIR} \
    --cleanup_inference \
    --cleanup_codec_comparison \
    --cleanup_best_model

sleep 5