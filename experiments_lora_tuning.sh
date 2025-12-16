#!/bin/bash
################################################################################
# FocalCodec LoRA Fine-tuning Experiment Script
#
# Important notes:
# 1. Fully frozen: Use --freeze_* for all components (baseline)
# 2. LoRA tune: Use --train_* + --lora_* (do not use --freeze_*)
#
# Key adjustable parameters:
# - learning_rate: (1e-3 ~ 1e-6)
# - lora_rank: (4, 8, 16, 32, 64)
# - lora_alpha: (typically 2x rank)
# - batch_size: (8, 16, 32, 64)
# - num_epochs: training epochs
# - weight_feature: feature loss weight (0.5 ~ 2.0)
# - weight_mel: mel loss weight (0.1 ~ 0.5)
# - patience: early stopping patience
# - overlap: training data chunk overlap (0.0 ~ 0.5)
# - gradient_clip: (0.5 ~ 5.0)
# - weight_decay: (1e-6 ~ 1e-4)
################################################################################

# Common parameters
GPU_ID=3
TRAIN_ENV="focalcodec"
EVAL_ENV="codec_eval"
CODEC_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison"
NUM_EPOCHS=300
BATCH_SIZE=32
OVERLAP=0.5
PATIENCE=20

# Base parameters (baseline)
BASE_LR=1e-4
BASE_LORA_RANK=16
BASE_LORA_ALPHA=32
BASE_WEIGHT_FEATURE=1.0
BASE_WEIGHT_MEL=0.15
BASE_GRADIENT_CLIP=1.0
BASE_WEIGHT_DECAY=1e-5

################################################################################
# Experiment Group 1: Adjust Learning Rate (LoRA tune encoder)
################################################################################

# Experiment 1a: lr = 1e-3 (higher)
echo "=========================================="
echo "Experiment 1a: LoRA Encoder - LR 1e-3"
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

# Experiment 1b: lr = 1e-4
echo "=========================================="
echo "Experiment 1b: LoRA Encoder - LR 1e-4"
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

# Experiment 1c: lr = 1e-5 (baseline)
echo "=========================================="
echo "Experiment 1c: LoRA Encoder - LR 1e-5 (baseline)"
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

# Experiment 1d: lr = 1e-6
echo "=========================================="
echo "Experiment 1d: LoRA Encoder - LR 1e-6"
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

# Experiment 1e: lr = 1e-7
echo "=========================================="
echo "Experiment 1e: LoRA Encoder - LR 1e-7"
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

# Experiment 1f: lr = 1e-8 (lower)
echo "=========================================="
echo "Experiment 1f: LoRA Encoder - LR 1e-8"
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

# Experiment 1g: lr = 1e-9 (lower)
echo "=========================================="
echo "Experiment 1g: LoRA Encoder - LR 1e-9"
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

# Experiment 1h: lr = 1e-10 (lower)
echo "=========================================="
echo "Experiment 1h: LoRA Encoder - LR 1e-10"
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