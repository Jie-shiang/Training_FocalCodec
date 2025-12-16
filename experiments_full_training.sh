#!/bin/bash
################################################################################
# FocalCodec Full Fine-tuning Experiment Script
################################################################################

# Common parameters
GPU_ID=3
TRAIN_ENV="focalcodec"
EVAL_ENV="codec_eval"
CODEC_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison"
NUM_EPOCHS=100
BATCH_SIZE=32
OVERLAP=0.5
PATIENCE=15

# Base parameters
BASE_WEIGHT_FEATURE=1.0
BASE_WEIGHT_TIME=0.3
BASE_WEIGHT_MEL=0.3
BASE_WEIGHT_ASR=0.5
BASE_GRADIENT_CLIP=1.0
BASE_WEIGHT_DECAY=1e-5

################################################################################
# Experiment 1: Train Decoder Only
################################################################################

echo "=========================================="
echo "Experiment 1a: Full Training - Decoder Only - LR 1e-5"
echo "=========================================="

python run_full_pipeline.py \
    --freeze_encoder \
    --freeze_compressor \
    --freeze_decompressor \
    --train_decoder \
    --use_feature_loss \
    --use_time_loss \
    --use_mel_loss \
    --use_asr_loss \
    --weight_feature ${BASE_WEIGHT_FEATURE} \
    --weight_time ${BASE_WEIGHT_TIME} \
    --weight_mel ${BASE_WEIGHT_MEL} \
    --weight_asr ${BASE_WEIGHT_ASR} \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate 1e-5 \
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

echo "=========================================="
echo "Experiment 1b: Full Training - Decoder Only - LR 1e-6"
echo "=========================================="

python run_full_pipeline.py \
    --freeze_encoder \
    --freeze_compressor \
    --freeze_decompressor \
    --train_decoder \
    --use_feature_loss \
    --use_time_loss \
    --use_mel_loss \
    --use_asr_loss \
    --weight_feature ${BASE_WEIGHT_FEATURE} \
    --weight_time ${BASE_WEIGHT_TIME} \
    --weight_mel ${BASE_WEIGHT_MEL} \
    --weight_asr ${BASE_WEIGHT_ASR} \
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

################################################################################
# Experiment 2: Train Decoder + Decompressor
################################################################################

echo "=========================================="
echo "Experiment 2a: Full Training - Decoder + Decompressor - LR 1e-5"
echo "=========================================="

python run_full_pipeline.py \
    --freeze_encoder \
    --freeze_compressor \
    --train_decompressor \
    --train_decoder \
    --use_feature_loss \
    --use_time_loss \
    --use_mel_loss \
    --use_asr_loss \
    --weight_feature ${BASE_WEIGHT_FEATURE} \
    --weight_time ${BASE_WEIGHT_TIME} \
    --weight_mel ${BASE_WEIGHT_MEL} \
    --weight_asr ${BASE_WEIGHT_ASR} \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate 1e-5 \
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

echo "=========================================="
echo "Experiment 2b: Full Training - Decoder + Decompressor - LR 1e-6"
echo "=========================================="

python run_full_pipeline.py \
    --freeze_encoder \
    --freeze_compressor \
    --train_decompressor \
    --train_decoder \
    --use_feature_loss \
    --use_time_loss \
    --use_mel_loss \
    --use_asr_loss \
    --weight_feature ${BASE_WEIGHT_FEATURE} \
    --weight_time ${BASE_WEIGHT_TIME} \
    --weight_mel ${BASE_WEIGHT_MEL} \
    --weight_asr ${BASE_WEIGHT_ASR} \
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

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
