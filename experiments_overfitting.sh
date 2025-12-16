#!/bin/bash
################################################################################
# FocalCodec Overfitting Test
# Verify model can learn from seen data (Train = Val)
################################################################################

set -e

GPU_ID=2
BASE_DIR="/home/jieshiang/Desktop/GitHub/FocalCodec"
EPOCHS=100
PATIENCE=50

echo "=========================================="
echo "FocalCodec Overfitting Test"
echo "=========================================="
echo "GPU: $GPU_ID"
echo "Epochs: $EPOCHS"
echo "Patience: $PATIENCE"
echo "=========================================="
echo ""

################################################################################
# Experiment 1: Decoder Only, LR=1e-6, Time+Mel
################################################################################

echo "[1/4] Decoder only, LR=1e-6, Time+Mel"
python $BASE_DIR/run_full_pipeline.py \
  --freeze_encoder \
  --freeze_compressor \
  --freeze_decompressor \
  --train_decoder \
  --learning_rate 1e-6 \
  --num_epochs $EPOCHS \
  --patience $PATIENCE \
  --use_time_loss \
  --use_mel_loss \
  --weight_time 0.3 \
  --weight_mel 1.0 \
  --early_stopping \
  --gpu_id $GPU_ID \
  --cleanup_inference \
  --cleanup_codec_comparison \
  --keep_best_model_only

echo ""
echo "✓ Experiment 1 completed"
echo ""


################################################################################
# Experiment 2: Decoder Only, LR=1e-5, Time+Mel
################################################################################

echo "[2/4] Decoder only, LR=1e-5, Time+Mel"
python $BASE_DIR/run_full_pipeline.py \
  --freeze_encoder \
  --freeze_compressor \
  --freeze_decompressor \
  --train_decoder \
  --learning_rate 1e-5 \
  --num_epochs $EPOCHS \
  --patience $PATIENCE \
  --use_time_loss \
  --use_mel_loss \
  --weight_time 0.3 \
  --weight_mel 1.0 \
  --early_stopping \
  --gpu_id $GPU_ID \
  --cleanup_inference \
  --cleanup_codec_comparison \
  --keep_best_model_only

echo ""
echo "✓ Experiment 2 completed"
echo ""

################################################################################
# Experiment 3: Joint Training, LR=1e-6, Time+Mel
################################################################################

echo "[3/4] Decomp+Dec, LR=1e-6, Time+Mel"
python $BASE_DIR/run_full_pipeline.py \
  --freeze_encoder \
  --freeze_compressor \
  --train_decompressor \
  --train_decoder \
  --learning_rate 1e-6 \
  --num_epochs $EPOCHS \
  --patience $PATIENCE \
  --use_time_loss \
  --use_mel_loss \
  --weight_time 0.3 \
  --weight_mel 1.0 \
  --early_stopping \
  --gpu_id $GPU_ID \
  --cleanup_inference \
  --cleanup_codec_comparison \
  --keep_best_model_only

echo ""
echo "✓ Experiment 3 completed"
echo ""

################################################################################
# Experiment 4: Joint Training, LR=1e-5, Time+Mel
################################################################################

echo "[4/4] Decomp+Dec, LR=1e-5, Time+Mel"
python $BASE_DIR/run_full_pipeline.py \
  --freeze_encoder \
  --freeze_compressor \
  --train_decompressor \
  --train_decoder \
  --learning_rate 1e-5 \
  --num_epochs $EPOCHS \
  --patience $PATIENCE \
  --use_time_loss \
  --use_mel_loss \
  --weight_time 0.3 \
  --weight_mel 1.0 \
  --early_stopping \
  --gpu_id $GPU_ID \
  --cleanup_inference \
  --cleanup_codec_comparison \
  --keep_best_model_only

echo ""
echo "✓ Experiment 4 completed"
echo ""

################################################################################
# Summary
################################################################################

echo ""
echo "================================================"
echo "All 4 overfitting experiments completed!"
echo "================================================"
echo ""
echo "Experiment Summary:"
echo "  Group 1: Decoder only (LR: 1e-6, 1e-5)"
echo "  Group 2: Decomp+Dec (LR: 1e-6, 1e-5)"
echo ""
echo "Check results in: experiments/"
echo "================================================"
