#!/bin/bash
################################################################################
# FocalCodec Two-Stage Training
#
# Stage 1 (50Hz):
#   - 訓練: Decompressor only (推薦)
#   - 凍結: Encoder + Compressor + Decoder
#   - 這是原始 7% dCER 結果使用的訓練策略
#
# Stage 2 (25Hz):
#   - 在 Stage 1 基礎上添加第 4 層 FocalNet
#   - 訓練: 新增的第 4 層
#   - 凍結: Encoder + 前 3 層 Compressor/Decompressor + Decoder
#
# Training Modes (Stage 1 only):
#   decompressor_only (default): Only train Decompressor
#   both: Train Compressor + Decompressor (no STE)
#   both_ste: Train with STE (may damage Compressor)
#
# Usage:
#   bash train.sh 1                    # Train Stage 1 (decompressor_only)
#   bash train.sh 1 both_ste           # Train Stage 1 with STE
#   bash train.sh 2                    # Train Stage 2 (requires Stage 1 done)
#
# Output:
#   Stage 1: output_dir/stage1_50hz/
#   Stage 2: output_dir/stage2_25hz/
################################################################################

set -e

# Parse arguments
STAGE=${1:-1}
TRAIN_MODE=${2:-"decompressor_only"}

# Validate stage
if [ "$STAGE" != "1" ] && [ "$STAGE" != "2" ]; then
    echo "ERROR: Invalid stage '$STAGE'"
    echo "Usage: bash train.sh [1|2] [train_mode]"
    echo "  Stage 1: 50Hz training"
    echo "  Stage 2: 25Hz training"
    exit 1
fi

# Stage 1: Use train_stage1.sh
if [ "$STAGE" = "1" ]; then
    echo "Executing Stage 1 training..."
    bash train_stage1.sh $TRAIN_MODE
    exit $?
fi

# Stage 2: Use train_stage2.sh
if [ "$STAGE" = "2" ]; then
    # Validate train_mode not specified
    if [ "$TRAIN_MODE" != "decompressor_only" ]; then
        echo "WARNING: Stage 2 does not support train_mode, using default strategy"
    fi
    echo "Executing Stage 2 training..."
    bash train_stage2.sh
    exit $?
fi
