#!/bin/bash
################################################################################
# Stage 1: Fine-tune 50Hz Causal 2k on AISHELL
#
# Training Modes:
#   decompressor_only (default, recommended):
#     - 訓練: Decompressor only
#     - 凍結: Encoder + Compressor + Decoder
#     - 這是原始 7% dCER 結果使用的訓練策略
#     - Compressor 保持 pretrained 能力，不會被破壞
#
#   both:
#     - 訓練: Compressor + Decompressor (無 STE)
#     - 凍結: Encoder + Decoder
#     - 警告: Compressor 沒有梯度，實際只訓練 Decompressor
#
#   both_ste:
#     - 訓練: Compressor + Decompressor (使用 STE)
#     - 凍結: Encoder + Decoder
#     - 警告: 可能破壞 Compressor 的 pretrained 能力!
#
# Feature Loss: MSE(qfeats, feats) = MSE(decompressor_output, encoder_features)
#
# 使用官方 FocalCodec API:
#   sig_to_feats → feats_to_lats → lats_to_codes → codes_to_qfeats → feats_to_sig
#
# Usage:
#   bash train_stage1.sh                       # Default: decompressor_only (recommended)
#   bash train_stage1.sh decompressor_only     # Only train Decompressor
#   bash train_stage1.sh both                  # Train both (no STE)
#   bash train_stage1.sh both_ste              # Train both with STE
#
# Output: Fine-tuned 50Hz model at output_dir/stage1_50hz/
################################################################################

set -e

# Training mode (default: decompressor_only)
TRAIN_MODE=${1:-"decompressor_only"}

# Validate mode
if [ "$TRAIN_MODE" != "decompressor_only" ] && [ "$TRAIN_MODE" != "both" ] && [ "$TRAIN_MODE" != "both_ste" ]; then
    echo "ERROR: Invalid train_mode '$TRAIN_MODE'"
    echo "Valid options: decompressor_only, both, both_ste"
    exit 1
fi

# Environment
export TOKENIZERS_PARALLELISM=false
source /opt/conda/anaconda3/etc/profile.d/conda.sh
conda activate focalcodec

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse config
get_config() {
    python -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
keys = '$1'.split('.')
value = config
for k in keys:
    value = value.get(k, {})
print(value if value else '')
"
}

GPU_ID=$(get_config "stage1.gpu_id")
BATCH_SIZE=$(get_config "stage1.batch_size")
NUM_EPOCHS=$(get_config "stage1.num_epochs")
LR=$(get_config "stage1.learning_rate")

echo "========================================================================"
echo "   Stage 1: Fine-tune 50Hz Causal 2k on AISHELL"
echo "========================================================================"
echo ""
echo "Training Mode: ${TRAIN_MODE}"
echo ""

if [ "$TRAIN_MODE" = "decompressor_only" ]; then
    echo "訓練策略 (推薦模式 - 原始 7% dCER):"
    echo "  訓練: Decompressor"
    echo "  凍結: Encoder + Compressor + Decoder"
    echo "  優點: Compressor 保持 pretrained 能力，不會被破壞"
elif [ "$TRAIN_MODE" = "both" ]; then
    echo "訓練策略:"
    echo "  訓練: Compressor + Decompressor (無 STE)"
    echo "  凍結: Encoder + Decoder"
    echo "  注意: Compressor 沒有梯度，實際只訓練 Decompressor"
elif [ "$TRAIN_MODE" = "both_ste" ]; then
    echo "訓練策略:"
    echo "  訓練: Compressor + Decompressor (使用 STE)"
    echo "  凍結: Encoder + Decoder"
    echo "  ⚠️  警告: 可能破壞 Compressor 的 pretrained 能力!"
fi

echo ""
echo "Feature Loss: MSE(decompressor_output, encoder_features)"
echo ""
echo "官方 FocalCodec API:"
echo "  sig_to_feats → feats_to_lats → lats_to_codes → codes_to_qfeats → feats_to_sig"
echo ""
echo "Configuration:"
echo "  GPU: ${GPU_ID}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Learning Rate: ${LR}"
echo ""
echo "========================================================================"
echo ""

# Run training
CUDA_VISIBLE_DEVICES=$GPU_ID python train_stage1_50hz.py --train_mode $TRAIN_MODE

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "   Stage 1 Completed!"
    echo "========================================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Run inference: bash infer.sh 1"
    echo "  2. Run evaluation: bash eval.sh 1"
    echo "  3. If dCER < 10%, proceed to Stage 2: bash train_stage2.sh"
    echo ""
    echo "========================================================================"
else
    echo ""
    echo "ERROR: Stage 1 training failed!"
    exit 1
fi
