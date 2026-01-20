#!/bin/bash
################################################################################
# Stage 2: Add 4th layer for 25Hz downscale (官方方法)
#
# 訓練策略:
#   - 從 Stage 1 checkpoint 開始
#   - 添加第 4 層 FocalNet 層
#   - 訓練: Compressor + Decompressor
#   - 凍結: Encoder + Decoder
#   - Feature Loss: MSE(qfeats, feats) = MSE(decompressor_output, encoder_features)
#
# 使用官方 FocalCodec API:
#   sig_to_feats → feats_to_lats → lats_to_codes → codes_to_qfeats → feats_to_sig
#
# Prerequisites: Stage 1 must be completed
# Output: Fine-tuned 25Hz model at output_dir/stage2_25hz/
################################################################################

set -e

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

GPU_ID=$(get_config "stage2.gpu_id")
BATCH_SIZE=$(get_config "stage2.batch_size")
NUM_EPOCHS=$(get_config "stage2.num_epochs")
LR_NEW=$(get_config "stage2.learning_rate_new")
LR_OLD=$(get_config "stage2.learning_rate_old")
WARMUP=$(get_config "stage2.warmup_epochs")
OUTPUT_DIR=$(get_config "paths.output_dir")

echo "========================================================================"
echo "   Stage 2: Add 4th layer for 25Hz downscale (官方方法)"
echo "========================================================================"
echo ""
echo "訓練策略:"
echo "  訓練: Compressor + Decompressor"
echo "  凍結: Encoder + Decoder"
echo "  Feature Loss: MSE(decompressor_output, encoder_features)"
echo ""
echo "官方 FocalCodec API:"
echo "  sig_to_feats → feats_to_lats → lats_to_codes → codes_to_qfeats → feats_to_sig"
echo ""
echo "Configuration:"
echo "  GPU: ${GPU_ID}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Warmup: ${WARMUP} epochs"
echo "  Learning Rate (new layer 4): ${LR_NEW}"
echo "  Learning Rate (old layers 0-2): ${LR_OLD}"
echo ""
echo "========================================================================"
echo ""

# Check if Stage 1 checkpoint exists
STAGE1_CHECKPOINT="${OUTPUT_DIR}/stage1_50hz/best_model.pt"
if [ ! -f "$STAGE1_CHECKPOINT" ]; then
    echo "ERROR: Stage 1 checkpoint not found at $STAGE1_CHECKPOINT"
    echo "Please run Stage 1 first: bash train_stage1.sh"
    exit 1
fi

echo "Found Stage 1 checkpoint: $STAGE1_CHECKPOINT"
echo ""

# Run training
CUDA_VISIBLE_DEVICES=$GPU_ID python train_stage2_25hz.py

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "   Stage 2 Completed!"
    echo "========================================================================"
    echo ""
    echo "Output: 25Hz model (275 bps)"
    echo ""
    echo "Next steps:"
    echo "  1. Run inference: bash infer.sh 2"
    echo "  2. Run evaluation: bash eval.sh 2"
    echo ""
    echo "========================================================================"
else
    echo ""
    echo "ERROR: Stage 2 training failed!"
    exit 1
fi
