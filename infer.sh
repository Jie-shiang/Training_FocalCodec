#!/bin/bash
################################################################################
# FocalCodec 25Hz Inference
#
# Reads configuration from config.yaml
#
# Usage:
#   bash infer.sh                    # Inference with config.yaml settings
#   bash infer.sh --stage 2          # Override stage
#   bash infer.sh --max_samples 500  # Override max samples
################################################################################

set -e

# Environment
export TOKENIZERS_PARALLELISM=false
source /opt/conda/anaconda3/etc/profile.d/conda.sh
conda activate focalcodec

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse config using Python
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

# Load config values
GPU_ID=$(get_config "inference.gpu_id")
STAGE=$(get_config "training.stage")
CODEBOOK_SIZE=$(get_config "model.codebook_size")
FRAME_RATE=$(get_config "model.frame_rate")
MAX_SAMPLES=$(get_config "inference.max_samples")

# Calculate bitrate
BITS=$(python3 -c "import math; print(int(math.log2(${CODEBOOK_SIZE})))")
BITRATE=$((FRAME_RATE * BITS))

echo "========================================================================"
echo "   FocalCodec 25Hz Inference"
echo "========================================================================"
echo ""
echo "Configuration (from config.yaml):"
echo "  Stage: ${STAGE}"
echo "  Frame Rate: ${FRAME_RATE} Hz"
echo "  Codebook: ${CODEBOOK_SIZE} (${BITS}-bit)"
echo "  Bitrate: ${BITRATE} bps"
echo "  Max Samples: ${MAX_SAMPLES}"
echo "  GPU: ${GPU_ID}"
echo ""
echo "========================================================================"
echo ""

# Run inference
CUDA_VISIBLE_DEVICES=$GPU_ID python infer_25hz_focalcodec.py "$@"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "   Inference Completed!"
    echo "========================================================================"
    echo ""
    echo "Next step: Evaluate with bash eval.sh"
    echo "========================================================================"
else
    echo ""
    echo "ERROR: Inference failed!"
    exit 1
fi
