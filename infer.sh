#!/bin/bash
################################################################################
# Inference Script for Two-Stage FocalCodec
#
# Usage:
#   bash infer.sh 1          # Inference with Stage 1 (50Hz)
#   bash infer.sh 2          # Inference with Stage 2 (25Hz)
#   bash infer.sh 2 500      # Inference Stage 2 with max 500 samples
################################################################################

set -e

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: bash infer.sh <stage> [max_samples]"
    echo "  stage: 1 (50Hz) or 2 (25Hz)"
    echo "  max_samples: optional, default from config.yaml"
    echo ""
    echo "Examples:"
    echo "  bash infer.sh 1          # Inference with Stage 1"
    echo "  bash infer.sh 2          # Inference with Stage 2"
    echo "  bash infer.sh 2 500      # Stage 2, max 500 samples"
    exit 1
fi

STAGE=$1
MAX_SAMPLES=${2:-""}

# Validate stage
if [ "$STAGE" != "1" ] && [ "$STAGE" != "2" ]; then
    echo "ERROR: Stage must be 1 or 2"
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

STAGE_KEY="stage${STAGE}"
GPU_ID=$(get_config "${STAGE_KEY}.gpu_id")
FRAME_RATE=$([ "$STAGE" = "1" ] && echo "50" || echo "25")
OUTPUT_DIR=$(get_config "paths.output_dir")
INFERENCE_DIR=$(get_config "paths.inference_dir")

echo "========================================================================"
echo "   Inference: Stage ${STAGE} (${FRAME_RATE}Hz)"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  GPU: ${GPU_ID}"
echo "  Frame Rate: ${FRAME_RATE} Hz"
echo "  Stage: ${STAGE}"
if [ -n "$MAX_SAMPLES" ]; then
    echo "  Max Samples: ${MAX_SAMPLES}"
fi
echo ""
echo "========================================================================"
echo ""

# Build command
CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} python infer_two_stage.py --stage ${STAGE}"
if [ -n "$MAX_SAMPLES" ]; then
    CMD="${CMD} --max_samples ${MAX_SAMPLES}"
fi

# Run inference
eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "   Inference Completed!"
    echo "========================================================================"
    echo ""
    echo "Results saved to: ${INFERENCE_DIR}/stage${STAGE}/"
    echo ""
    echo "Next step: Run evaluation"
    echo "  bash eval.sh ${STAGE}"
    echo "========================================================================"
else
    echo ""
    echo "ERROR: Inference failed!"
    exit 1
fi
