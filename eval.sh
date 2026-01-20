#!/bin/bash
################################################################################
# Evaluation Script for Two-Stage FocalCodec
#
# Uses Paraformer-zh (FunASR) by default - same ASR as the original 7% dCER.
# Supports batch processing for faster evaluation.
#
# Usage:
#   bash eval.sh 1                    # Evaluate Stage 1 (50Hz) with Paraformer
#   bash eval.sh 2                    # Evaluate Stage 2 (25Hz) with Paraformer
#   bash eval.sh 1 whisper            # Evaluate Stage 1 with Whisper
#   bash eval.sh 1 paraformer 500     # Evaluate Stage 1, max 500 samples
#   bash eval.sh 1 paraformer 2000 32 # Evaluate with batch_size=32
################################################################################

set -e

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: bash eval.sh <stage> [asr] [max_samples] [batch_size]"
    echo ""
    echo "Arguments:"
    echo "  stage:       1 (50Hz) or 2 (25Hz)"
    echo "  asr:         'paraformer' (default) or 'whisper'"
    echo "  max_samples: optional, default 2000"
    echo "  batch_size:  optional, default 16"
    echo ""
    echo "Examples:"
    echo "  bash eval.sh 1                    # Stage 1, Paraformer (recommended)"
    echo "  bash eval.sh 2                    # Stage 2, Paraformer"
    echo "  bash eval.sh 1 whisper            # Stage 1, Whisper"
    echo "  bash eval.sh 1 paraformer 500     # Stage 1, max 500 samples"
    echo "  bash eval.sh 1 paraformer 2000 32 # Stage 1, batch_size=32"
    echo ""
    echo "Note: Paraformer-zh is the same ASR used for the original 7% dCER result."
    exit 1
fi

STAGE=$1
ASR=${2:-"paraformer"}
MAX_SAMPLES=${3:-""}
BATCH_SIZE=${4:-"16"}

# Validate stage
if [ "$STAGE" != "1" ] && [ "$STAGE" != "2" ]; then
    echo "ERROR: Stage must be 1 or 2"
    exit 1
fi

# Validate ASR
if [ "$ASR" != "paraformer" ] && [ "$ASR" != "whisper" ]; then
    echo "ERROR: ASR must be 'paraformer' or 'whisper'"
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

GPU_ID=$(get_config "evaluation.gpu_id")
FRAME_RATE=$([ "$STAGE" = "1" ] && echo "50" || echo "25")
INFERENCE_DIR=$(get_config "paths.inference_dir")

echo "========================================================================"
echo "   dCER Evaluation: Stage ${STAGE} (${FRAME_RATE}Hz)"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Stage: ${STAGE}"
echo "  Frame Rate: ${FRAME_RATE} Hz"
echo "  ASR Model: ${ASR}"
if [ -n "$MAX_SAMPLES" ]; then
    echo "  Max Samples: ${MAX_SAMPLES}"
else
    echo "  Max Samples: 2000 (default)"
fi
echo "  Batch Size: ${BATCH_SIZE}"
echo "  GPU: ${GPU_ID}"
echo ""
if [ "$ASR" = "paraformer" ]; then
    echo "  Note: Using Paraformer-zh (same as original 7% dCER evaluation)"
fi
echo ""
echo "========================================================================"
echo ""

# Build command
CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_two_stage.py --stage ${STAGE} --asr ${ASR} --batch_size ${BATCH_SIZE}"
if [ -n "$MAX_SAMPLES" ]; then
    CMD="${CMD} --max_samples ${MAX_SAMPLES}"
fi

# Run evaluation
echo "Running: $CMD"
echo ""
eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "   Evaluation Completed!"
    echo "========================================================================"
    echo ""
    echo "Results saved to: ${INFERENCE_DIR}/stage${STAGE}/"
    echo "  - dcer_results_${ASR}.csv"
    echo "  - dcer_summary_${ASR}.json"
    echo "========================================================================"
else
    echo ""
    echo "ERROR: Evaluation failed!"
    exit 1
fi
