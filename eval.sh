#!/bin/bash
################################################################################
# FocalCodec 25Hz Evaluation (dCER)
#
# Reads configuration from config.yaml
#
# Usage:
#   bash eval.sh                    # Evaluate with config.yaml settings
#   bash eval.sh --stage 2          # Override stage
#   bash eval.sh --max_samples 500  # Override max samples
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
GPU_ID=$(get_config "evaluation.gpu_id")
STAGE=$(get_config "training.stage")
MAX_SAMPLES=$(get_config "evaluation.max_samples")

echo "========================================================================"
echo "   FocalCodec 25Hz dCER Evaluation"
echo "========================================================================"
echo ""
echo "Configuration (from config.yaml):"
echo "  Stage: ${STAGE}"
echo "  Max Samples: ${MAX_SAMPLES}"
echo "  GPU: ${GPU_ID}"
echo ""
echo "========================================================================"
echo ""

# Run evaluation
CUDA_VISIBLE_DEVICES=$GPU_ID python eval_25hz_focalcodec.py "$@"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "   Evaluation Completed!"
    echo "========================================================================"
else
    echo ""
    echo "ERROR: Evaluation failed!"
    exit 1
fi
