#!/bin/bash
################################################################################
# Experiment K: Stage 3 Training (12.5Hz) with Whisper ASR Loss
# 從 Exp J Stage 2 checkpoint 繼承，加入 Stage 3 壓縮層
# 目標: dCER < 10% (中文), dWER < 5% (英文) @ 137.5 bps
################################################################################

#SBATCH --job-name=S3_K_asr
#SBATCH --partition=normal2
#SBATCH --account=MST114558
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=logs/exp_K/train_s3_%j.log
#SBATCH --error=logs/exp_K/train_s3_%j.err

# 載入環境
ml purge
ml load miniconda3/24.11.1
conda activate focalcodec

export PYTHONPATH=$PYTHONPATH:$(pwd)
export HF_HOME=/work/u4162168/model_cache
export TRANSFORMERS_CACHE=/work/u4162168/model_cache
export TOKENIZERS_PARALLELISM=false

# 創建輸出目錄
mkdir -p logs/exp_K
mkdir -p output/exp_K/stage3_12.5hz
mkdir -p experiments/exp_K

CONFIG="config_exp_K.yaml"

echo "========================================================================"
echo "   Experiment K - Stage 3 with Whisper ASR Loss (12.5Hz)"
echo "========================================================================"
echo "Source: Exp J Stage 2 checkpoint (ASR-aligned)"
echo "Stage: 3 (12.5Hz, 137.5 bps)"
echo "Data: Chinese (~50h) + English (~50h) = ~100h mixed"
echo ""
echo "Loss Weights:"
echo "  Feature: 1.0"
echo "  Mel:     5.0"
echo "  STFT:    2.0"
echo "  ASR:     20.0 (降低以保留更多音質)"
echo ""
echo "LR Strategy:"
echo "  Old layers (inherited): 5e-5"
echo "  New layer (Stage 3):    5e-4"
echo "  Warmup epochs (new only): 5"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $SLURM_GPUS_PER_NODE"
echo "Started: $(date)"
echo "========================================================================"

# 確認 Exp J Stage 2 checkpoint 存在
SOURCE_CHECKPOINT="output/exp_J/stage2_25hz/best_model.pt"
if [ ! -f "$SOURCE_CHECKPOINT" ]; then
    echo "ERROR: Source checkpoint not found: $SOURCE_CHECKPOINT"
    echo "Please ensure Exp J Stage 2 training is complete"
    exit 1
fi
echo "Source checkpoint: $SOURCE_CHECKPOINT ($(du -h $SOURCE_CHECKPOINT | cut -f1))"
echo ""

# 確認訓練資料存在
TRAIN_CSV="experiments/data_mix_h/train_split.csv"
if [ ! -f "$TRAIN_CSV" ]; then
    echo "ERROR: Training CSV not found: $TRAIN_CSV"
    exit 1
fi
TRAIN_COUNT=$(wc -l < "$TRAIN_CSV")
echo "Training data: $((TRAIN_COUNT - 1)) entries"

# Whisper token cache: 重用 Exp J 的 cache (資料集相同)
EXPJ_CACHE_DIR="experiments/exp_J/whisper_cache"
TRAIN_CACHE="$EXPJ_CACHE_DIR/whisper_tokens_train_small.pt"
VAL_CACHE="$EXPJ_CACHE_DIR/whisper_tokens_val_small.pt"

if [ -f "$TRAIN_CACHE" ] && [ -f "$VAL_CACHE" ]; then
    echo "Reusing Exp J Whisper token cache:"
    echo "  Train: $TRAIN_CACHE ($(du -h $TRAIN_CACHE | cut -f1))"
    echo "  Val:   $VAL_CACHE ($(du -h $VAL_CACHE | cut -f1))"
else
    echo "WARNING: Exp J token cache not found at $EXPJ_CACHE_DIR"
    echo "Training will use online Whisper mode (slower)."
    echo "Consider running: python precompute_whisper_tokens.py --config $CONFIG"
fi
echo ""

# 單 GPU 訓練（train_stage3_asr.py 未實作 DDP）
# H100 80GB 測試: max batch=128, 安全值=104 (留15%緩衝) → config 已設定
export CUDA_VISIBLE_DEVICES=0

python train_stage3_asr.py \
    --config $CONFIG

TRAIN_EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "   Experiment K - Stage 3 ASR Training Completed"
    echo "========================================================================"
    echo "Checkpoint: output/exp_K/stage3_12.5hz/best_model.pt"
    echo "Log CSV: experiments/exp_K/stage3_12.5hz_asr_training.csv"
    echo ""
    echo "Next steps:"
    echo "  sbatch bash/slurm_infer_eval_stage3_K.sh"
else
    echo "   Experiment K - Stage 3 ASR Training FAILED"
    echo "========================================================================"
    echo "Exit code: $TRAIN_EXIT_CODE"
    echo "Check error log: logs/exp_K/train_s3_${SLURM_JOB_ID}.err"
    echo ""
    echo "To resume:"
    echo "  python train_stage3_asr.py --config $CONFIG --resume"
fi
echo "Finished: $(date)"
echo "========================================================================"

exit $TRAIN_EXIT_CODE
