#!/bin/bash
################################################################################
# Experiment K: Stage 3 DDP Training (12.5Hz) with Whisper ASR Loss
# DDP 版本 - 使用 torchrun 多 GPU 訓練
#
# normal partition 限制: MaxTRESPA=gres/gpu=16 (整個 Account MST114558 共用)
# 設定: 2 nodes × 8 GPU = 16 GPU total
#   per_gpu batch_size = 104 (H100 80GB 測試安全值)
#   effective batch_size = 104 × 16 = 1664
################################################################################
#SBATCH --job-name=S3_K_ddp
#SBATCH --partition=normal
#SBATCH --account=MST114558
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=logs/exp_K/train_s3_ddp_%j.log
#SBATCH --error=logs/exp_K/train_s3_ddp_%j.err

# 載入環境
ml purge
ml load miniconda3/24.11.1
conda activate focalcodec

export PYTHONPATH=$PYTHONPATH:$(pwd)
export HF_HOME=/work/u4162168/model_cache
export TRANSFORMERS_CACHE=/work/u4162168/model_cache
export TOKENIZERS_PARALLELISM=false

# NCCL 跨節點通訊設定
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker

# 創建輸出目錄
mkdir -p logs/exp_K
mkdir -p output/exp_K/stage3_12.5hz
mkdir -p experiments/exp_K

CONFIG="config_exp_K.yaml"

echo "========================================================================"
echo "   Experiment K - Stage 3 DDP Training (12.5Hz, ASR Loss)"
echo "========================================================================"
echo "Source: Exp J Stage 2 checkpoint (ASR-aligned)"
echo "Stage: 3 (12.5Hz, 137.5 bps)"
echo ""
echo "DDP Config:"
echo "  Nodes:            $SLURM_NNODES"
echo "  GPUs/node:        $SLURM_GPUS_PER_NODE"
echo "  Total GPUs:       $((SLURM_NNODES * SLURM_GPUS_PER_NODE))"
echo "  per_GPU batch:    104"
echo "  Effective batch:  $((SLURM_NNODES * SLURM_GPUS_PER_NODE * 104))"
echo ""
echo "Job ID:  $SLURM_JOB_ID"
echo "Nodes:   $SLURM_NODELIST"
echo "Started: $(date)"
echo "========================================================================"

# 確認 Exp J Stage 2 checkpoint 存在
SOURCE_CHECKPOINT="output/exp_J/stage2_25hz/best_model.pt"
if [ ! -f "$SOURCE_CHECKPOINT" ]; then
    echo "ERROR: Source checkpoint not found: $SOURCE_CHECKPOINT"
    exit 1
fi
echo "Source checkpoint: $SOURCE_CHECKPOINT ($(du -h $SOURCE_CHECKPOINT | cut -f1))"

# 確認訓練資料存在
TRAIN_CSV="experiments/data_mix_h/train_split.csv"
if [ ! -f "$TRAIN_CSV" ]; then
    echo "ERROR: Training CSV not found: $TRAIN_CSV"
    exit 1
fi
TRAIN_COUNT=$(wc -l < "$TRAIN_CSV")
echo "Training data: $((TRAIN_COUNT - 1)) entries"

# Whisper token cache
EXPJ_CACHE_DIR="experiments/exp_J/whisper_cache"
TRAIN_CACHE="$EXPJ_CACHE_DIR/whisper_tokens_train_small.pt"
VAL_CACHE="$EXPJ_CACHE_DIR/whisper_tokens_val_small.pt"

if [ -f "$TRAIN_CACHE" ] && [ -f "$VAL_CACHE" ]; then
    echo "Reusing Exp J Whisper token cache:"
    echo "  Train: $TRAIN_CACHE ($(du -h $TRAIN_CACHE | cut -f1))"
    echo "  Val:   $VAL_CACHE ($(du -h $VAL_CACHE | cut -f1))"
else
    echo "WARNING: Exp J token cache not found. Training will use online Whisper (slower)."
fi
echo ""

# torchrun 跨節點 DDP 啟動
# SLURM_PROCID=0 的節點作為 master
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
NGPUS_PER_NODE=$SLURM_GPUS_PER_NODE
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_NODEID

echo "Master node: $MASTER_ADDR:$MASTER_PORT"
echo "Node rank:   $NODE_RANK / $NNODES"
echo ""

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NGPUS_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_stage3_asr_ddp.py \
    --config $CONFIG

TRAIN_EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "   Experiment K - Stage 3 DDP Training Completed"
    echo "========================================================================"
    echo "Checkpoint: output/exp_K/stage3_12.5hz/best_model.pt"
    echo "Log CSV:    experiments/exp_K/stage3_12.5hz_asr_training_ddp.csv"
    echo ""
    echo "Next steps:"
    echo "  sbatch bash/slurm_infer_eval_stage3_K.sh"
else
    echo "   Experiment K - Stage 3 DDP Training FAILED"
    echo "========================================================================"
    echo "Exit code: $TRAIN_EXIT_CODE"
    echo "Check error log: logs/exp_K/train_s3_ddp_${SLURM_JOB_ID}.err"
    echo ""
    echo "To resume:"
    echo "  sbatch bash/slurm_train_stage3_K_ddp.sh  (add --resume to python args)"
fi
echo "Finished: $(date)"
echo "========================================================================"

exit $TRAIN_EXIT_CODE
