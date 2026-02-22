#!/bin/bash
################################################################################
# Experiment K: Stage 3 Inference + Evaluation (Bilingual, 12.5Hz)
# 中文 (AISHELL) + 英文 (LibriSpeech)
# Bitrate: 12.5 Hz × 11-bit = 137.5 bps
################################################################################

#SBATCH --job-name=InferEval_K
#SBATCH --partition=normal2
#SBATCH --account=MST114558
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=logs/exp_K/infer_eval_s3_%j.log
#SBATCH --error=logs/exp_K/infer_eval_s3_%j.err

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
mkdir -p inference/exp_K/stage3/aishell
mkdir -p inference/exp_K/stage3/librispeech

CONFIG="config_exp_K.yaml"
FAILED_STAGES=""

echo "========================================================================"
echo "   Experiment K - Stage 3 Inference + Evaluation (Bilingual)"
echo "========================================================================"
echo "Model: Stage 3 ASR Loss (Feature:1.0, Mel:5.0, STFT:2.0, ASR:20.0)"
echo "Bitrate: 12.5 Hz × 11-bit = 137.5 bps"
echo "Expected: dCER < 10% (ZH), dWER < 5% (EN)"
echo ""
echo "Compare with:"
echo "  Exp J Stage 2 (275 bps): dCER=4.15%, MOS=1.589"
echo "  Exp E Stage 3 (137.5 bps, no ASR): dCER=43.60%, MOS=2.421"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "========================================================================"

# 確認 checkpoint
CHECKPOINT="output/exp_K/stage3_12.5hz/best_model.pt"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Please run training first: sbatch bash/slurm_train_stage3_K.sh"
    exit 1
fi
echo "Checkpoint: $CHECKPOINT ($(du -h $CHECKPOINT | cut -f1))"
echo ""

export CUDA_VISIBLE_DEVICES=0

########################################
# Chinese Inference (AISHELL)
########################################
echo "--- Stage 3 Inference: Chinese (AISHELL) ---"

python infer_stage3.py \
    --config $CONFIG \
    --max_samples 2000 \
    --csv_path csv/aishell_filtered_clean.csv \
    --output_subdir aishell

INFER_ZH_EXIT=$?

if [ $INFER_ZH_EXIT -ne 0 ]; then
    echo "Stage 3 Chinese inference failed!"
    FAILED_STAGES="$FAILED_STAGES Infer-ZH"
else
    echo "Stage 3 Chinese inference completed"
fi

echo ""

########################################
# English Inference (LibriSpeech)
########################################
echo "--- Stage 3 Inference: English (LibriSpeech) ---"

python infer_stage3.py \
    --config $CONFIG \
    --max_samples 2000 \
    --csv_path csv/librispeech_filtered_clean.csv \
    --output_subdir librispeech

INFER_EN_EXIT=$?

if [ $INFER_EN_EXIT -ne 0 ]; then
    echo "Stage 3 English inference failed!"
    FAILED_STAGES="$FAILED_STAGES Infer-EN"
else
    echo "Stage 3 English inference completed"
fi

echo ""

########################################
# Chinese Evaluation (dCER + MOS)
########################################
if [ $INFER_ZH_EXIT -eq 0 ]; then
    echo "--- Stage 3 Evaluation: Chinese ---"

    python eval_two_stage.py \
        --config $CONFIG \
        --stage 3 \
        --inference_subdir aishell

    EVAL_ZH_EXIT=$?

    if [ $EVAL_ZH_EXIT -ne 0 ]; then
        echo "Stage 3 Chinese evaluation failed!"
        FAILED_STAGES="$FAILED_STAGES Eval-ZH"
    else
        echo "Stage 3 Chinese evaluation completed"
        if [ -f "inference/exp_K/stage3/aishell/dcer_summary_paraformer.json" ]; then
            echo ""
            echo "=== Chinese Results (dCER) ==="
            cat inference/exp_K/stage3/aishell/dcer_summary_paraformer.json
            echo ""
            echo "Compare with:"
            echo "  Exp E Stage 3 (no ASR): ~43.60% dCER"
            echo "  Exp J Stage 2 (ASR):    ~4.15% dCER (but 275 bps)"
            echo "  Target:                 < 10% dCER @ 137.5 bps"
        fi
    fi
fi

echo ""

########################################
# English Evaluation (dWER)
########################################
if [ $INFER_EN_EXIT -eq 0 ]; then
    echo "--- Stage 3 Evaluation: English ---"

    python eval_two_stage.py \
        --config $CONFIG \
        --stage 3 \
        --inference_subdir librispeech

    EVAL_EN_EXIT=$?

    if [ $EVAL_EN_EXIT -ne 0 ]; then
        echo "Stage 3 English evaluation failed!"
        FAILED_STAGES="$FAILED_STAGES Eval-EN"
    else
        echo "Stage 3 English evaluation completed"
        if [ -f "inference/exp_K/stage3/librispeech/dwer_summary_whisper.json" ]; then
            echo ""
            echo "=== English Results (dWER) ==="
            cat inference/exp_K/stage3/librispeech/dwer_summary_whisper.json
            echo ""
            echo "Compare with:"
            echo "  Exp E Stage 3 (no ASR): ~96.21% dWER"
            echo "  Exp J Stage 2 (ASR):    ~3.02% dWER (but 275 bps)"
            echo "  Target:                 < 5% dWER @ 137.5 bps"
        fi
    fi
fi

echo ""

########################################
# Comprehensive Evaluation (MOS + PESQ + STOI)
########################################
echo "========================================================================"
echo "   Run comprehensive evaluation with metrics_evaluator_v3.py"
echo "========================================================================"
echo ""
echo "Chinese (AISHELL):"
echo "  python /work/u4162168/Codec_comparison/metrics_evaluator_v3.py \\"
echo "    --model_name FocalCodec-S \\"
echo "    --frequency 12.5Hz_137.5bps_expK \\"
echo "    --dataset_path /work/u4162168/ASR_Dataset/AISHELL-1/test \\"
echo "    --reconstructed_path inference/exp_K/stage3/aishell \\"
echo "    --dataset_name aishell \\"
echo "    --output_dir Evaluation_Result"
echo ""
echo "English (LibriSpeech):"
echo "  python /work/u4162168/Codec_comparison/metrics_evaluator_v3.py \\"
echo "    --model_name FocalCodec-S \\"
echo "    --frequency 12.5Hz_137.5bps_expK \\"
echo "    --dataset_path /work/u4162168/ASR_Dataset/LibriSpeech/test-clean \\"
echo "    --reconstructed_path inference/exp_K/stage3/librispeech \\"
echo "    --dataset_name librispeech \\"
echo "    --output_dir Evaluation_Result"

########################################
# Summary
########################################
echo ""
echo "========================================================================"
echo "   Experiment K - Stage 3 Results Summary"
echo "========================================================================"

if [ -z "$FAILED_STAGES" ]; then
    echo "All steps completed successfully!"
else
    echo "Failed steps:$FAILED_STAGES"
fi

echo ""
echo "Output files:"
echo "  Chinese: inference/exp_K/stage3/aishell/"
echo "  English: inference/exp_K/stage3/librispeech/"
echo ""
echo "Finished: $(date)"
echo "========================================================================"

exit 0
