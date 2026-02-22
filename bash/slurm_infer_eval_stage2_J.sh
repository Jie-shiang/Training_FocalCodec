#!/bin/bash
################################################################################
# Experiment J: Stage 2 Inference + Evaluation (Bilingual)
# Runs inference and evaluation on both Chinese and English test sets
# Model: Exp J with Whisper ASR Loss
################################################################################

#SBATCH --job-name=InferEval_J
#SBATCH --partition=dev
#SBATCH --account=MST114558
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=logs/exp_J/infer_eval_s2_%j.log
#SBATCH --error=logs/exp_J/infer_eval_s2_%j.err

# 載入環境
ml purge
ml load miniconda3/24.11.1
conda activate focalcodec

export PYTHONPATH=$PYTHONPATH:$(pwd)
export HF_HOME=/work/u4162168/model_cache
export TRANSFORMERS_CACHE=/work/u4162168/model_cache
export TOKENIZERS_PARALLELISM=false

# 創建輸出目錄
mkdir -p logs/exp_J
mkdir -p inference/exp_J/stage2/aishell
mkdir -p inference/exp_J/stage2/librispeech

CONFIG="config_exp_J.yaml"
FAILED_STAGES=""

echo "========================================================================"
echo "   Experiment J - Stage 2 Inference + Evaluation (Bilingual)"
echo "========================================================================"
echo "Model: Whisper ASR Loss (Feature:1.0, Mel:5.0, STFT:2.0, ASR:30.0)"
echo "Expected: Significant dCER/dWER reduction from ASR Loss"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "========================================================================"

# 確認 checkpoint
CHECKPOINT="output/exp_J/stage2_25hz/best_model.pt"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Please run training first: sbatch bash/slurm_train_stage2_J.sh"
    exit 1
fi
echo "Checkpoint: $CHECKPOINT"
echo "Checkpoint size: $(du -h $CHECKPOINT | cut -f1)"
echo ""

export CUDA_VISIBLE_DEVICES=0

########################################
# Chinese Inference (AISHELL)
########################################
echo "--- Stage 2 Inference: Chinese (AISHELL) ---"

python infer_two_stage.py \
    --config $CONFIG \
    --stage 2 \
    --max_samples 2000 \
    --csv_path csv/aishell_filtered_clean.csv \
    --output_subdir aishell

INFER_ZH_EXIT=$?

if [ $INFER_ZH_EXIT -ne 0 ]; then
    echo "Stage 2 Chinese inference failed!"
    FAILED_STAGES="$FAILED_STAGES Infer-ZH"
else
    echo "Stage 2 Chinese inference completed"
fi

echo ""

########################################
# English Inference (LibriSpeech)
########################################
echo "--- Stage 2 Inference: English (LibriSpeech) ---"

python infer_two_stage.py \
    --config $CONFIG \
    --stage 2 \
    --max_samples 2000 \
    --csv_path csv/librispeech_filtered_clean.csv \
    --output_subdir librispeech

INFER_EN_EXIT=$?

if [ $INFER_EN_EXIT -ne 0 ]; then
    echo "Stage 2 English inference failed!"
    FAILED_STAGES="$FAILED_STAGES Infer-EN"
else
    echo "Stage 2 English inference completed"
fi

echo ""

########################################
# Chinese Evaluation (dCER with Paraformer)
########################################
if [ $INFER_ZH_EXIT -eq 0 ]; then
    echo "--- Stage 2 Evaluation: Chinese (dCER) ---"

    python eval_two_stage.py \
        --config $CONFIG \
        --stage 2 \
        --inference_subdir aishell

    EVAL_ZH_EXIT=$?

    if [ $EVAL_ZH_EXIT -ne 0 ]; then
        echo "Stage 2 Chinese evaluation failed!"
        FAILED_STAGES="$FAILED_STAGES Eval-ZH"
    else
        echo "Stage 2 Chinese evaluation completed"
        if [ -f "inference/exp_J/stage2/aishell/dcer_summary_paraformer.json" ]; then
            echo ""
            echo "=== Chinese Results (dCER) ==="
            cat inference/exp_J/stage2/aishell/dcer_summary_paraformer.json
            echo ""
            echo "Compare with:"
            echo "  Exp G: ~17.39% dCER mean (baseline)"
            echo "  Exp I: ~16.56% dCER mean (loss rebalance)"
            echo "  Target: <10% dCER (ASR Loss expected improvement)"
        fi
    fi
fi

echo ""

########################################
# English Evaluation (dWER with Whisper)
########################################
if [ $INFER_EN_EXIT -eq 0 ]; then
    echo "--- Stage 2 Evaluation: English (dWER) ---"

    python eval_two_stage.py \
        --config $CONFIG \
        --stage 2 \
        --inference_subdir librispeech

    EVAL_EN_EXIT=$?

    if [ $EVAL_EN_EXIT -ne 0 ]; then
        echo "Stage 2 English evaluation failed!"
        FAILED_STAGES="$FAILED_STAGES Eval-EN"
    else
        echo "Stage 2 English evaluation completed"
        if [ -f "inference/exp_J/stage2/librispeech/dwer_summary_whisper.json" ]; then
            echo ""
            echo "=== English Results (dWER) ==="
            cat inference/exp_J/stage2/librispeech/dwer_summary_whisper.json
            echo ""
            echo "Compare with:"
            echo "  Exp G: ~7.04% dWER mean (baseline)"
            echo "  Exp I: ~6.89% dWER mean (loss rebalance)"
        fi
    fi
fi

echo ""

########################################
# Summary
########################################
echo "========================================================================"
echo "   Experiment J - Stage 2 Results Summary"
echo "========================================================================"

if [ -z "$FAILED_STAGES" ]; then
    echo "All steps completed successfully!"
else
    echo "Failed steps:$FAILED_STAGES"
fi

echo ""
echo "Output files:"
echo "  Chinese: inference/exp_J/stage2/aishell/"
echo "  English: inference/exp_J/stage2/librispeech/"
echo ""
echo "Comprehensive evaluation:"
echo "  python /work/u4162168/Codec_comparison/metrics_evaluator_v3.py \\"
echo "    --model_name FocalCodec-S \\"
echo "    --frequency 50Hz_2k_finetune_expJ \\"
echo "    --dataset_path /work/u4162168/ASR_Dataset/AISHELL-1/test \\"
echo "    --reconstructed_path inference/exp_J/stage2/aishell \\"
echo "    --dataset_name aishell \\"
echo "    --output_dir Evaluation_Result"
echo ""
echo "Finished: $(date)"
echo "========================================================================"

exit 0
