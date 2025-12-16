# FocalCodec LoRA Fine-tuning

LoRA fine-tuning framework based on [FocalCodec](https://github.com/lucadellalib/focalcodec) with flexible component training and evaluation pipeline.

## Project Structure

```
FocalCodec/
├── focalcodec/                      # Original FocalCodec model library (git clone)
│   ├── focalcodec/
│   │   ├── codec.py
│   │   ├── vocos.py
│   │   └── ...
│   └── README.md
│
├── experiments/                     # Experiment results
│   ├── bs32_lr1e-04_ep300_*/       # Results for each experiment
│   │   ├── config_*.json
│   │   ├── experiment_results.csv
│   │   ├── commonvoice_detailed.csv
│   │   └── librispeech_detailed.csv
│   └── data_*/                     # Train/validation dataset splits
│
├── train_focalcodec.py             # Core training script
├── run_full_pipeline.py            # Full pipeline (training + evaluation)
├── simple_inference.py             # Inference script
├── experiments_lora_tuning.sh      # Batch experiment script
└── README.md                        # This document

External dependencies:
~/Desktop/GitHub/Codec_comparison/  # Evaluation tool (clone separately)
```

## Environment Setup

```bash
cd ~/Desktop/GitHub
git clone <your-repo-url> FocalCodec
cd FocalCodec

# Clone original FocalCodec model library
git clone https://github.com/lucadellalib/focalcodec.git focalcodec

# Create conda environment
conda create -n focalcodec python=3.10 -y
conda activate focalcodec

cd focalcodec
pip install -r requirements.txt
```

## Path Configuration

**Important:** Modify hardcoded paths in the following files before running:

| File | Lines | Content | Default Value |
|------|-------|---------|---------------|
| train_focalcodec.py | 30-36 | Training/validation CSV paths | `/home/jieshiang/Desktop/GitHub/FocalCodec/experiments/data_commonvoice/` |
| train_focalcodec.py | 34 | Model cache directory | `/mnt/Internal/jieshiang/Model/FocalCodec` |
| train_focalcodec.py | 35 | Default checkpoint directory | `/mnt/Internal/jieshiang/Model/FocalCodec/checkpoints_focalcodec` |
| train_focalcodec.py | 36 | Log directory | `/mnt/Internal/jieshiang/Model/FocalCodec/logs` |
| train_focalcodec.py | 73 | Base audio directory | `/mnt/Internal/ASR` |
| run_full_pipeline.py | 145-149 | Checkpoint directory | `/mnt/Internal/jieshiang/Model/FocalCodec/checkpoints_{exp_name}` |
| run_full_pipeline.py | 148 | Inference output directory | `/mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50Hz_2K_finetuned/{exp_name}` |
| run_full_pipeline.py | 183, 255-262, 275-282 | Dataset CSV paths, model cache, base audio directory | Various paths under `/home/jieshiang/` and `/mnt/Internal/` |
| experiments_lora_tuning.sh | 24, 27 | GPU ID, Conda environments | GPU 3, `focalcodec`, `codec_eval` |
| experiments_lora_tuning.sh | 27 | Codec_comparison path | `/home/jieshiang/Desktop/GitHub/Codec_comparison` |
| experiments_full_training.sh | 7-10 | GPU ID, Conda environments, Codec_comparison path | GPU 3, paths under `/home/jieshiang/` |
| experiments_overfitting.sh | 9-10 | GPU ID, base directory | GPU 2, `/home/jieshiang/Desktop/GitHub/FocalCodec` |
| simple_inference.py | 86 | Model cache directory (default) | `/mnt/Internal/jieshiang/Model/FocalCodec` |

## Prepare Dataset

Create training/validation CSV files:

```csv
file_name,file_path,duration
sample1,./CommonVoice/clips/sample1.mp3,3.5
sample2,./LibriSpeech/train/sample2.flac,4.2
```

Place CSV files at:
- `experiments/data_commonvoice/train_split.csv`
- `experiments/data_commonvoice/val_split.csv`

## Start Training

Basic training (Decoder only with LoRA):

```bash
conda activate focalcodec

python train_focalcodec.py \
    --freeze_encoder \
    --freeze_compressor \
    --freeze_decompressor \
    --train_decoder \
    --use_lora \
    --lora_decoder \
    --lora_rank 16 \
    --lora_alpha 32 \
    --use_feature_loss \
    --use_mel_loss \
    --weight_feature 1.0 \
    --weight_mel 0.15 \
    --num_epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --early_stopping \
    --patience 20 \
    --gpu_id 0
```

Full pipeline (training + inference + evaluation):

```bash
python run_full_pipeline.py \
    --freeze_encoder --freeze_compressor --freeze_decompressor --train_decoder \
    --use_lora --lora_decoder --lora_rank 16 --lora_alpha 32 \
    --use_feature_loss --use_mel_loss --weight_feature 1.0 --weight_mel 0.15 \
    --num_epochs 50 --batch_size 32 --learning_rate 1e-4 \
    --early_stopping --patience 20 --overlap 0.5 --gpu_id 0 \
    --train_env focalcodec --eval_env codec_eval \
    --codec_comparison_dir /path/to/Codec_comparison \
    --cleanup_inference --cleanup_codec_comparison --cleanup_best_model
```

## Configuration Options

### Component Training Control

FocalCodec has 4 main components:

| Component | Function | Train Flag | Freeze Flag |
|-----------|----------|------------|-------------|
| Encoder | Audio to features | --train_encoder | --freeze_encoder |
| Compressor | Feature compression | --train_compressor | --freeze_compressor |
| Decompressor | Feature decompression | --train_decompressor | --freeze_decompressor |
| Decoder | Features to audio | --train_decoder | --freeze_decoder |

### LoRA Configuration

```bash
--use_lora              # Enable LoRA
--lora_rank 16          # Rank: 4, 8, 16, 32, 64 (recommended: 16)
--lora_alpha 32         # Alpha: rank x 2 (recommended)
--lora_decoder          # Apply LoRA to Decoder
```

### Loss Configuration

```bash
--use_feature_loss --weight_feature 1.0     # Feature Loss (required)
--use_mel_loss --weight_mel 0.15            # Mel Loss (recommended)
--use_time_loss --weight_time 0.3           # Time Loss (optional)
```

### Training Parameters

```bash
--num_epochs 100        # Training epochs
--batch_size 32         # Batch size (8, 16, 32, 64)
--learning_rate 1e-4    # Learning rate (1e-3 to 1e-6)
--early_stopping        # Enable early stopping
--patience 20           # Early stopping patience
```

## Experiment Results

Each experiment generates:

```
experiments/bs32_lr1e-04_ep50_dec_feat+mel_lora16_YYYYMMDD_HHMMSS/
├── config_*.json                   # Experiment configuration
├── experiment_results.csv          # Evaluation results summary
├── commonvoice_detailed.csv        # Chinese detailed results
├── commonvoice_summary.csv         # Chinese statistics summary
├── librispeech_detailed.csv        # English detailed results
└── librispeech_summary.csv         # English statistics summary
```


## References

- [FocalCodec Paper](https://arxiv.org/abs/2502.04465)
- [FocalCodec GitHub](https://github.com/lucadellalib/focalcodec)
- [PEFT Library](https://github.com/huggingface/peft)