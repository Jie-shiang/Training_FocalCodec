# FocalCodec 25Hz 2048 Training

Train a 25Hz streaming semantic codec based on FocalCodec-S with 2048 codebook size (11-bit).

**Target:** 275 bps (25 Hz × 11 bits) streaming semantic codec with good ASR accuracy.

## Quick Start

```bash
# 1. Verify setup
python scripts/setup_check.py

# 2. Train
bash train.sh

# 3. Inference
bash infer.sh

# 4. Evaluate dCER
bash eval.sh
```

All configuration is in `config.yaml`.

---

## Installation

### Step 1: Clone FocalCodec Repository

```bash
cd /home/jieshiang/Desktop/GitHub/FocalCodec_main

# Clone official FocalCodec into focalcodec/ directory
git clone https://github.com/lucadellalib/focalcodec.git focalcodec

# Install FocalCodec
cd focalcodec && pip install -e . && cd ..
```

### Step 2: Download Pretrained Models

```bash
python scripts/download_models.py
```

Models are cached at `/mnt/Internal/jieshiang/Model/FocalCodec/`.

Required models:
- `lucadellalib/focalcodec_50hz_2k_causal` - Base model (50Hz causal)
- `lucadellalib/focalcodec_25hz` - Teacher model (25Hz non-causal)

### Step 3: Install Dependencies

```bash
# Create conda environment
conda create -n focalcodec python=3.10
conda activate focalcodec

# Install PyTorch
conda install pytorch torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install requirements
pip install -r requirements.txt
```

### Step 4: Verify Setup

```bash
python scripts/setup_check.py
```

This checks:
1. FocalCodec repository clone
2. Pretrained models downloaded
3. Python packages installed
4. Training data accessible
5. FocalCodec import works

---

## Configuration

All settings are in `config.yaml`:

```yaml
# Paths
paths:
  base_dir: "/home/jieshiang/Desktop/GitHub/FocalCodec_main"
  focalcodec_dir: "..."
  model_cache_dir: "/mnt/Internal/jieshiang/Model/FocalCodec"
  output_dir: "/mnt/Internal/jieshiang/Model/FocalCodec/25hz_2048"
  inference_dir: "/mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/25hz_2048"

# Data
data:
  audio_base_path: "/mnt/Internal/ASR"
  train_csv: "experiments/data_aishell/train_split.csv"
  val_csv: "experiments/data_aishell/val_split.csv"

# Model
model:
  codebook_size: 2048  # 2048=11-bit, 256=8-bit
  frame_rate: 25

# Training
training:
  stage: 1
  gpu_id: 1
  batch_size: 250
  num_epochs: 500
  learning_rate: 0.0005
  patience: 25
```

Edit `config.yaml` to customize your setup.

---

## Training

```bash
# Train with config.yaml settings
bash train.sh

# Override stage
bash train.sh --stage 2

# Resume from checkpoint
bash train.sh --resume
```

Or run Python directly:

```bash
python train_25hz_focalcodec.py
python train_25hz_focalcodec.py --stage 2 --resume
```

---

## Inference

```bash
# Inference with config.yaml settings
bash infer.sh

# Override settings
bash infer.sh --stage 1 --max_samples 500
```

Output: `{filename}_inference.wav` in inference directory.

---

## Evaluation

```bash
# Evaluate dCER
bash eval.sh

# Override settings
bash eval.sh --max_samples 500
```

**dCER = CER(reconstructed) - CER(original)**

- dCER < 0.1: Excellent
- dCER < 0.3: Good
- dCER < 0.5: Moderate
- dCER > 0.5: Poor

Results:
- `dcer_results.csv`: Per-sample results
- `dcer_summary.json`: Summary statistics

---

## Checkpoints

```bash
# Location of trained models
/mnt/Internal/jieshiang/Model/FocalCodec/25hz_2048/stage_1/
├── best_model.pt      # Best validation loss
└── last_model.pt      # Latest checkpoint (for resuming)
```

Checkpoint contents:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `scheduler_state_dict`: LR scheduler state
- `epoch`: Current epoch
- `val_loss`: Validation loss

---

## Project Structure

```
FocalCodec_main/
├── config.yaml                    # All configuration
├── train_25hz_focalcodec.py       # Training script
├── infer_25hz_focalcodec.py       # Inference script
├── eval_25hz_focalcodec.py        # Evaluation script
├── train.sh                       # Training launcher
├── infer.sh                       # Inference launcher
├── eval.sh                        # Evaluation launcher
├── requirements.txt               # Dependencies
├── README.md                      # This file
│
├── utils/
│   ├── __init__.py
│   └── config_loader.py           # Config loader
│
├── scripts/
│   ├── setup_check.py             # Setup verification
│   └── download_models.py         # Model downloader
│
├── experiments/
│   └── data_aishell/              # CSV files
│       ├── train_split.csv
│       └── val_split.csv
│
└── focalcodec/                    # FocalCodec source (git clone)
```

---

## Results

### 25Hz 2048 (Stage 1, 500 epochs)

| Metric | Value |
|--------|-------|
| dCER Mean | 0.4482 |
| dCER Std | 0.3096 |
| Bitrate | 275 bps |

---

## Troubleshooting

### CUDA Out of Memory
Edit `config.yaml`:
```yaml
training:
  batch_size: 128  # Reduce from 250
```

### Audio Not Found
Edit `config.yaml`:
```yaml
data:
  audio_base_path: "/your/path/to/ASR"
```

### FocalCodec Import Error
```bash
cd focalcodec && pip install -e .
```

---

## License

FocalCodec is licensed under the Apache 2.0 License.
