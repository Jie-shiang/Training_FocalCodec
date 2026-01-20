# Installation Guide

## Prerequisites

- Linux system (tested on Ubuntu)
- CUDA-capable GPU
- Conda or Miniconda
- Git

---

## Step 1: Clone This Repository

```bash
git clone https://github.com/YOUR_USERNAME/FocalCodec.git
cd FocalCodec
```

---

## Step 2: Clone FocalCodec Official Repository

The official FocalCodec source code is not included in this repository. You need to clone it separately:

```bash
# Clone into focalcodec/ directory (already in .gitignore)
git clone https://github.com/lucadellalib/focalcodec.git focalcodec
```

Or clone to a custom location and update `config.yaml`:

```bash
# Option 1: Clone to parent directory
cd ..
git clone https://github.com/lucadellalib/focalcodec.git FocalCodec_main/focalcodec
cd FocalCodec

# Then update config.yaml:
# focalcodec_dir: "/path/to/FocalCodec_main/focalcodec"
```

---

## Step 3: Create Conda Environment

```bash
# Create environment
conda create -n focalcodec python=3.10
conda activate focalcodec

# Install PyTorch with CUDA support
conda install pytorch torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install requirements
pip install -r requirements.txt

# Install FocalCodec
cd focalcodec
pip install -e .
cd ..
```

---

## Step 4: Prepare Dataset

This project uses AISHELL-1 dataset. Update `config.yaml` with your dataset paths:

```yaml
# Data paths
audio_base_path: "/path/to/your/ASR"  # Parent directory of aishell
train_csv: "experiments/data_aishell/train_split.csv"
val_csv: "experiments/data_aishell/val_split.csv"
```

Dataset structure:
```
/path/to/your/ASR/
└── aishell/
    └── data_aishell/
        └── wav/
            ├── train/
            ├── dev/
            └── test/
```

---

## Step 5: Update Configuration

Edit `config.yaml` to match your environment:

```yaml
# FocalCodec source directory
focalcodec_dir: "/path/to/focalcodec"  # Where you cloned official repo

# Model cache directory (for pretrained models)
model_cache_dir: "/path/to/model/cache"

# Output directories
output_dir: "/path/to/output"
inference_dir: "/path/to/inference"

# Data paths
audio_base_path: "/path/to/ASR"
train_csv: "experiments/data_aishell/train_split.csv"
val_csv: "experiments/data_aishell/val_split.csv"
```

---

## Step 6: Verify Setup

```bash
# Check if FocalCodec can be imported
python -c "from focalcodec import FocalCodec; print('✓ FocalCodec OK')"

# Check if training CSV exists
python -c "import pandas as pd; df = pd.read_csv('experiments/data_aishell/train_split.csv'); print(f'✓ Found {len(df)} training samples')"
```

---

## Step 7: Start Training

```bash
# Stage 1 (50Hz)
bash train.sh 1

# Inference
bash infer.sh 1

# Evaluation
bash eval.sh 1

# Stage 2 (25Hz) - requires Stage 1 completed
bash train.sh 2
bash infer.sh 2
bash eval.sh 2
```

---

## Troubleshooting

### FocalCodec Import Error

If you get `ModuleNotFoundError: No module named 'focalcodec'`:

```bash
# Make sure you installed it in editable mode
cd focalcodec
pip install -e .
cd ..
```

### Path Not Found Errors

Update all paths in `config.yaml` to match your environment.

### CUDA Out of Memory

Reduce batch size in `config.yaml`:

```yaml
stage1:
  batch_size: 8  # Reduce from 16
```

---

## Directory Structure After Setup

```
FocalCodec/
├── focalcodec/              # ← Clone official repo here (gitignored)
├── config.yaml              # ← Update paths here
├── train.sh
├── infer.sh
├── eval.sh
└── ...
```

---

## Quick Start

After setup, see [USAGE.md](USAGE.md) for common commands.
