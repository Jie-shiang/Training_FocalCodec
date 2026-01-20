# FocalCodec Two-Stage Training (25Hz 2048)

Train a 25Hz streaming semantic codec based on FocalCodec-S with 2048 codebook size (11-bit).

**Target:** 275 bps (25 Hz × 11 bits) streaming semantic codec with good ASR accuracy.

> **First time setup?** See [INSTALLATION.md](INSTALLATION.md) for complete installation guide.

## Quick Start

```bash
# 1. Train Stage 1 (50Hz)
bash train.sh 1

# 2. Inference Stage 1
bash infer.sh 1

# 3. Evaluate Stage 1
bash eval.sh 1

# 4. Train Stage 2 (25Hz) - requires Stage 1 done
bash train.sh 2

# 5. Inference & Evaluate Stage 2
bash infer.sh 2
bash eval.sh 2
```

All configuration is in `config.yaml`.

---

## Two-Stage Training Strategy

### Stage 1: 50Hz Causal 2k (550 bps)

**Training Mode: decompressor_only (推薦)**

- **訓練**: Decompressor only
- **凍結**: Encoder + Compressor + Decoder
- **優點**: 保持 Compressor 的 pretrained 能力，不會被破壞
- **目標**: dCER < 10% (原始結果: 7%)

```bash
# Default mode: decompressor_only (recommended)
bash train.sh 1

# Alternative: Train with STE (may damage Compressor)
bash train.sh 1 both_ste
```

**為什麼推薦 decompressor_only?**
- 原始 7% dCER 結果使用此策略
- 使用 STE 同時訓練 Compressor 會破壞其 pretrained 能力
- 實驗顯示 STE 訓練導致 dCER 20%+，部分樣本輸出英文而非中文

### Stage 2: 25Hz Causal 2k (275 bps)

**基於 Stage 1 添加第 4 層 FocalNet**

- **訓練**: 新增的第 4 層 Compressor/Decompressor
- **凍結**: Encoder + 前 3 層 + Decoder
- **目標**: dCER < 15%

```bash
# Requires Stage 1 completed
bash train.sh 2
```

---

## Installation

### Step 1: Clone FocalCodec Repository

```bash
cd /home/jieshiang/Desktop/GitHub/FocalCodec

# Clone official FocalCodec (if not already done)
# Already cloned at: /home/jieshiang/Desktop/GitHub/FocalCodec_main/focalcodec
```

### Step 2: Install Dependencies

```bash
# Create conda environment
conda create -n focalcodec python=3.10
conda activate focalcodec

# Install PyTorch
conda install pytorch torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install requirements
pip install -r requirements.txt

# Install FocalCodec
cd /home/jieshiang/Desktop/GitHub/FocalCodec_main/focalcodec
pip install -e .
```

### Step 3: Download Pretrained Models

Models are cached at `/mnt/Internal/jieshiang/Model/FocalCodec/`.

Required model:
- `lucadellalib/focalcodec_50hz_2k_causal` - Base model

The model will be downloaded automatically on first training run.

---

## Configuration

All settings are in `config.yaml`:

```yaml
# Paths
focalcodec_dir: "/home/jieshiang/Desktop/GitHub/FocalCodec_main/focalcodec"
model_cache_dir: "/mnt/Internal/jieshiang/Model/FocalCodec"
base_model: "lucadellalib/focalcodec_50hz_2k_causal"

# Output directories
output_dir: "/mnt/Internal/jieshiang/Model/FocalCodec/two_stage"
inference_dir: "/mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/two_stage"

# Data
audio_base_path: "/mnt/Internal/ASR"
train_csv: "experiments/data_aishell/train_split.csv"
val_csv: "experiments/data_aishell/val_split.csv"

# Stage 1: 50Hz
stage1:
  gpu_id: 1
  batch_size: 16
  num_epochs: 100
  learning_rate: 0.0005
  weight_feature: 1.0

# Stage 2: 25Hz
stage2:
  gpu_id: 1
  batch_size: 16
  num_epochs: 100
  learning_rate: 0.0001
```

---

## Training

### Stage 1 (50Hz)

```bash
# Recommended: Only train Decompressor
bash train.sh 1

# Alternative modes
bash train.sh 1 decompressor_only  # Same as default
bash train.sh 1 both              # Train both (no STE)
bash train.sh 1 both_ste          # Train with STE (不推薦)
```

**Training Modes:**

| Mode | Compressor | Decompressor | Gradient Flow | Result |
|------|------------|--------------|---------------|--------|
| decompressor_only | 凍結 | 訓練 | Only Decompressor | ~7% dCER ✅ |
| both | 訓練 | 訓練 | Only Decompressor | Same as decompressor_only |
| both_ste | 訓練 | 訓練 | Both (with STE) | ~20% dCER ❌ |

### Stage 2 (25Hz)

```bash
# Requires Stage 1 completed
bash train.sh 2
```

---

## Inference

```bash
# Stage 1 inference
bash infer.sh 1

# Stage 2 inference
bash infer.sh 2

# Override checkpoint
bash infer.sh 1 --checkpoint last_model.pt

# Limit samples
bash infer.sh 1 --max_samples 500
```

Output: `{filename}_inference.wav` in inference directory.

---

## Evaluation

```bash
# Stage 1 evaluation
bash eval.sh 1

# Stage 2 evaluation
bash eval.sh 2

# Use Whisper ASR (default: Paraformer)
bash eval.sh 1 whisper

# Limit samples
bash eval.sh 1 paraformer 500
```

**dCER = CER(reconstructed) - CER(original)**

- dCER < 10%: Excellent
- dCER < 20%: Good
- dCER < 50%: Moderate
- dCER > 50%: Poor

Results:
- `dcer_results_paraformer.csv`: Per-sample results
- `dcer_summary_paraformer.json`: Summary statistics

---

## Project Structure

```
FocalCodec/
├── config.yaml                    # All configuration
├── train.sh                       # Training launcher (Stage 1/2)
├── infer.sh                       # Inference launcher
├── eval.sh                        # Evaluation launcher
├── README.md                      # This file
│
├── train_stage1_50hz.py           # Stage 1 training script
├── train_stage2_25hz.py           # Stage 2 training script
├── infer_two_stage.py             # Inference script
├── eval_two_stage.py              # Evaluation script
│
├── train_stage1.sh                # Stage 1 internal script
├── train_stage2.sh                # Stage 2 internal script
│
├── utils/
│   ├── __init__.py
│   ├── config_loader.py           # Config loader
│   ├── dataset.py                 # Dataset class
│   └── asr_evaluator.py           # ASR evaluator
│
└── experiments/
    └── data_aishell/              # CSV files
        ├── train_split.csv
        └── val_split.csv
```

---

## Checkpoints

### Stage 1 (50Hz)
```bash
/mnt/Internal/jieshiang/Model/FocalCodec/two_stage/stage1_50hz/
├── best_model.pt      # Best validation loss
└── last_model.pt      # Latest checkpoint
```

### Stage 2 (25Hz)
```bash
/mnt/Internal/jieshiang/Model/FocalCodec/two_stage/stage2_25hz/
├── best_model.pt      # Best validation loss
└── last_model.pt      # Latest checkpoint
```

Checkpoint contents:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `epoch`: Current epoch
- `val_loss`: Validation loss
- `metrics`: Training/validation metrics

---

## Results

### Stage 1 (50Hz, decompressor_only mode)

| Metric | Value |
|--------|-------|
| Frame Rate | 50 Hz |
| Codebook | 2048 (11-bit) |
| Bitrate | 550 bps |
| dCER Mean | ~7% (expected) |

### Stage 2 (25Hz)

| Metric | Value |
|--------|-------|
| Frame Rate | 25 Hz |
| Codebook | 2048 (11-bit) |
| Bitrate | 275 bps |
| dCER Mean | ~15% (expected) |

---

## Troubleshooting

### CUDA Out of Memory
Edit `config.yaml`:
```yaml
stage1:
  batch_size: 8  # Reduce from 16
```

### Audio Not Found
Edit `config.yaml`:
```yaml
audio_base_path: "/your/path/to/ASR"
```

### FocalCodec Import Error
```bash
cd /home/jieshiang/Desktop/GitHub/FocalCodec_main/focalcodec
pip install -e .
```

### Stage 2 Fails
Make sure Stage 1 is completed:
```bash
ls -la /mnt/Internal/jieshiang/Model/FocalCodec/two_stage/stage1_50hz/best_model.pt
```

### High dCER in Stage 1 (> 20%)
- Make sure you're using `decompressor_only` mode (default)
- Check if you accidentally used `both_ste` mode
- Re-train with: `bash train.sh 1 decompressor_only`

---

## API Reference

### Official FocalCodec API

```python
from focalcodec import FocalCodec

model = FocalCodec.from_pretrained("lucadellalib/focalcodec_50hz_2k_causal")

# Forward pass
feats = model.sig_to_feats(audio)          # Encoder
lats = model.feats_to_lats(feats)          # Compressor
codes = model.lats_to_codes(lats)          # Quantizer
qfeats = model.codes_to_qfeats(codes)      # Decompressor
reconstructed = model.feats_to_sig(qfeats) # Decoder
```

### Training Loss

```python
# Feature Loss (官方方法)
feature_loss = MSE(qfeats, feats)  # MSE between decompressor output and encoder features
```

---

## License

FocalCodec is licensed under the Apache 2.0 License.
