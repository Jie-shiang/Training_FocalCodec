# FocalCodec-S Bilingual Fine-tuning

Fine-tuning [FocalCodec-S](https://github.com/lucadellalib/focalcodec) for bilingual (Chinese/English) semantic speech codec with Whisper ASR Loss supervision.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](focalcodec/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Demo Page](https://img.shields.io/badge/Demo-Audio%20Samples-blueviolet)](https://jie-shiang.github.io/FocalCodec_Release/demo/)

> **Audio Demo:** [https://jie-shiang.github.io/FocalCodec_Release/demo/](https://jie-shiang.github.io/FocalCodec_Release/demo/)

---

## Overview

FocalCodec-S is a streaming semantic speech codec. This project fine-tunes it for bilingual (Chinese + English) support using a progressive 3-stage training pipeline, culminating in Whisper Token Cross-Entropy Loss to directly constrain semantic intelligibility.

**Best results (Exp J, Stage 2, 25Hz / 275 bps):**
- Chinese (AISHELL-1): **dCER = 4.15%**
- English (LibriSpeech): **dWER = 3.02%**

**Ultra-low bitrate (Exp K, Stage 3, 12.5Hz / 137.5 bps):**
- Chinese: **dCER = 16.03%**
- English: **dWER = 10.01%**

---

## Architecture

FocalCodec-S uses progressive compression with causal FocalNet layers:

| Stage | Frame Rate | Bitrate | Compressor Layers | Decompressor Layers |
|:-----:|:----------:|:-------:|:-----------------:|:-------------------:|
| **1** | 50 Hz | 550 bps | 3 | 3 |
| **2** | 25 Hz | 275 bps | 4 (+1) | 4 (+1) |
| **3** | 12.5 Hz | 137.5 bps | 5 (+1) | 5 (+1) |

- **Encoder**: WavLM (frozen)
- **Compressor/Decompressor**: Causal FocalNet layers (trainable)
- **Quantizer**: Binary Scalar Quantization (BSQ, 2048-entry codebook)
- **Decoder**: WaveNeXt (frozen)

Each stage adds one new 2× compression layer. Training uses a dual-LR strategy: new layers use a higher LR (5e-4), inherited layers use a lower LR (5e-5).

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- SLURM (optional, for HPC cluster)

### Setup

```bash
# 1. Clone this repository
git clone <repository-url>
cd FocalCodec_Release

# 2. Clone and install FocalCodec (the core model package — NOT included in this repo)
git clone https://github.com/lucadellalib/focalcodec.git
cd focalcodec
pip install -e .
cd ..

# 3. Install training dependencies
pip install -r requirements.txt

# 4. Set environment variables (adjust paths to your environment)
export HF_HOME=/path/to/your/model_cache
export TRANSFORMERS_CACHE=/path/to/your/model_cache
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

> **Pretrained model:** The training scripts automatically download
> `lucadellalib/focalcodec_50hz_2k_causal` from HuggingFace on first run.
> Make sure `HF_HOME` points to a location with sufficient disk space (~2 GB).

### Dataset Preparation

You need to prepare CSV files with two columns: `filepath` and `duration`.

```
filepath,duration
/path/to/audio1.wav,3.21
/path/to/audio2.wav,4.87
```

**Recommended datasets:**
- **AISHELL-1** (Chinese): ~170h train, freely available
- **LibriSpeech train-clean-100** (English): ~100h train

Bilingual training (Exp G/H/J/K) used ~50–100h of mixed Chinese + English data.

---

## Quick Start

### Step 1: Configure paths

Copy and edit a config file:

```bash
cp configs/config_example_stage2_asr.yaml configs/my_config.yaml
# Edit all <YOUR_...> placeholders in my_config.yaml
```

The config contains clear comments on every field. The most important ones to change:

```yaml
paths:
  base_dir: "/your/project/root"
  model_cache_dir: "/your/model/cache"
  output_dir: "/your/project/root/output"

data:
  audio_base_path: "/your/dataset/root"
  train_csv: "/your/project/root/data/train_split.csv"
  val_csv: "/your/project/root/data/val_split.csv"
```

### Step 2: Train

**Option A — Stage 1 + Stage 2 from scratch (standard fine-tuning):**

```bash
# Stage 1: 50Hz (550 bps)
python train_stage1_50hz.py --config configs/my_config.yaml

# Stage 2: 25Hz (275 bps) — loads Stage 1 checkpoint automatically
python train_stage2_25hz.py --config configs/my_config.yaml
```

**Option B — Stage 2 with Whisper ASR Loss (continue from existing checkpoint):**

```bash
# Set source_experiment in config to point to your Stage 2 checkpoint
python train_stage3_asr.py --config configs/my_config.yaml
```

**Option C — Stage 3 with ASR Loss (continue from Stage 2 ASR checkpoint):**

```bash
# Single GPU
python train_stage3_asr.py --config configs/my_config.yaml

# Multi-GPU (DDP)
python train_stage3_asr_ddp.py --config configs/my_config.yaml
```

**SLURM (HPC):** Example scripts are in `bash/`. Edit the `#SBATCH` directives and paths, then:

```bash
sbatch bash/slurm_train_stage3_K.sh
```

### Step 3: Inference

```bash
# Stage 1/2 inference
python infer_two_stage.py \
    --config configs/my_config.yaml \
    --stage 2 \
    --max_samples 2000 \
    --csv_path /path/to/test.csv \
    --output_subdir test_set_name

# Stage 3 inference
python infer_stage3.py \
    --config configs/my_config.yaml \
    --max_samples 2000 \
    --csv_path /path/to/test.csv \
    --output_subdir test_set_name
```

### Step 4: Evaluate

```bash
# dCER (Chinese) or dWER (English)
python eval_two_stage.py \
    --config configs/my_config.yaml \
    --stage 2 \
    --inference_subdir test_set_name

# Stage 3 evaluation
python eval_stage3.py \
    --config configs/my_config.yaml \
    --inference_subdir test_set_name
```

---

## Project Structure

```
FocalCodec_Release/
├── README.md
├── requirements.txt
│
├── configs/                                  # Example configurations
│   ├── config_example_stage2_finetune.yaml  # Stage 1→2 standard fine-tuning (Exp E style)
│   ├── config_example_stage2_asr.yaml       # Stage 2 + Whisper ASR Loss (Exp J style)
│   └── config_example_stage3_asr.yaml       # Stage 3 + Whisper ASR Loss (Exp K style)
│
├── train_stage1_50hz.py     # Train Stage 1 (50Hz) from pretrained FocalCodec-S
├── train_stage2_25hz.py     # Train Stage 2 (25Hz) from Stage 1 checkpoint
├── train_stage3_asr.py      # Train Stage 2 or 3 with Whisper ASR Loss (single GPU)
├── train_stage3_asr_ddp.py  # Train Stage 3 with Whisper ASR Loss (multi-GPU DDP)
│
├── infer_two_stage.py       # Inference for Stage 1/2
├── infer_stage3.py          # Inference for Stage 3
├── eval_two_stage.py        # Evaluation (dCER/dWER) for Stage 1/2
├── eval_stage3.py           # Evaluation for Stage 3
│
├── audio_losses.py          # Mel + STFT loss implementations
├── whisper_asr_loss.py      # Whisper Token Cross-Entropy Loss
│
├── utils/
│   ├── __init__.py
│   └── config_loader.py     # YAML config loading utility
│
├── bash/                    # Example SLURM scripts (Exp J/K)
│   ├── slurm_infer_eval_stage2_J.sh
│   ├── slurm_train_stage3_K.sh
│   ├── slurm_train_stage3_K_ddp.sh
│   └── slurm_infer_eval_stage3_K.sh
│
└── focalcodec/              # FocalCodec-S source (from lucadellalib/focalcodec)
    ├── focalcodec/          # Core model code
    ├── setup.py
    └── ...
```

---

## Experiment Results

### Experiment Roadmap

```
Pretrained FocalCodec-S (50Hz)
    │
    ├─ Exp A: Feature Loss only → poor audio quality, high dCER
    │
    ├─ Exp D/E: + Mel + STFT Loss (1:10:5 / 1:5:2)
    │       Stage 1 dCER ≈ 3-4% (AISHELL overfitting test)
    │
    └─ Full Training (AISHELL-1 ~25h Chinese)
           │
           ├─ Exp E (decompressor_only):  Stage2 dCER=13.33%, dWER=30.57%
           └─ Exp F (both_ste):           Stage2 dCER=14.47%, dWER=27.68%
                  │
                  ├─ Problem: Stage 2/3 new layers never seen English → dWER explodes
                  │
                  └─ Bilingual Training (Chinese + English mixed)
                         │
                         ├─ Exp G: 50h bilingual (from Exp E S1)
                         │       dCER=17.39%, dWER=10.10%
                         │
                         ├─ Exp H: 100h bilingual (from Exp G)
                         │       dCER=16.42%, dWER=11.78%
                         │
                         ├─ Exp I: 100h + adjusted loss weights (from Exp G)
                         │       dCER=16.56%, dWER=11.34%
                         │       → Hit the ceiling of Feature/Mel/STFT loss
                         │
                         └─ Whisper ASR Loss (ICASSP 2026)
                                │
                                ├─ Exp J (Stage 2, 275 bps):
                                │   dCER=4.15%, dWER=3.02%  ← BEST semantics
                                │   MOS_Q=1.59 (audio quality trade-off)
                                │
                                └─ Exp K (Stage 3, 137.5 bps):
                                    dCER=16.03%, dWER=10.01%
                                    MOS_Q=1.33
```

### Full Results Table

#### AISHELL-1 Test (Chinese, 2000 utterances)

| Exp | Stage | Freq | Bitrate | dCER ↓ | MOS_Q ↑ | PESQ ↑ | STOI ↑ | Notes |
|:---:|:-----:|:----:|:-------:|:------:|:-------:|:------:|:------:|:------|
| Pretrained | - | 50Hz | 550 bps | 25.29% | 2.781 | 1.154 | 0.685 | Baseline (no fine-tuning) |
| **E** | 1 | 50Hz | 550 bps | **6.17%** | 2.961 | 1.259 | 0.726 | decompressor_only, CN 25h |
| F | 1 | 50Hz | 550 bps | 7.51% | 2.893 | 1.312 | 0.738 | both_ste, CN 25h |
| **E** | 2 | 25Hz | 275 bps | **13.33%** | 2.739 | 1.223 | 0.701 | |
| F | 2 | 25Hz | 275 bps | 14.47% | 2.681 | 1.235 | 0.699 | |
| G | 2 | 25Hz | 275 bps | 17.39% | 2.651 | 1.203 | 0.691 | Bilingual 50h |
| H | 2 | 25Hz | 275 bps | 16.42% | 2.707 | 1.209 | 0.695 | Bilingual 100h |
| I | 2 | 25Hz | 275 bps | 16.56% | **2.954** | 1.193 | 0.688 | Bilingual 100h + adjusted loss |
| **J** | 2 | 25Hz | 275 bps | **4.15%** | 1.589 | 1.091 | 0.601 | + Whisper ASR Loss (λ=30) |
| E | 3 | 12.5Hz | 137.5 bps | 43.60% | 2.421 | 1.159 | 0.634 | Stage 3, no bilingual |
| **K** | 3 | 12.5Hz | 137.5 bps | **16.03%** | 1.330 | 1.066 | 0.388 | + Whisper ASR Loss (λ=20) |

#### LibriSpeech Test-Clean (English, 2000 utterances)

| Exp | Stage | Freq | Bitrate | dWER ↓ | MOS_Q ↑ | PESQ ↑ | STOI ↑ | Notes |
|:---:|:-----:|:----:|:-------:|:------:|:-------:|:------:|:------:|:------|
| Pretrained | - | 50Hz | 550 bps | 1.64% | 3.706 | 1.241 | 0.804 | Baseline |
| E | 1 | 50Hz | 550 bps | 3.87% | 3.319 | 1.165 | 0.749 | CN-only training |
| E | 2 | 25Hz | 275 bps | 30.57% | 2.953 | 1.118 | 0.673 | English severely degraded |
| G | 2 | 25Hz | 275 bps | 10.10% | 3.021 | 1.166 | 0.742 | Bilingual 50h recovers EN |
| H | 2 | 25Hz | 275 bps | 11.78% | 3.068 | 1.170 | 0.744 | |
| I | 2 | 25Hz | 275 bps | 11.34% | **3.307** | 1.162 | 0.739 | |
| **J** | 2 | 25Hz | 275 bps | **3.02%** | 1.998 | 1.098 | 0.668 | + Whisper ASR Loss |
| E | 3 | 12.5Hz | 137.5 bps | 96.21% | 2.507 | 1.083 | 0.574 | Stage 3, completely broken |
| **K** | 3 | 12.5Hz | 137.5 bps | **10.01%** | 1.437 | 1.059 | 0.409 | + Whisper ASR Loss |

### Key Takeaways

1. **Feature/Mel/STFT loss alone hits a ceiling** at ~16% dCER / ~11% dWER regardless of more data or weight tuning (Exp G/H/I).
2. **Whisper ASR Loss breaks the ceiling**: Exp J reduces dCER from 16% → **4.15%** and dWER from 11% → **3.02%**, but at the cost of audio quality (MOS_Q: 2.95 → 1.59).
3. **Bilingual data is essential**: Training on Chinese only causes English dWER to explode (30% at Stage 2, 96% at Stage 3). Mixing ~50h EN + 50h CN significantly recovers both.
4. **Stage 3 compression costs**: Going from 275 bps (Stage 2) to 137.5 bps (Stage 3) adds ~12pp dCER and ~7pp dWER even with ASR loss.

---

## Loss Configuration

### Standard Fine-tuning (Exp E/G/H/I style)

```yaml
weight_feature: 1.0   # WavLM feature MSE — semantic preservation anchor
weight_mel: 5.0       # Mel-spectrogram loss — timbre & prosody
weight_stft: 2.0      # Multi-resolution STFT — high-frequency details
```

### ASR Loss Fine-tuning (Exp J/K style)

```yaml
weight_feature: 1.0
weight_mel: 5.0
weight_stft: 2.0
use_asr_loss: true
weight_asr: 30.0      # Stage 2; use 20.0 for Stage 3 to preserve more audio quality
whisper_model: "small"
```

Increasing `weight_asr` improves semantic metrics but degrades audio quality. The recommended range is 20–30.

---

## Training Modes

### Mode 1: Full training from pretrained (Stage 1 → 2 → 3)

Suitable for adapting to a new language or dataset from scratch.

```bash
python train_stage1_50hz.py --config configs/config_example_stage2_finetune.yaml
python train_stage2_25hz.py --config configs/config_example_stage2_finetune.yaml
```

### Mode 2: Continual training with ASR Loss (from existing Stage 2 checkpoint)

The most effective approach. Requires an existing Stage 2 checkpoint.

```bash
# Edit source_experiment in config, then:
python train_stage3_asr.py --config configs/config_example_stage2_asr.yaml
```

### Mode 3: Stage 3 training (ultra-low bitrate)

Continue from a Stage 2 ASR checkpoint, adding the 5th compression layer.

```bash
python train_stage3_asr.py --config configs/config_example_stage3_asr.yaml
# or multi-GPU:
python train_stage3_asr_ddp.py --config configs/config_example_stage3_asr.yaml
```

---

## Hardware Requirements

| Task | GPU | VRAM | Notes |
|:-----|:----|:-----|:------|
| Stage 1/2 training (batch=256) | H100 / A100 | 40–80 GB | |
| Stage 2 ASR training (batch=128) | H100 / A100 | 40–80 GB | Whisper-small adds ~8GB |
| Stage 3 DDP (batch=104 × 2 GPU) | 2× H100 | 2× 80 GB | |
| Inference | Any CUDA GPU | 8 GB+ | |

---

## Troubleshooting

**CUDA Out of Memory:**
```yaml
stage2:
  batch_size: 64   # halve it; effective LR stays the same with plateau scheduler
```

**Training diverges (NaN loss):**
```yaml
stage2:
  learning_rate: 1.0e-05   # reduce LR
  gradient_clip: 1.0        # tighten gradient clipping
```

**Poor English performance after Stage 2:**
- This is expected if trained on Chinese-only data.
- Use bilingual data (e.g., 50h CN + 50h EN) when training Stage 2.

**Checkpoint not found:**
```bash
# Verify checkpoint location — the training script saves to:
ls output/<experiment_name>/stage2_25hz/best_model.pt
ls output/<experiment_name>/stage3_12.5hz/best_model.pt
```

---

## Future Work

1. **Improve Stage 3 (12.5Hz) audio quality** — current MOS_Q ~1.3 is low; exploring better loss balancing or vocoder post-processing.
2. **Add prosody/timbre information to Exp J** — simulate full TTS-style pipeline with pitch and speaker conditioning.
3. **Reduce ASR loss quality trade-off** — investigate curriculum scheduling of `weight_asr` instead of a fixed value.

---

## Citation

If you use this work, please cite the original FocalCodec paper:

```bibtex
@article{focalcodec2024,
  title={FocalCodec: Low-Bitrate Speech Coding via Focal Modulation Networks},
  author={Della Libera, Luca and others},
  journal={arXiv preprint arXiv:2407.12753},
  year={2024}
}
```

If you use the ASR Loss approach, also cite:

```bibtex
@inproceedings{lmloss2026,
  title={From Hallucination to Articulation: Scaling Speech Language Models with ASR Loss},
  booktitle={ICASSP 2026},
  year={2026}
}
```

Original FocalCodec repository: [lucadellalib/focalcodec](https://github.com/lucadellalib/focalcodec)

---

## License

This project follows FocalCodec's Apache 2.0 License. See [focalcodec/LICENSE](focalcodec/LICENSE) for details.

---

## Acknowledgments

- FocalCodec team (Luca Della Libera et al.) for the original implementation
- AISHELL-1 and LibriSpeech for open speech datasets
- NCHC (National Center for High-performance Computing, Taiwan) for computational resources
- BIIC Lab, NTHU for project support
