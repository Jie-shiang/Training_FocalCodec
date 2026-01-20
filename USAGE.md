# FocalCodec Two-Stage Training - Quick Usage Guide

## 快速開始 (3 步驟)

```bash
# 1. 訓練 Stage 1 (50Hz, 預設使用 decompressor_only 模式)
bash train.sh 1

# 2. 推論 Stage 1
bash infer.sh 1

# 3. 評估 Stage 1
bash eval.sh 1
```

如果 Stage 1 的 dCER < 10%，可以繼續 Stage 2：

```bash
# 4. 訓練 Stage 2 (25Hz)
bash train.sh 2

# 5. 推論 & 評估 Stage 2
bash infer.sh 2
bash eval.sh 2
```

---

## 訓練模式說明 (Stage 1)

### decompressor_only (預設，推薦) ✅

```bash
bash train.sh 1
```

- **訓練**: Decompressor only
- **凍結**: Encoder + Compressor + Decoder
- **優點**: 保持 Compressor 的 pretrained 能力
- **結果**: ~7% dCER

### both_ste (不推薦) ❌

```bash
bash train.sh 1 both_ste
```

- **訓練**: Compressor + Decompressor (with STE)
- **凍結**: Encoder + Decoder
- **問題**: 破壞 Compressor 的 pretrained 能力
- **結果**: ~20% dCER (部分樣本輸出英文!)

---

## 常用命令

### 訓練

```bash
# Stage 1 (推薦模式)
bash train.sh 1

# Stage 2
bash train.sh 2
```

### 推論

```bash
# Stage 1 推論 (前 2000 個樣本)
bash infer.sh 1

# Stage 2 推論
bash infer.sh 2

# 指定推論樣本數
bash infer.sh 1 --max_samples 500
```

### 評估

```bash
# Stage 1 評估 (使用 Paraformer ASR)
bash eval.sh 1

# 使用 Whisper ASR
bash eval.sh 1 whisper

# 限制評估樣本數
bash eval.sh 1 paraformer 500
```

---

## 檢查點位置

```bash
# Stage 1 檢查點
/mnt/Internal/jieshiang/Model/FocalCodec/two_stage/stage1_50hz/
├── best_model.pt      # 最佳驗證損失
└── last_model.pt      # 最新檢查點

# Stage 2 檢查點
/mnt/Internal/jieshiang/Model/FocalCodec/two_stage/stage2_25hz/
├── best_model.pt
└── last_model.pt
```

---

## 推論輸出位置

```bash
# Stage 1 推論輸出
/mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/two_stage/stage1/
└── {filename}_inference.wav

# Stage 2 推論輸出
/mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/two_stage/stage2/
└── {filename}_inference.wav
```

---

## 評估結果位置

```bash
# Stage 1 評估結果 (Paraformer)
/mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/two_stage/stage1/
├── dcer_results_paraformer.csv      # 逐樣本結果
└── dcer_summary_paraformer.json     # 摘要統計

# Whisper 評估結果
├── dcer_results_whisper.csv
└── dcer_summary_whisper.json
```

---

## 配置文件

所有設定都在 [`config.yaml`](config.yaml):

```yaml
# 路徑
focalcodec_dir: "/home/jieshiang/Desktop/GitHub/FocalCodec_main/focalcodec"
model_cache_dir: "/mnt/Internal/jieshiang/Model/FocalCodec"
output_dir: "/mnt/Internal/jieshiang/Model/FocalCodec/two_stage"
inference_dir: "/mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/two_stage"

# 數據
audio_base_path: "/mnt/Internal/ASR"
train_csv: "experiments/data_aishell/train_split.csv"
val_csv: "experiments/data_aishell/val_split.csv"

# Stage 1 訓練參數
stage1:
  gpu_id: 1
  batch_size: 16
  num_epochs: 100
  learning_rate: 0.0005

# Stage 2 訓練參數
stage2:
  gpu_id: 1
  batch_size: 16
  num_epochs: 100
  learning_rate: 0.0001
```

---

## 常見問題

### 1. CUDA Out of Memory

編輯 `config.yaml`:
```yaml
stage1:
  batch_size: 8  # 從 16 減少到 8
```

### 2. Stage 1 dCER > 20%

這表示使用了錯誤的訓練模式。重新訓練：
```bash
bash train.sh 1 decompressor_only
```

### 3. Stage 2 無法啟動

確認 Stage 1 已完成：
```bash
ls -la /mnt/Internal/jieshiang/Model/FocalCodec/two_stage/stage1_50hz/best_model.pt
```

### 4. 找不到音檔

編輯 `config.yaml`:
```yaml
audio_base_path: "/your/path/to/ASR"
```

---

## dCER 評估標準

| dCER 範圍 | 評價 |
|----------|------|
| < 10% | 優秀 ✅ |
| 10-20% | 良好 ✓ |
| 20-50% | 中等 ⚠️ |
| > 50% | 不佳 ❌ |

**dCER = CER(重建音檔) - CER(原始音檔)**

---

## 完整工作流程

```bash
# 1. Stage 1 訓練
bash train.sh 1
# 預期: Epoch 100, feature_loss < 0.5, dCER ~7%

# 2. Stage 1 推論
bash infer.sh 1
# 輸出: 2000 個 .wav 檔案

# 3. Stage 1 評估
bash eval.sh 1
# 檢查: dcer_summary_paraformer.json
# 如果 dCER Mean < 10%，繼續 Stage 2

# 4. Stage 2 訓練
bash train.sh 2
# 預期: Epoch 100, feature_loss < 1.0, dCER ~15%

# 5. Stage 2 推論 & 評估
bash infer.sh 2
bash eval.sh 2
# 檢查: dcer_summary_paraformer.json
```

---

## 詳細文檔

完整說明請參考 [README.md](README.md)
