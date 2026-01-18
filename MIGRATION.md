# 遷移到其他 Server 指南

將訓練好的 25Hz FocalCodec 模型遷移到其他 server 繼續訓練。

## 需要傳輸的檔案

### 1. 專案代碼 (必須)

```bash
# 在目標 server 上 clone
git clone <your-repo-url> FocalCodec_main
cd FocalCodec_main

# Clone focalcodec 子模組
git clone https://github.com/lucadellalib/focalcodec.git focalcodec
cd focalcodec && pip install -e . && cd ..
```

### 2. 訓練 Checkpoint (必須)

從原 server 複製 checkpoint：

```bash
# 原 server 上的 checkpoint 位置
/mnt/Internal/jieshiang/Model/FocalCodec/25hz_2048/stage_1/
├── best_model.pt      # 最佳模型 (~500MB)
└── last_model.pt      # 最新 checkpoint，用於繼續訓練

# 使用 scp 傳輸
scp -r user@source-server:/mnt/Internal/jieshiang/Model/FocalCodec/25hz_2048 /path/to/new/location/
```

### 3. 預訓練模型快取 (可選，會自動下載)

```bash
# 原位置
/mnt/Internal/jieshiang/Model/FocalCodec/models--lucadellalib--focalcodec_*

# 如果不傳輸，第一次執行會自動從 HuggingFace 下載
```

### 4. 訓練資料 (必須)

```bash
# AISHELL 資料位置
/mnt/Internal/ASR/aishell/data_aishell/wav/

# CSV 檔案已包含在專案中
experiments/data_aishell/train_split.csv
experiments/data_aishell/val_split.csv
```

## 在新 Server 上設定

### Step 1: 修改 config.yaml

根據新 server 的路徑修改：

```yaml
paths:
  # 專案目錄
  base_dir: "/path/to/FocalCodec_main"
  focalcodec_dir: "/path/to/FocalCodec_main/focalcodec"

  # 模型快取目錄
  model_cache_dir: "/path/to/model/cache"

  # 訓練輸出目錄
  output_dir: "/path/to/output/25hz_2048"

  # 推論輸出目錄
  inference_dir: "/path/to/inference/25hz_2048"

data:
  # 音檔根目錄
  audio_base_path: "/path/to/ASR/data"
```

### Step 2: 安裝環境

```bash
# 建立 conda 環境
conda create -n focalcodec python=3.10
conda activate focalcodec

# 安裝 PyTorch
conda install pytorch torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 安裝依賴
pip install -r requirements.txt

# 安裝 focalcodec
cd focalcodec && pip install -e . && cd ..
```

### Step 3: 驗證設定

```bash
python scripts/setup_check.py
```

### Step 4: 繼續訓練

```bash
# 使用 --resume 從 last_model.pt 繼續訓練
bash train.sh --resume

# 或直接執行 Python
python train_25hz_focalcodec.py --resume
```

## Checkpoint 內容

`last_model.pt` 包含：
- `model_state_dict`: 模型權重
- `optimizer_state_dict`: 優化器狀態
- `scheduler_state_dict`: 學習率排程器狀態
- `epoch`: 當前 epoch (500)
- `val_loss`: 驗證損失
- `stage`: 訓練階段

使用 `--resume` 會載入所有狀態，從 epoch 501 繼續訓練。

## 繼續訓練設定

在 `config.yaml` 中調整：

```yaml
# 繼續訓練的配置
continue_training:
  gpu_id: 0              # 新 server 的 GPU ID
  num_epochs: 1000       # 目標 epoch 數
  patience: 50           # 早停耐心值
  resume_checkpoint: "last_model.pt"
```

## 快速遷移清單

- [ ] Clone 專案代碼
- [ ] Clone focalcodec 並安裝
- [ ] 傳輸 checkpoint (`last_model.pt`)
- [ ] 傳輸或準備訓練資料 (AISHELL)
- [ ] 修改 `config.yaml` 中的路徑
- [ ] 安裝 Python 環境
- [ ] 執行 `python scripts/setup_check.py` 驗證
- [ ] 執行 `bash train.sh --resume` 繼續訓練
