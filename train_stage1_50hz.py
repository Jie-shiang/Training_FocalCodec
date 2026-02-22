#!/usr/bin/env python3
"""
Stage 1: Fine-tune 50Hz Causal 2k on AISHELL

使用官方 FocalCodec API 進行訓練:
- sig_to_feats: audio -> encoder features
- feats_to_lats: features -> compressed latents
- lats_to_codes: latents -> discrete codes
- codes_to_qfeats: codes -> quantized features (decompressor output)
- feats_to_sig: quantized features -> reconstructed audio

正確的 Feature Loss:
- MSE(qfeats, feats) = MSE(decompressor_output, encoder_features)
- 這是官方 SpeechBrain recipe 的方法

訓練策略:
- 訓練: Compressor + Decompressor (+ Quantizer)
- 凍結: Encoder + Decoder
- LR: 0.0005 (官方配置)

Output: Fine-tuned 50Hz model ready for Stage 2

Usage:
    python train_stage1_50hz.py
    python train_stage1_50hz.py --config custom_config.yaml
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
from typing import Dict, List, Optional, Tuple
import math

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent

# Import audio losses
from audio_losses import MelSpectrogramLoss, MultiResolutionSTFTLoss
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

from utils.config_loader import load_config


class AudioDataset(Dataset):
    """Dataset for audio files with transcriptions."""

    def __init__(
        self,
        csv_path: str,
        base_path: str,
        chunk_duration: float = 3.0,
        overlap: float = 0.5,
        sample_rate: int = 16000,
        max_chunks: Optional[int] = None
    ):
        self.csv_path = csv_path
        self.base_path = base_path
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.chunk_size = int(chunk_duration * sample_rate)
        self.hop_size = int((chunk_duration - overlap) * sample_rate)
        self.chunks = []

        # Load CSV
        df = pd.read_csv(csv_path)

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading dataset"):
            relative_path = row['file_path']

            # Handle transcription (optional)
            transcription = ""
            if 'transcription' in df.columns and pd.notna(row.get('transcription')):
                transcription = str(row['transcription'])

            # Convert relative path to absolute path
            if relative_path.startswith('./'):
                audio_path = os.path.join(base_path, relative_path.lstrip('./'))
            else:
                audio_path = os.path.join(base_path, relative_path)

            try:
                info = torchaudio.info(audio_path)
                file_sr = info.sample_rate
                file_num_frames = info.num_frames
            except Exception:
                continue

            expected_frames = int(file_num_frames * sample_rate / file_sr)

            if expected_frames >= self.chunk_size:
                chunk_frames_in_source = int(self.chunk_size * file_sr / sample_rate)
                hop_frames_in_source = int(self.hop_size * file_sr / sample_rate)
                n_chunks = (file_num_frames - chunk_frames_in_source) // hop_frames_in_source + 1

                for i in range(n_chunks):
                    start_frame = i * hop_frames_in_source
                    self.chunks.append({
                        'file_path': audio_path,
                        'start': start_frame,
                        'num_frames': chunk_frames_in_source,
                        'file_sr': file_sr,
                        'transcription': transcription
                    })

                    if max_chunks and len(self.chunks) >= max_chunks:
                        break

            if max_chunks and len(self.chunks) >= max_chunks:
                break

        if max_chunks:
            self.chunks = self.chunks[:max_chunks]

        print(f"Dataset initialized: {len(self.chunks)} chunks from {len(df)} files")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk_info = self.chunks[idx]

        waveform, sr = torchaudio.load(
            chunk_info['file_path'],
            frame_offset=chunk_info['start'],
            num_frames=chunk_info['num_frames']
        )

        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if waveform.shape[-1] < self.chunk_size:
            padding = self.chunk_size - waveform.shape[-1]
            waveform = F.pad(waveform, (0, padding))
        elif waveform.shape[-1] > self.chunk_size:
            waveform = waveform[:, :self.chunk_size]

        waveform = waveform.squeeze(0)

        return waveform.contiguous(), chunk_info['transcription']


class FocalCodecTrainer:
    """
    使用官方 FocalCodec API 進行訓練

    正確的 Feature Loss:
    - feats = model.sig_to_feats(audio)  # encoder features
    - lats = model.feats_to_lats(feats)   # compressed latents
    - codes = model.lats_to_codes(lats)   # discrete codes
    - qfeats = model.codes_to_qfeats(codes)  # decompressor output
    - feature_loss = MSE(qfeats, feats)   # ✅ 官方方法
    """

    def __init__(
        self,
        model,
        device: str = 'cuda',
        enable_codebook_monitor: bool = False,
        codebook_size: int = 2048,
        weight_feature: float = 1.0,
        weight_time: float = 0.0,
        weight_mel: float = 0.0,
        weight_stft: float = 0.0,
        use_mel_loss: bool = False,
        use_stft_loss: bool = False,
        sample_rate: int = 16000,
        train_mode: str = 'decompressor_only'
    ):
        self.model = model.to(device)
        self.device = device
        self.enable_codebook_monitor = enable_codebook_monitor
        self.codebook_size = codebook_size
        self.codebook_usage_history = []
        self.train_mode = train_mode
        self.sample_rate = sample_rate  # ✅ 儲存 sample_rate

        # Loss weights
        self.weight_feature = weight_feature
        self.weight_time = weight_time
        self.weight_mel = weight_mel
        self.weight_stft = weight_stft
        self.use_mel_loss = use_mel_loss
        self.use_stft_loss = use_stft_loss

        # Initialize advanced audio losses
        if use_mel_loss:
            self.mel_loss_fn = MelSpectrogramLoss(sample_rate=sample_rate).to(device)

        if use_stft_loss:
            self.stft_loss_fn = MultiResolutionSTFTLoss()

        print("\n" + "="*60)
        print("FocalCodec Training Configuration (官方方法)")
        print("="*60)
        print(f"使用官方 FocalCodec API:")
        print(f"  sig_to_feats → feats_to_lats → lats_to_codes → codes_to_qfeats → feats_to_sig")
        print(f"\nFeature Loss: MSE(qfeats, feats)")
        print(f"  = MSE(decompressor_output, encoder_features)")
        print(f"\nLoss Weights:")
        print(f"  Feature Loss: {weight_feature}")
        print(f"  Time Loss: {weight_time}")
        print(f"  Mel Loss: {weight_mel} {'(enabled)' if use_mel_loss else '(disabled)'}")
        print(f"  STFT Loss: {weight_stft} {'(enabled)' if use_stft_loss else '(disabled)'}")
        print("="*60 + "\n")

    def compute_loss(
        self,
        audio: torch.Tensor,
        reconstructed_audio: torch.Tensor,
        feats: torch.Tensor,
        qfeats: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        計算訓練 Loss

        正確的 Feature Loss:
        - MSE(qfeats, feats) = MSE(decompressor_output, encoder_features)
        """
        total_loss = 0
        loss_dict = {}

        # ✅ Feature Loss: MSE(decompressor_output, encoder_features)
        # 這是官方 SpeechBrain recipe 的方法
        if self.weight_feature > 0:
            # Align temporal dimensions
            min_frames = min(feats.shape[1], qfeats.shape[1])
            feats_aligned = feats[:, :min_frames, :]
            qfeats_aligned = qfeats[:, :min_frames, :]

            feature_loss = F.mse_loss(qfeats_aligned, feats_aligned)
            total_loss += self.weight_feature * feature_loss
            loss_dict['feature_loss'] = feature_loss.item()

        # Time domain loss (optional)
        if self.weight_time > 0:
            # ✅ 修正：將原始音訊重採樣到輸出採樣率 (24kHz)
            audio_upsampled = torchaudio.functional.resample(
                audio,
                self.sample_rate,  # 16000
                self.model.sample_rate_output  # 24000
            )
            min_len = min(reconstructed_audio.shape[-1], audio_upsampled.shape[-1])
            recon_aligned = reconstructed_audio[..., :min_len]
            orig_aligned = audio_upsampled[..., :min_len]

            time_loss = F.l1_loss(recon_aligned, orig_aligned)
            total_loss += self.weight_time * time_loss
            loss_dict['time_loss'] = time_loss.item()

        # Mel spectrogram loss (advanced)
        if self.use_mel_loss and self.weight_mel > 0:
            # ✅ 修正：將原始音訊重採樣到輸出採樣率 (24kHz)
            audio_upsampled = torchaudio.functional.resample(
                audio,
                self.sample_rate,  # 16000
                self.model.sample_rate_output  # 24000
            )
            min_len = min(reconstructed_audio.shape[-1], audio_upsampled.shape[-1])
            recon_aligned = reconstructed_audio[..., :min_len]
            orig_aligned = audio_upsampled[..., :min_len]

            mel_loss = self.mel_loss_fn(recon_aligned, orig_aligned)
            total_loss += self.weight_mel * mel_loss
            loss_dict['mel_loss'] = mel_loss.item()

        # Multi-resolution STFT loss (advanced)
        if self.use_stft_loss and self.weight_stft > 0:
            # ✅ 修正：將原始音訊重採樣到輸出採樣率 (24kHz)
            audio_upsampled = torchaudio.functional.resample(
                audio,
                self.sample_rate,  # 16000
                self.model.sample_rate_output  # 24000
            )
            min_len = min(reconstructed_audio.shape[-1], audio_upsampled.shape[-1])
            recon_aligned = reconstructed_audio[..., :min_len]
            orig_aligned = audio_upsampled[..., :min_len]

            stft_loss = self.stft_loss_fn(recon_aligned, orig_aligned)
            total_loss += self.weight_stft * stft_loss
            loss_dict['stft_loss'] = stft_loss.item()

        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict

    def train_epoch(self, dataloader, optimizer, epoch, gradient_clip=5.0):
        """Train one epoch using official FocalCodec API."""
        self.model.train()

        # Keep encoder and decoder in eval mode (frozen)
        self.model.encoder.eval()
        self.model.decoder.eval()

        total_loss = 0
        loss_components = {}

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, (audio, transcriptions) in enumerate(pbar):
            audio = audio.to(self.device)

            optimizer.zero_grad()

            # ============================================================
            # 使用官方 FocalCodec API
            # ============================================================

            # 1. Encoder: audio -> features
            feats = self.model.sig_to_feats(audio)

            # 2. Compressor: features -> latents
            lats = self.model.feats_to_lats(feats)

            # 3. Quantizer: latents -> codes
            codes = self.model.lats_to_codes(lats)

            # 4. Decompressor: codes -> quantized features
            if self.train_mode == 'both_ste':
                # ✅ Straight-Through Estimator (STE):
                # Forward: use discrete codes
                # Backward: gradient flows through lats
                # WARNING: This may damage Compressor's pretrained ability!
                codes_input = lats + (codes - lats).detach()
            else:
                # Standard: use discrete codes directly
                # Gradient only flows to Decompressor
                codes_input = codes

            qfeats = self.model.codes_to_qfeats(codes_input)

            # 5. Decoder: quantized features -> reconstructed audio
            reconstructed_audio = self.model.feats_to_sig(qfeats)

            # ============================================================
            # 計算 Loss
            # ✅ Feature Loss = MSE(qfeats, feats)
            # ============================================================
            loss, loss_dict = self.compute_loss(
                audio,
                reconstructed_audio,
                feats,
                qfeats
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=gradient_clip
                )

            optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0
                loss_components[key] += value

            # Codebook monitoring
            if self.enable_codebook_monitor and batch_idx % 10 == 0:
                toks = self.model.codes_to_toks(codes)
                codebook_stats = self.monitor_codebook(toks)
                loss_dict.update(codebook_stats)

            # Update progress bar
            postfix = {'loss': f"{loss.item():.4f}"}
            if 'feature_loss' in loss_dict:
                postfix['feat'] = f"{loss_dict['feature_loss']:.4f}"
            if self.enable_codebook_monitor and 'codebook_perplexity' in loss_dict:
                postfix['cb_ppl'] = f"{loss_dict['codebook_perplexity']:.0f}"
                postfix['cb_use'] = f"{loss_dict['codebook_usage']:.2f}"
            pbar.set_postfix(postfix)

        # Calculate epoch averages
        n_batches = len(dataloader)
        metrics = {'total_loss': total_loss / n_batches}
        for key, value in loss_components.items():
            if key != 'total_loss':
                metrics[key] = value / n_batches

        return metrics

    def monitor_codebook(self, toks):
        """Monitor codebook usage."""
        toks_flat = toks.flatten().long()

        if toks_flat.numel() > 0:
            min_tok = toks_flat.min().item()
            max_tok = toks_flat.max().item()
            if min_tok < 0 or max_tok >= self.codebook_size:
                toks_flat = torch.clamp(toks_flat, 0, self.codebook_size - 1)

        usage = torch.bincount(toks_flat, minlength=self.codebook_size)
        self.codebook_usage_history.append(usage.cpu().numpy())

        used_codes = (usage > 0).sum().item()
        usage_ratio = used_codes / self.codebook_size

        probs = usage.float() / usage.sum()
        probs = probs[probs > 0]
        if len(probs) > 0:
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            perplexity = torch.exp(entropy).item()
        else:
            perplexity = 0.0

        return {
            'codebook_usage': usage_ratio,
            'codebook_used_codes': used_codes,
            'codebook_perplexity': perplexity
        }

    @torch.no_grad()
    def validate(self, dataloader):
        """Validate model using official FocalCodec API."""
        self.model.eval()

        total_loss = 0
        loss_components = {}
        all_toks = []

        for audio, transcriptions in tqdm(dataloader, desc="Validation"):
            audio = audio.to(self.device)

            # Forward pass using official API + STE
            feats = self.model.sig_to_feats(audio)
            lats = self.model.feats_to_lats(feats)
            codes = self.model.lats_to_codes(lats)
            # STE: use discrete codes in forward, but allow gradient to flow
            codes_ste = lats + (codes - lats).detach()
            qfeats = self.model.codes_to_qfeats(codes_ste)
            reconstructed_audio = self.model.feats_to_sig(qfeats)

            # Collect tokens for monitoring
            if self.enable_codebook_monitor:
                toks = self.model.codes_to_toks(codes)
                all_toks.append(toks.cpu())

            # Compute loss
            loss, loss_dict = self.compute_loss(
                audio,
                reconstructed_audio,
                feats,
                qfeats
            )

            # Accumulate losses
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0
                loss_components[key] += value

        # Calculate averages
        n_batches = len(dataloader)
        metrics = {'total_loss': total_loss / n_batches}
        for key, value in loss_components.items():
            if key != 'total_loss':
                metrics[key] = value / n_batches

        # Codebook statistics on validation set
        if self.enable_codebook_monitor and all_toks:
            all_toks = torch.cat(all_toks, dim=0)
            codebook_stats = self.monitor_codebook(all_toks)
            metrics.update(codebook_stats)

        return metrics

    def save_checkpoint(self, path, epoch, optimizer, metrics):
        """Save checkpoint."""
        import shutil

        os.makedirs(os.path.dirname(path), exist_ok=True)

        stat = shutil.disk_usage(os.path.dirname(path))
        available_gb = stat.free / (1024**3)

        if available_gb < 1.0:
            print(f"WARNING: Low disk space ({available_gb:.2f} GB available)")
            print(f"Skipping checkpoint save to avoid disk full error")
            return

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': metrics['train']['total_loss'],
            'val_loss': metrics['val']['total_loss'],
            'metrics': metrics,
            'stage': '1_50hz_official_api'
        }

        try:
            torch.save(checkpoint, path)
            print(f"Checkpoint saved: {path} ({available_gb:.2f} GB remaining)")
        except RuntimeError as e:
            if "file write failed" in str(e) or "PytorchStreamWriter failed" in str(e):
                print(f"ERROR: Failed to save checkpoint - disk may be full")
            else:
                raise e


def parse_args():
    parser = argparse.ArgumentParser(description='Stage 1: Fine-tune 50Hz on AISHELL (官方方法)')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')

    # Override config parameters
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--max_chunks', type=int, default=None)

    # Training mode: which components to train
    parser.add_argument('--train_mode', type=str, default='decompressor_only',
                       choices=['decompressor_only', 'both', 'both_ste'],
                       help='Training mode: decompressor_only (recommended, matches original 7%% dCER), '
                            'both (train both without STE), both_ste (train both with STE)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Extract experiment ID from config filename (e.g., config_exp_A.yaml -> exp_A)
    config_filename = os.path.basename(args.config)
    if 'exp_' in config_filename:
        exp_id = config_filename.split('exp_')[1].split('.')[0]  # Extract A, B, or C
        experiment_name = f"exp_{exp_id}"
    else:
        experiment_name = "default"

    print(f"Experiment: {experiment_name}")

    # Add focalcodec to path
    sys.path.insert(0, config.focalcodec_dir)
    from focalcodec import FocalCodec

    # Get Stage 1 config with command line overrides
    gpu_id = config.get('stage1.gpu_id', 1)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} (GPU {gpu_id} via CUDA_VISIBLE_DEVICES)")
    print(f"Stage 1: Fine-tune 50Hz Causal 2k on AISHELL (官方方法)")

    # Load model
    print(f"\nLoading model: {config.base_model}")
    model = FocalCodec.from_pretrained(
        config.base_model,
        cache_dir=config.model_cache_dir
    )

    # ============================================================
    # 設置訓練組件 (官方方法)
    # 訓練: Compressor + Decompressor
    # 凍結: Encoder + Decoder
    # ============================================================

    # ============================================================
    # Training Mode Selection
    # ============================================================
    train_mode = args.train_mode
    print("\n" + "="*60)
    print(f"訓練策略: {train_mode}")
    print("="*60)

    # Freeze Encoder (always)
    print("Encoder: 凍結")
    model.encoder.eval()
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Compressor: depends on train_mode
    if train_mode == 'decompressor_only':
        print("Compressor: 凍結 (保持 pretrained 能力)")
        model.compressor.eval()
        for param in model.compressor.parameters():
            param.requires_grad = False
    else:
        print("Compressor: 訓練")
        model.compressor.train()
        for param in model.compressor.parameters():
            param.requires_grad = True

    # Train Decompressor (always)
    print("Decompressor: 訓練")
    model.decompressor.train()
    for param in model.decompressor.parameters():
        param.requires_grad = True

    # Freeze Decoder (always)
    print("Decoder: 凍結")
    model.decoder.eval()
    for param in model.decoder.parameters():
        param.requires_grad = False

    if train_mode == 'decompressor_only':
        print("\n⚠️  注意: 只訓練 Decompressor (推薦模式)")
        print("   這是原始 7% dCER 結果使用的訓練策略")
        print("   Compressor 保持 pretrained，不會被破壞")
    elif train_mode == 'both_ste':
        print("\n⚠️  注意: 使用 STE 同時訓練 Compressor + Decompressor")
        print("   警告: 這可能會破壞 Compressor 的 pretrained 能力!")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print("="*60)

    # Training parameters (官方配置)
    batch_size = args.batch_size or config.get('stage1.batch_size', 16)
    num_epochs = args.num_epochs or config.get('stage1.num_epochs', 100)
    learning_rate = args.learning_rate or config.get('stage1.learning_rate', 0.0005)  # 官方: 0.0005
    max_chunks = args.max_chunks or config.get('stage1.max_chunks', 0)

    print(f"\n訓練參數 (官方配置):")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning Rate: {learning_rate} (官方: 0.0005)")
    print(f"  Max Chunks: {max_chunks if max_chunks > 0 else 'all'}")

    # Create trainer
    trainer = FocalCodecTrainer(
        model=model,
        device=device,
        enable_codebook_monitor=config.get('stage1.enable_codebook_monitor', True),
        codebook_size=2048,
        weight_feature=config.get('stage1.weight_feature', 1.0),
        weight_time=config.get('stage1.weight_time', 0.0),
        weight_mel=config.get('stage1.weight_mel', 0.0),
        weight_stft=config.get('stage1.weight_stft', 0.0),
        use_mel_loss=config.get('stage1.use_mel_loss', False),
        use_stft_loss=config.get('stage1.use_stft_loss', False),
        train_mode=train_mode
    )

    # Prepare datasets
    print("\nPreparing datasets...")

    max_chunks_train = max_chunks if max_chunks > 0 else None
    max_chunks_val = int(max_chunks * 0.2) if max_chunks > 0 else None

    train_dataset = AudioDataset(
        csv_path=config.train_csv,
        base_path=config.audio_base_path,
        chunk_duration=config.get('stage1.chunk_duration', 3.0),
        overlap=config.get('stage1.overlap', 0.5),
        max_chunks=max_chunks_train
    )

    val_dataset = AudioDataset(
        csv_path=config.val_csv,
        base_path=config.audio_base_path,
        chunk_duration=config.get('stage1.chunk_duration', 3.0),
        overlap=0.0,
        max_chunks=max_chunks_val
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.get('stage1.num_workers', 4),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get('stage1.num_workers', 4),
        pin_memory=True
    )

    # Optimizer (官方配置: AdamW with weight_decay=0.01)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=config.get('stage1.weight_decay', 0.01),
        betas=(0.9, 0.98)  # 官方配置
    )

    # Scheduler: Cosine or ReduceLROnPlateau
    scheduler_type = config.get('stage1.scheduler_type', 'cosine')

    if scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get('stage1.scheduler_factor', 0.5),
            patience=config.get('stage1.scheduler_patience', 5),
            threshold=config.get('stage1.scheduler_threshold', 0.001),
            cooldown=config.get('stage1.scheduler_cooldown', 2),
            min_lr=config.get('stage1.scheduler_min_lr', 1e-6),
            verbose=True
        )
        print(f"Using ReduceLROnPlateau scheduler:")
        print(f"  Factor: {config.get('stage1.scheduler_factor', 0.5)}")
        print(f"  Patience: {config.get('stage1.scheduler_patience', 5)} epochs")
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        print(f"Using CosineAnnealingLR scheduler")

    # Training loop - use experiment-specific directories
    checkpoint_dir = os.path.join(config.output_dir, experiment_name, 'stage1_50hz')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup logging
    experiments_dir = os.path.join(PROJECT_ROOT, 'experiments', experiment_name)
    os.makedirs(experiments_dir, exist_ok=True)
    log_file = os.path.join(experiments_dir, 'stage1_50hz_official_training.csv')

    with open(log_file, 'w') as f:
        f.write('epoch,train_loss,val_loss,feature_loss,codebook_usage,codebook_perplexity\n')
    print(f"Logging to: {log_file}")

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    patience = config.get('stage1.patience', 15)
    min_delta = config.get('stage1.min_delta', 0.00001)

    print(f"\n" + "="*60)
    print(f"Starting training for Stage 1 (官方方法)")
    print(f"="*60)
    print(f"Epochs: {num_epochs}")
    print(f"Patience: {patience}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"\n成功標準 (基於官方論文):")
    print(f"  Epoch 20:  feature_loss < 0.5")
    print(f"  Epoch 50:  feature_loss < 0.1")
    print(f"  Epoch 100: feature_loss < 0.05")
    print(f"  Codebook: usage > 0.90, perplexity > 1500")
    print("="*60)

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")

        train_metrics = trainer.train_epoch(
            train_loader, optimizer, epoch,
            gradient_clip=config.get('stage1.gradient_clip', 5.0)
        )
        val_metrics = trainer.validate(val_loader)

        # Update scheduler
        if scheduler_type == 'plateau':
            scheduler.step(val_metrics['total_loss'])  # Pass validation loss
        else:
            scheduler.step()

        # Display current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nCurrent learning rate: {current_lr:.2e}")

        print(f"\nTrain metrics:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value:.6f}")

        print(f"\nValidation metrics:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.6f}")

        # Log to CSV
        with open(log_file, 'a') as f:
            f.write(f"{epoch},"
                   f"{train_metrics.get('total_loss', 0):.6f},"
                   f"{val_metrics.get('total_loss', 0):.6f},"
                   f"{val_metrics.get('feature_loss', 0):.6f},"
                   f"{val_metrics.get('codebook_usage', 0):.6f},"
                   f"{val_metrics.get('codebook_perplexity', 0):.2f}\n")

        current_val_loss = val_metrics['total_loss']

        if current_val_loss < best_val_loss - min_delta:
            improvement = best_val_loss - current_val_loss
            best_val_loss = current_val_loss
            best_epoch = epoch
            patience_counter = 0

            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            trainer.save_checkpoint(
                best_path,
                epoch,
                optimizer,
                {'train': train_metrics, 'val': val_metrics}
            )
            print(f"New best model saved! (val_loss: {best_val_loss:.6f}, improved by {improvement:.6f})")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s) (best: {best_val_loss:.6f} at epoch {best_epoch})")

            if patience_counter >= patience:
                print(f"\nEarly stopping triggered!")
                print(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
                break

        # Save last model
        last_path = os.path.join(checkpoint_dir, "last_model.pt")
        trainer.save_checkpoint(
            last_path,
            epoch,
            optimizer,
            {'train': train_metrics, 'val': val_metrics}
        )

    print("\n" + "="*60)
    print("Stage 1 Training completed!")
    print("="*60)
    print(f"Best model: epoch {best_epoch} with val_loss {best_val_loss:.6f}")
    print(f"Checkpoint: {os.path.join(checkpoint_dir, 'best_model.pt')}")
    print(f"\nNext step: Run inference and evaluation")
    print(f"  bash infer.sh 1")
    print(f"  bash eval.sh 1")
    print("="*60)


if __name__ == '__main__':
    main()
