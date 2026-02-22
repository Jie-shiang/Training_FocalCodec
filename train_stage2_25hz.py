#!/usr/bin/env python3
"""
Stage 2: Add 4th layer for 25Hz downscale

使用官方 FocalCodec API 進行訓練:
- sig_to_feats: audio -> encoder features
- feats_to_lats: features -> compressed latents
- lats_to_codes: latents -> discrete codes
- codes_to_qfeats: codes -> quantized features (decompressor output)
- feats_to_sig: quantized features -> reconstructed audio

此階段從 Stage 1 fine-tuned 50Hz 模型開始，添加第 4 層 FocalNet 層
以實現 25Hz 下採樣 (275 bps)。

架構:
- Layers 1-3: 從 Stage 1 繼承 (使用較小 LR 繼續訓練)
- Layer 4: 新增層 (使用較大 LR 訓練)

正確的 Feature Loss:
- MSE(qfeats, feats) = MSE(decompressor_output, encoder_features)

Usage:
    python train_stage2_25hz.py
    python train_stage2_25hz.py --config custom_config.yaml
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
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
from typing import Dict, List, Optional, Tuple
import math
import copy

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

from utils.config_loader import load_config

# Import audio losses
from audio_losses import MelSpectrogramLoss, MultiResolutionSTFTLoss


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

        df = pd.read_csv(csv_path)

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading dataset"):
            relative_path = row['file_path']

            transcription = ""
            if 'transcription' in df.columns and pd.notna(row.get('transcription')):
                transcription = str(row['transcription'])

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


def add_fourth_layer_to_compressor(model, device):
    """Add a 4th FocalDownScale layer to the compressor for 2x downscaling."""
    from focalcodec.focalnet import FocalDownScale

    comp = model.compressor

    # Create new 4th layer with downscale=2
    layer_4 = FocalDownScale(
        input_dim=1024,
        output_dim=1024,
        downscale_factor=2,
        focal_window=comp.focal_window,
        focal_level=comp.focal_level,
        focal_factor=comp.focal_factor,
        dropout=comp.dropout_,
        use_post_norm=comp.use_post_norm,
        use_layerscale=comp.use_layerscale,
        layerscale_init=comp.layerscale_init,
        tanhscale_init=comp.tanhscale_init,
        normalize_modulator=comp.normalize_modulator,
        causal=comp.causal,
        window_size=comp.window_size,
    ).to(device)

    comp.layers.append(layer_4)
    comp.downscale_factors = list(comp.downscale_factors) + [2]
    comp.downsample_factor = torch.Size(comp.downscale_factors).numel()
    comp.chunk_size = comp.downsample_factor

    # Check current out_proj dimensions and only expand if needed
    old_out_proj = comp.out_proj
    old_out_dim = old_out_proj.weight.shape[0]
    target_out_dim = 11

    if old_out_dim == target_out_dim:
        # Already 11 dimensions, no need to expand
        print(f"Added 4th layer to compressor: downscale_factors = {comp.downscale_factors}")
        print(f"out_proj already {target_out_dim} dimensions, keeping as-is")
    else:
        # Expand from old_out_dim to target_out_dim
        comp.out_proj = nn.Linear(1024, target_out_dim).to(device)
        with torch.no_grad():
            comp.out_proj.weight[:old_out_dim, :].copy_(old_out_proj.weight)
            comp.out_proj.bias[:old_out_dim].copy_(old_out_proj.bias)
            if target_out_dim > old_out_dim:
                nn.init.normal_(comp.out_proj.weight[old_out_dim:, :], mean=0, std=0.02)
                nn.init.constant_(comp.out_proj.bias[old_out_dim:], 0)
        print(f"Added 4th layer to compressor: downscale_factors = {comp.downscale_factors}")
        print(f"Updated out_proj: {old_out_dim} -> {target_out_dim} dimensions")

    return model


def add_fourth_layer_to_decompressor(model, device):
    """Add a 4th FocalUpScale layer to the decompressor."""
    from focalcodec.focalnet import FocalUpScale

    decomp = model.decompressor

    layer_0 = FocalUpScale(
        input_dim=1024,
        output_dim=1024,
        upscale_factor=2,
        focal_window=decomp.focal_window,
        focal_level=decomp.focal_level,
        focal_factor=decomp.focal_factor,
        dropout=decomp.dropout_,
        use_post_norm=decomp.use_post_norm,
        use_layerscale=decomp.use_layerscale,
        layerscale_init=decomp.layerscale_init,
        tanhscale_init=decomp.tanhscale_init,
        normalize_modulator=decomp.normalize_modulator,
        causal=decomp.causal,
        window_size=decomp.window_size,
    ).to(device)

    decomp.layers.insert(0, layer_0)
    decomp.upscale_factors = [2] + list(decomp.upscale_factors)

    # Check current in_proj dimensions and only expand if needed
    old_in_proj = decomp.in_proj
    old_in_dim = old_in_proj.weight.shape[1]
    target_in_dim = 11

    if old_in_dim == target_in_dim:
        # Already 11 dimensions, no need to expand
        print(f"Added 4th layer to decompressor: upscale_factors = {decomp.upscale_factors}")
        print(f"in_proj already {target_in_dim} dimensions, keeping as-is")
    else:
        # Expand from old_in_dim to target_in_dim
        decomp.in_proj = nn.Linear(target_in_dim, 1024).to(device)
        with torch.no_grad():
            decomp.in_proj.weight[:, :old_in_dim].copy_(old_in_proj.weight)
            decomp.in_proj.bias.copy_(old_in_proj.bias)
            if target_in_dim > old_in_dim:
                nn.init.normal_(decomp.in_proj.weight[:, old_in_dim:], mean=0, std=0.02)
        print(f"Added 4th layer to decompressor: upscale_factors = {decomp.upscale_factors}")
        print(f"Updated in_proj: {old_in_dim} -> {target_in_dim} dimensions")

    return model


class FocalCodecTrainer:
    """使用官方 FocalCodec API 進行 Stage 2 訓練"""

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
        sample_rate: int = 16000
    ):
        self.model = model.to(device)
        self.device = device
        self.enable_codebook_monitor = enable_codebook_monitor
        self.codebook_size = codebook_size
        self.codebook_usage_history = []
        self.sample_rate = sample_rate  # ✅ 儲存 sample_rate

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
        print("FocalCodec Stage 2 Training Configuration (官方方法)")
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
        """計算訓練 Loss - 使用官方 Feature Loss"""
        total_loss = 0
        loss_dict = {}

        # Feature Loss: MSE(decompressor_output, encoder_features)
        if self.weight_feature > 0:
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

    def train_epoch(self, dataloader, optimizer, epoch, gradient_clip=5.0, is_warmup=False):
        """Train one epoch using official FocalCodec API."""
        self.model.train()
        self.model.encoder.eval()
        self.model.decoder.eval()

        total_loss = 0
        loss_components = {}

        pbar = tqdm(dataloader, desc=f"Epoch {epoch} {'[Warmup]' if is_warmup else ''}")
        for batch_idx, (audio, transcriptions) in enumerate(pbar):
            audio = audio.to(self.device)

            optimizer.zero_grad()

            # 使用官方 FocalCodec API + Straight-Through Estimator (STE)
            feats = self.model.sig_to_feats(audio)
            lats = self.model.feats_to_lats(feats)
            codes = self.model.lats_to_codes(lats)

            # ✅ Straight-Through Estimator (STE):
            # Forward: use discrete codes
            # Backward: gradient flows through lats
            # This allows compressor to receive gradients!
            codes_ste = lats + (codes - lats).detach()

            qfeats = self.model.codes_to_qfeats(codes_ste)
            reconstructed_audio = self.model.feats_to_sig(qfeats)

            # 計算 Loss: MSE(qfeats, feats)
            loss, loss_dict = self.compute_loss(
                audio,
                reconstructed_audio,
                feats,
                qfeats
            )

            loss.backward()

            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=gradient_clip
                )

            optimizer.step()

            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0
                loss_components[key] += value

            if self.enable_codebook_monitor and batch_idx % 10 == 0:
                toks = self.model.codes_to_toks(codes)
                codebook_stats = self.monitor_codebook(toks)
                loss_dict.update(codebook_stats)

            postfix = {'loss': f"{loss.item():.4f}"}
            if 'feature_loss' in loss_dict:
                postfix['feat'] = f"{loss_dict['feature_loss']:.4f}"
            if self.enable_codebook_monitor and 'codebook_perplexity' in loss_dict:
                postfix['cb_ppl'] = f"{loss_dict['codebook_perplexity']:.0f}"
            pbar.set_postfix(postfix)

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

            feats = self.model.sig_to_feats(audio)
            lats = self.model.feats_to_lats(feats)
            codes = self.model.lats_to_codes(lats)
            # STE: use discrete codes in forward, but allow gradient to flow
            codes_ste = lats + (codes - lats).detach()
            qfeats = self.model.codes_to_qfeats(codes_ste)
            reconstructed_audio = self.model.feats_to_sig(qfeats)

            if self.enable_codebook_monitor:
                toks = self.model.codes_to_toks(codes)
                all_toks.append(toks.cpu())

            loss, loss_dict = self.compute_loss(
                audio,
                reconstructed_audio,
                feats,
                qfeats
            )

            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0
                loss_components[key] += value

        n_batches = len(dataloader)
        metrics = {'total_loss': total_loss / n_batches}
        for key, value in loss_components.items():
            if key != 'total_loss':
                metrics[key] = value / n_batches

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
            return

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': metrics['train']['total_loss'],
            'val_loss': metrics['val']['total_loss'],
            'metrics': metrics,
            'stage': '2_25hz_official_api'
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
    parser = argparse.ArgumentParser(description='Stage 2: Add 4th layer for 25Hz (官方方法)')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--max_chunks', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)

    # Extract experiment ID from config filename (e.g., config_exp_A.yaml -> exp_A)
    config_filename = os.path.basename(args.config)
    if 'exp_' in config_filename:
        exp_id = config_filename.split('exp_')[1].split('.')[0]  # Extract A, B, or C
        experiment_name = f"exp_{exp_id}"
    else:
        experiment_name = "default"

    print(f"Experiment: {experiment_name}")

    sys.path.insert(0, config.focalcodec_dir)
    from focalcodec import FocalCodec

    gpu_id = config.get('stage2.gpu_id', 1)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} (GPU {gpu_id} via CUDA_VISIBLE_DEVICES)")
    print(f"Stage 2: Add 4th layer for 25Hz downscale (官方方法)")

    # Load Stage 1 checkpoint - use experiment-specific path
    stage1_ckpt = os.path.join(config.output_dir, experiment_name, 'stage1_50hz', 'best_model.pt')
    if not os.path.exists(stage1_ckpt):
        print(f"ERROR: Stage 1 checkpoint not found: {stage1_ckpt}")
        print(f"Please run Stage 1 first for experiment {experiment_name}")
        return

    print(f"\nLoading Stage 1 checkpoint: {stage1_ckpt}")
    checkpoint = torch.load(stage1_ckpt, map_location='cpu')

    # Create model from base
    model = FocalCodec.from_pretrained(
        config.base_model,
        cache_dir=config.model_cache_dir
    ).to(device)

    # Load Stage 1 weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"Loaded Stage 1 weights (epoch {checkpoint.get('epoch', 'N/A')})")

    # Add 4th layer
    print("\nAdding 4th layer to Compressor and Decompressor...")
    model = add_fourth_layer_to_compressor(model, device)
    model = add_fourth_layer_to_decompressor(model, device)

    # ============================================================
    # 設置訓練組件 (官方方法)
    # 訓練: Compressor + Decompressor
    # 凍結: Encoder + Decoder
    # ============================================================

    print("\n" + "="*60)
    print("訓練策略 (官方 SpeechBrain Recipe)")
    print("="*60)

    # Freeze Encoder
    print("Encoder: 凍結")
    model.encoder.eval()
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Freeze Decoder
    print("Decoder: 凍結")
    model.decoder.eval()
    for param in model.decoder.parameters():
        param.requires_grad = False

    # Training setup: different LR for old and new layers
    print("\nSetting up optimizer with different learning rates...")

    # Old layers (Layers 0-2 in compressor, Layers 1-3 in decompressor)
    old_params = []
    for i in range(3):
        old_params.extend(list(model.compressor.layers[i].parameters()))
        old_params.extend(list(model.decompressor.layers[i+1].parameters()))

    # New layers (Layer 3 in compressor, Layer 0 in decompressor)
    new_params = list(model.compressor.layers[3].parameters()) + \
                 list(model.decompressor.layers[0].parameters()) + \
                 list(model.compressor.out_proj.parameters()) + \
                 list(model.decompressor.in_proj.parameters())

    lr_old = config.get('stage2.learning_rate_old', 0.00005)
    lr_new = config.get('stage2.learning_rate_new', 0.0005)

    param_groups = [
        {'params': old_params, 'lr': lr_old},
        {'params': new_params, 'lr': lr_new}
    ]

    print(f"Compressor: 訓練")
    print(f"  - Layers 0-2 (old): LR = {lr_old}")
    print(f"  - Layer 3 (new): LR = {lr_new}")
    print(f"Decompressor: 訓練")
    print(f"  - Layer 0 (new): LR = {lr_new}")
    print(f"  - Layers 1-3 (old): LR = {lr_old}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print("="*60)

    # Training parameters
    batch_size = args.batch_size or config.get('stage2.batch_size', 16)
    num_epochs = args.num_epochs or config.get('stage2.num_epochs', 100)
    warmup_epochs = config.get('stage2.warmup_epochs', 10)
    max_chunks = args.max_chunks or config.get('stage2.max_chunks', 0)

    print(f"\n訓練參數:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Warmup Epochs: {warmup_epochs}")
    print(f"  Max Chunks: {max_chunks if max_chunks > 0 else 'all'}")

    # Create trainer
    trainer = FocalCodecTrainer(
        model=model,
        device=device,
        enable_codebook_monitor=config.get('stage2.enable_codebook_monitor', True),
        codebook_size=2048,
        weight_feature=config.get('stage2.weight_feature', 1.0),
        weight_time=config.get('stage2.weight_time', 0.0),
        weight_mel=config.get('stage2.weight_mel', 0.0),
        weight_stft=config.get('stage2.weight_stft', 0.0),
        use_mel_loss=config.get('stage2.use_mel_loss', False),
        use_stft_loss=config.get('stage2.use_stft_loss', False)
    )

    # Prepare datasets
    print("\nPreparing datasets...")

    max_chunks_train = max_chunks if max_chunks > 0 else None
    max_chunks_val = int(max_chunks * 0.2) if max_chunks > 0 else None

    train_dataset = AudioDataset(
        csv_path=config.train_csv,
        base_path=config.audio_base_path,
        chunk_duration=config.get('stage2.chunk_duration', 3.0),
        overlap=config.get('stage2.overlap', 0.5),
        max_chunks=max_chunks_train
    )

    val_dataset = AudioDataset(
        csv_path=config.val_csv,
        base_path=config.audio_base_path,
        chunk_duration=config.get('stage2.chunk_duration', 3.0),
        overlap=0.0,
        max_chunks=max_chunks_val
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.get('stage2.num_workers', 4),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get('stage2.num_workers', 4),
        pin_memory=True
    )

    # Optimizer
    optimizer = AdamW(param_groups, weight_decay=config.get('stage2.weight_decay', 0.01))
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop - use experiment-specific directories
    checkpoint_dir = os.path.join(config.output_dir, experiment_name, 'stage2_25hz')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup logging
    experiments_dir = os.path.join(PROJECT_ROOT, 'experiments', experiment_name)
    os.makedirs(experiments_dir, exist_ok=True)
    log_file = os.path.join(experiments_dir, 'stage2_25hz_official_training.csv')

    with open(log_file, 'w') as f:
        f.write('epoch,train_loss,val_loss,feature_loss,codebook_usage,codebook_perplexity\n')
    print(f"Logging to: {log_file}")

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    patience = config.get('stage2.patience', 25)
    min_delta = config.get('stage2.min_delta', 0.00001)

    print(f"\n" + "="*60)
    print(f"Starting training for Stage 2 (官方方法)")
    print(f"="*60)
    print(f"Epochs: {num_epochs}, Warmup: {warmup_epochs}")
    print(f"Patience: {patience}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print("="*60)

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")

        is_warmup = epoch <= warmup_epochs
        if is_warmup:
            # During warmup, freeze old layers
            for param in old_params:
                param.requires_grad = False
        else:
            # After warmup, train all layers
            for param in old_params:
                param.requires_grad = True

        train_metrics = trainer.train_epoch(
            train_loader, optimizer, epoch,
            gradient_clip=config.get('stage2.gradient_clip', 5.0),
            is_warmup=is_warmup
        )
        val_metrics = trainer.validate(val_loader)

        scheduler.step()

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
    print("Stage 2 Training completed!")
    print("="*60)
    print(f"Best model: epoch {best_epoch} with val_loss {best_val_loss:.6f}")
    print(f"Checkpoint: {os.path.join(checkpoint_dir, 'best_model.pt')}")
    print(f"\nNext step: Run inference and evaluation")
    print(f"  bash infer.sh 2")
    print(f"  bash eval.sh 2")
    print("="*60)


if __name__ == '__main__':
    main()
