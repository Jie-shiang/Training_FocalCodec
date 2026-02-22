#!/usr/bin/env python3
"""
Stage 3: Add 5th layer for 12.5Hz with Whisper ASR Loss — DDP 版本

與 train_stage3_asr.py 完全分開，使用 torchrun 啟動多 GPU 訓練。

單節點 8 GPU:
    torchrun --nproc_per_node=8 train_stage3_asr_ddp.py --config config_exp_K.yaml

多節點 (由 SLURM 腳本 bash/slurm_train_stage3_K_ddp.sh 管理):
    sbatch bash/slurm_train_stage3_K_ddp.sh

從 Exp J Stage 2 checkpoint 繼承，加入第 5 層實現 12.5Hz 下採樣，
同時使用 Whisper Token Cross-Entropy Loss 保持語意可辨識性。

架構:
- Layers 0-3 (compressor) / Layers 1-4 (decompressor): 繼承自 Exp J Stage 2
- Layer 4 (compressor) / Layer 0 (decompressor): 新增 Stage 3 層

Bitrate: 12.5 Hz × 11-bit BSQ = 137.5 bps

Usage:
    # Step 1: 重用 Exp J 的 Whisper token cache (資料集相同)
    # (不需要重新計算，直接用 experiments/exp_J/whisper_cache/)

    # Step 2: Train
    python train_stage3_asr.py --config config_exp_K.yaml

    # Resume
    python train_stage3_asr.py --config config_exp_K.yaml --resume
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

from utils.config_loader import load_config
from audio_losses import MelSpectrogramLoss, MultiResolutionSTFTLoss
from whisper_asr_loss import WhisperASRLoss


class FocalCodecTrainWrapper(nn.Module):
    """Wraps FocalCodec forward pass for training.

    Combines all FocalCodec steps into one call:
    audio -> feats -> lats -> codes (STE) -> qfeats -> reconstructed_audio
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio):
        with torch.no_grad():
            feats = self.model.sig_to_feats(audio)
        lats = self.model.feats_to_lats(feats)
        codes = self.model.lats_to_codes(lats)
        # Straight-Through Estimator (STE): forward uses discrete, backward uses continuous
        codes_ste = lats + (codes - lats).detach()
        qfeats = self.model.codes_to_qfeats(codes_ste)
        reconstructed_audio = self.model.feats_to_sig(qfeats)
        return reconstructed_audio, feats, qfeats, codes


class AudioDatasetWithLang(Dataset):
    """Dataset for audio files with language labels and chunk indices."""

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

            language = "en"
            if 'language' in df.columns and pd.notna(row.get('language')):
                language = str(row['language'])

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
                        'language': language,
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
        return waveform.contiguous(), chunk_info['language'], idx


class FocalCodecASRTrainer:
    """Trainer for Stage 3 with ASR Loss."""

    def __init__(
        self,
        model,
        model_wrapper=None,
        device: str = 'cuda',
        enable_codebook_monitor: bool = True,
        codebook_size: int = 2048,
        weight_feature: float = 1.0,
        weight_time: float = 0.0,
        weight_mel: float = 5.0,
        weight_stft: float = 2.0,
        weight_asr: float = 20.0,
        use_mel_loss: bool = True,
        use_stft_loss: bool = True,
        use_asr_loss: bool = True,
        whisper_model: str = "small",
        whisper_cache_dir: str = None,
        sample_rate: int = 16000,
        token_cache_train: list = None,
        token_cache_val: list = None,
        use_amp: bool = True,
    ):
        self.model = model
        self.model_wrapper = model_wrapper
        self.device = device
        self.use_amp = use_amp
        self.enable_codebook_monitor = enable_codebook_monitor
        self.codebook_size = codebook_size
        self.sample_rate = sample_rate

        self.weight_feature = weight_feature
        self.weight_time = weight_time
        self.weight_mel = weight_mel
        self.weight_stft = weight_stft
        self.weight_asr = weight_asr
        self.use_mel_loss = use_mel_loss
        self.use_stft_loss = use_stft_loss
        self.use_asr_loss = use_asr_loss

        self.token_cache_train = token_cache_train
        self.token_cache_val = token_cache_val
        self.use_cache = token_cache_train is not None

        if use_mel_loss:
            self.mel_loss_fn = MelSpectrogramLoss(sample_rate=sample_rate).to(device)
        if use_stft_loss:
            self.stft_loss_fn = MultiResolutionSTFTLoss()
        if use_asr_loss:
            self.asr_loss_fn = WhisperASRLoss(
                model_name=whisper_model,
                default_language="en",
                download_root=whisper_cache_dir,
            ).to(device)

        print("\n" + "=" * 60)
        print("FocalCodec Stage 3 Training with ASR Loss")
        print("=" * 60)
        print(f"Loss Weights:")
        print(f"  Feature Loss: {weight_feature}")
        print(f"  Mel Loss: {weight_mel} {'(enabled)' if use_mel_loss else '(disabled)'}")
        print(f"  STFT Loss: {weight_stft} {'(enabled)' if use_stft_loss else '(disabled)'}")
        print(f"  ASR Loss: {weight_asr} {'(enabled)' if use_asr_loss else '(disabled)'}")
        if use_asr_loss:
            print(f"  Whisper Model: {whisper_model}")
            print(f"  Token Cache: {'ENABLED' if self.use_cache else 'DISABLED (online mode)'}")
            if self.use_cache:
                print(f"    Train cache: {len(token_cache_train)} entries")
                if token_cache_val:
                    print(f"    Val cache:   {len(token_cache_val)} entries")
        print(f"Mixed Precision (BF16 AMP): {'ENABLED' if use_amp else 'DISABLED'}")
        print("=" * 60 + "\n")

    def _get_cached_tokens(self, indices, cache):
        """Retrieve cached tokens by chunk indices."""
        tokens = []
        languages = []
        for idx in indices:
            idx = int(idx)
            if idx < len(cache):
                entry = cache[idx]
                tokens.append(entry['tokens'])
                languages.append(entry['language'])
            else:
                tokens.append(torch.tensor([], dtype=torch.int32))
                languages.append('en')
        return tokens, languages

    def compute_loss(
        self,
        audio: torch.Tensor,
        reconstructed_audio: torch.Tensor,
        feats: torch.Tensor,
        qfeats: torch.Tensor,
        languages: list = None,
        cached_tokens: list = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
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

        # Time domain loss (usually disabled)
        if self.weight_time > 0:
            audio_upsampled = torchaudio.functional.resample(
                audio, self.sample_rate, self.model.sample_rate_output
            )
            min_len = min(reconstructed_audio.shape[-1], audio_upsampled.shape[-1])
            time_loss = F.l1_loss(reconstructed_audio[..., :min_len], audio_upsampled[..., :min_len])
            total_loss += self.weight_time * time_loss
            loss_dict['time_loss'] = time_loss.item()

        # Mel spectrogram loss
        if self.use_mel_loss and self.weight_mel > 0:
            audio_upsampled = torchaudio.functional.resample(
                audio, self.sample_rate, self.model.sample_rate_output
            )
            min_len = min(reconstructed_audio.shape[-1], audio_upsampled.shape[-1])
            mel_loss = self.mel_loss_fn(
                reconstructed_audio[..., :min_len], audio_upsampled[..., :min_len]
            )
            total_loss += self.weight_mel * mel_loss
            loss_dict['mel_loss'] = mel_loss.item()

        # Multi-resolution STFT loss
        if self.use_stft_loss and self.weight_stft > 0:
            audio_upsampled = torchaudio.functional.resample(
                audio, self.sample_rate, self.model.sample_rate_output
            )
            min_len = min(reconstructed_audio.shape[-1], audio_upsampled.shape[-1])
            stft_loss = self.stft_loss_fn(
                reconstructed_audio[..., :min_len], audio_upsampled[..., :min_len]
            )
            total_loss += self.weight_stft * stft_loss
            loss_dict['stft_loss'] = stft_loss.item()

        # ASR Loss (Whisper Token Cross-Entropy)
        if self.use_asr_loss and self.weight_asr > 0:
            asr_loss = self.asr_loss_fn(
                clean_audio_16k=audio if cached_tokens is None else None,
                reconstructed_audio=reconstructed_audio,
                languages=languages,
                reconstructed_sr=self.model.sample_rate_output,
                cached_tokens=cached_tokens,
            )
            total_loss += self.weight_asr * asr_loss
            loss_dict['asr_loss'] = asr_loss.item()

        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict

    def train_epoch(self, dataloader, optimizer, epoch, gradient_clip=5.0,
                    old_params=None, warmup=False, rank=0, world_size=1):
        """Train one epoch.

        Args:
            old_params: list of parameters from old layers (for warmup control)
            warmup: if True, freeze old_params (only train new Stage 3 layers)
            rank: DDP local rank (0 = master)
            world_size: total number of processes
        """
        self.model.train()
        self.model.encoder.eval()
        self.model.decoder.eval()

        # Control warmup: freeze/unfreeze old layers
        if old_params is not None:
            for p in old_params:
                p.requires_grad = not warmup

        total_loss = 0
        num_batches = 0
        all_metrics = {}

        is_master = (rank == 0)
        iterable = tqdm(dataloader, desc=f"Train Epoch {epoch} {'[Warmup]' if warmup else ''}") \
            if is_master else dataloader

        for batch_idx, (audio, languages, indices) in enumerate(iterable):
            audio = audio.to(self.device)

            cached_tokens = None
            lang_list = list(languages)
            if self.use_cache and self.token_cache_train is not None:
                cached_tokens, lang_list = self._get_cached_tokens(
                    indices, self.token_cache_train
                )

            if self.use_amp:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    reconstructed_audio, feats, qfeats, codes = self.model_wrapper(audio)
                    loss, loss_dict = self.compute_loss(
                        audio, reconstructed_audio, feats, qfeats,
                        languages=lang_list,
                        cached_tokens=cached_tokens,
                    )
            else:
                reconstructed_audio, feats, qfeats, codes = self.model_wrapper(audio)
                loss, loss_dict = self.compute_loss(
                    audio, reconstructed_audio, feats, qfeats,
                    languages=lang_list,
                    cached_tokens=cached_tokens,
                )

            optimizer.zero_grad()
            loss.backward()

            if gradient_clip > 0:
                # Use model_wrapper (DDP or plain) to clip grads correctly
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model_wrapper.parameters() if p.requires_grad],
                    gradient_clip
                )

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            for key, value in loss_dict.items():
                if key not in all_metrics:
                    all_metrics[key] = 0
                all_metrics[key] += value

            if is_master:
                postfix = {'loss': f"{loss.item():.4f}"}
                if 'feature_loss' in loss_dict:
                    postfix['feat'] = f"{loss_dict['feature_loss']:.4f}"
                if 'asr_loss' in loss_dict:
                    postfix['asr'] = f"{loss_dict['asr_loss']:.4f}"
                iterable.set_postfix(postfix)

        avg_metrics = {k: v / num_batches for k, v in all_metrics.items()}

        # All-reduce metrics across all ranks so rank 0 has the global average
        if world_size > 1:
            for key in avg_metrics:
                t = torch.tensor(avg_metrics[key], device=self.device)
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
                avg_metrics[key] = t.item()

        return avg_metrics

    @torch.no_grad()
    def validate(self, dataloader, rank=0, world_size=1):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_metrics = {}
        all_toks = []

        is_master = (rank == 0)
        iterable = tqdm(dataloader, desc="Validation") if is_master else dataloader

        for audio, languages, indices in iterable:
            audio = audio.to(self.device)

            cached_tokens = None
            lang_list = list(languages)
            if self.use_cache and self.token_cache_val is not None:
                cached_tokens, lang_list = self._get_cached_tokens(
                    indices, self.token_cache_val
                )

            if self.use_amp:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    reconstructed_audio, feats, qfeats, codes = self.model_wrapper(audio)
                    loss, loss_dict = self.compute_loss(
                        audio, reconstructed_audio, feats, qfeats,
                        languages=lang_list,
                        cached_tokens=cached_tokens,
                    )
            else:
                reconstructed_audio, feats, qfeats, codes = self.model_wrapper(audio)
                loss, loss_dict = self.compute_loss(
                    audio, reconstructed_audio, feats, qfeats,
                    languages=lang_list,
                    cached_tokens=cached_tokens,
                )

            total_loss += loss.item()
            num_batches += 1

            for key, value in loss_dict.items():
                if key not in all_metrics:
                    all_metrics[key] = 0
                all_metrics[key] += value

            if self.enable_codebook_monitor:
                toks = self.model.codes_to_toks(codes)
                all_toks.append(toks.cpu())

        avg_metrics = {k: v / num_batches for k, v in all_metrics.items()}

        # All-reduce metrics across all ranks
        if world_size > 1:
            for key in avg_metrics:
                t = torch.tensor(avg_metrics[key], device=self.device)
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
                avg_metrics[key] = t.item()

        if self.enable_codebook_monitor and all_toks:
            all_toks = torch.cat(all_toks, dim=0)
            toks_flat = all_toks.flatten().long()
            toks_flat = torch.clamp(toks_flat, 0, self.codebook_size - 1)
            used_codes = torch.bincount(toks_flat, minlength=self.codebook_size)
            # All-reduce codebook counts so rank 0 sees all GPUs' data
            if world_size > 1:
                dist.all_reduce(used_codes, op=dist.ReduceOp.SUM)
            num_used = (used_codes > 0).sum().item()
            avg_metrics['codebook_usage'] = num_used / self.codebook_size
            avg_metrics['codebook_used_codes'] = num_used
            probs = used_codes.float() / used_codes.sum()
            probs = probs[probs > 0]
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            avg_metrics['codebook_perplexity'] = torch.exp(entropy).item()

        return avg_metrics

    def save_checkpoint(self, path, epoch, optimizer, metrics):
        import shutil
        os.makedirs(os.path.dirname(path), exist_ok=True)

        stat = shutil.disk_usage(os.path.dirname(path))
        available_gb = stat.free / (1024 ** 3)

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
            'stage': '3_12.5hz_asr'
        }

        try:
            torch.save(checkpoint, path)
            print(f"Checkpoint saved: {path} ({available_gb:.2f} GB remaining)")
        except RuntimeError as e:
            if "file write failed" in str(e) or "PytorchStreamWriter failed" in str(e):
                print(f"ERROR: Failed to save checkpoint - disk may be full")
            else:
                raise e


def load_token_cache(cache_dir, whisper_model, split_name):
    """Load pre-computed Whisper token cache if available."""
    cache_path = os.path.join(cache_dir, f'whisper_tokens_{split_name}_{whisper_model}.pt')
    if os.path.exists(cache_path):
        print(f"Loading token cache: {cache_path}")
        cache = torch.load(cache_path, map_location='cpu', weights_only=False)
        print(f"  Loaded {len(cache)} cached entries")
        return cache
    else:
        print(f"Token cache not found: {cache_path}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description='Stage 3 Fine-tuning with ASR Loss (12.5Hz)')
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--max_chunks', type=int, default=None)
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last_model.pt')
    # DDP args (set automatically by torchrun)
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for DDP (set by torchrun)')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    config_filename = os.path.basename(args.config)
    if 'exp_' in config_filename:
        exp_id = config_filename.split('exp_')[1].split('.')[0]
        experiment_name = f"exp_{exp_id}"
    else:
        experiment_name = "default"

    # ============================================================
    # DDP Initialization
    # ============================================================
    # torchrun sets LOCAL_RANK / RANK / WORLD_SIZE env vars automatically
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_ddp = world_size > 1
    is_master = (rank == 0)

    if is_ddp:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if is_master:
        print(f"Experiment: {experiment_name}")
        print(f"DDP: world_size={world_size}, rank={rank}, local_rank={local_rank}")
        print(f"Using device: {device}")
        print(f"Stage 3: Add 5th layer for 12.5Hz with ASR Loss")

    sys.path.insert(0, config.focalcodec_dir)
    from focalcodec import FocalCodec
    from focalcodec.focalnet import FocalDownScale, FocalUpScale

    # ============================================================
    # Load Whisper token cache
    # Reuse Exp J cache (same dataset, same whisper model)
    # Each rank loads its own copy (read-only, no sync needed)
    # ============================================================
    whisper_model_name = config.get('stage3.whisper_model', 'small')

    # Try experiment-specific cache first, then fall back to Exp J cache
    cache_dir = os.path.join(PROJECT_ROOT, 'experiments', experiment_name, 'whisper_cache')
    cache_dir_expJ = os.path.join(PROJECT_ROOT, 'experiments', 'exp_J', 'whisper_cache')

    token_cache_train = load_token_cache(cache_dir, whisper_model_name, 'train')
    token_cache_val = load_token_cache(cache_dir, whisper_model_name, 'val')

    if token_cache_train is None:
        if is_master:
            print(f"Trying Exp J cache (same dataset): {cache_dir_expJ}")
        token_cache_train = load_token_cache(cache_dir_expJ, whisper_model_name, 'train')
        token_cache_val = load_token_cache(cache_dir_expJ, whisper_model_name, 'val')

    if token_cache_train is None and is_master:
        print("\nWARNING: No token cache found, using online Whisper mode (SLOW)")
        print("Consider running: python precompute_whisper_tokens.py --config config_exp_K.yaml\n")

    # ============================================================
    # Load source checkpoint (Exp J Stage 2)
    # ============================================================
    source_exp = config.get('stage3.source_experiment', 'exp_J')
    source_stage = config.get('stage3.source_stage', 'stage2_25hz')

    if args.resume:
        ckpt_path = os.path.join(
            config.output_dir, experiment_name, 'stage3_12.5hz', 'last_model.pt'
        )
        print(f"\nResuming from: {ckpt_path}")
    else:
        ckpt_path = os.path.join(config.output_dir, source_exp, source_stage, 'best_model.pt')
        print(f"\nLoading source checkpoint: {ckpt_path}")
        print(f"  Source: {source_exp}/{source_stage}")

    if not os.path.exists(ckpt_path):
        if is_master:
            print(f"ERROR: Checkpoint not found: {ckpt_path}")
        if is_ddp:
            dist.destroy_process_group()
        return

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    source_epoch = checkpoint.get('epoch', 'N/A')
    source_val_loss = checkpoint.get('val_loss', 'N/A')
    if is_master:
        print(f"Source model: epoch {source_epoch}, val_loss {source_val_loss}")

    # ============================================================
    # Reconstruct Stage 2 model architecture
    # ============================================================
    model = FocalCodec.from_pretrained(
        config.base_model,
        cache_dir=config.model_cache_dir
    ).to(device)

    # Add 4th layer (Stage 2 architecture: 25Hz)
    if is_master:
        print("\nReconstructing Stage 2 architecture (4 layers, 25Hz)...")
    comp = model.compressor
    layer_4 = FocalDownScale(
        input_dim=1024, output_dim=1024, downscale_factor=2,
        focal_window=comp.focal_window, focal_level=comp.focal_level,
        focal_factor=comp.focal_factor, dropout=comp.dropout_,
        use_post_norm=comp.use_post_norm, use_layerscale=comp.use_layerscale,
        layerscale_init=comp.layerscale_init, tanhscale_init=comp.tanhscale_init,
        normalize_modulator=comp.normalize_modulator, causal=comp.causal,
        window_size=comp.window_size,
    ).to(device)
    comp.layers.append(layer_4)
    comp.downscale_factors = list(comp.downscale_factors) + [2]
    comp.downsample_factor = torch.Size(comp.downscale_factors).numel()
    comp.chunk_size = comp.downsample_factor

    decomp = model.decompressor
    layer_0 = FocalUpScale(
        input_dim=1024, output_dim=1024, upscale_factor=2,
        focal_window=decomp.focal_window, focal_level=decomp.focal_level,
        focal_factor=decomp.focal_factor, dropout=decomp.dropout_,
        use_post_norm=decomp.use_post_norm, use_layerscale=decomp.use_layerscale,
        layerscale_init=decomp.layerscale_init, tanhscale_init=decomp.tanhscale_init,
        normalize_modulator=decomp.normalize_modulator, causal=decomp.causal,
        window_size=decomp.window_size,
    ).to(device)
    decomp.layers.insert(0, layer_0)
    decomp.upscale_factors = [2] + list(decomp.upscale_factors)

    # Load Stage 2 weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if is_master:
        print(f"Loaded Stage 2 weights (epoch {source_epoch}, val_loss {source_val_loss})")

    # ============================================================
    # Add Stage 3 layers (5th layer: 12.5Hz)
    # ============================================================
    if is_master:
        print("\nAdding Stage 3 layers (5th layer, 12.5Hz)...")

    # Compressor: append new layer_5 at end
    layer_5 = FocalDownScale(
        input_dim=1024, output_dim=1024, downscale_factor=2,
        focal_window=comp.focal_window, focal_level=comp.focal_level,
        focal_factor=comp.focal_factor, dropout=comp.dropout_,
        use_post_norm=comp.use_post_norm, use_layerscale=comp.use_layerscale,
        layerscale_init=comp.layerscale_init, tanhscale_init=comp.tanhscale_init,
        normalize_modulator=comp.normalize_modulator, causal=comp.causal,
        window_size=comp.window_size,
    ).to(device)
    comp.layers.append(layer_5)
    comp.downscale_factors = list(comp.downscale_factors) + [2]
    comp.downsample_factor = torch.Size(comp.downscale_factors).numel()
    comp.chunk_size = comp.downsample_factor
    print(f"  Compressor: downscale_factors = {comp.downscale_factors}")

    # Decompressor: prepend new layer_0 at beginning
    new_decomp_layer_0 = FocalUpScale(
        input_dim=1024, output_dim=1024, upscale_factor=2,
        focal_window=decomp.focal_window, focal_level=decomp.focal_level,
        focal_factor=decomp.focal_factor, dropout=decomp.dropout_,
        use_post_norm=decomp.use_post_norm, use_layerscale=decomp.use_layerscale,
        layerscale_init=decomp.layerscale_init, tanhscale_init=decomp.tanhscale_init,
        normalize_modulator=decomp.normalize_modulator, causal=decomp.causal,
        window_size=decomp.window_size,
    ).to(device)
    decomp.layers.insert(0, new_decomp_layer_0)
    decomp.upscale_factors = [2] + list(decomp.upscale_factors)
    print(f"  Decompressor: upscale_factors = {decomp.upscale_factors}")

    # Resume: load Stage 3 weights if available
    if args.resume and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        if is_master:
            print(f"Resumed Stage 3 weights")

    # ============================================================
    # Training strategy: Dual LR (old layers vs new layer)
    # ============================================================
    if is_master:
        print("\n" + "=" * 60)
        print("Training Strategy: Stage 3 with Dual LR")
        print("=" * 60)

    model.encoder.eval()
    for param in model.encoder.parameters():
        param.requires_grad = False
    if is_master:
        print("Encoder: frozen")

    model.decoder.eval()
    for param in model.decoder.parameters():
        param.requires_grad = False
    if is_master:
        print("Decoder: frozen")

    # Old layers: compressor layers[0..3], decompressor layers[1..4]
    # (indices after Stage 3 insertion: comp[0-3] are old, decomp[1-4] are old)
    old_params = []
    for i in range(4):
        old_params.extend(list(comp.layers[i].parameters()))
        old_params.extend(list(decomp.layers[i + 1].parameters()))
    # Also include quantizer and projection layers (inherited from Stage 2)
    old_params.extend(list(model.quantizer.parameters()))

    # New layers: compressor layer[4] and decompressor layer[0]
    new_params = (
        list(comp.layers[4].parameters()) +
        list(decomp.layers[0].parameters()) +
        list(comp.out_proj.parameters()) +
        list(decomp.in_proj.parameters())
    )

    lr_old = config.get('stage3.learning_rate_old', 5e-5)
    lr_new = config.get('stage3.learning_rate_new', 5e-4)

    if is_master:
        print(f"Compressor layers [0-3] (old): LR = {lr_old}")
        print(f"Compressor layer [4] (new):    LR = {lr_new}")
        print(f"Decompressor layer [0] (new):  LR = {lr_new}")
        print(f"Decompressor layers [1-4] (old): LR = {lr_old}")

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"\nTrainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        print("=" * 60)

    # ============================================================
    # Dataset & DataLoader
    # ============================================================
    batch_size = args.batch_size or config.get('stage3.batch_size', 64)
    num_epochs = args.num_epochs or config.get('stage3.num_epochs', 10000)
    max_chunks = args.max_chunks or config.get('stage3.max_chunks', 0)
    warmup_epochs = config.get('stage3.warmup_epochs', 5)

    if is_master:
        print(f"\nTraining parameters:")
        print(f"  Batch Size per GPU: {batch_size}")
        print(f"  Effective Batch Size: {batch_size * world_size}")
        print(f"  World Size (GPUs): {world_size}")
        print(f"  Warmup Epochs (new layers only): {warmup_epochs}")
        print(f"  Max Chunks: {max_chunks if max_chunks > 0 else 'all'}")

    max_chunks_train = max_chunks if max_chunks > 0 else None

    train_dataset = AudioDatasetWithLang(
        csv_path=config.train_csv,
        base_path=config.audio_base_path,
        chunk_duration=config.get('stage3.chunk_duration', 3.0),
        overlap=config.get('stage3.overlap', 0.5),
        max_chunks=max_chunks_train
    )

    val_dataset = AudioDatasetWithLang(
        csv_path=config.val_csv,
        base_path=config.audio_base_path,
        chunk_duration=config.get('stage3.chunk_duration', 3.0),
        overlap=0.0,
        max_chunks=None
    )

    # Verify cache size — only invalidate if cache is smaller than dataset
    # (cache > dataset is OK when max_chunks trims the dataset)
    if token_cache_train is not None and len(token_cache_train) < len(train_dataset):
        if is_master:
            print(f"ERROR: Token cache size ({len(token_cache_train)}) < "
                  f"dataset size ({len(train_dataset)})")
            print("Falling back to online Whisper mode.")
        token_cache_train = None
        token_cache_val = None

    # DDP: use DistributedSampler so each rank sees a different shard
    if is_ddp:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        train_shuffle = False  # sampler handles shuffling
    else:
        train_sampler = None
        val_sampler = None
        train_shuffle = True

    num_workers = config.get('stage3.num_workers', 4)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers, pin_memory=True
    )

    # ============================================================
    # Model wrapper + Trainer
    # ============================================================
    wrapper = FocalCodecTrainWrapper(model)
    wrapper = wrapper.to(device)
    if is_ddp:
        wrapper = DDP(wrapper, device_ids=[local_rank], find_unused_parameters=False)
        if is_master:
            print(f"\nUsing DDP with {world_size} GPUs")
    else:
        if is_master:
            print("\nUsing single GPU")

    use_amp = config.get('stage3.use_amp', True)
    # Pass the underlying model (not DDP wrapper) to trainer for param access
    trainer = FocalCodecASRTrainer(
        model=model, model_wrapper=wrapper, device=device,
        enable_codebook_monitor=config.get('stage3.enable_codebook_monitor', True),
        codebook_size=2048,
        weight_feature=config.get('stage3.weight_feature', 1.0),
        weight_time=config.get('stage3.weight_time', 0.0),
        weight_mel=config.get('stage3.weight_mel', 5.0),
        weight_stft=config.get('stage3.weight_stft', 2.0),
        weight_asr=config.get('stage3.weight_asr', 20.0),
        use_mel_loss=config.get('stage3.use_mel_loss', True),
        use_stft_loss=config.get('stage3.use_stft_loss', True),
        use_asr_loss=config.get('stage3.use_asr_loss', True),
        whisper_model=whisper_model_name,
        whisper_cache_dir=config.get('paths.asr_cache_dir', None),
        token_cache_train=token_cache_train,
        token_cache_val=token_cache_val,
        use_amp=use_amp,
    )

    # ============================================================
    # Optimizer + Scheduler
    # ============================================================
    optimizer = AdamW(
        [
            {'params': old_params, 'lr': lr_old},
            {'params': new_params, 'lr': lr_new},
        ],
        weight_decay=config.get('stage3.weight_decay', 0.01)
    )

    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        factor=config.get('stage3.scheduler_factor', 0.5),
        patience=config.get('stage3.scheduler_patience', 8),
        threshold=config.get('stage3.scheduler_threshold', 0.001),
        cooldown=config.get('stage3.scheduler_cooldown', 3),
        min_lr=config.get('stage3.scheduler_min_lr', 1e-6),
        verbose=False
    )

    start_epoch = 1
    if args.resume and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        if is_master:
            print(f"Resumed optimizer, starting from epoch {start_epoch}")

    # ============================================================
    # Checkpoint & Logging (rank 0 only)
    # ============================================================
    checkpoint_dir = os.path.join(config.output_dir, experiment_name, 'stage3_12.5hz')
    if is_master:
        os.makedirs(checkpoint_dir, exist_ok=True)

    experiments_dir = os.path.join(PROJECT_ROOT, 'experiments', experiment_name)
    if is_master:
        os.makedirs(experiments_dir, exist_ok=True)
    log_file = os.path.join(experiments_dir, 'stage3_12.5hz_asr_training_ddp.csv')

    if is_master:
        if args.resume and os.path.exists(log_file):
            print(f"Appending to existing log: {log_file}")
        else:
            with open(log_file, 'w') as f:
                f.write('epoch,train_loss,val_loss,feature_loss,asr_loss,'
                        'codebook_usage,codebook_perplexity,lr_old,lr_new\n')
        print(f"Logging to: {log_file}")

    # ============================================================
    # Training Loop
    # ============================================================
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    patience = config.get('stage3.patience', 40)
    min_delta = config.get('stage3.min_delta', 0.0001)

    if is_master:
        print(f"\n{'='*60}")
        print(f"Starting Stage 3 DDP Training (12.5Hz, ASR Loss)")
        print(f"{'='*60}")
        print(f"Epochs: {num_epochs}, Warmup: {warmup_epochs}")
        print(f"Patience: {patience}")
        print(f"Checkpoint: {checkpoint_dir}")
        print(f"Target: 12.5 Hz × 11-bit BSQ = 137.5 bps")
        print(f"Goal: dCER < 10% (ZH), dWER < 5% (EN)")
        print("=" * 60)

    for epoch in range(start_epoch, num_epochs + 1):
        # DistributedSampler must be told the epoch for proper shuffling
        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if is_master:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")

        is_warmup = (epoch <= warmup_epochs) and not args.resume

        train_metrics = trainer.train_epoch(
            train_loader, optimizer, epoch,
            gradient_clip=config.get('stage3.gradient_clip', 5.0),
            old_params=old_params,
            warmup=is_warmup,
            rank=rank,
            world_size=world_size,
        )
        val_metrics = trainer.validate(val_loader, rank=rank, world_size=world_size)

        # Scheduler and logging: rank 0 only
        if is_master:
            scheduler.step(val_metrics['total_loss'])

            lr_old_cur = optimizer.param_groups[0]['lr']
            lr_new_cur = optimizer.param_groups[1]['lr']
            print(f"\nLR: old={lr_old_cur:.2e}, new={lr_new_cur:.2e}")

            print(f"\nTrain: " + ", ".join(
                f"{k}={v:.4f}" for k, v in train_metrics.items()
            ))
            print(f"Val:   " + ", ".join(
                f"{k}={v:.4f}" for k, v in val_metrics.items()
            ))

            with open(log_file, 'a') as f:
                f.write(
                    f"{epoch},"
                    f"{train_metrics.get('total_loss', 0):.6f},"
                    f"{val_metrics.get('total_loss', 0):.6f},"
                    f"{val_metrics.get('feature_loss', 0):.6f},"
                    f"{val_metrics.get('asr_loss', 0):.6f},"
                    f"{val_metrics.get('codebook_usage', 0):.6f},"
                    f"{val_metrics.get('codebook_perplexity', 0):.2f},"
                    f"{lr_old_cur:.2e},"
                    f"{lr_new_cur:.2e}\n"
                )

            current_val_loss = val_metrics['total_loss']

            if current_val_loss < best_val_loss - min_delta:
                improvement = best_val_loss - current_val_loss
                best_val_loss = current_val_loss
                best_epoch = epoch
                patience_counter = 0

                best_path = os.path.join(checkpoint_dir, "best_model.pt")
                trainer.save_checkpoint(
                    best_path, epoch, optimizer,
                    {'train': train_metrics, 'val': val_metrics}
                )
                print(f"New best! val_loss={best_val_loss:.6f} (improved {improvement:.6f})")
            else:
                patience_counter += 1
                print(f"No improvement {patience_counter}/{patience} "
                      f"(best={best_val_loss:.6f} @ epoch {best_epoch})")

            last_path = os.path.join(checkpoint_dir, "last_model.pt")
            trainer.save_checkpoint(
                last_path, epoch, optimizer,
                {'train': train_metrics, 'val': val_metrics}
            )

        # Broadcast early-stopping decision from rank 0 to all ranks
        if is_ddp:
            stop_tensor = torch.tensor(
                1 if (is_master and patience_counter >= patience) else 0,
                device=device
            )
            dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item() == 1:
                if is_master:
                    print(f"\nEarly stopping triggered!")
                break
        else:
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered!")
                break

    if is_master:
        print("\n" + "=" * 60)
        print("Stage 3 DDP Training Completed!")
        print("=" * 60)
        print(f"Best model: epoch {best_epoch}, val_loss={best_val_loss:.6f}")
        print(f"Checkpoint: {os.path.join(checkpoint_dir, 'best_model.pt')}")
        print(f"\nNext steps:")
        print(f"  sbatch bash/slurm_infer_eval_stage3_K.sh")
        print("=" * 60)

    if is_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
