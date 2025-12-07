#!/usr/bin/env python3

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import warnings
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/jieshiang/Desktop/GitHub/FocalCodec/focalcodec')
from focalcodec import FocalCodec


TRAIN_CSV = '/home/jieshiang/Desktop/GitHub/FocalCodec/experiments/data_commonvoice/train_split.csv'
VAL_CSV = '/home/jieshiang/Desktop/GitHub/FocalCodec/experiments/data_commonvoice/val_split.csv'
MODEL_CONFIG = "lucadellalib/focalcodec_50hz_2k_causal"
MODEL_CACHE_DIR = "/mnt/Internal/jieshiang/Model/FocalCodec"
DEFAULT_CHECKPOINT_DIR = '/mnt/Internal/jieshiang/Model/FocalCodec/checkpoints_focalcodec'
LOG_DIR = '/mnt/Internal/jieshiang/Model/FocalCodec/logs'


class AudioDataset(Dataset):
    """Dataset for loading audio chunks from CSV file."""

    def __init__(
        self,
        csv_path: str,
        sample_rate: int = 16000,
        chunk_duration: float = 3.0,
        overlap: float = 0.5,
        augment: bool = False,
        max_chunks: Optional[int] = None
    ):
        self.csv_path = csv_path
        self.sample_rate = sample_rate
        self.chunk_size = int(chunk_duration * sample_rate)
        self.hop_size = int((chunk_duration - overlap) * sample_rate)
        self.augment = augment
        self.chunks = []

        if self.augment:
            self.augmentation = nn.Sequential(
                # Simple augmentation pipeline (can be extended)
            )

        df = pd.read_csv(csv_path)

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading dataset"):
            file_path = row['file_path']  # CSV column name is 'file_path'

            # Convert relative path to absolute path (assuming base is /mnt/Internal/ASR)
            if file_path.startswith('./'):
                file_path = file_path.replace('./', '/mnt/Internal/ASR/', 1)

            try:
                info = torchaudio.info(file_path)
                file_sr = info.sample_rate
                file_num_frames = info.num_frames
            except:
                continue  # Skip files that can't be read

            expected_frames = int(file_num_frames * sample_rate / file_sr)

            if expected_frames >= self.chunk_size:
                chunk_frames_in_source = int(self.chunk_size * file_sr / sample_rate)
                hop_frames_in_source = int(self.hop_size * file_sr / sample_rate)

                n_chunks = (file_num_frames - chunk_frames_in_source) // hop_frames_in_source + 1

                for i in range(n_chunks):
                    start_frame = i * hop_frames_in_source
                    self.chunks.append({
                        'file_path': file_path,
                        'start': start_frame,
                        'num_frames': chunk_frames_in_source,
                        'file_sr': file_sr
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

        if self.augment:
            waveform = self.augmentation(waveform)

        return waveform.contiguous()


class FlexibleLoss(nn.Module):

    def __init__(
        self,
        use_feature: bool = True,
        use_time: bool = False,
        use_mel: bool = False,
        weight_feature: float = 1.0,
        weight_time: float = 0.3,
        weight_mel: float = 0.3,
        sample_rate: int = 16000
    ):
        super().__init__()
        self.use_feature = use_feature
        self.use_time = use_time
        self.use_mel = use_mel
        self.weight_feature = weight_feature
        self.weight_time = weight_time
        self.weight_mel = weight_mel

        if use_mel:
            self.mel_transform = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=1024,
                hop_length=256,
                n_mels=80
            )

    def forward(
        self,
        original_audio: torch.Tensor,
        reconstructed_audio: torch.Tensor,
        original_features: torch.Tensor,
        reconstructed_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = 0
        loss_dict = {}

        if self.use_feature:
            feature_loss = F.mse_loss(reconstructed_features, original_features)
            total_loss += self.weight_feature * feature_loss
            loss_dict['feature_loss'] = feature_loss.item()

        if self.use_time:
            time_loss = F.l1_loss(reconstructed_audio, original_audio)
            total_loss += self.weight_time * time_loss
            loss_dict['time_loss'] = time_loss.item()

        if self.use_mel:
            mel_orig = self.mel_transform(original_audio)
            mel_recon = self.mel_transform(reconstructed_audio)
            mel_loss = F.l1_loss(mel_recon, mel_orig)
            total_loss += self.weight_mel * mel_loss
            loss_dict['mel_loss'] = mel_loss.item()

        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict


def apply_lora_to_component(component: nn.Module, lora_rank: int, lora_alpha: int):
    try:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
        )

        component = get_peft_model(component, lora_config)
        print(f"  LoRA applied: rank={lora_rank}, alpha={lora_alpha}")
        return component
    except ImportError:
        print("  Warning: peft not installed, LoRA not applied")
        return component


class FocalCodecTrainer:

    def __init__(
        self,
        model: FocalCodec,
        train_components: Dict[str, bool],
        freeze_components: Dict[str, bool],
        lora_config: Dict[str, Optional[Tuple[int, int]]],
        loss_config: Dict[str, any],
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.device = device
        self.train_components = train_components
        self.freeze_components = freeze_components

        print("\n" + "="*60)
        print("FocalCodec Trainer Initialization")
        print("="*60)

        self._setup_training_state()

        self._apply_lora(lora_config)

        self.criterion = FlexibleLoss(**loss_config).to(device)

        self._print_parameter_stats()

    def _setup_training_state(self):
        print("\nComponent Training Status:")
        print("-" * 60)

        components = {
            'encoder': self.model.encoder,
            'compressor': self.model.compressor,
            'decompressor': self.model.decompressor,
            'decoder': self.model.decoder
        }

        for name, component in components.items():
            for param in component.parameters():
                param.requires_grad = True

            if self.freeze_components.get(name, False):
                for param in component.parameters():
                    param.requires_grad = False
                status = "❄️  FROZEN"
            elif self.train_components.get(name, False):
                status = " TRAINING"
            else:
                for param in component.parameters():
                    param.requires_grad = False
                status = "⏸️  DISABLED"

            print(f"  {name:15s}: {status}")

        print(f"  {'quantizer':15s}: ⚙️  NON-PARAMETRIC (always frozen)")

    def _apply_lora(self, lora_config: Dict[str, Optional[Tuple[int, int]]]):
        print("\nLoRA Configuration:")
        print("-" * 60)

        components = {
            'encoder': self.model.encoder,
            'compressor': self.model.compressor,
            'decompressor': self.model.decompressor,
            'decoder': self.model.decoder
        }

        for name, component in components.items():
            if not self.train_components.get(name, False):
                print(f"  {name:15s}: Skipped (not training)")
                continue

            lora_params = lora_config.get(name)
            if lora_params:
                rank, alpha = lora_params
                print(f"  {name:15s}: rank={rank}, alpha={alpha}")
                setattr(self.model, name, apply_lora_to_component(component, rank, alpha))
            else:
                print(f"  {name:15s}: No LoRA")

    def _print_parameter_stats(self):
        print("\nParameter Statistics:")
        print("-" * 60)

        components = {
            'encoder': self.model.encoder,
            'compressor': self.model.compressor,
            'decompressor': self.model.decompressor,
            'decoder': self.model.decoder
        }

        total_params = 0
        trainable_params = 0

        for name, component in components.items():
            params = sum(p.numel() for p in component.parameters())
            train_params = sum(p.numel() for p in component.parameters() if p.requires_grad)

            total_params += params
            trainable_params += train_params

            print(f"  {name:15s}: {params/1e6:>8.2f}M total, {train_params/1e6:>8.2f}M trainable")

        print(f"  {'TOTAL':15s}: {total_params/1e6:>8.2f}M total, {trainable_params/1e6:>8.2f}M trainable")
        print(f"  Trainable ratio: {100*trainable_params/total_params:.2f}%")
        print("="*60 + "\n")

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int
    ) -> Dict[str, float]:
        self.model.train()

        total_losses = {}
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, waveform in enumerate(pbar):
            waveform = waveform.to(self.device)

            optimizer.zero_grad()

            original_features = self.model.sig_to_feats(waveform)

            codes = self.model.sig_to_codes(waveform)
            reconstructed_features = self.model.codes_to_qfeats(codes)

            reconstructed_audio = self.model.codes_to_sig(codes)

            min_len = min(waveform.shape[-1], reconstructed_audio.shape[-1])
            waveform = waveform[..., :min_len]
            reconstructed_audio = reconstructed_audio[..., :min_len]

            loss, loss_dict = self.criterion(
                waveform,
                reconstructed_audio,
                original_features,
                reconstructed_features
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            for key, value in loss_dict.items():
                if key not in total_losses:
                    total_losses[key] = 0
                total_losses[key] += value

            num_batches += 1

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()

        total_losses = {}
        num_batches = 0

        for waveform in tqdm(val_loader, desc="Validating"):
            waveform = waveform.to(self.device)

            original_features = self.model.sig_to_feats(waveform)
            codes = self.model.sig_to_codes(waveform)
            reconstructed_features = self.model.codes_to_qfeats(codes)
            reconstructed_audio = self.model.codes_to_sig(codes)

            min_len = min(waveform.shape[-1], reconstructed_audio.shape[-1])
            waveform = waveform[..., :min_len]
            reconstructed_audio = reconstructed_audio[..., :min_len]

            _, loss_dict = self.criterion(
                waveform,
                reconstructed_audio,
                original_features,
                reconstructed_features
            )

            for key, value in loss_dict.items():
                if key not in total_losses:
                    total_losses[key] = 0
                total_losses[key] += value

            num_batches += 1

        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses

    def save_checkpoint(self, path: str, epoch: int, optimizer: torch.optim.Optimizer, metrics: Dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'train_components': self.train_components,
            'freeze_components': self.freeze_components,
        }

        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description='FocalCodec Direct Fine-tuning')

    # Dataset parameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--chunk_duration', type=float, default=3.0)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--augment', action='store_true')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--gradient_clip', type=float, default=1.0)

    # Component training flags
    parser.add_argument('--train_encoder', action='store_true')
    parser.add_argument('--train_compressor', action='store_true')
    parser.add_argument('--train_decompressor', action='store_true')
    parser.add_argument('--train_decoder', action='store_true')

    # Component freezing flags
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--freeze_compressor', action='store_true')
    parser.add_argument('--freeze_decompressor', action='store_true')
    parser.add_argument('--freeze_decoder', action='store_true')

    # LoRA parameters
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_encoder', action='store_true')
    parser.add_argument('--lora_compressor', action='store_true')
    parser.add_argument('--lora_decompressor', action='store_true')
    parser.add_argument('--lora_decoder', action='store_true')

    # Loss parameters
    parser.add_argument('--use_feature_loss', action='store_true')
    parser.add_argument('--use_time_loss', action='store_true')
    parser.add_argument('--use_mel_loss', action='store_true')
    parser.add_argument('--weight_feature', type=float, default=1.0)
    parser.add_argument('--weight_time', type=float, default=0.3)
    parser.add_argument('--weight_mel', type=float, default=0.3)

    # Other parameters
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--checkpoint_dir', type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')

    # Early stopping
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (epochs)')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='Minimum change to qualify as improvement')

    return parser.parse_args()


def main():
    args = parse_args()

    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if not any([args.train_encoder, args.train_compressor, args.train_decompressor, args.train_decoder]):
        print("ERROR: At least one component must be trained!")
        print("Use --train_encoder, --train_compressor, --train_decompressor, or --train_decoder")
        return

    if not any([args.use_feature_loss, args.use_time_loss, args.use_mel_loss]):
        print("ERROR: At least one loss must be enabled!")
        print("Use --use_feature_loss, --use_time_loss, or --use_mel_loss")
        return

    print(f"\nLoading model: {MODEL_CONFIG}")
    model = FocalCodec.from_pretrained(
        MODEL_CONFIG,
        cache_dir=MODEL_CACHE_DIR
    )

    train_components = {
        'encoder': args.train_encoder,
        'compressor': args.train_compressor,
        'decompressor': args.train_decompressor,
        'decoder': args.train_decoder
    }

    freeze_components = {
        'encoder': args.freeze_encoder,
        'compressor': args.freeze_compressor,
        'decompressor': args.freeze_decompressor,
        'decoder': args.freeze_decoder
    }

    lora_config = {}
    if args.use_lora:
        if args.lora_encoder:
            lora_config['encoder'] = (args.lora_rank, args.lora_alpha)
        if args.lora_compressor:
            lora_config['compressor'] = (args.lora_rank, args.lora_alpha)
        if args.lora_decompressor:
            lora_config['decompressor'] = (args.lora_rank, args.lora_alpha)
        if args.lora_decoder:
            lora_config['decoder'] = (args.lora_rank, args.lora_alpha)

    loss_config = {
        'use_feature': args.use_feature_loss,
        'use_time': args.use_time_loss,
        'use_mel': args.use_mel_loss,
        'weight_feature': args.weight_feature,
        'weight_time': args.weight_time,
        'weight_mel': args.weight_mel,
        'sample_rate': 16000
    }

    trainer = FocalCodecTrainer(
        model=model,
        train_components=train_components,
        freeze_components=freeze_components,
        lora_config=lora_config,
        loss_config=loss_config,
        device=device
    )

    print("\nPreparing datasets...")
    train_dataset = AudioDataset(
        csv_path=TRAIN_CSV,
        sample_rate=16000,
        chunk_duration=args.chunk_duration,
        overlap=args.overlap,
        augment=args.augment
    )

    val_dataset = AudioDataset(
        csv_path=VAL_CSV,
        sample_rate=16000,
        chunk_duration=args.chunk_duration,
        overlap=0.0,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, trainer.model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    start_epoch = 1
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    print(f"\nStarting training for {args.num_epochs} epochs...")
    if args.early_stopping:
        print(f"Early stopping enabled: patience={args.patience}, min_delta={args.min_delta}")

    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in range(start_epoch, args.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*60}")

        train_metrics = trainer.train_epoch(train_loader, optimizer, epoch)

        val_metrics = trainer.validate(val_loader)

        print(f"\nTrain metrics:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value:.6f}")

        print(f"\nValidation metrics:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.6f}")

        current_val_loss = val_metrics['total_loss']

        if current_val_loss < best_val_loss - args.min_delta:
            improvement = best_val_loss - current_val_loss
            best_val_loss = current_val_loss
            best_epoch = epoch
            patience_counter = 0

            best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            trainer.save_checkpoint(
                best_path,
                epoch,
                optimizer,
                {'train': train_metrics, 'val': val_metrics}
            )
            print(f"✅ New best model saved! (val_loss: {best_val_loss:.6f}, improved by {improvement:.6f})")
        else:
            patience_counter += 1
            print(f"⚠️  No improvement for {patience_counter} epoch(s) (best: {best_val_loss:.6f} at epoch {best_epoch})")

            if args.early_stopping and patience_counter >= args.patience:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered!")
                print(f"{'='*60}")
                print(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
                print(f"No improvement for {patience_counter} consecutive epochs")
                break

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Best model: epoch {best_epoch} with val_loss {best_val_loss:.6f}")
    if args.early_stopping and patience_counter >= args.patience:
        print(f"Training stopped early at epoch {epoch}/{args.num_epochs}")
    else:
        print(f"Completed all {args.num_epochs} epochs")


if __name__ == '__main__':
    main()
