#!/usr/bin/env python3
"""
FocalCodec Training with Whisper-tiny ASR Loss (Cross Entropy)
Uses ground truth transcriptions to compute cross entropy loss
"""

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
    """Dataset for loading audio chunks with transcriptions."""

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
            self.augmentation = nn.Sequential()

        df = pd.read_csv(csv_path)
        
        # 確認 CSV 有 transcription 欄位
        if 'transcription' not in df.columns:
            raise ValueError(f"CSV must have 'transcription' column! Found: {df.columns.tolist()}")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading dataset"):
            file_path = row['file_path']
            transcription = str(row['transcription']) if pd.notna(row['transcription']) else ""

            # Convert relative path to absolute path
            if file_path.startswith('./'):
                file_path = file_path.replace('./', '/mnt/Internal/ASR/', 1)

            try:
                info = torchaudio.info(file_path)
                file_sr = info.sample_rate
                file_num_frames = info.num_frames
            except:
                continue

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

        if self.augment:
            waveform = self.augmentation(waveform)

        return waveform.contiguous(), chunk_info['transcription']


class WhisperASRLoss(nn.Module):
    """
    Whisper ASR-based loss using Cross Entropy
    Computes CE between ground truth text and reconstructed audio transcription
    """
    
    def __init__(self, device: str = "cuda", cache_dir: Optional[str] = None):
        super().__init__()
        self.device = device
        self.cache_dir = cache_dir or "/mnt/Internal/jieshiang/Model/ASR"
        
        print(f"\n{'='*60}")
        print(f"Loading Whisper-tiny ASR model")
        print(f"{'='*60}")
        
        self._load_whisper()
        
        # Freeze Whisper model
        self.whisper_model.eval()
        for param in self.whisper_model.parameters():
            param.requires_grad = False
        
        print(f"Whisper model loaded and frozen")
        print(f"{'='*60}\n")
    
    def _load_whisper(self):
        """Load Whisper-tiny model"""
        try:
            import whisper
            
            # 設定下載目錄
            whisper_cache = os.path.join(self.cache_dir, "whisper")
            os.makedirs(whisper_cache, exist_ok=True)
            
            os.environ['XDG_CACHE_HOME'] = whisper_cache
            
            self.whisper_model = whisper.load_model(
                "tiny", 
                device=self.device,
                download_root=whisper_cache
            )
            
            # 獲取 tokenizer
            self.tokenizer = whisper.tokenizer.get_tokenizer(
                multilingual=self.whisper_model.is_multilingual
            )
            
            print(f"Whisper model cached at: {whisper_cache}")
            
        except ImportError:
            raise ImportError("Please install openai-whisper: pip install openai-whisper")
    
    @torch.no_grad()
    def encode_text_to_tokens(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        將文字編碼為 token IDs
        
        Returns:
            tokens: [batch_size, max_len] token IDs
            lengths: [batch_size] 每個序列的有效長度
        """
        import whisper
        
        batch_tokens = []
        batch_lengths = []
        
        for text in texts:
            # Whisper tokenization
            # 加入 SOT (start of transcript) token
            tokens = [self.tokenizer.sot]
            tokens += self.tokenizer.encode(text)
            tokens.append(self.tokenizer.eot)  # EOT (end of transcript)
            
            batch_tokens.append(tokens)
            batch_lengths.append(len(tokens))
        
        # Pad to same length
        max_len = max(batch_lengths)
        padded_tokens = []
        
        for tokens in batch_tokens:
            padded = tokens + [self.tokenizer.eot] * (max_len - len(tokens))
            padded_tokens.append(padded)
        
        return (
            torch.tensor(padded_tokens, device=self.device),
            torch.tensor(batch_lengths, device=self.device)
        )
    
    @torch.no_grad()
    def get_whisper_encoder_output(self, audio: torch.Tensor) -> torch.Tensor:
        """
        獲取 Whisper encoder 的輸出

        Args:
            audio: [batch_size, samples]

        Returns:
            encoder_output: [batch_size, n_frames, n_embed]
        """
        import whisper

        batch_size = audio.shape[0]
        all_encoder_outputs = []

        for i in range(batch_size):
            audio_np = audio[i].cpu().numpy()

            # Pad or trim audio to 30 seconds (480000 samples at 16kHz)
            # Whisper expects exactly 30 seconds of audio
            target_length = 16000 * 30  # 480000 samples
            if len(audio_np) < target_length:
                # Pad with zeros
                audio_np = np.pad(audio_np, (0, target_length - len(audio_np)), mode='constant')
            else:
                # Trim to 30 seconds
                audio_np = audio_np[:target_length]

            # 獲取 mel spectrogram
            mel = whisper.log_mel_spectrogram(audio_np).to(self.device)

            # Encoder forward
            encoder_output = self.whisper_model.encoder(mel.unsqueeze(0))
            all_encoder_outputs.append(encoder_output)

        return torch.cat(all_encoder_outputs, dim=0)
    
    def forward(
        self,
        original_audio: torch.Tensor,
        reconstructed_audio: torch.Tensor,
        ground_truth_texts: List[str]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        計算 ASR Cross Entropy Loss
        
        Args:
            original_audio: [batch_size, samples] - 不使用,但保留介面一致性
            reconstructed_audio: [batch_size, samples] - 重建的音訊
            ground_truth_texts: List[str] - Ground truth 文字
        
        Returns:
            loss: ASR cross entropy loss
            info: Loss information dictionary
        """
        
        if not ground_truth_texts or all(t == "" for t in ground_truth_texts):
            # 如果沒有 ground truth,返回零 loss
            return torch.tensor(0.0, device=self.device), {'asr_loss': 0.0}
        
        batch_size = reconstructed_audio.shape[0]
        
        # 1. 將 ground truth 文字編碼為 tokens
        gt_tokens, gt_lengths = self.encode_text_to_tokens(ground_truth_texts)
        # gt_tokens: [batch_size, max_len]
        
        # 2. 獲取重建音訊的 encoder output
        encoder_output = self.get_whisper_encoder_output(reconstructed_audio)
        # encoder_output: [batch_size, n_audio_frames, n_embed]
        
        # 3. Decoder forward pass 獲取 logits
        # Whisper decoder 是 autoregressive,需要逐步 decode
        total_loss = 0.0
        total_tokens = 0
        
        for i in range(batch_size):
            # 當前樣本的 encoder output
            enc_out = encoder_output[i:i+1]  # [1, n_frames, n_embed]
            
            # Ground truth tokens (包含 SOT 和 EOT)
            tokens = gt_tokens[i]  # [max_len]
            length = gt_lengths[i].item()
            
            # 使用 teacher forcing: input = tokens[:-1], target = tokens[1:]
            # 即: 用前 N-1 個 token 預測第 N 個 token
            input_tokens = tokens[:-1].unsqueeze(0)  # [1, seq_len-1]
            target_tokens = tokens[1:length]  # [seq_len-1] (去掉最後的 padding)
            
            if len(target_tokens) == 0:
                continue
            
            # Decoder forward
            logits = self.whisper_model.decoder(input_tokens, enc_out)
            # logits: [1, seq_len-1, vocab_size]
            
            # 計算 cross entropy
            logits = logits[0, :len(target_tokens), :]  # [seq_len-1, vocab_size]
            
            # Cross entropy loss
            loss = F.cross_entropy(
                logits, 
                target_tokens,
                reduction='sum'
            )
            
            total_loss += loss
            total_tokens += len(target_tokens)
        
        # Average over tokens
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
        else:
            avg_loss = torch.tensor(0.0, device=self.device)
        
        info = {
            'asr_loss': avg_loss.item(),
            'asr_tokens': total_tokens
        }
        
        return avg_loss, info


class FlexibleLossWithASR(nn.Module):
    """Loss function with Whisper ASR loss"""
    
    def __init__(
        self,
        use_feature: bool = True,
        use_time: bool = False,
        use_mel: bool = False,
        use_asr: bool = False,
        weight_feature: float = 1.0,
        weight_time: float = 0.3,
        weight_mel: float = 0.3,
        weight_asr: float = 0.5,
        sample_rate: int = 16000,
        device: str = "cuda",
        asr_cache_dir: Optional[str] = None
    ):
        super().__init__()
        self.use_feature = use_feature
        self.use_time = use_time
        self.use_mel = use_mel
        self.use_asr = use_asr
        self.weight_feature = weight_feature
        self.weight_time = weight_time
        self.weight_mel = weight_mel
        self.weight_asr = weight_asr
        self.device = device

        if use_mel:
            self.mel_transform = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=1024,
                hop_length=256,
                n_mels=80
            ).to(device)  # 移到正確的 device
        
        if use_asr:
            self.asr_loss_module = WhisperASRLoss(
                device=device,
                cache_dir=asr_cache_dir
            )

    def forward(
        self,
        original_audio: torch.Tensor,
        reconstructed_audio: torch.Tensor,
        original_features: torch.Tensor,
        reconstructed_features: torch.Tensor,
        ground_truth_texts: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = 0
        loss_dict = {}

        # Feature loss (encoder feature level) - handle different shapes
        if self.use_feature:
            # Features shape: [batch, n_frames, n_features]
            # Align to same temporal length
            min_frames = min(original_features.shape[1], reconstructed_features.shape[1])
            orig_aligned = original_features[:, :min_frames, :]
            recon_aligned = reconstructed_features[:, :min_frames, :]
            
            feature_loss = F.mse_loss(recon_aligned, orig_aligned)
            total_loss += self.weight_feature * feature_loss
            loss_dict['feature_loss'] = feature_loss.item()

        # Time domain loss
        if self.use_time:
            # Align audio lengths (reconstructed may be longer/shorter)
            min_len = min(reconstructed_audio.shape[-1], original_audio.shape[-1])
            recon_aligned = reconstructed_audio[..., :min_len]
            orig_aligned = original_audio[..., :min_len]
            
            time_loss = F.l1_loss(recon_aligned, orig_aligned)
            total_loss += self.weight_time * time_loss
            loss_dict['time_loss'] = time_loss.item()

        # Mel spectrogram loss
        if self.use_mel:
            # Align audio lengths first
            min_len = min(reconstructed_audio.shape[-1], original_audio.shape[-1])
            recon_aligned = reconstructed_audio[..., :min_len]
            orig_aligned = original_audio[..., :min_len]
            
            mel_orig = self.mel_transform(orig_aligned)
            mel_recon = self.mel_transform(recon_aligned)
            mel_loss = F.l1_loss(mel_recon, mel_orig)
            total_loss += self.weight_mel * mel_loss
            loss_dict['mel_loss'] = mel_loss.item()
        
        # ASR loss (cross entropy with ground truth)
        if self.use_asr and ground_truth_texts:
            asr_loss, asr_info = self.asr_loss_module(
                original_audio,
                reconstructed_audio,
                ground_truth_texts
            )
            total_loss += self.weight_asr * asr_loss
            loss_dict.update(asr_info)

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
        print("FocalCodec Training Configuration")
        print("="*60)

        # Setup training/freezing for each component
        components = {
            'encoder': self.model.encoder,
            'compressor': self.model.compressor,
            'decompressor': self.model.decompressor,
            'decoder': self.model.decoder
        }

        for name, component in components.items():
            print(f"\n{name.upper()}:")
            
            # Apply LoRA if specified
            if name in lora_config:
                rank, alpha = lora_config[name]
                components[name] = apply_lora_to_component(component, rank, alpha)
            
            # Set training mode
            if train_components.get(name, False):
                component.train()
                for param in component.parameters():
                    param.requires_grad = True
                print(f"  Mode: TRAINING")
            else:
                component.eval()
                for param in component.parameters():
                    param.requires_grad = False
                print(f"  Mode: FROZEN (eval)")
            
            # Additional freeze if specified
            if freeze_components.get(name, False):
                for param in component.parameters():
                    param.requires_grad = False
                print(f"  Additional freeze applied")

        # Create loss function
        self.loss_fn = FlexibleLossWithASR(**loss_config)
        
        print("\n" + "="*60)
        print("Loss Configuration:")
        for key, value in loss_config.items():
            if key not in ['device', 'asr_cache_dir']:
                print(f"  {key}: {value}")
        print("="*60 + "\n")

    def train_epoch(self, dataloader, optimizer, epoch):
        self.model.train()
        
        # Set components to appropriate modes
        for name, should_train in self.train_components.items():
            component = getattr(self.model, name)
            if should_train:
                component.train()
            else:
                component.eval()
        
        total_loss = 0
        loss_components = {}
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, (audio, transcriptions) in enumerate(pbar):
            audio = audio.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass through codec using FocalCodec feature space API
            # sig_to_feats: audio -> encoder features
            # feats_to_lats: features -> compressed latents
            # lats_to_codes: latents -> discrete codes
            # codes_to_qfeats: codes -> quantized features (for decoder)
            feats = self.model.sig_to_feats(audio)
            lats = self.model.feats_to_lats(feats)
            codes = self.model.lats_to_codes(lats)
            qfeats = self.model.codes_to_qfeats(codes)
            
            # Reconstruct audio from quantized features
            reconstructed_audio = self.model.feats_to_sig(qfeats)
            
            # For feature loss comparison
            reconstructed_feats = self.model.sig_to_feats(reconstructed_audio)
            
            # Compute loss
            loss, loss_dict = self.loss_fn(
                audio,
                reconstructed_audio,
                feats,
                reconstructed_feats,
                transcriptions
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )
            
            optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0
                loss_components[key] += value
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                **{k: f"{v:.4f}" for k, v in loss_dict.items() if k != 'total_loss'}
            })
        
        # Calculate epoch averages
        n_batches = len(dataloader)
        metrics = {
            'total_loss': total_loss / n_batches
        }
        for key, value in loss_components.items():
            if key != 'total_loss':
                metrics[key] = value / n_batches
        
        return metrics

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        
        total_loss = 0
        loss_components = {}
        
        for audio, transcriptions in tqdm(dataloader, desc="Validation"):
            audio = audio.to(self.device)
            
            # Forward pass using feature space API
            feats = self.model.sig_to_feats(audio)
            lats = self.model.feats_to_lats(feats)
            codes = self.model.lats_to_codes(lats)
            qfeats = self.model.codes_to_qfeats(codes)
            reconstructed_audio = self.model.feats_to_sig(qfeats)
            reconstructed_feats = self.model.sig_to_feats(reconstructed_audio)
            
            # Compute loss
            loss, loss_dict = self.loss_fn(
                audio,
                reconstructed_audio,
                feats,
                reconstructed_feats,
                transcriptions
            )
            
            # Accumulate losses
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0
                loss_components[key] += value
        
        # Calculate averages
        n_batches = len(dataloader)
        metrics = {
            'total_loss': total_loss / n_batches
        }
        for key, value in loss_components.items():
            if key != 'total_loss':
                metrics[key] = value / n_batches
        
        return metrics

    def save_checkpoint(self, path, epoch, optimizer, metrics):
        import shutil

        # Check available disk space
        stat = shutil.disk_usage(os.path.dirname(path))
        available_gb = stat.free / (1024**3)

        if available_gb < 1.0:  # Less than 1GB available
            print(f"⚠️  WARNING: Low disk space ({available_gb:.2f} GB available)")
            print(f"⚠️  Skipping checkpoint save to avoid disk full error")
            return

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': metrics['train']['total_loss'],
            'val_loss': metrics['val']['total_loss'],
            'metrics': metrics
        }

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save checkpoint
            torch.save(checkpoint, path)
            print(f"✅ Checkpoint saved: {path} ({available_gb:.2f} GB remaining)")
        except RuntimeError as e:
            if "file write failed" in str(e) or "PytorchStreamWriter failed" in str(e):
                print(f"❌ ERROR: Failed to save checkpoint - disk may be full")
                print(f"   Available space: {available_gb:.2f} GB")
                print(f"   Try using --checkpoint_dir to specify a different location")
            else:
                raise e


def parse_args():
    parser = argparse.ArgumentParser(description='Train FocalCodec with Whisper ASR Loss')
    
    # Data parameters
    parser.add_argument('--batch_size', type=int, default=16)
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
    parser.add_argument('--use_asr_loss', action='store_true', help='Enable Whisper ASR loss')
    parser.add_argument('--weight_feature', type=float, default=1.0)
    parser.add_argument('--weight_time', type=float, default=0.3)
    parser.add_argument('--weight_mel', type=float, default=0.3)
    parser.add_argument('--weight_asr', type=float, default=0.5, help='Weight for ASR loss')
    parser.add_argument('--asr_cache_dir', type=str, default='/mnt/Internal/jieshiang/Model/ASR')

    # Other parameters
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--checkpoint_dir', type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')

    # Early stopping
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min_delta', type=float, default=1e-4)

    return parser.parse_args()


def main():
    args = parse_args()

    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if not any([args.train_encoder, args.train_compressor, args.train_decompressor, args.train_decoder]):
        print("ERROR: At least one component must be trained!")
        return

    if not any([args.use_feature_loss, args.use_time_loss, args.use_mel_loss, args.use_asr_loss]):
        print("ERROR: At least one loss must be enabled!")
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
        'use_asr': args.use_asr_loss,
        'weight_feature': args.weight_feature,
        'weight_time': args.weight_time,
        'weight_mel': args.weight_mel,
        'weight_asr': args.weight_asr,
        'sample_rate': 16000,
        'device': device,
        'asr_cache_dir': args.asr_cache_dir
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
                print(f"\nEarly stopping triggered!")
                print(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
                break

    print("\nTraining completed!")
    print(f"Best model: epoch {best_epoch} with val_loss {best_val_loss:.6f}")


if __name__ == '__main__':
    main()