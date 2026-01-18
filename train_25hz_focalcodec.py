#!/usr/bin/env python3
"""
FocalCodec 25Hz Streaming Semantic Codec Training

Configuration is loaded from config.yaml.

Training Stages:
- Stage 1: Feature Matching with 25Hz teacher
- Stage 2: ASR Fine-tuning (semantic focus)
- Stage 3: ASR + HuBERT Distillation

Usage:
    python train_25hz_focalcodec.py
    python train_25hz_focalcodec.py --config /path/to/config.yaml
    python train_25hz_focalcodec.py --stage 2
    python train_25hz_focalcodec.py --resume

Author: Research
Date: 2026-01-18
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
import json
import yaml
import warnings
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import math

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
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
        max_chunks: Optional[int] = None,
        sample_rate: int = 16000
    ):
        self.csv_path = csv_path
        self.base_path = base_path
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.sample_rate = sample_rate

        # Load CSV
        self.df = pd.read_csv(csv_path)
        self.file_list = []

        # Create chunked file list
        for idx, row in self.df.iterrows():
            relative_path = row['file_path']
            audio_path = os.path.join(base_path, relative_path.lstrip('./'))
            transcription = row.get('transcription', '')

            # Get audio duration
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate

            # Calculate chunks
            chunk_samples = int(chunk_duration * sample_rate)
            hop_samples = int(chunk_duration * (1 - overlap) * sample_rate)

            num_chunks = max(1, int((duration - chunk_duration) / (chunk_duration * (1 - overlap))) + 1)

            for chunk_idx in range(num_chunks):
                start_sample = chunk_idx * hop_samples
                self.file_list.append({
                    'audio_path': audio_path,
                    'transcription': transcription,
                    'start_sample': start_sample,
                    'num_samples': chunk_samples
                })

                if max_chunks and len(self.file_list) >= max_chunks:
                    break

            if max_chunks and len(self.file_list) >= max_chunks:
                break

        print(f"Dataset: {len(self.file_list)} chunks from {len(self.df)} files")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        item = self.file_list[idx]

        # Load audio chunk
        waveform, sr = torchaudio.load(
            item['audio_path'],
            frame_offset=item['start_sample'],
            num_frames=item['num_samples']
        )

        # Resample if needed
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad if too short
        if waveform.shape[1] < item['num_samples']:
            pad_length = item['num_samples'] - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_length))

        return waveform.squeeze(0), item['transcription']


class LowBitrateSemanticLoss(nn.Module):
    """Loss function for low bitrate semantic codec training."""

    def __init__(
        self,
        mode: str = 'feature',
        weight_feature: float = 1.0,
        weight_asr: float = 1.0,
        weight_hubert: float = 0.5,
        device: str = 'cuda',
        asr_cache_dir: str = None
    ):
        super().__init__()
        self.mode = mode
        self.weight_feature = weight_feature
        self.weight_asr = weight_asr
        self.weight_hubert = weight_hubert
        self.device = device

        # Load ASR model if needed
        if mode in ['asr_only', 'asr_hubert']:
            import whisper
            print(f"Loading Whisper model for {mode} mode...")
            self.whisper_model = whisper.load_model("small", device=device)
            self.whisper_model.eval()
            for param in self.whisper_model.parameters():
                param.requires_grad = False
            print("Whisper model loaded")

            if mode == 'asr_only':
                print("NOTE: ASR-only mode uses small feature loss (0.1) for gradient flow")
                self.gradient_feature_weight = 0.1
        else:
            self.whisper_model = None

        # Load HuBERT if needed
        if mode == 'asr_hubert':
            from transformers import HubertModel
            print("Loading HuBERT model...")
            self.hubert_model = HubertModel.from_pretrained(
                "facebook/hubert-base-ls960",
                cache_dir=asr_cache_dir
            ).to(device)
            self.hubert_model.eval()
            for param in self.hubert_model.parameters():
                param.requires_grad = False
            print("HuBERT model loaded")
        else:
            self.hubert_model = None

    def compute_asr_loss(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        ground_truth: List[str]
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute ASR-based loss using Whisper encoder-decoder with teacher forcing."""
        import whisper

        if not ground_truth or all(t == "" for t in ground_truth):
            return torch.tensor(0.0, device=self.device, requires_grad=True), {'asr_loss': 0.0}

        batch_size = reconstructed.shape[0]
        tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language="zh")

        total_loss = 0.0
        total_tokens = 0

        for i in range(batch_size):
            gt_text = ground_truth[i] if i < len(ground_truth) else ""
            if not gt_text:
                continue

            tokens = tokenizer.encode(gt_text)
            sot_token = tokenizer.sot
            eot_token = tokenizer.eot
            lang_token = tokenizer.sot + 1 + list(tokenizer.all_language_tokens).index(tokenizer.to_language_token("zh"))
            task_token = tokenizer.transcribe

            full_tokens = [sot_token, lang_token, task_token] + tokens + [eot_token]
            full_tokens = torch.tensor(full_tokens, device=self.device, dtype=torch.long)

            audio_np = reconstructed[i].detach().cpu().numpy()

            target_length = 16000 * 30
            if len(audio_np) < target_length:
                audio_np = np.pad(audio_np, (0, target_length - len(audio_np)), mode='constant')
            else:
                audio_np = audio_np[:target_length]

            mel = whisper.log_mel_spectrogram(audio_np).to(self.device)
            with torch.no_grad():
                encoder_output = self.whisper_model.encoder(mel.unsqueeze(0))

            input_tokens = full_tokens[:-1].unsqueeze(0)
            target_tokens = full_tokens[1:]

            logits = self.whisper_model.decoder(input_tokens, encoder_output)
            logits = logits[0]

            loss = F.cross_entropy(logits, target_tokens, reduction='sum')

            total_loss = total_loss + loss
            total_tokens += len(target_tokens)

        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
        else:
            avg_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        info = {
            'asr_loss': avg_loss.item() if isinstance(avg_loss, torch.Tensor) else avg_loss,
            'asr_tokens': total_tokens
        }

        return avg_loss, info

    def compute_hubert_loss(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """Compute HuBERT feature matching loss."""
        min_length = min(original.shape[-1], reconstructed.shape[-1])
        orig_aligned = original[..., :min_length]
        recon_aligned = reconstructed[..., :min_length]

        with torch.no_grad():
            target_features = self.hubert_model(orig_aligned).last_hidden_state

        pred_features = self.hubert_model(recon_aligned).last_hidden_state

        min_seq_len = min(pred_features.shape[1], target_features.shape[1])
        pred_aligned = pred_features[:, :min_seq_len, :]
        target_aligned = target_features[:, :min_seq_len, :]

        loss = F.mse_loss(pred_aligned, target_aligned)
        return loss

    def forward(
        self,
        encoder_output: torch.Tensor,
        teacher_encoder_output: Optional[torch.Tensor],
        decompressor_output: torch.Tensor,
        reconstructed_audio: Optional[torch.Tensor] = None,
        original_audio: Optional[torch.Tensor] = None,
        ground_truth_texts: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Forward pass."""
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        info = {}

        # Stage 1: Feature Matching
        if self.mode == 'feature':
            if teacher_encoder_output is not None:
                min_channels = min(encoder_output.shape[1], teacher_encoder_output.shape[1])
                min_time = min(encoder_output.shape[-1], teacher_encoder_output.shape[-1])
                enc_aligned = encoder_output[:, :min_channels, :min_time]
                teacher_enc_aligned = teacher_encoder_output[:, :min_channels, :min_time]
                loss_enc = F.mse_loss(enc_aligned, teacher_enc_aligned)
                total_loss = total_loss + self.weight_feature * loss_enc
                info['loss_enc'] = loss_enc.item()

            if teacher_encoder_output is not None:
                min_channels = min(decompressor_output.shape[1], teacher_encoder_output.shape[1])
                min_time = min(decompressor_output.shape[-1], teacher_encoder_output.shape[-1])
                decomp_aligned = decompressor_output[:, :min_channels, :min_time]
                teacher_decomp_aligned = teacher_encoder_output[:, :min_channels, :min_time]
                loss_recon = F.mse_loss(decomp_aligned, teacher_decomp_aligned)
                total_loss = total_loss + self.weight_feature * loss_recon
                info['loss_recon'] = loss_recon.item()

            if reconstructed_audio is not None and original_audio is not None:
                min_length = min(reconstructed_audio.shape[-1], original_audio.shape[-1])
                recon_aligned = reconstructed_audio[..., :min_length]
                orig_aligned = original_audio[..., :min_length]
                loss_audio = F.mse_loss(recon_aligned, orig_aligned)
                total_loss = total_loss + 0.1 * loss_audio
                info['loss_audio'] = loss_audio.item()

        # Stage 2: ASR-Only
        elif self.mode == 'asr_only':
            if teacher_encoder_output is not None:
                min_channels = min(encoder_output.shape[1], teacher_encoder_output.shape[1])
                min_time = min(encoder_output.shape[-1], teacher_encoder_output.shape[-1])
                enc_aligned = encoder_output[:, :min_channels, :min_time]
                teacher_aligned = teacher_encoder_output[:, :min_channels, :min_time]
                feature_loss = F.mse_loss(enc_aligned, teacher_aligned)
                total_loss = total_loss + self.gradient_feature_weight * feature_loss
                info['loss_feature_bridge'] = feature_loss.item()

            if reconstructed_audio is not None and original_audio is not None and ground_truth_texts:
                asr_loss, asr_info = self.compute_asr_loss(
                    original_audio,
                    reconstructed_audio,
                    ground_truth_texts
                )
                total_loss = total_loss + self.weight_asr * asr_loss
                info['loss_asr'] = asr_loss.item() if hasattr(asr_loss, 'item') else asr_loss
                info['asr_tokens'] = asr_info.get('asr_tokens', 0)

        # Stage 3: ASR + HuBERT
        elif self.mode == 'asr_hubert':
            if teacher_encoder_output is not None:
                min_channels = min(encoder_output.shape[1], teacher_encoder_output.shape[1])
                min_time = min(encoder_output.shape[-1], teacher_encoder_output.shape[-1])
                enc_aligned = encoder_output[:, :min_channels, :min_time]
                teacher_aligned = teacher_encoder_output[:, :min_channels, :min_time]
                feature_loss = F.mse_loss(enc_aligned, teacher_aligned)
                total_loss = total_loss + 0.1 * feature_loss
                info['loss_feature'] = feature_loss.item()

            if reconstructed_audio is not None and original_audio is not None and ground_truth_texts:
                asr_loss, asr_info = self.compute_asr_loss(
                    original_audio,
                    reconstructed_audio,
                    ground_truth_texts
                )
                total_loss = total_loss + self.weight_asr * asr_loss
                info['loss_asr'] = asr_loss.item() if hasattr(asr_loss, 'item') else asr_loss
                info['asr_tokens'] = asr_info.get('asr_tokens', 0)

            if reconstructed_audio is not None and original_audio is not None:
                hubert_loss = self.compute_hubert_loss(original_audio, reconstructed_audio)
                total_loss = total_loss + self.weight_hubert * hubert_loss
                info['loss_hubert'] = hubert_loss.item()

        info['loss_total'] = total_loss.item()
        return total_loss, info


class LowBitrateCodecTrainer:
    """Trainer for 25Hz low bitrate codec."""

    def __init__(
        self,
        student_model,
        teacher_model,
        loss_fn: LowBitrateSemanticLoss,
        device: str = 'cuda'
    ):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device) if teacher_model else None
        self.device = device
        self.loss_fn = loss_fn

        # Freeze teacher
        if self.teacher:
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False
            print("Teacher model frozen")

        # Freeze student encoder and decoder
        print("\nFreezing Student Encoder and Decoder...")
        for param in self.student.encoder.parameters():
            param.requires_grad = False
        for param in self.student.decoder.parameters():
            param.requires_grad = False
        self.student.encoder.eval()
        self.student.decoder.eval()

        # Train student compressor and decompressor
        print("Training Student Compressor and Decompressor...")
        for param in self.student.compressor.parameters():
            param.requires_grad = True
        for param in self.student.decompressor.parameters():
            param.requires_grad = True
        self.student.compressor.train()
        self.student.decompressor.train()

        print("Quantizer (BSQ) has no learnable parameters")

        # Count parameters
        trainable = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.student.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        self.code_usage_history = []

    def compute_quantizer_stats(self, codes: torch.Tensor, codebook_size: int) -> Dict:
        """Compute quantizer statistics for monitoring codebook collapse."""
        codes_flat = codes.reshape(-1).cpu().numpy()
        unique_codes = np.unique(codes_flat)
        num_unique = len(unique_codes)
        usage_ratio = num_unique / codebook_size

        code_counts = np.bincount(codes_flat.astype(int), minlength=codebook_size)
        code_probs = code_counts / code_counts.sum()

        entropy = -np.sum(code_probs[code_probs > 0] * np.log2(code_probs[code_probs > 0]))
        max_entropy = np.log2(codebook_size)
        normalized_entropy = entropy / max_entropy

        perplexity = 2 ** entropy

        return {
            'unique_codes': num_unique,
            'usage_ratio': usage_ratio,
            'entropy': entropy,
            'max_entropy': max_entropy,
            'normalized_entropy': normalized_entropy,
            'perplexity': perplexity,
        }

    def train_epoch(self, dataloader, optimizer, epoch, gradient_clip: float = 5.0, codebook_size: int = 2048):
        """Train one epoch."""
        self.student.compressor.train()
        self.student.decompressor.train()

        total_loss = 0
        loss_components = {}
        all_codes = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, (audio, transcriptions) in enumerate(pbar):
            audio = audio.to(self.device)

            optimizer.zero_grad()

            # Teacher forward
            teacher_enc = None
            if self.teacher:
                with torch.no_grad():
                    teacher_enc = self.teacher.encoder(audio)
                    if isinstance(teacher_enc, tuple):
                        teacher_enc = teacher_enc[0]

            # Student forward
            student_enc = self.student.encoder(audio)
            if isinstance(student_enc, tuple):
                student_enc = student_enc[0]

            student_comp = self.student.compressor(student_enc)
            if isinstance(student_comp, tuple):
                student_comp = student_comp[0]

            quant_result = self.student.quantizer(student_comp)
            if isinstance(quant_result, tuple):
                tokens = quant_result[0]
                codes = quant_result[1]
            else:
                tokens = quant_result
                codes = quant_result

            if batch_idx % 10 == 0:
                all_codes.append(tokens.detach())

            student_decomp = self.student.decompressor(codes)
            if isinstance(student_decomp, tuple):
                student_decomp = student_decomp[0]

            reconstructed = self.student.decoder(student_decomp)
            if isinstance(reconstructed, tuple):
                reconstructed = reconstructed[0]

            loss, info = self.loss_fn(
                encoder_output=student_enc,
                teacher_encoder_output=teacher_enc,
                decompressor_output=student_decomp,
                reconstructed_audio=reconstructed,
                original_audio=audio,
                ground_truth_texts=list(transcriptions)
            )

            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), gradient_clip)
            optimizer.step()

            total_loss += loss.item()
            for key, value in info.items():
                if key not in loss_components:
                    loss_components[key] = 0
                loss_components[key] += value

            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(dataloader)
        avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}

        if all_codes:
            codes_tensor = torch.cat(all_codes, dim=0)
            quant_stats = self.compute_quantizer_stats(codes_tensor, codebook_size)
            avg_components.update({
                'quant_unique': quant_stats['unique_codes'],
                'quant_usage': quant_stats['usage_ratio'],
                'quant_entropy': quant_stats['entropy'],
                'quant_norm_entropy': quant_stats['normalized_entropy'],
                'quant_perplexity': quant_stats['perplexity'],
            })

        return avg_loss, avg_components

    @torch.no_grad()
    def validate(self, dataloader):
        """Validate model."""
        self.student.eval()

        total_loss = 0
        loss_components = {}

        for audio, transcriptions in tqdm(dataloader, desc="Validating"):
            audio = audio.to(self.device)

            teacher_enc = None
            if self.teacher:
                teacher_enc = self.teacher.encoder(audio)
                if isinstance(teacher_enc, tuple):
                    teacher_enc = teacher_enc[0]

            student_enc = self.student.encoder(audio)
            if isinstance(student_enc, tuple):
                student_enc = student_enc[0]

            student_comp = self.student.compressor(student_enc)
            if isinstance(student_comp, tuple):
                student_comp = student_comp[0]

            quant_result = self.student.quantizer(student_comp)
            if isinstance(quant_result, tuple):
                codes = quant_result[1]
            else:
                codes = quant_result

            student_decomp = self.student.decompressor(codes)
            if isinstance(student_decomp, tuple):
                student_decomp = student_decomp[0]

            reconstructed = self.student.decoder(student_decomp)
            if isinstance(reconstructed, tuple):
                reconstructed = reconstructed[0]

            loss, info = self.loss_fn(
                encoder_output=student_enc,
                teacher_encoder_output=teacher_enc,
                decompressor_output=student_decomp,
                reconstructed_audio=reconstructed,
                original_audio=audio,
                ground_truth_texts=list(transcriptions)
            )

            total_loss += loss.item()
            for key, value in info.items():
                if key not in loss_components:
                    loss_components[key] = 0
                loss_components[key] += value

        avg_loss = total_loss / len(dataloader)
        avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}

        self.student.train()
        return avg_loss, avg_components


def create_25hz_model(base_model_path: str, codebook_size: int, model_cache_dir: str, device: str = 'cuda'):
    """Create 25Hz causal model from 50Hz 2k base."""
    # Import FocalCodec
    from focalcodec import FocalCodec

    bits = int(math.log2(codebook_size))

    print(f"\nCreating 25Hz {bits}-bit causal model from {base_model_path}")
    print(f"  Codebook size: {codebook_size}")

    base_model = FocalCodec.from_pretrained(
        base_model_path,
        cache_dir=model_cache_dir
    )

    encoder_config = base_model.encoder_config
    decoder_config = base_model.decoder_config

    compressor_config = base_model.compressor_config.copy() if isinstance(base_model.compressor_config, dict) else {}
    compressor_config['downscale_factors'] = [2, 1, 1]
    compressor_config['output_dim'] = bits

    quantizer_config = base_model.quantizer_config.copy() if isinstance(base_model.quantizer_config, dict) else {}
    quantizer_config['codebook_size'] = codebook_size

    decompressor_config = base_model.decompressor_config.copy() if isinstance(base_model.decompressor_config, dict) else {}
    decompressor_config['input_dim'] = bits
    decompressor_config['upscale_factors'] = [2, 1, 1]

    from focalcodec import FocalCodec as FC
    model = FC(
        encoder_name=base_model.encoder_name,
        encoder_config=encoder_config,
        compressor_name=base_model.compressor_name,
        compressor_config=compressor_config,
        quantizer_name=base_model.quantizer_name,
        quantizer_config=quantizer_config,
        decompressor_name=base_model.decompressor_name,
        decompressor_config=decompressor_config,
        decoder_name=base_model.decoder_name,
        decoder_config=decoder_config,
    )

    model_state = model.state_dict()
    base_state = base_model.state_dict()

    for key in base_state.keys():
        if key.startswith('encoder.') and key in model_state:
            if base_state[key].shape == model_state[key].shape:
                model_state[key] = base_state[key]

    for key in base_state.keys():
        if key.startswith('decoder.') and key in model_state:
            if base_state[key].shape == model_state[key].shape:
                model_state[key] = base_state[key]

    model.load_state_dict(model_state)

    bitrate = 25 * bits
    print(f"Model created:")
    print(f"  Frame Rate: 25 Hz")
    print(f"  Codebook: {codebook_size} ({bits}-bit)")
    print(f"  Bitrate: {bitrate} bps")
    print(f"  Causal: {model.causal}")

    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Train 25Hz FocalCodec')

    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.yaml')
    parser.add_argument('--stage', type=int, default=None,
                       help='Training stage (overrides config)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    parser.add_argument('--gpu_id', type=int, default=None,
                       help='GPU ID (overrides config)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Add focalcodec to path
    sys.path.insert(0, config.focalcodec_dir)
    from focalcodec import FocalCodec

    # Override with command line args
    stage = args.stage if args.stage is not None else config.stage
    gpu_id = args.gpu_id if args.gpu_id is not None else config.gpu_id

    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Training Stage: {stage}")

    # Map stage to loss mode
    stage_to_mode = {
        1: 'feature',
        2: 'asr_only',
        3: 'asr_hubert'
    }
    loss_mode = stage_to_mode[stage]

    # Checkpoint directory
    checkpoint_dir = config.get_checkpoint_dir(stage)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Resume checkpoint
    resume_from = None
    if args.resume:
        resume_from = os.path.join(checkpoint_dir, config.get('continue_training.resume_checkpoint', 'last_model.pt'))
        if not os.path.exists(resume_from):
            print(f"Warning: Resume checkpoint not found: {resume_from}")
            resume_from = None

    # Create/Load student model
    if resume_from and os.path.exists(resume_from):
        print(f"\nLoading student from: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        student = create_25hz_model(
            config.base_model,
            config.codebook_size,
            config.model_cache_dir,
            device
        )
        student.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Student weights loaded")
    else:
        student = create_25hz_model(
            config.base_model,
            config.codebook_size,
            config.model_cache_dir,
            device
        )

    # Load teacher model
    print(f"\nLoading teacher: {config.teacher_model}")
    teacher = FocalCodec.from_pretrained(
        config.teacher_model,
        cache_dir=config.model_cache_dir
    )
    print("Teacher model loaded (25Hz non-causal)")

    # Create loss function
    loss_fn = LowBitrateSemanticLoss(
        mode=loss_mode,
        weight_feature=config.weight_feature,
        weight_asr=config.weight_asr,
        weight_hubert=config.weight_hubert,
        device=device,
        asr_cache_dir=config.asr_cache_dir
    )

    # Create trainer
    trainer = LowBitrateCodecTrainer(
        student_model=student,
        teacher_model=teacher,
        loss_fn=loss_fn,
        device=device
    )

    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = AudioDataset(
        csv_path=config.train_csv,
        base_path=config.audio_base_path,
        chunk_duration=config.chunk_duration,
        overlap=config.overlap
    )

    val_dataset = AudioDataset(
        csv_path=config.val_csv,
        base_path=config.audio_base_path,
        chunk_duration=config.chunk_duration,
        overlap=0.0
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # Optimizer and scheduler
    optimizer = AdamW(
        student.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    num_epochs = config.get('continue_training.num_epochs', config.num_epochs) if args.resume else config.num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    start_epoch = 1
    patience = config.get('continue_training.patience', config.patience) if args.resume else config.patience

    # Resume training state
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded")
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state loaded")
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']
            best_epoch = checkpoint.get('epoch', 0)
            print(f"Best val_loss so far: {best_val_loss:.6f} (epoch {best_epoch})")

    print(f"\nStarting training for Stage {stage} ({loss_mode} mode)")
    print(f"Epochs: {start_epoch} to {num_epochs}, Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}, Weight decay: {config.weight_decay}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print("="*80)

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss, train_info = trainer.train_epoch(
            train_loader,
            optimizer,
            epoch,
            gradient_clip=config.gradient_clip,
            codebook_size=config.codebook_size
        )

        val_loss, val_info = trainer.validate(val_loader)

        scheduler.step()

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        for key, value in val_info.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")

        if 'quant_unique' in train_info:
            print(f"  [Quantizer Stats]")
            print(f"    Unique codes: {train_info['quant_unique']}/{config.codebook_size} ({train_info['quant_usage']*100:.1f}%)")
            print(f"    Entropy: {train_info['quant_entropy']:.3f} / {np.log2(config.codebook_size):.3f} (norm: {train_info['quant_norm_entropy']:.3f})")
            print(f"    Perplexity: {train_info['quant_perplexity']:.1f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'stage': stage,
        }

        torch.save(checkpoint, os.path.join(checkpoint_dir, 'last_model.pt'))

        # Check for improvement
        improvement = best_val_loss - val_loss
        if improvement > config.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pt'))
            print(f"New best! val_loss: {best_val_loss:.6f} (improved by {improvement:.6f})")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs (best: {best_val_loss:.6f} at epoch {best_epoch})")

            if patience_counter >= patience:
                print(f"\nEarly stopping! Best: epoch {best_epoch}, val_loss {best_val_loss:.6f}")
                break

    print("\nTraining completed!")
    print(f"Best model: epoch {best_epoch} with val_loss {best_val_loss:.6f}")
    print(f"Checkpoint: {os.path.join(checkpoint_dir, 'best_model.pt')}")


if __name__ == '__main__':
    main()
