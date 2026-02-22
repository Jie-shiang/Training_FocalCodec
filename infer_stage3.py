#!/usr/bin/env python3
"""
Inference script for Stage 3 FocalCodec (12.5Hz) - outputs reconstructed audio files.

使用官方 FocalCodec API:
- sig_to_feats: audio -> encoder features
- feats_to_lats: features -> compressed latents
- lats_to_codes: latents -> discrete codes
- codes_to_qfeats: codes -> quantized features (decompressor output)
- feats_to_sig: quantized features -> reconstructed audio

Supports Stage 3 (12.5Hz) model.
Configuration is loaded from config.yaml.

Usage:
    python infer_stage3.py --config config_exp_E.yaml
    python infer_stage3.py --config config_exp_E.yaml --max_samples 500
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import math

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

from utils.config_loader import load_config


def build_full_stage3_architecture(model, device):
    """
    Build complete Stage 3 architecture from base model.
    Adds 2 new layers: 4th layer (Stage 2) + 5th layer (Stage 3).
    This creates the structure to hold checkpoint weights.
    """
    from focalcodec.focalnet import FocalDownScale, FocalUpScale

    comp = model.compressor
    decomp = model.decompressor

    # === Compressor: Add 4th layer (Stage 2) ===
    layer_4_comp = FocalDownScale(
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

    comp.layers.append(layer_4_comp)
    comp.downscale_factors = list(comp.downscale_factors) + [2]

    # === Compressor: Add 5th layer (Stage 3) ===
    layer_5_comp = FocalDownScale(
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

    comp.layers.append(layer_5_comp)
    comp.downscale_factors = list(comp.downscale_factors) + [2]
    comp.downsample_factor = torch.Size(comp.downscale_factors).numel()
    comp.chunk_size = comp.downsample_factor

    # Update out_proj to 11 dimensions (will be overwritten by checkpoint)
    comp.out_proj = nn.Linear(1024, 11).to(device)

    # === Decompressor: Add layer 0 (Stage 3) ===
    layer_0_decomp_s3 = FocalUpScale(
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

    decomp.layers.insert(0, layer_0_decomp_s3)
    decomp.upscale_factors = [2] + list(decomp.upscale_factors)

    # === Decompressor: Add layer 1 (Stage 2) ===
    layer_1_decomp_s2 = FocalUpScale(
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

    decomp.layers.insert(1, layer_1_decomp_s2)
    decomp.upscale_factors = [2] + list(decomp.upscale_factors)

    # Update in_proj to 11 dimensions (will be overwritten by checkpoint)
    decomp.in_proj = nn.Linear(11, 1024).to(device)

    print(f"Stage 3 architecture built:")
    print(f"  Compressor: {len(comp.layers)} layers, downscale_factors={comp.downscale_factors}")
    print(f"  Decompressor: {len(decomp.layers)} layers, upscale_factors={decomp.upscale_factors}")

    return model


def reconstruct_stage2_architecture(model, device):
    """
    Reconstruct Stage 2 architecture (add 4th layer).
    This matches train_stage3_12.5hz.py lines 590-618.
    Does NOT update out_proj - that will be loaded from checkpoint.
    """
    from focalcodec.focalnet import FocalDownScale, FocalUpScale

    comp = model.compressor
    decomp = model.decompressor

    # Add 4th layer to compressor (Stage 2: 50Hz -> 25Hz)
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

    # Add 4th layer to decompressor (prepended as layer 0)
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

    # Update in_proj to 11 dimensions (will be overwritten by checkpoint)
    decomp.in_proj = nn.Linear(11, 1024).to(device)

    # NOTE: out_proj will be updated when loading checkpoint
    # Don't update it here - let checkpoint loading handle it

    return model


def add_stage3_layers(model, device):
    """
    Add Stage 3 layers (5th layer) to the model.
    This matches train_stage3_12.5hz.py add_fifth_layer functions.
    """
    from focalcodec.focalnet import FocalDownScale, FocalUpScale

    comp = model.compressor
    decomp = model.decompressor

    # Add 5th layer to compressor (Stage 3: 25Hz -> 12.5Hz)
    layer_5 = FocalDownScale(
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

    comp.layers.append(layer_5)
    comp.downscale_factors = list(comp.downscale_factors) + [2]
    comp.downsample_factor = torch.Size(comp.downscale_factors).numel()
    comp.chunk_size = comp.downsample_factor

    # Add 5th layer to decompressor (prepended as layer 0)
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

    # NOTE: Stage 3 keeps 11-dimensional tokens (2048 codebook)
    # in_proj and out_proj remain 11-dimensional

    print(f"Stage 3 architecture complete:")
    print(f"  Compressor layers: {len(comp.layers)}")
    print(f"  Compressor downscale_factors: {comp.downscale_factors}")
    print(f"  Decompressor layers: {len(decomp.layers)}")
    print(f"  Decompressor upscale_factors: {decomp.upscale_factors}")
    print(f"  out_proj: {comp.out_proj.out_features} dimensions")
    print(f"  in_proj: {decomp.in_proj.in_features} dimensions")

    return model


class CodecInference:
    """
    使用官方 FocalCodec API 進行推理

    API:
    - sig_to_feats: audio -> encoder features
    - feats_to_lats: features -> compressed latents
    - lats_to_codes: latents -> discrete codes
    - codes_to_qfeats: codes -> quantized features
    - feats_to_sig: quantized features -> reconstructed audio
    """

    def __init__(self, checkpoint_path: str, model_cache_dir: str, base_model: str, device: str = 'cuda'):
        self.device = device
        self.resamplers = {}

        print(f"Loading Stage 3 (12.5Hz) model from: {checkpoint_path}")

        # Load base model
        from focalcodec import FocalCodec
        model = FocalCodec.from_pretrained(base_model, cache_dir=model_cache_dir).to(device)

        # Build full Stage 3 architecture (5 compressor + 5 decompressor layers)
        print("Building Stage 3 architecture (5 layers)...")
        model = build_full_stage3_architecture(model, device)

        # Load checkpoint weights
        print("Loading Stage 3 checkpoint weights...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        self.model = model.to(device)
        self.model.eval()

        epoch = checkpoint.get('epoch', 'unknown')
        val_loss = checkpoint.get('val_loss', 'N/A')
        print(f"Model loaded (Epoch: {epoch}, Val Loss: {val_loss})")

        frame_rate = 12.5
        bits = 11  # 2048 codebook = 2^11 = 11-bit
        bitrate = frame_rate * bits
        print(f"Config: Stage 3, {frame_rate} Hz, {bits}-bit (2048 codes), Bitrate: {bitrate} bps")

    def load_audio(self, path: str, sr: int = 16000) -> np.ndarray:
        """Load and resample audio."""
        waveform, sample_rate = torchaudio.load(path)

        if sample_rate != sr:
            key = f"{sample_rate}_{sr}"
            if key not in self.resamplers:
                self.resamplers[key] = T.Resample(sample_rate, sr)
            waveform = self.resamplers[key](waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform.squeeze(0).numpy()

    @torch.no_grad()
    def reconstruct(self, audio: np.ndarray, use_8bit: bool = False) -> np.ndarray:
        """
        使用官方 FocalCodec API 重建音訊

        Args:
            audio: Input audio numpy array
            use_8bit: If True, mask codes to 8-bit (for MRL evaluation)

        Returns:
            Reconstructed audio numpy array
        """
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)
        original_length = audio_tensor.shape[1]

        # ============================================================
        # 使用官方 FocalCodec API
        # ============================================================

        # 1. Encoder: audio -> features
        feats = self.model.sig_to_feats(audio_tensor)

        # 2. Compressor: features -> latents
        lats = self.model.feats_to_lats(feats)

        # 3. Quantizer: latents -> codes
        codes = self.model.lats_to_codes(lats)

        # Optional: Apply 8-bit mask for MRL evaluation
        if use_8bit:
            if codes.dim() == 3:  # [batch, time, bits]
                codes[:, :, 8:] = 0
            elif codes.dim() == 2:  # [batch, bits]
                codes[:, 8:] = 0

        # 4. Decompressor: codes -> quantized features
        qfeats = self.model.codes_to_qfeats(codes)

        # 5. Decoder: quantized features -> reconstructed audio
        reconstructed = self.model.feats_to_sig(qfeats)

        return reconstructed.squeeze(0).cpu().numpy()

    def process_dataset(self, csv_path: str, base_path: str, output_dir: str, max_samples: int = 2000):
        """Process dataset and save reconstructed audio.

        Args:
            csv_path: Path to CSV file
            base_path: Base path for audio files
            output_dir: Output directory (can include subdirectory like aishell/librispeech)
            max_samples: Maximum number of samples to process
        """
        os.makedirs(output_dir, exist_ok=True)

        df = pd.read_csv(csv_path)
        if max_samples > 0:
            df = df.head(max_samples)

        print(f"\nProcessing {len(df)} samples...")
        print(f"Using official FocalCodec API: sig_to_feats -> feats_to_lats -> lats_to_codes -> codes_to_qfeats -> feats_to_sig")

        processed = 0
        failed = 0

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            relative_path = row['file_path']
            audio_path = os.path.join(base_path, relative_path.lstrip('./'))

            try:
                original = self.load_audio(audio_path)
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                failed += 1
                continue

            try:
                reconstructed = self.reconstruct(original)
            except Exception as e:
                print(f"Error reconstructing {audio_path}: {e}")
                failed += 1
                continue

            basename = Path(audio_path).stem
            output_path = os.path.join(output_dir, f"{basename}_inference.wav")

            recon_tensor = torch.from_numpy(reconstructed).unsqueeze(0)
            torchaudio.save(output_path, recon_tensor, self.model.sample_rate_output)

            processed += 1

        print(f"\nCompleted: {processed} processed, {failed} failed")
        return processed, failed


def parse_args():
    parser = argparse.ArgumentParser(description='Inference for Stage 3 FocalCodec (12.5Hz, 官方 API)')

    parser.add_argument('--config', type=str, required=True,
                       help='Path to config.yaml')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to process (overrides config)')
    parser.add_argument('--gpu_id', type=int, default=None,
                       help='GPU ID (overrides config)')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt',
                       help='Checkpoint filename (default: best_model.pt)')
    parser.add_argument('--use_8bit', action='store_true',
                       help='Use 8-bit codes (for MRL evaluation)')
    parser.add_argument('--csv_path', type=str, default=None,
                       help='CSV file path (overrides config)')
    parser.add_argument('--audio_base_path', type=str, default=None,
                       help='Audio base path (overrides config)')
    parser.add_argument('--output_subdir', type=str, default=None,
                       help='Output subdirectory (e.g., aishell, librispeech)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Extract experiment ID from config filename (e.g., config_exp_E.yaml -> exp_E)
    config_filename = os.path.basename(args.config)
    if 'exp_' in config_filename:
        exp_id = config_filename.split('exp_')[1].split('.')[0]
        experiment_name = f"exp_{exp_id}"
    else:
        experiment_name = "default"

    print(f"Experiment: {experiment_name}")

    # Add focalcodec to path
    sys.path.insert(0, config.focalcodec_dir)

    max_samples = args.max_samples if args.max_samples is not None else config.get('inference.max_samples', 2000)
    gpu_id = args.gpu_id if args.gpu_id is not None else config.get('stage3.gpu_id', config.get('stage2.gpu_id', 1))

    # When CUDA_VISIBLE_DEVICES is set in shell script, use cuda:0
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} (GPU {gpu_id} via CUDA_VISIBLE_DEVICES)")

    # Paths - use experiment-specific directories
    checkpoint_dir = os.path.join(config.output_dir, experiment_name, 'stage3_12.5hz')
    checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)

    # inference_dir from config already includes experiment_name (e.g., /path/to/inference/exp_E)
    inference_dir = os.path.join(config.inference_dir, 'stage3')

    # Add subdirectory if specified (e.g., aishell, librispeech)
    if args.output_subdir:
        inference_dir = os.path.join(inference_dir, args.output_subdir)

    # Override CSV and audio paths if provided
    csv_path = args.csv_path if args.csv_path else config.test_csv
    audio_base_path = args.audio_base_path if args.audio_base_path else config.audio_base_path

    # Calculate bitrate for display
    frame_rate = 12.5
    bits = 11  # 2048 codebook = 2^11 = 11-bit
    bitrate = frame_rate * bits

    # Determine dataset name for display
    dataset_name = args.output_subdir.upper() if args.output_subdir else "Chinese (AISHELL)"

    print("="*60)
    print(f"  Inference: Stage 3 ({frame_rate}Hz {bits}-bit) - {dataset_name}")
    print("="*60)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Codebook: 2048 ({bits}-bit)")
    print(f"  Bitrate: {bitrate} bps")
    print(f"  Samples: {max_samples}")
    print(f"  CSV: {csv_path}")
    print(f"  Audio base: {audio_base_path}")
    print(f"  Output: {inference_dir}")
    print(f"  8-bit mode: {args.use_8bit}")
    print("="*60)

    # Check checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Make sure Stage 3 training is completed")
        return 1

    # Create inference
    inference = CodecInference(
        checkpoint_path=checkpoint_path,
        model_cache_dir=config.model_cache_dir,
        base_model=config.base_model,
        device=device
    )

    # Process dataset
    processed, failed = inference.process_dataset(
        csv_path=csv_path,
        base_path=audio_base_path,
        output_dir=inference_dir,
        max_samples=max_samples
    )

    print(f"\nDone! {processed} files saved to {inference_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
