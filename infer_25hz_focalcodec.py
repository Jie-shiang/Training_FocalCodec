#!/usr/bin/env python3
"""
Inference script for 25Hz FocalCodec - outputs reconstructed audio files.

Configuration is loaded from config.yaml.

Usage:
    python infer_25hz_focalcodec.py
    python infer_25hz_focalcodec.py --config /path/to/config.yaml
    python infer_25hz_focalcodec.py --stage 1 --max_samples 500

Author: Research
Date: 2026-01-18
"""

import os
import sys
import argparse
import torch
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


def create_25hz_model(base_model_path: str, codebook_size: int, model_cache_dir: str, device: str = 'cuda'):
    """Create 25Hz causal model from 50Hz 2k base."""
    from focalcodec import FocalCodec

    bits = int(math.log2(codebook_size))

    print(f"Creating 25Hz {bits}-bit model")
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

    model = FocalCodec(
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

    return model


class CodecInference:
    def __init__(self, checkpoint_path: str, codebook_size: int, model_cache_dir: str, base_model: str, device: str = 'cuda'):
        self.device = device
        self.codebook_size = codebook_size
        self.resamplers = {}

        print(f"Loading model from: {checkpoint_path}")
        self.model = create_25hz_model(base_model, codebook_size, model_cache_dir, device)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model = self.model.to(device)
        self.model.eval()

        epoch = checkpoint.get('epoch', 'unknown')
        val_loss = checkpoint.get('val_loss', 'N/A')
        print(f"Model loaded (Epoch: {epoch}, Val Loss: {val_loss})")

        bits = int(math.log2(codebook_size))
        bitrate = 25 * bits
        print(f"Config: 25 Hz, {bits}-bit ({codebook_size} codes), Bitrate: {bitrate} bps")

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
    def reconstruct(self, audio: np.ndarray) -> np.ndarray:
        """Reconstruct audio through codec."""
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)

        enc_out = self.model.encoder(audio_tensor)
        if isinstance(enc_out, tuple):
            enc_out = enc_out[0]

        comp_out = self.model.compressor(enc_out)
        if isinstance(comp_out, tuple):
            comp_out = comp_out[0]

        quant_result = self.model.quantizer(comp_out)
        if isinstance(quant_result, tuple):
            codes = quant_result[1]
        else:
            codes = quant_result

        decomp_out = self.model.decompressor(codes)
        if isinstance(decomp_out, tuple):
            decomp_out = decomp_out[0]

        reconstructed = self.model.decoder(decomp_out)
        if isinstance(reconstructed, tuple):
            reconstructed = reconstructed[0]

        return reconstructed.squeeze(0).cpu().numpy()

    def process_dataset(self, csv_path: str, base_path: str, output_dir: str, max_samples: int = 2000):
        """Process dataset and save reconstructed audio."""
        os.makedirs(output_dir, exist_ok=True)

        df = pd.read_csv(csv_path)
        if max_samples > 0:
            df = df.head(max_samples)

        print(f"\nProcessing {len(df)} samples...")

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
            torchaudio.save(output_path, recon_tensor, 16000)

            processed += 1

        print(f"\nCompleted: {processed} processed, {failed} failed")
        return processed, failed


def parse_args():
    parser = argparse.ArgumentParser(description='Inference for 25Hz FocalCodec')

    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.yaml')
    parser.add_argument('--stage', type=int, default=None,
                       help='Stage to load checkpoint from (overrides config)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to process (overrides config)')
    parser.add_argument('--gpu_id', type=int, default=None,
                       help='GPU ID (overrides config)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Add focalcodec to path
    sys.path.insert(0, config.focalcodec_dir)

    # Override with command line args
    stage = args.stage if args.stage is not None else config.stage
    max_samples = args.max_samples if args.max_samples is not None else config.get('inference.max_samples', 2000)
    gpu_id = args.gpu_id if args.gpu_id is not None else config.get('inference.gpu_id', 0)

    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Paths
    checkpoint_dir = config.get_checkpoint_dir(stage)
    checkpoint_name = config.get('inference.checkpoint', 'best_model.pt')
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    output_dir = config.get_inference_dir(stage)

    # Calculate bitrate for display
    bits = int(math.log2(config.codebook_size))
    bitrate = 25 * bits

    print("="*60)
    print(f"  Inference: 25Hz {bits}-bit Codec")
    print("="*60)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Codebook: {config.codebook_size} ({bits}-bit)")
    print(f"  Bitrate: {bitrate} bps")
    print(f"  Samples: {max_samples}")
    print(f"  Output: {output_dir}")
    print("="*60)

    # Check checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return 1

    # Create inference
    inference = CodecInference(
        checkpoint_path=checkpoint_path,
        codebook_size=config.codebook_size,
        model_cache_dir=config.model_cache_dir,
        base_model=config.base_model,
        device=device
    )

    # Process dataset
    processed, failed = inference.process_dataset(
        csv_path=config.train_csv,
        base_path=config.audio_base_path,
        output_dir=output_dir,
        max_samples=max_samples
    )

    print(f"\nDone! {processed} files saved to {output_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
