#!/usr/bin/env python3
"""
dCER Evaluation Script for two-stage FocalCodec

Evaluates semantic preservation by computing delta Character Error Rate (dCER).
dCER = CER(reconstructed) - CER(original)
Lower dCER means better semantic preservation.

Uses Paraformer-zh (FunASR) for Chinese ASR - same as the original 7% dCER evaluation.
Supports batch processing for faster evaluation.

Supports both Stage 1 (50Hz) and Stage 2 (25Hz) models.
Configuration is loaded from config.yaml.

Usage:
    python eval_two_stage.py --stage 1  # Evaluate Stage 1 (50Hz)
    python eval_two_stage.py --stage 2  # Evaluate Stage 2 (25Hz)
    python eval_two_stage.py --stage 1 --max_samples 500 --gpu_id 1
    python eval_two_stage.py --stage 1 --asr whisper  # Use Whisper instead

Author: Research
Date: 2026-01-20
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json
import torch
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

from utils.config_loader import load_config


def compute_cer(hypothesis: str, reference: str) -> float:
    """Compute Character Error Rate."""
    import Levenshtein

    # Handle NaN or non-string types (convert to string)
    if not isinstance(hypothesis, str):
        hypothesis = str(hypothesis) if hypothesis is not None else ''
    if not isinstance(reference, str):
        reference = str(reference) if reference is not None else ''

    # Handle empty reference
    if not reference or reference == 'nan':
        return 0.0

    # Normalize: remove spaces and convert to lowercase for Chinese
    hypothesis = hypothesis.replace(' ', '').lower()
    reference = reference.replace(' ', '').lower()
    distance = Levenshtein.distance(hypothesis, reference)
    return distance / len(reference) if len(reference) > 0 else 0.0


class ASREngine:
    """ASR Engine supporting both Paraformer-zh and Whisper."""

    def __init__(self, asr_type='paraformer', gpu_id=0, cache_dir=None):
        """
        Initialize ASR engine.

        Args:
            asr_type: 'paraformer' or 'whisper'
            gpu_id: GPU ID to use (note: when CUDA_VISIBLE_DEVICES is set, use cuda:0)
            cache_dir: Model cache directory
        """
        self.asr_type = asr_type
        self.gpu_id = gpu_id
        self.cache_dir = cache_dir
        self.model = None
        # When CUDA_VISIBLE_DEVICES is set, the visible GPU becomes cuda:0
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self._load_model()

    def _load_model(self):
        """Load ASR model."""
        if self.asr_type == 'paraformer':
            print(f"Loading Paraformer-zh (FunASR) on GPU {self.gpu_id}...")
            try:
                from funasr import AutoModel
                self.model = AutoModel(
                    model="paraformer-zh",
                    hub="hf",  # Use Hugging Face hub (already cached)
                    device=self.device,
                    disable_update=True
                )
                print(f"Paraformer-zh loaded on {self.device}")
            except ImportError:
                print("ERROR: FunASR not installed. Install with: pip install funasr")
                raise
            except Exception as e:
                print(f"ERROR loading Paraformer: {e}")
                raise
        else:  # whisper
            print(f"Loading Whisper on GPU {self.gpu_id}...")
            from transformers import pipeline
            self.model = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-small",
                device=self.device,
                model_kwargs={"cache_dir": self.cache_dir} if self.cache_dir else {}
            )
            print(f"Whisper loaded on {self.device}")

    def transcribe_batch(self, audio_paths: list, batch_size: int = 16) -> list:
        """
        Transcribe a batch of audio files.

        Args:
            audio_paths: List of audio file paths
            batch_size: Batch size for processing

        Returns:
            List of transcriptions
        """
        results = []

        if self.asr_type == 'paraformer':
            # Paraformer supports batch processing
            for i in tqdm(range(0, len(audio_paths), batch_size),
                         desc="Transcribing (Paraformer)", leave=False):
                batch_paths = audio_paths[i:i+batch_size]
                try:
                    batch_results = self.model.generate(input=batch_paths, batch_size=batch_size)
                    for res in batch_results:
                        if isinstance(res, dict) and 'text' in res:
                            results.append(res['text'])
                        elif isinstance(res, list) and len(res) > 0:
                            results.append(res[0].get('text', ''))
                        else:
                            results.append('')
                except Exception as e:
                    print(f"Batch error: {e}")
                    # Fall back to individual processing
                    for path in batch_paths:
                        try:
                            res = self.model.generate(input=path)
                            if isinstance(res, list) and len(res) > 0:
                                results.append(res[0].get('text', ''))
                            else:
                                results.append('')
                        except:
                            results.append('')
        else:  # whisper
            for path in tqdm(audio_paths, desc="Transcribing (Whisper)", leave=False):
                try:
                    result = self.model(path, generate_kwargs={"language": "zh"})
                    results.append(result['text'])
                except Exception as e:
                    print(f"Error transcribing {path}: {e}")
                    results.append('')

        return results

    def transcribe_single(self, audio_path: str) -> str:
        """Transcribe a single audio file."""
        results = self.transcribe_batch([audio_path], batch_size=1)
        return results[0] if results else ''


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate dCER for two-stage FocalCodec')

    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.yaml')
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2],
                       help='Stage to evaluate (1=50Hz, 2=25Hz)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to evaluate (overrides config)')
    parser.add_argument('--gpu_id', type=int, default=None,
                       help='GPU ID (overrides config)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for ASR processing')
    parser.add_argument('--asr', type=str, default='paraformer', choices=['paraformer', 'whisper'],
                       help='ASR model to use (default: paraformer for consistency with original 7%% dCER)')
    parser.add_argument('--inference_subdir', type=str, default=None,
                       help='Inference subdirectory (e.g., aishell, librispeech)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Get stage config
    stage = args.stage
    stage_key = f'stage{stage}'

    max_samples = args.max_samples if args.max_samples is not None else config.get('evaluation.max_samples', 2000)
    gpu_id = args.gpu_id if args.gpu_id is not None else config.get('evaluation.gpu_id', 1)
    batch_size = args.batch_size

    # Paths
    inference_dir = os.path.join(config.inference_dir, f'stage{stage}')

    # Add subdirectory if specified (e.g., aishell, librispeech)
    if args.inference_subdir:
        inference_dir = os.path.join(inference_dir, args.inference_subdir)

    csv_path = config.test_csv
    base_path = config.audio_base_path
    asr_cache_dir = config.asr_cache_dir

    frame_rate = 50 if stage == 1 else 25
    bits = 11
    bitrate = frame_rate * bits

    print("="*60)
    print(f"  dCER Evaluation: Stage {stage} ({frame_rate}Hz {bits}-bit)")
    print("="*60)
    print(f"  Inference directory: {inference_dir}")
    print(f"  CSV path: {csv_path}")
    print(f"  Max samples: {max_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  GPU ID: {gpu_id}")
    print(f"  ASR model: {args.asr}")
    print(f"  Bitrate: {bitrate} bps")
    print("="*60)

    # Check inference directory exists
    if not os.path.exists(inference_dir):
        print(f"ERROR: Inference directory not found: {inference_dir}")
        print(f"Run inference first: python infer_two_stage.py --stage {stage}")
        return 1

    # Import Levenshtein
    try:
        import Levenshtein
    except ImportError:
        print("ERROR: Levenshtein not installed. Install with: pip install python-Levenshtein")
        return 1

    # Initialize ASR Engine
    print()
    asr_engine = ASREngine(asr_type=args.asr, gpu_id=gpu_id, cache_dir=asr_cache_dir)

    # Load CSV
    df = pd.read_csv(csv_path)
    if max_samples > 0:
        df = df.head(max_samples)

    print(f"\nPreparing {len(df)} samples...")

    # Collect valid samples
    valid_samples = []
    for idx, row in df.iterrows():
        file_name = row['file_name']
        relative_path = row['file_path']
        ground_truth = row.get('transcription', '')

        # Handle NaN or non-string ground_truth
        if pd.isna(ground_truth):
            ground_truth = ''
        elif not isinstance(ground_truth, str):
            ground_truth = str(ground_truth)

        original_path = os.path.join(base_path, relative_path.lstrip('./'))
        recon_path = os.path.join(inference_dir, f"{file_name}_inference.wav")

        if os.path.exists(original_path) and os.path.exists(recon_path):
            valid_samples.append({
                'file_name': file_name,
                'original_path': original_path,
                'recon_path': recon_path,
                'ground_truth': ground_truth
            })

    print(f"Found {len(valid_samples)} valid samples (with both original and reconstructed audio)")

    if len(valid_samples) == 0:
        print("ERROR: No valid samples found!")
        return 1

    # Batch transcribe original audio
    print("\n[1/2] Transcribing original audio...")
    original_paths = [s['original_path'] for s in valid_samples]
    original_texts = asr_engine.transcribe_batch(original_paths, batch_size=batch_size)

    # Batch transcribe reconstructed audio
    print("\n[2/2] Transcribing reconstructed audio...")
    recon_paths = [s['recon_path'] for s in valid_samples]
    recon_texts = asr_engine.transcribe_batch(recon_paths, batch_size=batch_size)

    # Compute dCER
    print("\nComputing dCER...")
    results = []
    dcer_values = []

    for i, sample in enumerate(tqdm(valid_samples, desc="Computing CER")):
        ground_truth = sample['ground_truth']
        original_text = original_texts[i]
        recon_text = recon_texts[i]

        cer_original = compute_cer(original_text, ground_truth)
        cer_recon = compute_cer(recon_text, ground_truth)
        dcer = cer_recon - cer_original

        dcer_values.append(dcer)

        results.append({
            'file_name': sample['file_name'],
            'ground_truth': ground_truth,
            'original_text': original_text,
            'reconstructed_text': recon_text,
            'cer_original': cer_original,
            'cer_reconstructed': cer_recon,
            'dCER': dcer
        })

    # Compute statistics
    dcer_array = np.array(dcer_values)

    summary = {
        'num_samples': len(results),
        'asr_model': args.asr,
        'dCER_mean': float(dcer_array.mean()),
        'dCER_std': float(dcer_array.std()),
        'dCER_min': float(dcer_array.min()),
        'dCER_max': float(dcer_array.max()),
        'dCER_median': float(np.median(dcer_array)),
        'dCER_q25': float(np.percentile(dcer_array, 25)),
        'dCER_q75': float(np.percentile(dcer_array, 75)),
        'timestamp': datetime.now().isoformat(),
        'inference_dir': inference_dir,
        'stage': stage,
        'frame_rate': frame_rate,
        'bitrate': bitrate,
    }

    # Save results
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(inference_dir, f'dcer_results_{args.asr}.csv')
    results_df.to_csv(results_csv, index=False)

    summary_json = os.path.join(inference_dir, f'dcer_summary_{args.asr}.json')
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "="*60)
    print(f"  EVALUATION RESULTS - Stage {stage} ({args.asr.upper()})")
    print("="*60)
    print(f"  ASR Model: {args.asr}")
    print(f"  Samples evaluated: {summary['num_samples']}")
    print(f"  Frame rate: {frame_rate} Hz")
    print(f"  Bitrate: {bitrate} bps")
    print()
    print(f"  dCER Mean:   {summary['dCER_mean']:.4f} ({summary['dCER_mean']*100:.2f}%) ± {summary['dCER_std']:.4f}")
    print(f"  dCER Median: {summary['dCER_median']:.4f} ({summary['dCER_median']*100:.2f}%)")
    print(f"  dCER Range:  [{summary['dCER_min']:.4f}, {summary['dCER_max']:.4f}]")
    print(f"  dCER IQR:    [{summary['dCER_q25']:.4f}, {summary['dCER_q75']:.4f}]")
    print()

    if summary['dCER_mean'] < 0.1:
        quality = "Excellent (< 10%)"
    elif summary['dCER_mean'] < 0.2:
        quality = "Good (10-20%)"
    elif summary['dCER_mean'] < 0.3:
        quality = "Moderate (20-30%)"
    elif summary['dCER_mean'] < 0.5:
        quality = "Fair (30-50%)"
    else:
        quality = "Poor (> 50%)"

    print(f"  Quality: {quality}")
    print()
    print(f"  Results saved to:")
    print(f"    - {results_csv}")
    print(f"    - {summary_json}")
    print("="*60)

    # Compare with baseline if available
    baseline_csv = os.path.join(inference_dir, 'dcer_results_paraformer.csv')
    if args.asr != 'paraformer' and os.path.exists(baseline_csv):
        baseline_df = pd.read_csv(baseline_csv)
        baseline_dcer = baseline_df['dCER'].mean()
        print(f"\n  Comparison with Paraformer baseline:")
        print(f"    Paraformer dCER: {baseline_dcer:.4f} ({baseline_dcer*100:.2f}%)")
        print(f"    Current dCER:    {summary['dCER_mean']:.4f} ({summary['dCER_mean']*100:.2f}%)")
        print("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
