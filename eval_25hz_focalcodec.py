#!/usr/bin/env python3
"""
dCER Evaluation Script for 25Hz FocalCodec

Evaluates semantic preservation by computing delta Character Error Rate (dCER).
dCER = CER(reconstructed) - CER(original)
Lower dCER means better semantic preservation.

Configuration is loaded from config.yaml.

Usage:
    python eval_25hz_focalcodec.py
    python eval_25hz_focalcodec.py --config /path/to/config.yaml
    python eval_25hz_focalcodec.py --stage 1 --max_samples 500

Author: Research
Date: 2026-01-18
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

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

from utils.config_loader import load_config


def compute_cer(hypothesis: str, reference: str) -> float:
    """Compute Character Error Rate."""
    import Levenshtein
    if not reference:
        return 0.0
    distance = Levenshtein.distance(hypothesis, reference)
    return distance / len(reference)


def transcribe_audio(audio_path: str, asr_pipeline) -> str:
    """Transcribe audio file using Whisper."""
    try:
        result = asr_pipeline(audio_path, generate_kwargs={"language": "zh"})
        return result['text']
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return ""


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate dCER for 25Hz FocalCodec')

    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.yaml')
    parser.add_argument('--stage', type=int, default=None,
                       help='Stage to evaluate (overrides config)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to evaluate (overrides config)')
    parser.add_argument('--gpu_id', type=int, default=None,
                       help='GPU ID (overrides config)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override with command line args
    stage = args.stage if args.stage is not None else config.stage
    max_samples = args.max_samples if args.max_samples is not None else config.get('evaluation.max_samples', 2000)
    gpu_id = args.gpu_id if args.gpu_id is not None else config.get('evaluation.gpu_id', 0)

    # Paths
    inference_dir = config.get_inference_dir(stage)
    csv_path = config.train_csv
    base_path = config.audio_base_path
    asr_model = config.get('evaluation.asr_model', 'openai/whisper-small')
    asr_cache_dir = config.asr_cache_dir

    print("="*60)
    print("  dCER Evaluation for 25Hz FocalCodec")
    print("="*60)
    print(f"  Inference directory: {inference_dir}")
    print(f"  CSV path: {csv_path}")
    print(f"  Max samples: {max_samples}")
    print("="*60)

    # Check inference directory exists
    if not os.path.exists(inference_dir):
        print(f"ERROR: Inference directory not found: {inference_dir}")
        print("Run inference first: python infer_25hz_focalcodec.py")
        return 1

    # Load ASR model
    print("\nLoading Whisper ASR model...")
    try:
        from transformers import pipeline
        import Levenshtein
    except ImportError as e:
        print(f"ERROR: Missing package: {e}")
        print("Install with: pip install transformers Levenshtein")
        return 1

    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=asr_model,
        device=gpu_id if gpu_id >= 0 else -1,
        model_kwargs={"cache_dir": asr_cache_dir}
    )
    print("ASR model loaded")

    # Load CSV
    df = pd.read_csv(csv_path)
    if max_samples > 0:
        df = df.head(max_samples)

    print(f"\nEvaluating {len(df)} samples...")

    # Process samples
    results = []
    dcer_values = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        file_name = row['file_name']
        relative_path = row['file_path']
        ground_truth = row.get('transcription', '')

        original_path = os.path.join(base_path, relative_path.lstrip('./'))
        recon_path = os.path.join(inference_dir, f"{file_name}_inference.wav")

        if not os.path.exists(original_path):
            continue
        if not os.path.exists(recon_path):
            continue

        original_text = transcribe_audio(original_path, asr_pipeline)
        recon_text = transcribe_audio(recon_path, asr_pipeline)

        cer_original = compute_cer(original_text, ground_truth)
        cer_recon = compute_cer(recon_text, ground_truth)
        dcer = cer_recon - cer_original

        dcer_values.append(dcer)

        results.append({
            'file_name': file_name,
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
    }

    # Save results
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(inference_dir, 'dcer_results.csv')
    results_df.to_csv(results_csv, index=False)

    summary_json = os.path.join(inference_dir, 'dcer_summary.json')
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("  EVALUATION RESULTS")
    print("="*60)
    print(f"  Samples evaluated: {summary['num_samples']}")
    print()
    print(f"  dCER Mean:   {summary['dCER_mean']:.4f} ± {summary['dCER_std']:.4f}")
    print(f"  dCER Median: {summary['dCER_median']:.4f}")
    print(f"  dCER Range:  [{summary['dCER_min']:.4f}, {summary['dCER_max']:.4f}]")
    print(f"  dCER IQR:    [{summary['dCER_q25']:.4f}, {summary['dCER_q75']:.4f}]")
    print()

    if summary['dCER_mean'] < 0.1:
        quality = "Excellent"
    elif summary['dCER_mean'] < 0.3:
        quality = "Good"
    elif summary['dCER_mean'] < 0.5:
        quality = "Moderate"
    else:
        quality = "Poor"

    print(f"  Quality: {quality} (lower dCER is better)")
    print()
    print(f"  Results saved to:")
    print(f"    - {results_csv}")
    print(f"    - {summary_json}")
    print("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
