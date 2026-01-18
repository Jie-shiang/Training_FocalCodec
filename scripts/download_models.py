#!/usr/bin/env python3
"""
Download FocalCodec pretrained models.

This script downloads the required pretrained models for 25Hz 2048 training.
Models are cached locally for faster loading.

Usage:
    python scripts/download_models.py

Author: Research
"""

import os
import sys
from pathlib import Path

# Add focalcodec to path
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir / "focalcodec"))

# Configuration
MODEL_CACHE_DIR = '/mnt/Internal/jieshiang/Model/FocalCodec'

MODELS = [
    {
        'name': 'lucadellalib/focalcodec_50hz_2k_causal',
        'description': '50Hz 2k Causal (base model for 25Hz training)'
    },
    {
        'name': 'lucadellalib/focalcodec_25hz',
        'description': '25Hz Non-Causal (teacher model)'
    }
]


def main():
    print("="*60)
    print("  FocalCodec Model Downloader")
    print("="*60)
    print(f"\nCache directory: {MODEL_CACHE_DIR}")

    # Create cache dir
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    # Import FocalCodec
    try:
        from focalcodec import FocalCodec
    except ImportError as e:
        print(f"\nERROR: Cannot import FocalCodec: {e}")
        print("Make sure focalcodec is installed:")
        print(f"  cd {script_dir / 'focalcodec'} && pip install -e .")
        return 1

    # Download each model
    for model_info in MODELS:
        name = model_info['name']
        desc = model_info['description']

        print(f"\n{'='*60}")
        print(f"Downloading: {name}")
        print(f"Description: {desc}")
        print(f"{'='*60}")

        try:
            model = FocalCodec.from_pretrained(
                name,
                cache_dir=MODEL_CACHE_DIR
            )
            print(f"✓ Successfully downloaded: {name}")
            print(f"  Causal: {model.causal}")

            # Clean up
            del model

        except Exception as e:
            print(f"✗ Failed to download {name}: {e}")
            return 1

    print("\n" + "="*60)
    print("  All models downloaded successfully!")
    print("="*60)
    print(f"\nModels cached at: {MODEL_CACHE_DIR}")
    print("\nYou can now run training:")
    print("  bash train_25hz_stage1.sh")

    return 0


if __name__ == '__main__':
    sys.exit(main())
