#!/usr/bin/env python3
"""
FocalCodec 25Hz 2048 Setup Verification Script

Checks all dependencies and configurations before training.
Run: python scripts/setup_check.py

Author: Research
"""

import os
import sys
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
CHECK = '✓'
CROSS = '✗'
WARN = '!'

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_ok(msg):
    print(f"  {GREEN}{CHECK}{RESET} {msg}")

def print_fail(msg):
    print(f"  {RED}{CROSS}{RESET} {msg}")

def print_warn(msg):
    print(f"  {YELLOW}{WARN}{RESET} {msg}")


def check_focalcodec_repo():
    """Check 1: FocalCodec repository clone."""
    print_header("1. FocalCodec Repository")

    # Expected path (relative to this script's parent's parent)
    script_dir = Path(__file__).parent.parent
    focalcodec_dir = script_dir / "focalcodec"

    if focalcodec_dir.exists():
        # Check key files
        key_files = [
            "focalcodec/__init__.py",
            "focalcodec/focalcodec.py",
            "setup.py"
        ]
        all_exist = True
        for f in key_files:
            if not (focalcodec_dir / f).exists():
                all_exist = False
                print_fail(f"Missing: {f}")

        if all_exist:
            print_ok(f"Found at: {focalcodec_dir}")
            return True
        else:
            print_fail("Repository incomplete")
            return False
    else:
        print_fail(f"Not found at: {focalcodec_dir}")
        print(f"\n  To fix, run:")
        print(f"    git clone https://github.com/lucadellalib/focalcodec.git {focalcodec_dir}")
        return False


def check_pretrained_models():
    """Check 2: Pretrained models downloaded."""
    print_header("2. Pretrained Models")

    model_dir = Path("/mnt/Internal/jieshiang/Model/FocalCodec")

    # Check for cached models from Hugging Face
    expected_models = [
        "lucadellalib/focalcodec_50hz_2k_causal",
        "lucadellalib/focalcodec_25hz"
    ]

    if not model_dir.exists():
        print_warn(f"Model cache directory not found: {model_dir}")
        print(f"\n  Models will be downloaded automatically on first run")
        print(f"  Or manually download with:")
        print(f"    python -c \"from focalcodec import FocalCodec; FocalCodec.from_pretrained('lucadellalib/focalcodec_50hz_2k_causal', cache_dir='{model_dir}')\"")
        print(f"    python -c \"from focalcodec import FocalCodec; FocalCodec.from_pretrained('lucadellalib/focalcodec_25hz', cache_dir='{model_dir}')\"")
        return False

    # Check hub cache structure
    hub_cache = model_dir / "models--lucadellalib--focalcodec_50hz_2k_causal"
    if hub_cache.exists():
        print_ok("50Hz 2k causal model cached")
    else:
        print_warn("50Hz 2k causal model not cached (will download on first run)")

    hub_cache2 = model_dir / "models--lucadellalib--focalcodec_25hz"
    if hub_cache2.exists():
        print_ok("25Hz non-causal teacher model cached")
    else:
        print_warn("25Hz teacher model not cached (will download on first run)")

    return True


def check_packages():
    """Check 3: Python packages."""
    print_header("3. Python Packages")

    required = {
        'torch': 'PyTorch',
        'torchaudio': 'TorchAudio',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'tqdm': 'TQDM',
        'transformers': 'Transformers (HuggingFace)',
        'huggingface_hub': 'Hugging Face Hub',
        'safetensors': 'Safetensors',
    }

    optional = {
        'whisper': 'OpenAI Whisper (for evaluation)',
        'Levenshtein': 'Levenshtein (for dCER)',
        'soundfile': 'Soundfile',
    }

    all_ok = True

    # Check required
    for pkg, name in required.items():
        try:
            module = __import__(pkg)
            version = getattr(module, '__version__', 'unknown')
            print_ok(f"{name}: {version}")
        except ImportError:
            print_fail(f"{name}: NOT INSTALLED")
            all_ok = False

    # Check optional
    print("\n  Optional packages:")
    for pkg, name in optional.items():
        try:
            module = __import__(pkg)
            version = getattr(module, '__version__', 'unknown')
            print_ok(f"{name}: {version}")
        except ImportError:
            print_warn(f"{name}: not installed")

    # Check CUDA
    print("\n  GPU:")
    try:
        import torch
        if torch.cuda.is_available():
            print_ok(f"CUDA available: {torch.cuda.get_device_name(0)}")
            print_ok(f"CUDA version: {torch.version.cuda}")
        else:
            print_warn("CUDA not available - will use CPU (slow)")
    except:
        print_warn("Could not check CUDA")

    return all_ok


def check_data():
    """Check 4: AISHELL data."""
    print_header("4. Training Data (AISHELL)")

    script_dir = Path(__file__).parent.parent

    # Check CSV files
    train_csv = script_dir / "experiments" / "data_aishell" / "train_split.csv"
    val_csv = script_dir / "experiments" / "data_aishell" / "val_split.csv"

    if not train_csv.exists():
        print_fail(f"Training CSV not found: {train_csv}")
        return False

    if not val_csv.exists():
        print_fail(f"Validation CSV not found: {val_csv}")
        return False

    print_ok(f"Train CSV: {train_csv}")
    print_ok(f"Val CSV: {val_csv}")

    # Check audio base path
    base_path = Path("/mnt/Internal/ASR")
    aishell_path = base_path / "aishell" / "data_aishell" / "wav"

    if not aishell_path.exists():
        print_fail(f"AISHELL audio not found at: {aishell_path}")
        print(f"\n  Expected structure:")
        print(f"    /mnt/Internal/ASR/aishell/data_aishell/wav/test/...")
        print(f"    /mnt/Internal/ASR/aishell/data_aishell/wav/train/...")
        return False

    print_ok(f"AISHELL audio: {aishell_path}")

    # Verify sample access
    import pandas as pd
    try:
        df = pd.read_csv(train_csv)
        sample_path = df.iloc[0]['file_path']
        full_path = base_path / sample_path.lstrip('./')

        if full_path.exists():
            print_ok(f"Sample audio accessible: {full_path.name}")
        else:
            print_fail(f"Sample audio not found: {full_path}")
            return False
    except Exception as e:
        print_fail(f"Error reading CSV: {e}")
        return False

    # Count samples
    print(f"\n  Dataset size:")
    print(f"    Train: {len(pd.read_csv(train_csv))} samples")
    print(f"    Val: {len(pd.read_csv(val_csv))} samples")

    return True


def check_focalcodec_import():
    """Check 5: FocalCodec import."""
    print_header("5. FocalCodec Import Test")

    script_dir = Path(__file__).parent.parent
    focalcodec_path = script_dir / "focalcodec"

    # Add to path
    sys.path.insert(0, str(focalcodec_path))

    try:
        from focalcodec import FocalCodec
        print_ok("FocalCodec imported successfully")
        return True
    except ImportError as e:
        print_fail(f"Import failed: {e}")
        print(f"\n  Make sure focalcodec is properly installed:")
        print(f"    cd {focalcodec_path} && pip install -e .")
        return False


def check_output_dirs():
    """Check 6: Output directories."""
    print_header("6. Output Directories")

    dirs = [
        "/mnt/Internal/jieshiang/Model/FocalCodec/25hz_2048/stage_1",
        "/mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/25hz_2048/stage_1",
    ]

    for d in dirs:
        path = Path(d)
        if path.exists():
            print_ok(f"Exists: {d}")
        else:
            print_warn(f"Will be created: {d}")

    return True


def main():
    print("\n" + "="*60)
    print("  FocalCodec 25Hz 2048 Setup Verification")
    print("="*60)

    results = []

    results.append(("FocalCodec Repository", check_focalcodec_repo()))
    results.append(("Pretrained Models", check_pretrained_models()))
    results.append(("Python Packages", check_packages()))
    results.append(("Training Data", check_data()))
    results.append(("FocalCodec Import", check_focalcodec_import()))
    results.append(("Output Directories", check_output_dirs()))

    # Summary
    print_header("SUMMARY")

    all_ok = True
    for name, status in results:
        if status:
            print_ok(name)
        else:
            print_fail(name)
            all_ok = False

    print()
    if all_ok:
        print(f"  {GREEN}All checks passed! Ready to train.{RESET}")
        print(f"\n  Start training with:")
        print(f"    bash train_25hz_stage1.sh")
    else:
        print(f"  {RED}Some checks failed. Please fix issues above.{RESET}")

    print()
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
