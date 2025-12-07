#!/usr/bin/env python3
"""
FocalCodec Batch Inference - Supports Fine-tuned Models
Processes multiple audio files with optional fine-tuned checkpoints

Usage:
    # With fine-tuned model
    python focalcodec_batch_inference.py \
        --base_audio_dir /mnt/Internal/ASR \
        --commonvoice_csv /path/to/chinese.csv \
        --models 50hz_2k \
        --output_dir /mnt/Internal/jieshiang/Inference_Result \
        --model_cache_dir /mnt/Internal/jieshiang/Model/FocalCodec \
        --checkpoint /path/to/best_model.pt \
        --device cuda:0
"""

import os
import sys
import argparse
import pandas as pd
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional
import warnings
import logging
import re

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Model configuration mapping
MODEL_CONFIGS = {
    "12.5hz": {
        "config": "lucadellalib/focalcodec_12_5hz",
        "output_path": "FocalCodec/12.5HZ",
        "is_streaming": False
    },
    "25hz": {
        "config": "lucadellalib/focalcodec_25hz",
        "output_path": "FocalCodec/25HZ",
        "is_streaming": False
    },
    "50hz": {
        "config": "lucadellalib/focalcodec_50hz",
        "output_path": "FocalCodec/50HZ",
        "is_streaming": False
    },
    "50hz_2k": {
        "config": "lucadellalib/focalcodec_50hz_2k_causal",
        "output_path": "FocalCodec-S/50HZ/2K",
        "is_streaming": True
    },
    "50hz_4k": {
        "config": "lucadellalib/focalcodec_50hz_4k_causal",
        "output_path": "FocalCodec-S/50HZ/4K",
        "is_streaming": True
    },
    "50hz_65k": {
        "config": "lucadellalib/focalcodec_50hz_65k_causal",
        "output_path": "FocalCodec-S/50HZ/65K",
        "is_streaming": True
    },
}


class FocalCodecInferenceProcessor:
    """FocalCodec inference processor with fine-tuned model support"""
    
    def __init__(
        self,
        model_name: str,
        model_cache_dir: str,
        device: str = "cuda:0",
        force_reload: bool = False,
        checkpoint_path: str = None
    ):
        """
        Initialize inference processor
        
        Args:
            model_name: Model name (e.g., "50hz", "50hz_4k")
            model_cache_dir: Model cache directory
            device: Device to use (e.g., "cuda:0", "cpu")
            force_reload: Force reload model from hub
            checkpoint_path: Fine-tuned checkpoint path (optional)
        """
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        
        # Get model config
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        
        model_info = MODEL_CONFIGS[model_name]
        self.model_config = model_info["config"]
        self.output_path = model_info["output_path"]
        self.is_streaming = model_info["is_streaming"]
        
        # Modify output path if using checkpoint
        if checkpoint_path:
            self.output_path = self.output_path + "_finetuned"
        
        self.device = torch.device(device)
        self.model_cache_dir = model_cache_dir
        
        logger.info(f"Loading model: {self.model_config}")
        logger.info(f"Output path: {self.output_path}")
        logger.info(f"Streaming model: {'Yes' if self.is_streaming else 'No'}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model cache: {self.model_cache_dir}")
        if checkpoint_path:
            logger.info(f"🔑 Fine-tuned checkpoint: {checkpoint_path}")
        
        # Set Hugging Face cache
        os.environ['HF_HOME'] = self.model_cache_dir
        os.environ['HUGGINGFACE_HUB_CACHE'] = self.model_cache_dir
        
        # Load model
        self.codec = self._load_model(force_reload)
        
    def _load_model(self, force_reload: bool):
        """Load FocalCodec model (with fine-tuned checkpoint support)"""
        try:
            # Load base model via torch.hub
            codec = torch.hub.load(
                repo_or_dir="lucadellalib/focalcodec",
                model="focalcodec",
                config=self.model_config,
                force_reload=force_reload,
            )
            
            # Load fine-tuned checkpoint if provided
            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                logger.info(f"Loading fine-tuned checkpoint: {self.checkpoint_path}")
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                
                # Load state dict
                codec.load_state_dict(checkpoint['model_state_dict'])
                
                # Log training info
                epoch = checkpoint.get('epoch', 'N/A')
                val_loss = checkpoint.get('val_loss', 'N/A')
                train_loss = checkpoint.get('train_loss', 'N/A')
                
                logger.info(f"✅ Checkpoint loaded successfully!")
                logger.info(f"   Epoch: {epoch}")
                logger.info(f"   Train Loss: {train_loss:.6f}" if isinstance(train_loss, float) else f"   Train Loss: {train_loss}")
                logger.info(f"   Val Loss: {val_loss:.6f}" if isinstance(val_loss, float) else f"   Val Loss: {val_loss}")
            elif self.checkpoint_path:
                logger.warning(f"⚠️ Checkpoint not found: {self.checkpoint_path}")
                logger.warning("   Using original pre-trained model")
            
            codec = codec.to(self.device)
            codec.eval()
            codec.requires_grad_(False)
            
            logger.info(f"Model loaded: {self.model_name}")
            logger.info(f"Input SR: {codec.sample_rate_input} Hz")
            logger.info(f"Output SR: {codec.sample_rate_output} Hz")
            
            return codec
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            raise
    
    @torch.no_grad()
    def inference_single_file(
        self,
        audio_path: str,
        output_path: str,
        original_sample_rate: int = None
    ) -> bool:
        """
        Inference on single audio file
        
        Args:
            audio_path: Input audio path
            output_path: Output audio path
            original_sample_rate: Original sample rate (None = use file SR)
        
        Returns:
            bool: Success status
        """
        try:
            # Load audio
            sig, sample_rate = torchaudio.load(audio_path)
            
            # Move to device
            sig = sig.to(self.device)
            
            # Resample to model input SR
            if sample_rate != self.codec.sample_rate_input:
                sig = torchaudio.functional.resample(
                    sig, sample_rate, self.codec.sample_rate_input
                ).to(self.device)
            
            # Encode
            toks = self.codec.sig_to_toks(sig)
            
            # Decode
            rec_sig = self.codec.toks_to_sig(toks)
            
            # Move to CPU
            rec_sig = rec_sig.cpu()
            
            # Resample back to target SR if needed
            target_sr = original_sample_rate if original_sample_rate else sample_rate
            if self.codec.sample_rate_output != target_sr:
                rec_sig = torchaudio.functional.resample(
                    rec_sig, self.codec.sample_rate_output, target_sr
                )
            
            # Ensure output dir exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save reconstructed audio
            torchaudio.save(output_path, rec_sig, target_sr)
            
            return True
            
        except Exception as e:
            logger.error(f"Processing failed {audio_path}: {e}")
            return False


def extract_segment_length_from_csv(csv_path: str) -> Optional[str]:
    """
    Extract segment length from CSV filename
    Example: librispeech_test_clean_filtered_0.5s.csv -> "0.5s"
    """
    filename = Path(csv_path).stem
    match = re.search(r'_(\d+\.?\d*s)$', filename)
    
    if match:
        segment_length = match.group(1)
        logger.info(f"Detected segment length: {segment_length}")
        return segment_length
    else:
        logger.info("No segment marker detected, assuming full audio")
        return None


def load_segment_csv_data(csv_path: str, base_audio_dir: str, dataset_type: str) -> List[Dict]:
    """Load CSV data (supports both segmented and full audio)"""
    logger.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Check if segmented CSV
    required_segmented_columns = ['segment_file_name', 'segment_file_path', 'original_file_name']
    is_segmented = all(col in df.columns for col in required_segmented_columns)
    
    if is_segmented:
        logger.info("Detected segmented CSV")
        audio_files = []
        segment_length = extract_segment_length_from_csv(csv_path)
        
        for idx, row in df.iterrows():
            segment_file_name = row['segment_file_name']
            segment_file_path = row['segment_file_path']
            original_file_name = row['original_file_name']
            
            if segment_file_path.startswith('./'):
                segment_file_path = segment_file_path[2:]
            
            full_path = os.path.join(base_audio_dir, segment_file_path)
            
            if os.path.exists(full_path):
                audio_files.append({
                    'segment_file_name': segment_file_name,
                    'original_file_name': original_file_name,
                    'input_path': full_path,
                    'relative_path': segment_file_path,
                    'dataset_type': dataset_type,
                    'segment_length': segment_length
                })
            else:
                logger.warning(f"Audio not found: {full_path}")
        
        logger.info(f"Found {len(audio_files)} valid segmented audio files")
    else:
        logger.info("Detected full audio CSV")
        required_complete_columns = ['file_name', 'file_path']
        if not all(col in df.columns for col in required_complete_columns):
            raise ValueError(
                f"CSV missing required columns. Need: {required_complete_columns}\n"
                f"Actual columns: {list(df.columns)}"
            )
        
        audio_files = []
        
        for idx, row in df.iterrows():
            file_name = row['file_name']
            file_path = row['file_path']
            
            if file_path.startswith('./'):
                file_path = file_path[2:]
            
            full_path = os.path.join(base_audio_dir, file_path)
            
            if os.path.exists(full_path):
                audio_files.append({
                    'file_name': file_name,
                    'input_path': full_path,
                    'relative_path': file_path,
                    'dataset_type': dataset_type
                })
            else:
                logger.warning(f"Audio not found: {full_path}")
        
        logger.info(f"Found {len(audio_files)} valid audio files")
    
    return audio_files


def process_segmented_dataset(
    audio_files: List[Dict],
    processor: FocalCodecInferenceProcessor,
    output_base_dir: str,
    model_name: str
) -> Dict[str, int]:
    """Process audio dataset"""
    stats = {'success': 0, 'failed': 0, 'total': len(audio_files)}
    
    if not audio_files:
        logger.warning("No audio files to process")
        return stats
    
    is_segmented = 'segment_file_name' in audio_files[0]
    
    logger.info(f"Processing {stats['total']} {'segmented' if is_segmented else 'full'} audio files...")
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    for file_info in tqdm(audio_files, desc=f"Processing {model_name}"):
        input_path = file_info['input_path']
        
        if is_segmented:
            file_name = file_info['segment_file_name']
        else:
            file_name = file_info['file_name']
        
        base_name = os.path.splitext(file_name)[0]
        output_filename = f"{base_name}_inference.wav"
        output_path = os.path.join(output_base_dir, output_filename)
        
        success = processor.inference_single_file(input_path, output_path)
        
        if success:
            stats['success'] += 1
        else:
            stats['failed'] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="FocalCodec Batch Inference - Supports Fine-tuned Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

1. With fine-tuned model:
    python focalcodec_batch_inference.py \\
        --base_audio_dir /mnt/Internal/ASR \\
        --commonvoice_csv /path/to/chinese.csv \\
        --models 50hz_2k \\
        --output_dir /mnt/Internal/jieshiang/Inference_Result \\
        --model_cache_dir /mnt/Internal/jieshiang/Model/FocalCodec \\
        --checkpoint /path/to/best_model.pt \\
        --device cuda:1

2. With original model (no --checkpoint):
    python focalcodec_batch_inference.py \\
        --base_audio_dir /mnt/Internal/ASR \\
        --commonvoice_csv /path/to/chinese.csv \\
        --models 50hz_2k \\
        --output_dir /mnt/Internal/jieshiang/Inference_Result \\
        --model_cache_dir /mnt/Internal/jieshiang/Model/FocalCodec \\
        --device cuda:1
        """
    )
    
    # Required args
    parser.add_argument('--base_audio_dir', type=str, required=True, help='Audio base directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output base directory')
    parser.add_argument('--model_cache_dir', type=str, required=True, help='Model cache directory')
    
    # Dataset args
    parser.add_argument('--librispeech_csv', type=str, default=None, help='LibriSpeech CSV file')
    parser.add_argument('--commonvoice_csv', type=str, default=None, help='Common Voice CSV file')
    
    # Model args
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        choices=list(MODEL_CONFIGS.keys()),
                        help=f'Models to use. Available: {list(MODEL_CONFIGS.keys())}')
    
    # Checkpoint arg
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Fine-tuned checkpoint path (e.g., best_model.pt)')
    
    # Device args
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    
    # Other args
    parser.add_argument('--force_reload', action='store_true', help='Force reload model from hub')
    parser.add_argument('--is_noise', action='store_true', help='Processing noise audio files')
    
    args = parser.parse_args()
    
    # Check at least one dataset specified
    if not args.librispeech_csv and not args.commonvoice_csv:
        parser.error("Must specify at least one dataset CSV (--librispeech_csv or --commonvoice_csv)")
    
    # Check CUDA availability
    if 'cuda' in args.device and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Warning: checkpoint should match model config
    if args.checkpoint:
        logger.info("=" * 80)
        logger.info("🔑 Fine-tuned Mode")
        logger.info("=" * 80)
        logger.info(f"Checkpoint: {args.checkpoint}")
        logger.info("⚠️  Note: Ensure checkpoint matches model config!")
        logger.info("   E.g., if checkpoint trained with 50hz_2k, use --models 50hz_2k")
        logger.info("=" * 80)
    
    logger.info("=" * 80)
    logger.info("FocalCodec Batch Inference - Fine-tuned Model Support")
    logger.info("=" * 80)
    logger.info(f"Audio base dir: {args.base_audio_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Model cache: {args.model_cache_dir}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Device: {args.device}")
    if args.checkpoint:
        logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info("=" * 80)
    
    # Load datasets
    all_datasets = []
    
    if args.librispeech_csv:
        librispeech_files = load_segment_csv_data(
            args.librispeech_csv, args.base_audio_dir, 'librispeech'
        )
        all_datasets.append(('librispeech', librispeech_files))
    
    if args.commonvoice_csv:
        commonvoice_files = load_segment_csv_data(
            args.commonvoice_csv, args.base_audio_dir, 'commonvoice'
        )
        all_datasets.append(('commonvoice', commonvoice_files))
    
    # Process each model
    for model_name in args.models:
        logger.info("=" * 80)
        logger.info(f"Processing model: {model_name}")
        logger.info("=" * 80)
        
        # Initialize processor (with checkpoint)
        processor = FocalCodecInferenceProcessor(
            model_name=model_name,
            model_cache_dir=args.model_cache_dir,
            device=args.device,
            force_reload=args.force_reload,
            checkpoint_path=args.checkpoint
        )
        
        # Process each dataset
        for dataset_name, audio_files in all_datasets:
            logger.info(f"\nProcessing {dataset_name} dataset...")
            
            is_segmented = any('segment_length' in f for f in audio_files) if audio_files else False
            
            # Build output path
            if args.is_noise:
                output_subdir = os.path.join(
                    args.output_dir, processor.output_path, dataset_name, "noise"
                )
            elif is_segmented:
                segment_length = audio_files[0].get('segment_length', 'segmented') if audio_files else 'segmented'
                output_subdir = os.path.join(
                    args.output_dir, processor.output_path, dataset_name, segment_length
                )
            else:
                output_subdir = os.path.join(
                    args.output_dir, processor.output_path, dataset_name
                )
            
            logger.info(f"Output dir: {output_subdir}")
            
            stats = process_segmented_dataset(
                audio_files=audio_files,
                processor=processor,
                output_base_dir=output_subdir,
                model_name=model_name
            )
            
            logger.info(f"\n{dataset_name} processing complete:")
            logger.info(f"  Total: {stats['total']}")
            logger.info(f"  Success: {stats['success']}")
            logger.info(f"  Failed: {stats['failed']}")
            if stats['total'] > 0:
                logger.info(f"  Success rate: {stats['success']/stats['total']*100:.2f}%")
            logger.info(f"  Output dir: {output_subdir}")
        
        # Release GPU memory
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.info("=" * 80)
    logger.info("All processing complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()