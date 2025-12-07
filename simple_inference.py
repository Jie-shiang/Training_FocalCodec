#!/usr/bin/env python3
"""
FocalCodec Simple Inference Script
Quick audio processing for single model inference
"""

import os
import sys
import argparse
import torch
import torchaudio
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_path, device, model_cache_dir):
    """Load FocalCodec model"""
    try:
        os.environ['HF_HOME'] = model_cache_dir
        os.environ['HUGGINGFACE_HUB_CACHE'] = model_cache_dir
        
        logger.info(f"Loading model: {model_path}")
        
        codec = torch.hub.load(
            repo_or_dir="lucadellalib/focalcodec",
            model="focalcodec",
            config=model_path,
            force_reload=False,
        )
        
        codec = codec.to(device)
        codec.eval()
        codec.requires_grad_(False)
        
        logger.info(f"Model loaded: Input SR={codec.sample_rate_input}Hz, Output SR={codec.sample_rate_output}Hz")
        
        return codec
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise


@torch.no_grad()
def process_audio_file(codec, input_path, output_path, device):
    """Process single audio file through codec"""
    try:
        sig, sample_rate = torchaudio.load(input_path)
        sig = sig.to(device)
        
        # Resample if needed
        if sample_rate != codec.sample_rate_input:
            sig = torchaudio.functional.resample(
                sig, sample_rate, codec.sample_rate_input
            ).to(device)
        
        # Encode & decode
        toks = codec.sig_to_toks(sig)
        rec_sig = codec.toks_to_sig(toks)
        
        # Save
        rec_sig = rec_sig.cpu()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_path), rec_sig, codec.sample_rate_output)
        
        return True
        
    except Exception as e:
        logger.error(f"Processing failed {input_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="FocalCodec Simple Inference")
    
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--csv_file', type=str, required=True)
    parser.add_argument('--base_audio_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_cache_dir', type=str, 
                       default='/mnt/Internal/jieshiang/Model/FocalCodec')
    parser.add_argument('--gpu_id', type=int, default=0)
    
    args = parser.parse_args()
    
    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    codec = load_model(args.model_path, device, args.model_cache_dir)
    
    # Load CSV
    logger.info(f"Loading CSV: {args.csv_file}")
    df = pd.read_csv(args.csv_file)
    
    required_columns = ['file_name', 'file_path']
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Missing columns. Required: {required_columns}, Found: {df.columns.tolist()}")
        sys.exit(1)
    
    # Collect audio files
    audio_files = []
    for idx, row in df.iterrows():
        file_path = row['file_path']
        if file_path.startswith('./'):
            file_path = file_path[2:]
        
        full_path = os.path.join(args.base_audio_dir, file_path)
        
        if os.path.exists(full_path):
            audio_files.append({
                'file_name': row['file_name'],
                'input_path': full_path
            })
        else:
            logger.warning(f"File not found: {full_path}")
    
    if not audio_files:
        logger.error("No valid audio files found")
        sys.exit(0)
    
    logger.info(f"Found {len(audio_files)} valid files")
    
    # Process files
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    for file_info in tqdm(audio_files, desc="Processing"):
        output_path = output_dir / f"{file_info['file_name']}.wav"
        if process_audio_file(codec, file_info['input_path'], output_path, device):
            success_count += 1
    
    logger.info(f"Completed: {success_count}/{len(audio_files)} files ({success_count/len(audio_files)*100:.1f}%)")
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()