#!/usr/bin/env python3

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
                       default='/mnt/Internal/jieshiang/Model/FocalCodec',
                       help='Model cache directory')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    codec = load_model(args.model_path, device, args.model_cache_dir)

    logger.info(f"Loading CSV: {args.csv_file}")
    df = pd.read_csv(args.csv_file)

    required_columns = ['file_name', 'file_path']
    if not all(col in df.columns for col in required_columns):
        sys.exit(1)

    audio_files = []
    for idx, row in df.iterrows():
        file_name = row['file_name']
        file_path = row['file_path']

        if file_path.startswith('./'):
            file_path = file_path[2:]

        full_path = os.path.join(args.base_audio_dir, file_path)

        if os.path.exists(full_path):
            audio_files.append({
                'file_name': file_name,
                'input_path': full_path,
                'relative_path': file_path
            })
        else:


    if len(audio_files) == 0:
        sys.exit(0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for file_info in tqdm(audio_files, desc="Processing audio"):
        input_path = file_info['input_path']
        file_name = file_info['file_name']

        output_path = output_dir / f"{file_name}.wav"

        if process_audio_file(codec, input_path, output_path, device):
            success_count += 1

    logger.info(f"✅ Processed {success_count}/{len(audio_files)} files successfully")
    logger.info(f"Output saved to: {output_dir}")

if __name__ == "__main__":
    main()
