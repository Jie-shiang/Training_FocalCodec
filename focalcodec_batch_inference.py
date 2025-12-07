"""
python focalcodec_batch_inference.py \
    --base_audio_dir /mnt/Internal/jieshiang/Split_Result \
    --librispeech_csv /home/jieshiang/Desktop/GitHub/Codec_comparison/csv/split/librispeech_test_clean_filtered_0.5s.csv \
    --commonvoice_csv /home/jieshiang/Desktop/GitHub/Codec_comparison/csv/split/common_voice_zh_CN_train_filtered_0.5s.csv \
    --models 12.5hz 25hz 50hz 50hz_2k 50hz_4k 50hz_65k \
    --output_dir /mnt/Internal/jieshiang/Inference_Result \
    --model_cache_dir /mnt/Internal/jieshiang/Model/FocalCodec \
    --device cuda:1

python focalcodec_batch_inference.py \
    --base_audio_dir /mnt/Internal/jieshiang/Split_Result \
    --librispeech_csv /home/jieshiang/Desktop/GitHub/Codec_comparison/csv/split/librispeech_test_clean_filtered_1.0s.csv \
    --commonvoice_csv /home/jieshiang/Desktop/GitHub/Codec_comparison/csv/split/common_voice_zh_CN_train_filtered_1.0s.csv \
    --models 12.5hz 25hz 50hz 50hz_2k 50hz_4k 50hz_65k \
    --output_dir /mnt/Internal/jieshiang/Inference_Result \
    --model_cache_dir /mnt/Internal/jieshiang/Model/FocalCodec \
    --device cuda:1

python focalcodec_batch_inference.py \
    --base_audio_dir /mnt/Internal/jieshiang/Split_Result \
    --librispeech_csv /home/jieshiang/Desktop/GitHub/Codec_comparison/csv/split/librispeech_test_clean_filtered_2.0s.csv \
    --commonvoice_csv /home/jieshiang/Desktop/GitHub/Codec_comparison/csv/split/common_voice_zh_CN_train_filtered_2.0s.csv \
    --models 12.5hz 25hz 50hz 50hz_2k 50hz_4k 50hz_65k \
    --output_dir /mnt/Internal/jieshiang/Inference_Result \
    --model_cache_dir /mnt/Internal/jieshiang/Model/FocalCodec \
    --device cuda:1

python focalcodec_batch_inference.py \
    --base_audio_dir /mnt/Internal/jieshiang/Noise_Result \
    --librispeech_csv /home/jieshiang/Desktop/GitHub/Codec_comparison/csv/noise/librispeech_test_clean_noise.csv \
    --commonvoice_csv /home/jieshiang/Desktop/GitHub/Codec_comparison/csv/noise/common_voice_zh_CN_train_noise.csv \
    --models 12.5hz 25hz 50hz 50hz_2k 50hz_4k 50hz_65k \
    --output_dir /mnt/Internal/jieshiang/Inference_Result \
    --model_cache_dir /mnt/Internal/jieshiang/Model/FocalCodec \
    --device cuda:1 \
    --is_noise


python focalcodec_batch_inference.py \
    --base_audio_dir /mnt/Internal/ASR \
    --librispeech_csv /home/jieshiang/Desktop/GitHub/Codec_comparison/csv/librispeech_test_clean_filtered.csv \
    --commonvoice_csv /home/jieshiang/Desktop/GitHub/Codec_comparison/csv/common_voice_zh_CN_train_filtered.csv \
    --models 50hz_2k \
    --output_dir /mnt/Internal/jieshiang/Inference_Result \
    --model_cache_dir /mnt/Internal/jieshiang/Model/FocalCodec \
    --checkpoint /mnt/Internal/jieshiang/Model/FocalCodec/checkpoints_encoder_finetune/best_model.pt \
    --device cuda:1

python focalcodec_batch_inference.py \
    --base_audio_dir /mnt/Internal/ASR \
    --librispeech_csv /home/jieshiang/Desktop/GitHub/Codec_comparison/csv/librispeech_test_clean_filtered.csv \
    --commonvoice_csv /home/jieshiang/Desktop/GitHub/Codec_comparison/csv/common_voice_zh_CN_train_filtered.csv \
    --models 50hz_2k \
    --output_dir /mnt/Internal/jieshiang/Inference_Result/baseline \
    --model_cache_dir /mnt/Internal/jieshiang/Model/FocalCodec \
    --device cuda:1

python focalcodec_batch_inference.py \
    --base_audio_dir /mnt/Internal/ASR \
    --librispeech_csv /home/jieshiang/Desktop/GitHub/Codec_comparison/csv/librispeech_test_clean_filtered.csv \
    --commonvoice_csv /home/jieshiang/Desktop/GitHub/Codec_comparison/csv/common_voice_zh_CN_train_filtered.csv \
    --models 50hz_2k \
    --output_dir /mnt/Internal/jieshiang/Inference_Result/finetuned \
    --model_cache_dir /mnt/Internal/jieshiang/Model/FocalCodec \
    --checkpoint /mnt/Internal/jieshiang/Model/FocalCodec/checkpoints_encoder_finetune/best_model.pt \
    --device cuda:1
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
    
    def __init__(
        self,
        model_name: str,
        model_cache_dir: str,
        device: str = "cuda:0",
        force_reload: bool = False,
    ):
        
        Args:
        """
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        
        if model_name not in MODEL_CONFIGS:
        
        model_info = MODEL_CONFIGS[model_name]
        self.model_config = model_info["config"]
        self.output_path = model_info["output_path"]
        self.is_streaming = model_info["is_streaming"]
        
        if checkpoint_path:
            self.output_path = self.output_path + "_finetuned"
        
        self.device = torch.device(device)
        self.model_cache_dir = model_cache_dir
        
        if checkpoint_path:
            logger.info(f" Fine-tuned Checkpoint: {checkpoint_path}")
        
        os.environ['HF_HOME'] = self.model_cache_dir
        os.environ['HUGGINGFACE_HUB_CACHE'] = self.model_cache_dir
        
        self.codec = self._load_model(force_reload)
        
    def _load_model(self, force_reload: bool):
        try:
            codec = torch.hub.load(
                repo_or_dir="lucadellalib/focalcodec",
                model="focalcodec",
                config=self.model_config,
                force_reload=force_reload,
            )
            
            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                
                codec.load_state_dict(checkpoint['model_state_dict'])
                
                epoch = checkpoint.get('epoch', 'N/A')
                val_loss = checkpoint.get('val_loss', 'N/A')
                train_loss = checkpoint.get('train_loss', 'N/A')
                
                logger.info(f"   Epoch: {epoch}")
                logger.info(f"   Train Loss: {train_loss:.6f}" if isinstance(train_loss, float) else f"   Train Loss: {train_loss}")
                logger.info(f"   Val Loss: {val_loss:.6f}" if isinstance(val_loss, float) else f"   Val Loss: {val_loss}")
            elif self.checkpoint_path:
            
            codec = codec.to(self.device)
            codec.eval()
            codec.requires_grad_(False)
            
            
            return codec
        except Exception as e:
            raise
    
    @torch.no_grad()
    def inference_single_file(
        self,
        audio_path: str,
        output_path: str,
        original_sample_rate: int = None
    ) -> bool:
        """
        
        Args:
        
        Returns:
        try:
            sig, sample_rate = torchaudio.load(audio_path)
            
            sig = sig.to(self.device)
            
            if sample_rate != self.codec.sample_rate_input:
                sig = torchaudio.functional.resample(
                    sig, sample_rate, self.codec.sample_rate_input
                ).to(self.device)
            
            toks = self.codec.sig_to_toks(sig)
            
            rec_sig = self.codec.toks_to_sig(toks)
            
            rec_sig = rec_sig.cpu()
            
            target_sr = original_sample_rate if original_sample_rate else sample_rate
            if self.codec.sample_rate_output != target_sr:
                rec_sig = torchaudio.functional.resample(
                    rec_sig, self.codec.sample_rate_output, target_sr
                )
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            torchaudio.save(output_path, rec_sig, target_sr)
            
            return True
            
        except Exception as e:
            return False


def extract_segment_length_from_csv(csv_path: str) -> Optional[str]:
    """
    filename = Path(csv_path).stem
    match = re.search(r'_(\d+\.?\d*s)$', filename)
    
    if match:
        segment_length = match.group(1)
        return segment_length
    else:
        return None


def load_segment_csv_data(csv_path: str, base_audio_dir: str, dataset_type: str) -> List[Dict]:
    df = pd.read_csv(csv_path)
    
    required_segmented_columns = ['segment_file_name', 'segment_file_path', 'original_file_name']
    is_segmented = all(col in df.columns for col in required_segmented_columns)
    
    if is_segmented:
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
        
    else:
        required_complete_columns = ['file_name', 'file_path']
        if not all(col in df.columns for col in required_complete_columns):
            raise ValueError(
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
        
    
    return audio_files


def process_segmented_dataset(
    audio_files: List[Dict],
    processor: FocalCodecInferenceProcessor,
    output_base_dir: str,
    model_name: str
) -> Dict[str, int]:
    stats = {'success': 0, 'failed': 0, 'total': len(audio_files)}
    
    if not audio_files:
        return stats
    
    is_segmented = 'segment_file_name' in audio_files[0]
    
    
    os.makedirs(output_base_dir, exist_ok=True)
    
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
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""

    python focalcodec_batch_inference_finetuned.py \\
        --base_audio_dir /mnt/Internal/ASR \\
        --commonvoice_csv /path/to/chinese.csv \\
        --models 50hz_2k \\
        --output_dir /mnt/Internal/jieshiang/Inference_Result \\
        --model_cache_dir /mnt/Internal/jieshiang/Model/FocalCodec \\
        --checkpoint /mnt/Internal/jieshiang/Model/FocalCodec/checkpoints_encoder_finetune/best_model.pt \\
        --device cuda:1

    python focalcodec_batch_inference_finetuned.py \\
        --base_audio_dir /mnt/Internal/ASR \\
        --commonvoice_csv /path/to/chinese.csv \\
        --models 50hz_2k \\
        --output_dir /mnt/Internal/jieshiang/Inference_Result \\
        --model_cache_dir /mnt/Internal/jieshiang/Model/FocalCodec \\
        --device cuda:1
    )
    
    
    
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        choices=list(MODEL_CONFIGS.keys()),
    
    parser.add_argument('--checkpoint', type=str, default=None,
    
    
    
    args = parser.parse_args()
    
    if not args.librispeech_csv and not args.commonvoice_csv:
    
    if 'cuda' in args.device and not torch.cuda.is_available():
        args.device = 'cpu'
    
    if args.checkpoint:
        logger.info("=" * 80)
        logger.info("=" * 80)
        logger.info(f"Checkpoint: {args.checkpoint}")
        logger.info("=" * 80)
    
    logger.info("=" * 80)
    logger.info("=" * 80)
    if args.checkpoint:
        logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info("=" * 80)
    
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
    
    for model_name in args.models:
        logger.info("=" * 80)
        logger.info("=" * 80)
        
        processor = FocalCodecInferenceProcessor(
            model_name=model_name,
            model_cache_dir=args.model_cache_dir,
            device=args.device,
            force_reload=args.force_reload,
        )
        
        for dataset_name, audio_files in all_datasets:
            
            is_segmented = any('segment_length' in f for f in audio_files) if audio_files else False
            
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
            
            
            stats = process_segmented_dataset(
                audio_files=audio_files,
                processor=processor,
                output_base_dir=output_subdir,
                model_name=model_name
            )
            
            if stats['total'] > 0:
        
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.info("=" * 80)
    logger.info("=" * 80)


if __name__ == '__main__':
    main()