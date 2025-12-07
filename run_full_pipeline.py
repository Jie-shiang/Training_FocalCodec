#!/usr/bin/env python3
"""FocalCodec Full Training & Evaluation Pipeline"""

import os
import sys
import argparse
import subprocess
import pandas as pd
import json
from datetime import datetime
from pathlib import Path


def run_command(cmd, description, conda_env=None):
    """Execute shell command"""
    print(f"\n{'='*80}\n {description}\n{'='*80}")
    if conda_env:
        cmd = f"conda run -n {conda_env} {cmd}"
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\nError: {description} failed")
        sys.exit(1)
    print(f"\n{description} completed")


def generate_experiment_id(args):
    """Generate experiment ID"""
    components = []
    if args.train_encoder: components.append("enc")
    if args.train_compressor: components.append("comp")
    if args.train_decompressor: components.append("decomp")
    if args.train_decoder: components.append("dec")
    
    losses = []
    if args.use_feature_loss: losses.append("feat")
    if args.use_time_loss: losses.append("time")
    if args.use_mel_loss: losses.append("mel")
    
    comp_str = "+".join(components) if components else "none"
    loss_str = "+".join(losses) if losses else "none"
    lora_str = f"_lora{args.lora_rank}" if args.use_lora else ""
    
    return f"bs{args.batch_size}_lr{args.learning_rate:.0e}_ep{args.num_epochs}_{comp_str}_{loss_str}{lora_str}"


def main():
    parser = argparse.ArgumentParser(description='FocalCodec Full Pipeline')
    
    # Data params
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--chunk_duration', type=float, default=3.0)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--augment', action='store_true')
    
    # Training params
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    
    # Component flags
    parser.add_argument('--train_encoder', action='store_true')
    parser.add_argument('--train_compressor', action='store_true')
    parser.add_argument('--train_decompressor', action='store_true')
    parser.add_argument('--train_decoder', action='store_true')
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--freeze_compressor', action='store_true')
    parser.add_argument('--freeze_decompressor', action='store_true')
    parser.add_argument('--freeze_decoder', action='store_true')
    
    # LoRA params
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_encoder', action='store_true')
    parser.add_argument('--lora_compressor', action='store_true')
    parser.add_argument('--lora_decompressor', action='store_true')
    parser.add_argument('--lora_decoder', action='store_true')
    
    # Loss params
    parser.add_argument('--use_feature_loss', action='store_true')
    parser.add_argument('--use_time_loss', action='store_true')
    parser.add_argument('--use_mel_loss', action='store_true')
    parser.add_argument('--weight_feature', type=float, default=1.0)
    parser.add_argument('--weight_time', type=float, default=0.3)
    parser.add_argument('--weight_mel', type=float, default=0.3)
    
    # Environment
    parser.add_argument('--train_env', type=str, default='focalcodec')
    parser.add_argument('--eval_env', type=str, default='codec_eval')
    parser.add_argument('--gpu_id', type=int, default=0)
    
    # Paths
    parser.add_argument('--output_dir', type=str, default='experiments/pipeline_results')
    parser.add_argument('--codec_comparison_dir', type=str, 
                       default='/home/jieshiang/Desktop/GitHub/Codec_comparison')
    
    # Control
    parser.add_argument('--skip_training', action='store_true')
    parser.add_argument('--model_checkpoint', type=str)
    
    # Cleanup
    parser.add_argument('--cleanup_inference', action='store_true')
    parser.add_argument('--cleanup_codec_comparison', action='store_true')
    parser.add_argument('--keep_best_model_only', action='store_true')
    parser.add_argument('--cleanup_best_model', action='store_true')
    
    # Early stopping
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min_delta', type=float, default=1e-4)
    
    args = parser.parse_args()
    
    # Setup directories
    exp_id = generate_experiment_id(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{exp_id}_{timestamp}"
    
    checkpoint_dir = Path(f"/mnt/Internal/jieshiang/Model/FocalCodec/checkpoints_{exp_name}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    inference_dir = Path(f"/mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50Hz_2K_finetuned/{exp_name}")
    inference_dir.mkdir(parents=True, exist_ok=True)
    
    experiment_dir = Path("experiments") / exp_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}\nFocalCodec Pipeline\n{'='*80}")
    print(f"Experiment: {exp_name}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Inference: {inference_dir}")
    print(f"Results: {experiment_dir}")
    print(f"{'='*80}\n")
    
    # Save config
    config = vars(args)
    config.update({
        'experiment_id': exp_name,
        'timestamp': timestamp,
        'checkpoint_dir': str(checkpoint_dir),
        'inference_dir': str(inference_dir),
        'experiment_dir': str(experiment_dir)
    })
    
    with open(experiment_dir / f"config_{exp_name}.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Step 1: Training
    if not args.skip_training:
        train_cmd = [
            "python /home/jieshiang/Desktop/GitHub/FocalCodec/train_focalcodec.py",
            f"--batch_size {args.batch_size}",
            f"--num_workers {args.num_workers}",
            f"--chunk_duration {args.chunk_duration}",
            f"--overlap {args.overlap}",
            f"--num_epochs {args.num_epochs}",
            f"--learning_rate {args.learning_rate}",
            f"--weight_decay {args.weight_decay}",
            f"--gradient_clip {args.gradient_clip}",
            f"--gpu_id {args.gpu_id}",
            f"--checkpoint_dir {checkpoint_dir}",
        ]
        
        if args.augment: train_cmd.append("--augment")
        if args.train_encoder: train_cmd.append("--train_encoder")
        if args.train_compressor: train_cmd.append("--train_compressor")
        if args.train_decompressor: train_cmd.append("--train_decompressor")
        if args.train_decoder: train_cmd.append("--train_decoder")
        if args.freeze_encoder: train_cmd.append("--freeze_encoder")
        if args.freeze_compressor: train_cmd.append("--freeze_compressor")
        if args.freeze_decompressor: train_cmd.append("--freeze_decompressor")
        if args.freeze_decoder: train_cmd.append("--freeze_decoder")
        
        if args.use_lora:
            train_cmd.extend([
                "--use_lora",
                f"--lora_rank {args.lora_rank}",
                f"--lora_alpha {args.lora_alpha}"
            ])
            if args.lora_encoder: train_cmd.append("--lora_encoder")
            if args.lora_compressor: train_cmd.append("--lora_compressor")
            if args.lora_decompressor: train_cmd.append("--lora_decompressor")
            if args.lora_decoder: train_cmd.append("--lora_decoder")
        
        if args.use_feature_loss:
            train_cmd.extend(["--use_feature_loss", f"--weight_feature {args.weight_feature}"])
        if args.use_time_loss:
            train_cmd.extend(["--use_time_loss", f"--weight_time {args.weight_time}"])
        if args.use_mel_loss:
            train_cmd.extend(["--use_mel_loss", f"--weight_mel {args.weight_mel}"])
        
        if args.early_stopping:
            train_cmd.extend([
                "--early_stopping",
                f"--patience {args.patience}",
                f"--min_delta {args.min_delta}"
            ])
        
        run_command(" ".join(train_cmd), "Training", args.train_env)
        model_checkpoint = checkpoint_dir / "best_model.pt"
    else:
        if not args.model_checkpoint:
            print("Error: --model_checkpoint required with --skip_training")
            sys.exit(1)
        model_checkpoint = Path(args.model_checkpoint)
        if not model_checkpoint.exists():
            print(f"Error: Checkpoint not found: {model_checkpoint}")
            sys.exit(1)
    
    # Step 2: Inference CommonVoice
    cv_dir = inference_dir / "commonvoice"
    cv_dir.mkdir(exist_ok=True)
    
    inf_cmd_cv = (
        f"python /home/jieshiang/Desktop/GitHub/FocalCodec/focalcodec_batch_inference.py "
        f"--base_audio_dir /mnt/Internal/ASR "
        f"--commonvoice_csv /home/jieshiang/Desktop/GitHub/Codec_comparison/csv/common_voice_zh_CN_train_filtered.csv "
        f"--models 50hz_2k "
        f"--output_dir {cv_dir} "
        f"--model_cache_dir /mnt/Internal/jieshiang/Model/FocalCodec "
        f"--checkpoint {model_checkpoint} "
        f"--device cuda:{args.gpu_id}"
    )
    run_command(inf_cmd_cv, "Inference CommonVoice", args.train_env)
    
    # Step 3: Inference LibriSpeech
    ls_dir = inference_dir / "librispeech"
    ls_dir.mkdir(exist_ok=True)
    
    inf_cmd_ls = (
        f"python /home/jieshiang/Desktop/GitHub/FocalCodec/focalcodec_batch_inference.py "
        f"--base_audio_dir /mnt/Internal/ASR "
        f"--librispeech_csv /home/jieshiang/Desktop/GitHub/Codec_comparison/csv/librispeech_test_clean_filtered.csv "
        f"--models 50hz_2k "
        f"--output_dir {ls_dir} "
        f"--model_cache_dir /mnt/Internal/jieshiang/Model/FocalCodec "
        f"--checkpoint {model_checkpoint} "
        f"--device cuda:{args.gpu_id}"
    )
    run_command(inf_cmd_ls, "Inference LibriSpeech", args.train_env)
    
    # Step 4: Evaluation config
    codec_dir = Path(args.codec_comparison_dir)
    config_dir = codec_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    eval_config = {
        "FocalCodec-S": {
            "50Hz_2k_finetuned": {
                "path": str(inference_dir),
                "dataset": {
                    "LibriSpeech": {"clean": str(ls_dir)},
                    "CommonVoice": {"clean": str(cv_dir)}
                }
            }
        }
    }
    
    with open(config_dir / "FocalCodec-S_50Hz_2k_finetuned_config.json", 'w') as f:
        json.dump(eval_config, f, indent=2)
    
    # Step 5: Evaluation
    eval_cmd = (
        f"cd {codec_dir} && "
        f"python enhanced_evaluation_pipeline.py "
        f"--mode complete --csv_dir ./csv --audio_dir ./audio "
        f"--result_dir ./result --config_dir ./configs "
        f"--model FocalCodec-S --version 50Hz_2k_finetuned --dataset both"
    )
    run_command(eval_cmd, "Evaluation", args.eval_env)
    
    # Step 6: Collect results
    result_dir = codec_dir / "result"
    
    try:
        exp_result = {
            'experiment_id': exp_name,
            'timestamp': timestamp,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'num_epochs': args.num_epochs,
        }
        
        cv_summary = result_dir / "summary_results_FocalCodec-S_50Hz_2k_finetuned_clean_commonvoice.csv"
        ls_summary = result_dir / "summary_results_FocalCodec-S_50Hz_2k_finetuned_clean_librispeech.csv"
        
        if cv_summary.exists():
            df = pd.read_csv(cv_summary)
            exp_result.update({
                'cv_cer': df['WER/CER'].values[0],
                'cv_pesq': df['PESQ'].values[0],
                'cv_stoi': df['STOI'].values[0],
                'cv_speaker_sim': df['Speaker_Sim'].values[0],
                'cv_dcer': df['dCER'].values[0]
            })
        
        if ls_summary.exists():
            df = pd.read_csv(ls_summary)
            exp_result.update({
                'ls_wer': df['WER/CER'].values[0],
                'ls_pesq': df['PESQ'].values[0],
                'ls_stoi': df['STOI'].values[0],
                'ls_speaker_sim': df['Speaker_Sim'].values[0]
            })
        
        pd.DataFrame([exp_result]).to_csv(experiment_dir / "experiment_results.csv", index=False)
        
        print(f"\n{'='*80}\nResults\n{'='*80}")
        for k, v in exp_result.items():
            print(f"  {k}: {v}")
        
    except Exception as e:
        print(f"Warning: Could not process results: {e}")
    
    # Step 7: Copy detailed results
    if result_dir.exists():
        import shutil
        files = [
            ("detailed_results_FocalCodec-S_50Hz_2k_finetuned_clean_commonvoice.csv", "cv_detailed.csv"),
            ("summary_results_FocalCodec-S_50Hz_2k_finetuned_clean_commonvoice.csv", "cv_summary.csv"),
            ("detailed_results_FocalCodec-S_50Hz_2k_finetuned_clean_librispeech.csv", "ls_detailed.csv"),
            ("summary_results_FocalCodec-S_50Hz_2k_finetuned_clean_librispeech.csv", "ls_summary.csv"),
        ]
        for src, dst in files:
            if (result_dir / src).exists():
                shutil.copy2(result_dir / src, experiment_dir / dst)
    
    # Step 8: Cleanup
    if args.cleanup_inference:
        import shutil
        for d in [cv_dir, ls_dir]:
            if d.exists():
                shutil.rmtree(d)
    
    if args.cleanup_codec_comparison:
        import shutil
        for path in [
            codec_dir / "audio" / "CommonVoice" / "FocalCodec-S" / "50Hz_2k_finetuned",
            codec_dir / "audio" / "LibriSpeech" / "FocalCodec-S" / "50Hz_2k_finetuned"
        ]:
            if path.exists():
                shutil.rmtree(path)
    
    if args.keep_best_model_only and not args.skip_training:
        for f in checkpoint_dir.glob("epoch_*.pt"):
            f.unlink()
    
    if args.cleanup_best_model and not args.skip_training:
        import shutil
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
    
    print(f"\n{'='*80}\nPipeline Complete!\n{'='*80}")
    print(f"Results: {experiment_dir}\n")


if __name__ == '__main__':
    main()