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


def run_command(cmd, description, conda_env=None, log_file=None):
    """Execute shell command with optional logging"""
    print(f"\n{'='*80}\n {description}\n{'='*80}")
    if conda_env:
        # Use bash -c to properly handle cd && python combined commands
        cmd = f"conda run -n {conda_env} bash -c '{cmd}'"
    print(f"Command: {cmd}\n")

    # Redirect output to log file if specified
    if log_file:
        cmd = f"{cmd} 2>&1 | tee -a {log_file}"

    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\nError: {description} failed")
        if log_file:
            print(f"Check log file: {log_file}")
        sys.exit(1)
    print(f"\n{description} completed")
    if log_file:
        print(f"Log saved to: {log_file}")


def generate_experiment_id(args):
    """Generate experiment ID based on training components and loss configuration"""
    components = []
    if args.train_encoder: components.append("enc")
    if args.train_compressor: components.append("comp")
    if args.train_decompressor: components.append("decomp")
    if args.train_decoder: components.append("dec")
    
    losses = []
    if args.use_feature_loss: losses.append("feat")
    if args.use_time_loss: losses.append("time")
    if args.use_mel_loss: losses.append("mel")
    # Include ASR loss in experiment ID
    if args.use_asr_loss: losses.append("asr")
    
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
    
    # ADDED: ASR Loss arguments for pipeline script (REQUIRED TO FIX YOUR ERROR)
    parser.add_argument('--use_asr_loss', action='store_true', help='Enable Whisper Cross Entropy ASR loss')
    
    parser.add_argument('--weight_feature', type=float, default=1.0)
    parser.add_argument('--weight_time', type=float, default=0.3)
    parser.add_argument('--weight_mel', type=float, default=0.3)
    
    # ADDED: ASR Loss weight
    parser.add_argument('--weight_asr', type=float, default=0.5, help='Weight for ASR loss')
    
    # ADDED: ASR Model parameters (Passed to the training script)
    parser.add_argument('--asr_cache_dir', type=str, default='/mnt/Internal/jieshiang/Model/ASR',
                       help='Cache directory for ASR models')
    
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

    # Create logs directory
    logs_dir = experiment_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    print(f"\n{'='*80}\nFocalCodec Pipeline\n{'='*80}")
    print(f"Experiment: {exp_name}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Inference: {inference_dir}")
    print(f"Results: {experiment_dir}")
    print(f"Logs: {logs_dir}")
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
        # NOTE: Using the dedicated script for ASR Loss training (train_focalcodec.py)
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
            
        # ADDED: ASR Loss configuration (Passed to the training script)
        if args.use_asr_loss:
            train_cmd.extend([
                "--use_asr_loss", 
                f"--weight_asr {args.weight_asr}",
                f"--asr_cache_dir {args.asr_cache_dir}"
            ])
        
        if args.early_stopping:
            train_cmd.extend([
                "--early_stopping",
                f"--patience {args.patience}",
                f"--min_delta {args.min_delta}"
            ])

        run_command(" ".join(train_cmd), "Training", args.train_env, log_file=logs_dir / "training.log")
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
    run_command(inf_cmd_cv, "Inference CommonVoice", args.train_env, log_file=logs_dir / "inference_commonvoice.log")

    # Inference script creates nested directory structure: cv_dir/FocalCodec-S/50HZ/2K_finetuned/commonvoice/
    # Find actual audio file directory
    cv_audio_dir = cv_dir / "FocalCodec-S" / "50HZ" / "2K_finetuned" / "commonvoice"
    
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
    run_command(inf_cmd_ls, "Inference LibriSpeech", args.train_env, log_file=logs_dir / "inference_librispeech.log")

    # Inference script creates nested directory structure: ls_dir/FocalCodec-S/50HZ/2K_finetuned/librispeech/
    # Find actual audio file directory
    ls_audio_dir = ls_dir / "FocalCodec-S" / "50HZ" / "2K_finetuned" / "librispeech"
    
    # Step 4: Evaluation config
    codec_dir = Path(args.codec_comparison_dir)
    config_dir = codec_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    eval_config = {
        "FocalCodec-S": {
            "50Hz_2k_finetuned": {
                "path": str(inference_dir),
                "dataset": {
                    "LibriSpeech": {"clean": str(ls_audio_dir)},
                    "CommonVoice": {"clean": str(cv_audio_dir)}
                }
            }
        }
    }

    config_path = config_dir / "FocalCodec-S_50Hz_2k_finetuned_config.json"
    with open(config_path, 'w') as f:
        json.dump(eval_config, f, indent=2)

    # Copy config to experiment directory
    import shutil
    shutil.copy2(config_path, experiment_dir / "eval_config.json")
    
    # Step 5: Evaluation - LibriSpeech
    eval_cmd_ls = (
        f"cd {codec_dir} && "
        f"python fast_evaluation_pipeline.py "
        f"--inference_dir {ls_audio_dir} "
        f"--csv_file librispeech_test_clean_filtered.csv "
        f"--original_dir /mnt/Internal/ASR "
        f"--model_name FocalCodec-S "
        f"--frequency 50Hz_2k_finetuned "
        f"--causality Causal "
        f"--bit_rate 0.55 "
        f"--quantizers 1 "
        f"--codebook_size 2048 "
        f'--n_params \\"N/A\\" '
        f'--training_set \\"N/A\\" '
        f'--testing_set \\"N/A\\" '
        f"--metrics dwer utmos pesq stoi speaker_similarity "
        f"--dataset_type clean "
        f"--use_gpu "
        f"--gpu_id {args.gpu_id} "
        f"--num_workers {args.num_workers} "
        f"--asr_batch_size 64"
    )
    run_command(eval_cmd_ls, "Evaluation - LibriSpeech", args.eval_env, log_file=logs_dir / "eval_librispeech.log")

    # Step 5b: Evaluation - Common Voice
    eval_cmd_cv = (
        f"cd {codec_dir} && "
        f"python fast_evaluation_pipeline.py "
        f"--inference_dir {cv_audio_dir} "
        f"--csv_file common_voice_zh_CN_train_filtered.csv "
        f"--original_dir /mnt/Internal/ASR "
        f"--model_name FocalCodec-S "
        f"--frequency 50Hz_2k_finetuned "
        f"--causality Causal "
        f"--bit_rate 0.55 "
        f"--quantizers 1 "
        f"--codebook_size 2048 "
        f'--n_params \\"N/A\\" '
        f'--training_set \\"N/A\\" '
        f'--testing_set \\"N/A\\" '
        f"--metrics dcer utmos pesq stoi speaker_similarity "
        f"--dataset_type clean "
        f"--use_gpu "
        f"--gpu_id {args.gpu_id} "
        f"--num_workers {args.num_workers} "
        f"--asr_batch_size 64"
    )
    run_command(eval_cmd_cv, "Evaluation - CommonVoice", args.eval_env, log_file=logs_dir / "eval_commonvoice.log")
    
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

        # Output format from fast_evaluation_pipeline.py
        cv_summary = result_dir / "summary_results_FocalCodec-S_50Hz_2k_finetuned_clean_commonvoice.csv"
        ls_summary = result_dir / "summary_results_FocalCodec-S_50Hz_2k_finetuned_clean_librispeech.csv"

        if cv_summary.exists():
            df = pd.read_csv(cv_summary)
            exp_result.update({
                'cv_cer': df['DCER_Mean'].values[0] if 'DCER_Mean' in df.columns else None,
                'cv_pesq': df['PESQ_Mean'].values[0] if 'PESQ_Mean' in df.columns else None,
                'cv_stoi': df['STOI_Mean'].values[0] if 'STOI_Mean' in df.columns else None,
                'cv_speaker_sim': df['Speaker_Sim_Mean'].values[0] if 'Speaker_Sim_Mean' in df.columns else None,
                'cv_utmos': df['UTMOS_Mean'].values[0] if 'UTMOS_Mean' in df.columns else None,
            })
        else:
            print(f"Warning: No CommonVoice summary results found: {cv_summary}")

        if ls_summary.exists():
            df = pd.read_csv(ls_summary)
            exp_result.update({
                'ls_wer': df['DWER_Mean'].values[0] if 'DWER_Mean' in df.columns else None,
                'ls_pesq': df['PESQ_Mean'].values[0] if 'PESQ_Mean' in df.columns else None,
                'ls_stoi': df['STOI_Mean'].values[0] if 'STOI_Mean' in df.columns else None,
                'ls_speaker_sim': df['Speaker_Sim_Mean'].values[0] if 'Speaker_Sim_Mean' in df.columns else None,
                'ls_utmos': df['UTMOS_Mean'].values[0] if 'UTMOS_Mean' in df.columns else None,
            })
        else:
            print(f"Warning: No LibriSpeech summary results found: {ls_summary}")

        pd.DataFrame([exp_result]).to_csv(experiment_dir / "experiment_results.csv", index=False)

        print(f"\n{'='*80}\nResults\n{'='*80}")
        for k, v in exp_result.items():
            print(f"  {k}: {v}")

    except Exception as e:
        print(f"Warning: Could not process results: {e}")
        import traceback
        traceback.print_exc()
    
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

    # Create summary file
    summary_path = experiment_dir / "EXPERIMENT_SUMMARY.txt"
    with open(summary_path, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"FocalCodec Experiment Summary\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Experiment ID: {exp_name}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"Directories:\n")
        f.write(f"  - Experiment: {experiment_dir}\n")
        f.write(f"  - Checkpoint: {checkpoint_dir}\n")
        f.write(f"  - Inference: {inference_dir}\n")
        f.write(f"  - Logs: {logs_dir}\n\n")
        f.write(f"Audio Files:\n")
        f.write(f"  - CommonVoice: {cv_audio_dir}\n")
        f.write(f"  - LibriSpeech: {ls_audio_dir}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  - Batch Size: {args.batch_size}\n")
        f.write(f"  - Learning Rate: {args.learning_rate}\n")
        f.write(f"  - Epochs: {args.num_epochs}\n")
        f.write(f"  - GPU ID: {args.gpu_id}\n\n")
        f.write(f"Results Files:\n")
        f.write(f"  - Config: config_{exp_name}.json\n")
        f.write(f"  - Results: experiment_results.csv\n")
        f.write(f"  - CommonVoice Details: cv_detailed.csv\n")
        f.write(f"  - CommonVoice Summary: cv_summary.csv\n")
        f.write(f"  - LibriSpeech Details: ls_detailed.csv\n")
        f.write(f"  - LibriSpeech Summary: ls_summary.csv\n\n")
        f.write(f"Log Files:\n")
        if not args.skip_training:
            f.write(f"  - Training: logs/training.log\n")
        f.write(f"  - Inference CommonVoice: logs/inference_commonvoice.log\n")
        f.write(f"  - Inference LibriSpeech: logs/inference_librispeech.log\n")
        f.write(f"  - Eval CommonVoice: logs/eval_commonvoice.log\n")
        f.write(f"  - Eval LibriSpeech: logs/eval_librispeech.log\n\n")
        f.write(f"{'='*80}\n")

    print(f"\n{'='*80}\nPipeline Complete!\n{'='*80}")
    print(f"Results: {experiment_dir}")
    print(f"Summary: {summary_path}\n")


if __name__ == '__main__':
    main()