#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import pandas as pd
import json
from datetime import datetime
from pathlib import Path


def run_command(cmd: str, description: str, conda_env: str = None):
    print("\n" + "="*80)
    print(f" {description}")
    if conda_env:
        print(f" Environment: {conda_env}")
    print("="*80)

    if conda_env:
        full_cmd = f"conda run -n {conda_env} {cmd}"
        print(f"Command: {full_cmd}\n")
    else:
        full_cmd = cmd
        print(f"Command: {cmd}\n")

    result = subprocess.run(full_cmd, shell=True)

    if result.returncode != 0:
        print(f"\n Error: {description} failed with code {result.returncode}")
        sys.exit(1)

    print(f"\n {description} completed successfully")


def generate_experiment_id(args):
    components = []
    if args.train_encoder:
        components.append("enc")
    if args.train_compressor:
        components.append("comp")
    if args.train_decompressor:
        components.append("decomp")
    if args.train_decoder:
        components.append("dec")

    losses = []
    if args.use_feature_loss:
        losses.append("feat")
    if args.use_time_loss:
        losses.append("time")
    if args.use_mel_loss:
        losses.append("mel")

    comp_str = "+".join(components) if components else "none"
    loss_str = "+".join(losses) if losses else "none"

    lora_str = f"_lora{args.lora_rank}" if args.use_lora else ""

    exp_id = f"bs{args.batch_size}_lr{args.learning_rate:.0e}_ep{args.num_epochs}_{comp_str}_{loss_str}{lora_str}"

    return exp_id


def main():
    parser = argparse.ArgumentParser(description='FocalCodec Full Pipeline (Cross-Environment)')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--chunk_duration', type=float, default=3.0)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--augment', action='store_true')

    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--gradient_clip', type=float, default=1.0)

    parser.add_argument('--train_encoder', action='store_true')
    parser.add_argument('--train_compressor', action='store_true')
    parser.add_argument('--train_decompressor', action='store_true')
    parser.add_argument('--train_decoder', action='store_true')

    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--freeze_compressor', action='store_true')
    parser.add_argument('--freeze_decompressor', action='store_true')
    parser.add_argument('--freeze_decoder', action='store_true')

    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_encoder', action='store_true')
    parser.add_argument('--lora_compressor', action='store_true')
    parser.add_argument('--lora_decompressor', action='store_true')
    parser.add_argument('--lora_decoder', action='store_true')

    parser.add_argument('--use_feature_loss', action='store_true')
    parser.add_argument('--use_time_loss', action='store_true')
    parser.add_argument('--use_mel_loss', action='store_true')
    parser.add_argument('--weight_feature', type=float, default=1.0)
    parser.add_argument('--weight_time', type=float, default=0.3)
    parser.add_argument('--weight_mel', type=float, default=0.3)

    parser.add_argument('--train_env', type=str, default='focalcodec', help='Conda env for training')
    parser.add_argument('--eval_env', type=str, default='codec_eval', help='Conda env for evaluation')

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='experiments/pipeline_results')
    parser.add_argument('--skip_training', action='store_true', help='Skip training, only run inference and eval')
    parser.add_argument('--model_checkpoint', type=str, help='Path to trained model checkpoint (for skip_training)')

    parser.add_argument('--codec_comparison_dir', type=str,
                       default='/home/jieshiang/Desktop/GitHub/Codec_comparison',
                       help='Path to Codec_comparison directory')

    parser.add_argument('--cleanup_inference', action='store_true',
                       help='Delete inference files after evaluation')
    parser.add_argument('--cleanup_codec_comparison', action='store_true',
                       help='Delete all generated files in Codec_comparison (audio, config, results)')
    parser.add_argument('--keep_best_model_only', action='store_true',
                       help='Delete all checkpoints except best_model.pt')
    parser.add_argument('--cleanup_best_model', action='store_true',
                       help='Delete best_model.pt and entire checkpoint directory after experiment')

    parser.add_argument('--early_stopping', action='store_true',
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience (epochs)')
    parser.add_argument('--min_delta', type=float, default=1e-4,
                       help='Minimum change to qualify as improvement')

    args = parser.parse_args()

    exp_id = generate_experiment_id(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{exp_id}_{timestamp}"

    checkpoint_dir = Path(f"/mnt/Internal/jieshiang/Model/FocalCodec/checkpoints_{exp_name}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    inference_base = Path("/mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50Hz_2K_finetuned")
    inference_dir = inference_base / exp_name
    inference_dir.mkdir(parents=True, exist_ok=True)

    experiment_dir = Path("experiments") / exp_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print(f"🎯 FocalCodec Full Pipeline (Cross-Environment)")
    print("="*80)
    print(f"Experiment ID: {exp_name}")
    print(f"Checkpoint Directory: {checkpoint_dir}")
    print(f"Inference Directory: {inference_dir}")
    print(f"Experiment Directory: {experiment_dir}")
    print(f"Training Environment: {args.train_env}")
    print(f"Evaluation Environment: {args.eval_env}")
    print("="*80)

    config = vars(args)
    config['experiment_id'] = exp_name
    config['timestamp'] = timestamp
    config['checkpoint_dir'] = str(checkpoint_dir)
    config['inference_dir'] = str(inference_dir)
    config['experiment_dir'] = str(experiment_dir)

    config_file = experiment_dir / f"config_{exp_name}.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n Config saved to: {config_file}")

    if not args.skip_training:
        train_cmd_parts = [
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

        if args.augment:
            train_cmd_parts.append("--augment")

        if args.train_encoder:
            train_cmd_parts.append("--train_encoder")
        if args.train_compressor:
            train_cmd_parts.append("--train_compressor")
        if args.train_decompressor:
            train_cmd_parts.append("--train_decompressor")
        if args.train_decoder:
            train_cmd_parts.append("--train_decoder")

        if args.freeze_encoder:
            train_cmd_parts.append("--freeze_encoder")
        if args.freeze_compressor:
            train_cmd_parts.append("--freeze_compressor")
        if args.freeze_decompressor:
            train_cmd_parts.append("--freeze_decompressor")
        if args.freeze_decoder:
            train_cmd_parts.append("--freeze_decoder")

        if args.use_lora:
            train_cmd_parts.append("--use_lora")
            train_cmd_parts.append(f"--lora_rank {args.lora_rank}")
            train_cmd_parts.append(f"--lora_alpha {args.lora_alpha}")
            if args.lora_encoder:
                train_cmd_parts.append("--lora_encoder")
            if args.lora_compressor:
                train_cmd_parts.append("--lora_compressor")
            if args.lora_decompressor:
                train_cmd_parts.append("--lora_decompressor")
            if args.lora_decoder:
                train_cmd_parts.append("--lora_decoder")

        if args.use_feature_loss:
            train_cmd_parts.append("--use_feature_loss")
            train_cmd_parts.append(f"--weight_feature {args.weight_feature}")
        if args.use_time_loss:
            train_cmd_parts.append("--use_time_loss")
            train_cmd_parts.append(f"--weight_time {args.weight_time}")
        if args.use_mel_loss:
            train_cmd_parts.append("--use_mel_loss")
            train_cmd_parts.append(f"--weight_mel {args.weight_mel}")

        if args.early_stopping:
            train_cmd_parts.append("--early_stopping")
            train_cmd_parts.append(f"--patience {args.patience}")
            train_cmd_parts.append(f"--min_delta {args.min_delta}")

        train_cmd = " ".join(train_cmd_parts)
        run_command(train_cmd, "Step 1: Training Model", conda_env=args.train_env)

        model_path = checkpoint_dir / "best_model.pt"
    else:
        print("\n⏭️  Skipping training (--skip_training enabled)")
        if not args.model_checkpoint:
            print(" Error: --model_checkpoint required when --skip_training is used")
            sys.exit(1)
        model_path = Path(args.model_checkpoint)
        if not model_path.exists():
            print(f" Error: Model checkpoint not found: {model_path}")
            sys.exit(1)

    cv_inference_dir = inference_dir / "commonvoice"
    cv_inference_dir.mkdir(exist_ok=True)

    cv_csv = "/home/jieshiang/Desktop/GitHub/Codec_comparison/csv/common_voice_zh_CN_train_filtered.csv"

    inference_cmd_cv = (
        f"python /home/jieshiang/Desktop/GitHub/FocalCodec/simple_inference.py "
        f"--model_path {model_path} "
        f"--csv_file {cv_csv} "
        f"--base_audio_dir /mnt/Internal/ASR "
        f"--output_dir {cv_inference_dir} "
        f"--model_cache_dir /mnt/Internal/jieshiang/Model/FocalCodec "
        f"--gpu_id {args.gpu_id}"
    )
    run_command(inference_cmd_cv, "Step 2a: Inference on Common Voice", conda_env=args.train_env)

    ls_inference_dir = inference_dir / "librispeech"
    ls_inference_dir.mkdir(exist_ok=True)

    ls_csv = "/home/jieshiang/Desktop/GitHub/Codec_comparison/csv/librispeech_test_clean_filtered.csv"

    inference_cmd_ls = (
        f"python /home/jieshiang/Desktop/GitHub/FocalCodec/simple_inference.py "
        f"--model_path {model_path} "
        f"--csv_file {ls_csv} "
        f"--base_audio_dir /mnt/Internal/ASR "
        f"--output_dir {ls_inference_dir} "
        f"--model_cache_dir /mnt/Internal/jieshiang/Model/FocalCodec "
        f"--gpu_id {args.gpu_id}"
    )
    run_command(inference_cmd_ls, "Step 2b: Inference on LibriSpeech", conda_env=args.train_env)

    codec_comparison_dir = Path(args.codec_comparison_dir)

    eval_cmd_cv = (
        f"python {codec_comparison_dir}/fast_evaluation_pipeline.py "
        f"--inference_dir {cv_inference_dir} "
        f"--csv_file common_voice_zh_CN_train_filtered.csv "
        f"--original_dir /mnt/Internal/ASR "
        f"--model_name FocalCodec-S "
        f"--frequency 50Hz_2k_finetuned "
        f"--causality Causal "
        f"--bit_rate 0.55 "
        f"--quantizers 1 "
        f"--codebook_size 2048 "
        f"--n_params N/A "
        f"--training_set N/A "
        f"--testing_set N/A "
        f"--metrics dcer utmos pesq stoi speaker_similarity "
        f"--dataset_type clean "
        f"--use_gpu "
        f"--gpu_id {args.gpu_id} "
        f"--num_workers 32 "
        f"--asr_batch_size 64"
    )
    run_command(eval_cmd_cv, "Step 3a: Evaluation on Common Voice", conda_env=args.eval_env)

    eval_cmd_ls = (
        f"python {codec_comparison_dir}/fast_evaluation_pipeline.py "
        f"--inference_dir {ls_inference_dir} "
        f"--csv_file librispeech_test_clean_filtered.csv "
        f"--original_dir /mnt/Internal/ASR "
        f"--model_name FocalCodec-S "
        f"--frequency 50Hz_2k_finetuned "
        f"--causality Causal "
        f"--bit_rate 0.55 "
        f"--quantizers 1 "
        f"--codebook_size 2048 "
        f"--n_params N/A "
        f"--training_set N/A "
        f"--testing_set N/A "
        f"--metrics dwer utmos pesq stoi speaker_similarity "
        f"--dataset_type clean "
        f"--use_gpu "
        f"--gpu_id {args.gpu_id} "
        f"--num_workers 32 "
        f"--asr_batch_size 64"
    )
    run_command(eval_cmd_ls, "Step 3b: Evaluation on LibriSpeech", conda_env=args.eval_env)

    print("\n" + "="*80)
    print("📊 Collecting Results and Generating Experiment Report")
    print("="*80)

    codec_config_dir = codec_comparison_dir / "configs"
    model_name_for_config = "FocalCodec-S_50Hz_2k_finetuned"

    import glob
    config_files = glob.glob(str(codec_config_dir / f"{model_name_for_config}*.json"))

    if not config_files:
        print(f"⚠️  Warning: Config file not found in {codec_config_dir}")
        print(f"  Looking for pattern: {model_name_for_config}*.json")
    else:
        config_file = config_files[0]
        print(f" Found config file: {config_file}")

        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)

            experiment_result = {
                'exp_id': exp_name,
                'timestamp': timestamp,

                'train_encoder': args.train_encoder,
                'train_compressor': args.train_compressor,
                'train_decompressor': args.train_decompressor,
                'train_decoder': args.train_decoder,

                'use_lora': args.use_lora,
                'lora_rank': args.lora_rank if args.use_lora else None,
                'lora_alpha': args.lora_alpha if args.use_lora else None,

                'use_feature_loss': args.use_feature_loss,
                'use_time_loss': args.use_time_loss,
                'use_mel_loss': args.use_mel_loss,
                'weight_feature': args.weight_feature if args.use_feature_loss else None,
                'weight_time': args.weight_time if args.use_time_loss else None,
                'weight_mel': args.weight_mel if args.use_mel_loss else None,

                'num_epochs': args.num_epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'augment': args.augment,
            }

            if 'LibriSpeech' in config_data and 'Total' in config_data['LibriSpeech']:
                ls_total = config_data['LibriSpeech']['Total']
                experiment_result['ls_utmos'] = float(ls_total.get('UTMOS', 0))
                experiment_result['ls_pesq'] = float(ls_total.get('PESQ', 0))
                experiment_result['ls_stoi'] = float(ls_total.get('STOI', 0))
                experiment_result['ls_speaker_sim'] = float(ls_total.get('Speaker_Sim', 0))
                experiment_result['ls_dwer'] = float(ls_total.get('dWER', 0))

                print(f"  UTMOS: {experiment_result['ls_utmos']:.3f}")
                print(f"  PESQ: {experiment_result['ls_pesq']:.3f}")
                print(f"  STOI: {experiment_result['ls_stoi']:.3f}")
                print(f"  Speaker_Sim: {experiment_result['ls_speaker_sim']:.3f}")
                print(f"  dWER: {experiment_result['ls_dwer']:.3f}")

            if 'CommonVoice' in config_data and 'Total' in config_data['CommonVoice']:
                cv_total = config_data['CommonVoice']['Total']
                experiment_result['cv_utmos'] = float(cv_total.get('UTMOS', 0))
                experiment_result['cv_pesq'] = float(cv_total.get('PESQ', 0))
                experiment_result['cv_stoi'] = float(cv_total.get('STOI', 0))
                experiment_result['cv_speaker_sim'] = float(cv_total.get('Speaker_Sim', 0))
                experiment_result['cv_dcer'] = float(cv_total.get('dCER', 0))

                print(f"  UTMOS: {experiment_result['cv_utmos']:.3f}")
                print(f"  PESQ: {experiment_result['cv_pesq']:.3f}")
                print(f"  STOI: {experiment_result['cv_stoi']:.3f}")
                print(f"  Speaker_Sim: {experiment_result['cv_speaker_sim']:.3f}")
                print(f"  dCER: {experiment_result['cv_dcer']:.3f}")

            exp_results_csv = experiment_dir / "experiment_results.csv"
            result_df = pd.DataFrame([experiment_result])
            result_df.to_csv(exp_results_csv, index=False)
            print(f"\n Experiment results saved to: {exp_results_csv}")

        except Exception as e:
            print(f"⚠️  Warning: Could not process config file: {e}")
            import traceback
            traceback.print_exc()

    codec_result_dir = codec_comparison_dir / "result"

    if codec_result_dir.exists():
        import shutil

        print("\n" + "="*80)
        print("📋 Copying detailed evaluation results")
        print("="*80)

        cv_detailed = codec_result_dir / "detailed_results_FocalCodec-S_50Hz_2k_finetuned_clean_commonvoice.csv"
        cv_summary = codec_result_dir / "summary_results_FocalCodec-S_50Hz_2k_finetuned_clean_commonvoice.csv"

        if cv_detailed.exists():
            shutil.copy2(cv_detailed, experiment_dir / "commonvoice_detailed.csv")
            print(f" Copied: commonvoice_detailed.csv")
        else:
            print(f"⚠️  Not found: {cv_detailed.name}")

        if cv_summary.exists():
            shutil.copy2(cv_summary, experiment_dir / "commonvoice_summary.csv")
            print(f" Copied: commonvoice_summary.csv")
        else:
            print(f"⚠️  Not found: {cv_summary.name}")

        ls_detailed = codec_result_dir / "detailed_results_FocalCodec-S_50Hz_2k_finetuned_clean_librispeech.csv"
        ls_summary = codec_result_dir / "summary_results_FocalCodec-S_50Hz_2k_finetuned_clean_librispeech.csv"

        if ls_detailed.exists():
            shutil.copy2(ls_detailed, experiment_dir / "librispeech_detailed.csv")
            print(f" Copied: librispeech_detailed.csv")
        else:
            print(f"⚠️  Not found: {ls_detailed.name}")

        if ls_summary.exists():
            shutil.copy2(ls_summary, experiment_dir / "librispeech_summary.csv")
            print(f" Copied: librispeech_summary.csv")
        else:
            print(f"⚠️  Not found: {ls_summary.name}")
    else:
        print(f"⚠️  Warning: Result directory not found: {codec_result_dir}")

    if args.cleanup_inference or args.cleanup_codec_comparison or args.keep_best_model_only or args.cleanup_best_model:
        print("\n" + "="*80)
        print("🧹 Cleanup Operations")
        print("="*80)

    if args.cleanup_inference:
        print("\n[1/3] Cleaning up inference files...")
        total_deleted = 0
        for inf_dir in [cv_inference_dir, ls_inference_dir]:
            if inf_dir.exists():
                import shutil
                try:
                    shutil.rmtree(inf_dir)
                    deleted_count = len(list(inf_dir.glob("*"))) if inf_dir.exists() else 0
                    total_deleted += deleted_count
                    print(f"   Deleted {inf_dir}")
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not delete {inf_dir}: {e}")
        print(f" Cleaned up inference directories")

    if args.cleanup_codec_comparison:
        print("\n[2/3] Cleaning up Codec_comparison generated files...")
        import shutil

        audio_paths = [
            codec_comparison_dir / "audio" / "CommonVoice" / "FocalCodec-S" / "50Hz_2k_finetuned",
            codec_comparison_dir / "audio" / "LibriSpeech" / "FocalCodec-S" / "50Hz_2k_finetuned"
        ]

        audio_count = 0
        for audio_path in audio_paths:
            if audio_path.exists():
                try:
                    file_count = len(list(audio_path.glob("*")))
                    shutil.rmtree(audio_path)
                    audio_count += file_count
                    print(f"   Deleted {audio_path}")
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not delete {audio_path}: {e}")

        print(f"   Cleaned up {audio_count} audio files")

        config_file = codec_comparison_dir / "configs" / "FocalCodec-S_50Hz_2k_finetuned_config.json"
        if config_file.exists():
            try:
                config_file.unlink()
                print(f"   Deleted config: {config_file.name}")
            except Exception as e:
                print(f"  ⚠️  Warning: Could not delete {config_file.name}: {e}")

        result_files = [
            codec_comparison_dir / "result" / "detailed_results_FocalCodec-S_50Hz_2k_finetuned_clean_commonvoice.csv",
            codec_comparison_dir / "result" / "detailed_results_FocalCodec-S_50Hz_2k_finetuned_clean_librispeech.csv",
            codec_comparison_dir / "result" / "summary_results_FocalCodec-S_50Hz_2k_finetuned_clean_commonvoice.csv",
            codec_comparison_dir / "result" / "summary_results_FocalCodec-S_50Hz_2k_finetuned_clean_librispeech.csv"
        ]

        result_count = 0
        for result_file in result_files:
            if result_file.exists():
                try:
                    result_file.unlink()
                    result_count += 1
                    print(f"   Deleted result: {result_file.name}")
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not delete {result_file.name}: {e}")

        print(f"   Cleaned up {result_count} result files")

    if args.keep_best_model_only and not args.skip_training:
        print("\n[3/3] Cleaning up training checkpoints (keeping best_model.pt only)...")
        if checkpoint_dir.exists():
            checkpoint_files = list(checkpoint_dir.glob("epoch_*.pt"))
            for f in checkpoint_files:
                try:
                    f.unlink()
                    print(f"  Deleted checkpoint: {f.name}")
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not delete {f.name}: {e}")
            print(f"   Cleaned up {len(checkpoint_files)} epoch checkpoints")
            print(f"   Kept: best_model.pt")

    if args.cleanup_best_model and not args.skip_training:
        print("\n[4/4] Cleaning up best_model.pt and checkpoint directory...")
        if checkpoint_dir.exists():
            import shutil
            try:
                shutil.rmtree(checkpoint_dir)
                print(f"   Deleted entire checkpoint directory: {checkpoint_dir}")
            except Exception as e:
                print(f"  ⚠️  Warning: Could not delete checkpoint directory: {e}")

    print("\n" + "="*80)
    print("🎉 Pipeline completed successfully!")
    print("="*80)
    print(f"\n📁 Experiment results saved to: {experiment_dir}")
    print(f"\n📂 Directory structure:")
    print(f"  ├─ checkpoints/")
    if args.keep_best_model_only:
        print(f"  │   └─ best_model.pt (kept)")
    else:
        print(f"  │   ├─ best_model.pt")
        print(f"  │   └─ epoch_*.pt (all epochs)")

    if args.cleanup_inference:
        print(f"  ├─ inference/ (deleted)")
    else:
        print(f"  ├─ inference/")
        print(f"  │   ├─ commonvoice/ (reconstructed audio)")
        print(f"  │   └─ librispeech/ (reconstructed audio)")

    print(f"  └─ results/")
    print(f"      ├─ commonvoice_detailed.csv")
    print(f"      ├─ commonvoice_summary.csv")
    print(f"      ├─ librispeech_detailed.csv")
    print(f"      └─ librispeech_summary.csv")

    print(f"\n📄 Experiment Summary:")
    print(f"  /home/jieshiang/Desktop/GitHub/FocalCodec/experiments/experiment_results.csv")

    if args.cleanup_codec_comparison:
        print(f"\n🗑️  Codec_comparison cleaned:")
        print(f"  ├─ audio/ (experiment files deleted)")
        print(f"  ├─ configs/ (experiment files deleted)")
        print(f"  └─ result/ (experiment CSV deleted)")
    else:
        print(f"\n⚠️  Codec_comparison NOT cleaned:")
        print(f"  ├─ audio/ (experiment files remain)")
        print(f"  ├─ configs/ (experiment files remain)")
        print(f"  └─ result/ (experiment CSV remain)")
        print(f"  Tip: Use --cleanup_codec_comparison to auto-clean")

    print("\n")


if __name__ == '__main__':
    main()
