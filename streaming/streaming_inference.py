
import os
import argparse
import time
import torch
import torchaudio
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StreamingInferenceProcessor:
#!/usr/bin/env python3
        Initialize streaming processor

        Args:
            model_config: Model config (e.g., "lucadellalib/focalcodec_50hz_2k_causal")
            device: Device to use (e.g., "cuda:0", "cpu")
        sig, sample_rate = torchaudio.load(audio_path)
        logger.info(f"\nInput audio: {audio_path}")
        logger.info(f"  Original sample rate: {sample_rate} Hz")
        logger.info(f"  Duration: {sig.shape[1] / sample_rate:.2f} seconds")

        if sample_rate != self.codec.sample_rate_input:
            sig = torchaudio.functional.resample(
                sig, sample_rate, self.codec.sample_rate_input
            )

        sig = sig.to(self.device)

        encoder_state = []
        compressor_state = []
        decompressor_state = []
        decoder_state = []

        chunk_size = self.codec.chunk_size
        total_samples = sig.shape[1]
        num_chunks = (total_samples + chunk_size - 1) // chunk_size  # Ceiling division

        logger.info(f"\nStreaming inference:")
        logger.info(f"  Chunk size: {chunk_size} samples")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Number of chunks: {num_chunks}")
        logger.info(f"  Chunk duration: {chunk_size / self.codec.sample_rate_input * 1000:.2f} ms")

        output_chunks = []
        chunk_latencies = []

        logger.info("\nProcessing chunks:")
        logger.info("-" * 80)

        i = 0
        chunk_idx = 0
        while i < total_samples:
            chunk_end = min(i + chunk_size, total_samples)
            chunk = sig[:, i:chunk_end]

            if chunk.shape[1] < chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_size - chunk.shape[1]))

            chunk_start_time = time.time()

            toks, encoder_state, compressor_state = self.codec.sig_to_toks(
                chunk,
                encoder_state,
                compressor_state,
                return_state=True  # Key: return updated state
            )

            rec_chunk, decompressor_state, decoder_state = self.codec.toks_to_sig(
                toks,
                decompressor_state,
                decoder_state,
                return_state=True  # Key: return updated state
            )

            chunk_end_time = time.time()
            chunk_latency_ms = (chunk_end_time - chunk_start_time) * 1000

            output_chunks.append(rec_chunk.cpu())
            chunk_latencies.append(chunk_latency_ms)

            if verbose:
                logger.info(
                    f"Chunk {chunk_idx + 1}/{num_chunks}: "
                    f"Latency = {chunk_latency_ms:.2f} ms | "
                    f"Tokens shape = {toks.shape}"
                )

            i += chunk_size
            chunk_idx += 1

        logger.info("-" * 80)

        rec_sig = torch.cat(output_chunks, dim=1)

        rec_sig = rec_sig[:, :int(total_samples * self.codec.sample_rate_output / self.codec.sample_rate_input)]

        if self.codec.sample_rate_output != sample_rate:
            rec_sig = torchaudio.functional.resample(
                rec_sig, self.codec.sample_rate_output, sample_rate
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchaudio.save(output_path, rec_sig, sample_rate)

        avg_latency = sum(chunk_latencies) / len(chunk_latencies)
        min_latency = min(chunk_latencies)
        max_latency = max(chunk_latencies)
        total_latency = sum(chunk_latencies)

        stats = {
            'num_chunks': num_chunks,
            'chunk_latencies_ms': chunk_latencies,
            'avg_latency_ms': avg_latency,
            'min_latency_ms': min_latency,
            'max_latency_ms': max_latency,
            'total_latency_ms': total_latency,
            'audio_duration_sec': sig.shape[1] / self.codec.sample_rate_input,
            'real_time_factor': total_latency / 1000 / (sig.shape[1] / self.codec.sample_rate_input)
        }

        logger.info("\n" + "=" * 80)
        logger.info("STREAMING INFERENCE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total chunks processed: {num_chunks}")
        logger.info(f"Average latency per chunk: {avg_latency:.2f} ms")
        logger.info(f"Min latency: {min_latency:.2f} ms")
        logger.info(f"Max latency: {max_latency:.2f} ms")
        logger.info(f"Total processing time: {total_latency:.2f} ms ({total_latency/1000:.2f} sec)")
        logger.info(f"Audio duration: {stats['audio_duration_sec']:.2f} sec")
        logger.info(f"Real-time factor (RTF): {stats['real_time_factor']:.4f}x")
        logger.info(f"  (RTF < 1.0 means faster than real-time)")
        logger.info(f"\nOutput saved to: {output_path}")
        logger.info("=" * 80)

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="FocalCodec Streaming Inference - Chunk-by-Chunk Processing"
    )

    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='Input audio file path'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Output audio file path'
    )

    parser.add_argument(
        '--model_config',
        type=str,
        default='lucadellalib/focalcodec_50hz_2k_causal',
        choices=[
            'lucadellalib/focalcodec_50hz_2k_causal',
            'lucadellalib/focalcodec_50hz_4k_causal',
            'lucadellalib/focalcodec_50hz_65k_causal'
        ],
        help='FocalCodec causal model config'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use (e.g., cuda:0, cpu)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print per-chunk latency information'
    )

    args = parser.parse_args()

    if 'cuda' in args.device and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'

    if not os.path.exists(args.input_path):
        logger.error(f"Input file not found: {args.input_path}")
        return

    processor = StreamingInferenceProcessor(
        model_config=args.model_config,
        device=args.device
    )

    stats = processor.inference_streaming(
        audio_path=args.input_path,
        output_path=args.output_path,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
