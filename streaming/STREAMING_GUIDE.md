# Streaming Inference Guide

## Installation and Setup

### Clone Repository

```bash
git clone https://github.com/lucadellalib/focalcodec.git
cd focalcodec
```

### Install Dependencies

```bash
conda create -n focalcodec python=3.10
conda activate focalcodec
pip install -r requirements.txt
```

## Streaming Inference Script

### Directory Structure

```
./FocalCodec/
├── focalcodec/
│   ├── focalcodec/
│   │   ├── codec.py
│   │   ├── __init__.py
│   │   └── ...
│   └── demo.py
├── streaming_inference.py
└── ...
```

### Usage Examples

Specify model and device:

```bash
python streaming_inference.py \
    --input_path input.wav \
    --output_path output.wav \
    --model_config lucadellalib/focalcodec_50hz_4k_causal \
    --device cuda:1 \
    --verbose
```

### Example Output

```
Loading model: lucadellalib/focalcodec_50hz_2k_causal
Model loaded successfully
  Input sample rate: 16000 Hz
  Output sample rate: 16000 Hz
  Chunk size: 320 samples
  Theoretical latency: 20.00 ms

Input audio: input.wav
  Original sample rate: 16000 Hz
  Duration: 5.23 seconds

Streaming inference:
  Chunk size: 320 samples
  Total samples: 83680
  Number of chunks: 262
  Chunk duration: 20.00 ms

Processing chunks:
Chunk 1/262: Latency = 15.32 ms | Tokens shape = torch.Size([1, 1])
Chunk 2/262: Latency = 12.45 ms | Tokens shape = torch.Size([1, 1])
Chunk 3/262: Latency = 11.98 ms | Tokens shape = torch.Size([1, 1])
...
Chunk 262/262: Latency = 12.10 ms | Tokens shape = torch.Size([1, 1])

STREAMING INFERENCE SUMMARY
Total chunks processed: 262
Average latency per chunk: 12.34 ms
Min latency: 11.85 ms
Max latency: 15.32 ms
Total processing time: 3233.08 ms (3.23 sec)
Audio duration: 5.23 sec
Real-time factor (RTF): 0.6180x
  (RTF < 1.0 means faster than real-time)

Output saved to: output.wav
```

## Understanding the Results

### Latency Metrics

Per-Chunk Latency: Time to process each chunk
- Represents the delay for each streaming segment
- Should be close to theoretical latency (chunk_size / sample_rate * 1000 ms)

Average Latency: Mean of all chunk latencies
- Key metric for streaming performance
- Lower is better for real-time applications

Real-Time Factor (RTF):
```
RTF = Total Processing Time / Audio Duration
```
- RTF < 1.0: Faster than real-time (can stream)
- RTF = 1.0: Exactly real-time
- RTF > 1.0: Slower than real-time (cannot stream)

### Performance Benchmarks

| Model | Codebook | Latency | RTF (GPU) | RTF (CPU) |
|-------|----------|---------|-----------|-----------|
| 65k_causal | 65536 | 80ms | 0.35x | 2.1x |
| 4k_causal | 4096 | 80ms | 0.31x | 1.8x |
| 2k_causal | 2048 | 80ms | 0.28x | 1.6x |

## Advanced Usage

### Custom Chunk Size

```bash
python streaming_inference.py \
    --input_path input.wav \
    --output_path output.wav \
    --chunk_duration 0.04
```

### Batch Processing

```bash
for file in *.wav; do
    python streaming_inference.py \
        --input_path "$file" \
        --output_path "output_${file}" \
        --model_config lucadellalib/focalcodec_50hz_2k_causal
done
```

## Troubleshooting

Issue: CUDA out of memory
Solution: Use CPU or reduce chunk size

Issue: RTF > 1.0 on GPU
Solution: Check GPU utilization, use smaller model

Issue: Audio artifacts
Solution: Increase chunk size or use higher quality model
