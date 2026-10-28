# Experiment 1: Vision encoder baseline

Measure vision encoder latency and the size of the image features produced before compression.

## Measurement setup

- **Model:** `Qwen/Qwen3.8-27B`
- **Input image:** `src/images/test1.jpg`
- **Warmup passes:** 5
- **Measured passes:** 1,000
- **Timing:** CUDA events around the vision encoder forward pass; excludes model loading and image preprocessing.

## Encoder latency

### Initial single-pass result

| Metric | Latency |
| --- | ---: |
| Single measured pass | 20.0581111907959 ms |

### Repeated measurements

| Metric | Latency |
| --- | ---: |
| Mean | 25.007 ms |
| Median | 20.840 ms |
| p95 | 37.263 ms |
| p99 | 39.985 ms |

The p95 and p99 values describe the upper end of the measured latency distribution: approximately 95% and 99% of passes, respectively, completed at or below these times.

## Image grid and output features

| Property | Value |
| --- | --- |
| Image grid `[time, height, width]` | `[[1, 22, 34]]` |
| Feature shape | `[187, 5120]` |
| Feature dtype | `torch.bfloat16` |
| Feature tensor size | 1,914,880 bytes (about 1.91 MB) |

Both the initial and repeated runs reported the same feature shape, dtype, and byte count. The grid describes the processed image patch layout; the output contains 187 feature vectors, each with 5,120 values.

## Estimated feature transmission time

Assuming a **10 Mbit/s** connection and transmitting the raw feature tensor:

```text
1,914,880 bytes × 8 bits/byte ÷ 10,000,000 bits/second
≈ 1.53 seconds
```

This is a theoretical payload transfer time, not a measured network result. Actual end-to-end time also includes serialization, protocol overhead, and network delays.
