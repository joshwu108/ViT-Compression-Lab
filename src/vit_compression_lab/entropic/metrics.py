"""Compression, reconstruction and timing metrics shared by the es_* scripts.

Timing summaries use the same keys as inject_eval.py (median_ms, p95_ms,
min_ms, runs) so that score_results.py can take medians across examples.
"""

import statistics

import torch
import torch.nn.functional as F

from ..int8_codec import raw_bytes, reconstruction_errors


def summarize_times(times_ms):
    ordered = sorted(times_ms)
    return {
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))],
        "min_ms": ordered[0],
        "runs": len(ordered),
    }


def compression_metrics(coded_bytes, container_overhead_bytes, ideal_bits, n_visual_tokens,
                        original_raw_bytes, original_serialized_bytes=None):
    coded_bits = 8 * coded_bytes
    out = {
        "compressed_serialized_bytes": int(coded_bytes),
        "container_overhead_bytes": int(container_overhead_bytes),
        "ideal_bits": float(ideal_bits),
        "coded_bits": int(coded_bits),
        "entropy_coding_overhead_ratio": coded_bits / ideal_bits if ideal_bits > 0 else float("inf"),
        "bits_per_visual_token": coded_bits / n_visual_tokens,
        "ideal_bits_per_visual_token": ideal_bits / n_visual_tokens,
        "original_raw_bytes": int(original_raw_bytes),
        "original_bits_per_visual_token": 8 * original_raw_bytes / n_visual_tokens,
        "raw_ratio": original_raw_bytes / coded_bytes if coded_bytes else float("inf"),
    }
    if original_serialized_bytes is not None:
        out["original_serialized_bytes"] = int(original_serialized_bytes)
        out["serialized_ratio"] = original_serialized_bytes / coded_bytes if coded_bytes else float("inf")
    return out


def reconstruction_metrics(original, reconstructed):
    """inject_eval's diagnostics plus per-token cosine similarity."""
    out = reconstruction_errors(original, reconstructed)
    cos = F.cosine_similarity(original.float(), reconstructed.float(), dim=-1)
    out["cosine_mean"] = cos.mean().item()
    out["cosine_min"] = cos.min().item()
    out["cosine_p05"] = torch.quantile(cos, 0.05).item() if cos.numel() > 1 else cos.item()
    return out


def feature_raw_bytes(features):
    return raw_bytes(features)
