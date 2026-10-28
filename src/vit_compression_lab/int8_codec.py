"""Symmetric per-token INT8 codec for [tokens, width] feature tensors.

quantize() and dequantize() are copied unchanged from
encoders/Qwen3.8-vision-compression/scripts/quantize_features.py, which runs
at import time and so cannot be imported directly. They work on any device.
"""

import io

import torch


def quantize(features):
    x = features.float()
    max_abs = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(
        max_abs == 0,
        torch.ones_like(max_abs),
        max_abs / 127,
    )
    q = (x / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def dequantize(q, scale, dtype):
    return (q.float() * scale).to(dtype)


def serialize(q, scale, dtype, grid_thw):
    """Bytes holding INT8 values, FP32 scales, and what is needed to rebuild."""
    payload = {
        "q": q.contiguous(),
        "scale": scale.contiguous(),
        "dtype": str(dtype).removeprefix("torch."),
        "shape": list(q.shape),
        "grid_thw": grid_thw,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def deserialize(data):
    """Inverse of serialize(): returns (q, scale, dtype, shape, grid_thw) on CPU."""
    payload = torch.load(io.BytesIO(data), weights_only=True, map_location="cpu")
    if list(payload["q"].shape) != payload["shape"]:
        raise ValueError("Payload shape does not match its stored INT8 tensor")
    dtype = getattr(torch, payload["dtype"])
    return payload["q"], payload["scale"], dtype, payload["shape"], payload["grid_thw"]


def raw_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def reconstruction_errors(original, reconstructed):
    """Diagnostics only; they do not measure task quality."""
    x = original.float()
    error = reconstructed.float() - x
    source_norm = torch.linalg.vector_norm(x).item()
    error_norm = torch.linalg.vector_norm(error).item()
    if source_norm > 0:
        relative_l2 = error_norm / source_norm
    else:
        # All-zero source: relative error is undefined unless the error is also zero.
        relative_l2 = 0.0 if error_norm == 0 else float("inf")
    return {
        "mae": error.abs().mean().item(),
        "max_abs_error": error.abs().max().item(),
        "mse": error.square().mean().item(),
        "relative_l2": relative_l2,
    }
