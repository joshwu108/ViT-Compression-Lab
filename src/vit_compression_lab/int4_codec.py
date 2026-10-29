"""Symmetric per-token INT4 codec with FP16 scales for [tokens, width] features.

Values are scaled by max|x| / 7 over either a whole token row (group=None)
or each group of `group` consecutive channels, rounded to the symmetric range
[-7, 7], and packed two per byte. The stored FP16 scale is the one used for
quantization, so encoder and decoder agree exactly. Interface matches
int8_codec except that quantize() returns the packed tensor and serialize()
needs the unpacked shape.
"""

import io

import torch

LEVELS = 7
OFFSET = 8  # maps [-7, 7] onto the unsigned nibble range [1, 15]


def quantize(features, group=None):
    """Returns (packed uint8 [tokens, width // 2], float16 scales).

    Scales are [tokens, 1] per token, or [tokens, width // group, 1] per group.
    """
    x = features.float()
    tokens, width = x.shape
    if width % 2:
        raise ValueError("Feature width must be even to pack two INT4 values per byte")
    if group is not None:
        if group < 2 or width % group:
            raise ValueError(
                f"Group size {group} must divide the feature width {width}"
            )
        x = x.reshape(tokens, width // group, group)
    max_abs = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / LEVELS)
    scale = scale.to(torch.float16)
    if not torch.isfinite(scale).all():
        raise ValueError("Quantization scale overflows float16")
    q = (x / scale.float()).round().clamp(-LEVELS, LEVELS).to(torch.int8)
    return pack(q.reshape(tokens, width)), scale


def pack(q):
    """int8 values in [-7, 7] -> uint8 with the even column in the low nibble."""
    u = (q + OFFSET).to(torch.uint8)
    return u[..., 0::2] | (u[..., 1::2] << 4)


def unpack(packed):
    low = (packed & 0x0F).to(torch.int8) - OFFSET
    high = (packed >> 4).to(torch.int8) - OFFSET
    return torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], -1)


def dequantize(packed, scale, dtype):
    q = unpack(packed).float()
    tokens = q.shape[0]
    if scale.dim() == 3:  # per-group scales: [tokens, groups, 1]
        q = q.reshape(tokens, scale.shape[1], -1)
    return (q * scale.float()).reshape(tokens, -1).to(dtype)


def serialize(packed, scale, dtype, shape, grid_thw):
    """Bytes holding packed INT4 values, FP16 scales, and what is needed to rebuild."""
    payload = {
        "q": packed.contiguous(),
        "scale": scale.contiguous(),
        "dtype": str(dtype).removeprefix("torch."),
        "shape": list(shape),
        "grid_thw": grid_thw,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def deserialize(data):
    """Inverse of serialize(): returns (packed, scale, dtype, shape, grid_thw) on CPU."""
    payload = torch.load(io.BytesIO(data), weights_only=True, map_location="cpu")
    shape = payload["shape"]
    if list(payload["q"].shape) != [*shape[:-1], shape[-1] // 2]:
        raise ValueError("Payload shape does not match its packed INT4 tensor")
    if payload["scale"].dtype != torch.float16:
        raise ValueError("INT4 payload scales must be float16")
    if payload["scale"].shape[0] != shape[0]:
        raise ValueError("Payload scales do not match the stored token count")
    dtype = getattr(torch, payload["dtype"])
    return payload["q"], payload["scale"], dtype, shape, payload["grid_thw"]
