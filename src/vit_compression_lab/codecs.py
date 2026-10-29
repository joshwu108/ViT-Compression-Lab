"""Selectable feature codecs sharing one interface.

quantize(features) -> (q, scale); dequantize(q, scale, dtype);
serialize(q, scale, dtype, shape, grid_thw) -> bytes;
deserialize(bytes) -> (q, scale, dtype, shape, grid_thw).
`q` is whatever the codec transmits (INT8 values, or packed INT4 bytes).
int4 scales per token; int4-g64 / int4-g32 scale per group of 64 / 32 channels.
"""

from functools import partial
from types import SimpleNamespace

from . import int4_codec, int8_codec


def _int8_serialize(q, scale, dtype, shape, grid_thw):
    return int8_codec.serialize(q, scale, dtype, grid_thw)


def _int4(group):
    where = "per token" if group is None else f"per group of {group} channels"
    return SimpleNamespace(
        name="int4" if group is None else f"int4-g{group}",
        scheme=f"symmetric INT4 packed two per byte, FP16 scales {where} (int4_codec.py)",
        file="int4_codec.py",
        quantize=partial(int4_codec.quantize, group=group),
        dequantize=int4_codec.dequantize,
        serialize=int4_codec.serialize,
        deserialize=int4_codec.deserialize,
    )


CODECS = {
    "int8": SimpleNamespace(
        name="int8",
        scheme="symmetric per-token INT8, FP32 scales (int8_codec.py)",
        file="int8_codec.py",
        quantize=int8_codec.quantize,
        dequantize=int8_codec.dequantize,
        serialize=_int8_serialize,
        deserialize=int8_codec.deserialize,
    ),
    "int4": _int4(None),
    "int4-g64": _int4(64),
    "int4-g32": _int4(32),
}
CODEC_NAMES = tuple(CODECS)


def get_codec(name):
    try:
        return CODECS[name]
    except KeyError:
        raise ValueError(f"Unknown codec {name!r}; choose from {CODEC_NAMES}") from None
