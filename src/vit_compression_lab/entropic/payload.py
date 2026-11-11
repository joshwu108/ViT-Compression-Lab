"""Container for a transmitted Entropic Student payload.

    magic "ESV1" | u32 header_len | header JSON (utf-8) | u32 n_strings |
    (u32 len | bytes) * n_strings

The header carries everything the server needs besides the model weights:
split spec, modality, ``grid_thw`` per image/video, latent counts, channel
count, and the dtype the reconstruction should be cast to. Video timestamps are
not here: they are text tokens in ``input_ids`` on the server side.
"""

import json
import struct

MAGIC = b"ESV1"


def pack(header, strings):
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    parts = [MAGIC, struct.pack("<I", len(header_bytes)), header_bytes, struct.pack("<I", len(strings))]
    for s in strings:
        parts.append(struct.pack("<I", len(s)))
        parts.append(bytes(s))
    return b"".join(parts)


def unpack(data):
    view = memoryview(data)
    if bytes(view[:4]) != MAGIC:
        raise ValueError("Not an Entropic Student payload")
    (header_len,) = struct.unpack("<I", view[4:8])
    header = json.loads(bytes(view[8 : 8 + header_len]).decode())
    pos = 8 + header_len
    (n,) = struct.unpack("<I", view[pos : pos + 4])
    pos += 4
    strings = []
    for _ in range(n):
        (length,) = struct.unpack("<I", view[pos : pos + 4])
        pos += 4
        strings.append(bytes(view[pos : pos + length]))
        pos += length
    if pos != len(view):
        raise ValueError("Trailing bytes after payload")
    return header, strings


def accounting(data, strings):
    string_bytes = sum(len(s) for s in strings)
    return {"total_bytes": len(data), "string_bytes": string_bytes,
            "container_overhead_bytes": len(data) - string_bytes}


def make_header(split, grid_thw, latent_counts, latent_channels, modality="image",
                dtype="bfloat16", encoder_type=None):
    return {
        "version": 1,
        "encoder_input": split.encoder_input,
        "decoder_target": split.decoder_target,
        "modality": modality,
        "grid_thw": [list(map(int, row)) for row in grid_thw.tolist()],
        "latent_counts": [int(n) for n in latent_counts],
        "latent_channels": int(latent_channels),
        "dtype": dtype,
        "encoder_type": encoder_type,
    }
