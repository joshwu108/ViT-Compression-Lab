import torch

from vit_compression_lab import int4_codec, int8_codec
from vit_compression_lab.codecs import CODEC_NAMES, get_codec
from vit_compression_lab.int4_codec import (
    dequantize,
    deserialize,
    pack,
    quantize,
    serialize,
    unpack,
)


def test_pack_unpack_round_trips_every_level():
    q = torch.arange(-7, 8, dtype=torch.int8).repeat(2)[:28].reshape(2, 14)
    assert torch.equal(unpack(pack(q)), q)
    assert pack(q).dtype == torch.uint8 and pack(q).shape == (2, 7)


def test_quantize_shapes_dtypes_and_range():
    features = torch.randn(5, 16).to(torch.bfloat16)
    packed, scale = quantize(features)
    assert packed.shape == (5, 8) and packed.dtype == torch.uint8
    assert scale.shape == (5, 1) and scale.dtype == torch.float16
    values = unpack(packed)
    assert values.min() >= -7 and values.max() <= 7
    # The largest-magnitude entry of each row sits at a saturation level.
    assert torch.all(values.abs().amax(dim=-1) == 7)


def test_reconstruction_error_is_bounded_by_half_a_step():
    torch.manual_seed(0)
    features = torch.randn(64, 32).to(torch.bfloat16)
    packed, scale = quantize(features)
    restored = dequantize(packed, scale, torch.float32)
    step = scale.float()
    # Allow for the float16 scale rounding on top of the half-step bound.
    assert torch.all((restored - features.float()).abs() <= 0.5 * step * 1.01)


def test_all_zero_token_round_trips_to_zero():
    features = torch.zeros(3, 8, dtype=torch.bfloat16)
    features[1] = torch.linspace(-2, 2, 8)
    packed, scale = quantize(features)
    assert scale[0].item() == 1.0 and scale[2].item() == 1.0
    restored = dequantize(packed, scale, features.dtype)
    assert torch.equal(restored[0], features[0]) and torch.equal(
        restored[2], features[2]
    )


def test_odd_width_is_rejected():
    try:
        quantize(torch.randn(2, 7))
    except ValueError as error:
        assert "even" in str(error)
    else:
        raise AssertionError("odd width should be rejected")


def test_serialization_preserves_values_and_metadata():
    features = torch.randn(5, 16).to(torch.bfloat16)
    packed, scale = quantize(features)
    data = serialize(packed, scale, features.dtype, features.shape, [[1, 2, 10]])
    packed2, scale2, dtype, shape, grid = deserialize(data)
    assert torch.equal(packed2, packed) and torch.equal(scale2, scale)
    assert dtype == torch.bfloat16 and shape == [5, 16] and grid == [[1, 2, 10]]
    assert torch.equal(
        dequantize(packed2, scale2, dtype), dequantize(packed, scale, dtype)
    )


def test_int4_payload_is_about_half_of_int8():
    features = torch.randn(300, 5120).to(torch.bfloat16)
    p4, s4 = int4_codec.quantize(features)
    q8, s8 = int8_codec.quantize(features)
    int4_bytes = p4.numel() + s4.numel() * 2
    int8_bytes = q8.numel() + s8.numel() * 4
    assert abs(int8_bytes / int4_bytes - 2.0) < 0.01
    assert abs(features.numel() * 2 / int4_bytes - 4.0) < 0.01


def test_group_scales_follow_each_group():
    torch.manual_seed(1)
    features = torch.randn(6, 128).to(torch.bfloat16)
    features[:, 5] = 60.0  # one outlier channel, as in the Qwen features
    packed, scale = quantize(features, group=32)
    assert scale.shape == (6, 4, 1) and scale.dtype == torch.float16
    restored = dequantize(packed, scale, torch.float32)
    error = (restored - features.float()).abs()
    step = scale.float().expand(6, 4, 32).reshape(6, 128)
    assert torch.all(error <= 0.5 * step * 1.01)
    # Groups without the outlier keep a fine step; per-token would not.
    assert scale[:, 1:].float().max() < scale[:, 0].float().min() / 10
    data = serialize(packed, scale, features.dtype, features.shape, [[1, 2, 3]])
    packed2, scale2, dtype, shape, grid = deserialize(data)
    assert torch.equal(scale2, scale) and shape == [6, 128]
    assert torch.equal(
        dequantize(packed2, scale2, dtype), dequantize(packed, scale, dtype)
    )


def test_group_must_divide_width():
    try:
        quantize(torch.randn(2, 100), group=64)
    except ValueError as error:
        assert "divide" in str(error)
    else:
        raise AssertionError("non-dividing group should be rejected")


def test_codec_registry_shares_one_interface():
    assert CODEC_NAMES == ("int8", "int4", "int4-g64", "int4-g32")
    features = torch.randn(4, 128).to(torch.bfloat16)
    for name in CODEC_NAMES:
        codec = get_codec(name)
        q, scale = codec.quantize(features)
        data = codec.serialize(
            q, scale, features.dtype, list(features.shape), [[1, 2, 2]]
        )
        q2, scale2, dtype, shape, grid = codec.deserialize(data)
        restored = codec.dequantize(q2, scale2, dtype)
        assert shape == [4, 128] and grid == [[1, 2, 2]]
        assert torch.equal(restored, codec.dequantize(q, scale, features.dtype))
        assert restored.shape == features.shape and restored.dtype == features.dtype
    try:
        get_codec("fp8")
    except ValueError as error:
        assert "fp8" in str(error)
    else:
        raise AssertionError("unknown codec should be rejected")
