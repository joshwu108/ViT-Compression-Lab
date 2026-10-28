import torch

from vit_compression_lab.int8_codec import (
    dequantize,
    deserialize,
    quantize,
    reconstruction_errors,
    serialize,
)


def test_all_zero_token_round_trips_to_zero():
    features = torch.zeros(3, 8, dtype=torch.bfloat16)
    features[1] = torch.linspace(-2, 2, 8)
    q, scale = quantize(features)
    assert scale[0].item() == 1.0 and scale[2].item() == 1.0
    restored = dequantize(q, scale, features.dtype)
    assert torch.equal(restored[0], features[0])
    assert torch.equal(restored[2], features[2])


def test_serialization_preserves_values_and_metadata():
    features = torch.randn(5, 16).to(torch.bfloat16)
    q, scale = quantize(features)
    data = serialize(q, scale, features.dtype, [[1, 2, 10]])
    q2, scale2, dtype, shape, grid = deserialize(data)
    assert torch.equal(q2, q) and torch.equal(scale2, scale)
    assert dtype == torch.bfloat16 and shape == [5, 16] and grid == [[1, 2, 10]]
    assert scale2.dtype == torch.float32 and q2.dtype == torch.int8


def test_relative_l2_handles_all_zero_source():
    zeros = torch.zeros(2, 4)
    assert reconstruction_errors(zeros, zeros)["relative_l2"] == 0.0
    assert reconstruction_errors(zeros, zeros + 1)["relative_l2"] == float("inf")
