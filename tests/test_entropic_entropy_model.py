import math

import torch

from vit_compression_lab.entropic.entropy_model import FactorizedEntropyModel


def test_likelihood_is_a_probability_and_sums_to_about_one():
    torch.manual_seed(0)
    model = FactorizedEntropyModel(channels=3)
    grid = torch.arange(-60, 61, dtype=torch.float32)
    z = grid[:, None].repeat(1, 3)
    _, likelihood = model(z, mode="round")  # integers around the (zero) medians
    assert likelihood.min() > 0 and likelihood.max() <= 1
    totals = likelihood.sum(0)
    assert torch.allclose(totals, torch.ones(3), atol=1e-3)


def test_noise_mode_perturbs_within_half_a_unit():
    model = FactorizedEntropyModel(channels=2)
    z = torch.randn(50, 2) * 5
    z_tilde = model.quantize(z, "noise")
    assert torch.all((z_tilde - z).abs() <= 0.5)


def test_round_and_symbols_are_consistent():
    model = FactorizedEntropyModel(channels=2)
    z = torch.randn(20, 2) * 3
    rounded = model.quantize(z, "round")
    symbols = model.quantize(z, "symbols")
    assert torch.allclose(model.dequantize(symbols), rounded)


def test_aux_loss_is_finite_and_differentiable():
    model = FactorizedEntropyModel(channels=2)
    loss = model.aux_loss()
    loss.backward()
    assert math.isfinite(loss.item())
    assert model.quantiles.grad is not None


def test_compress_decompress_round_trip_matches_rounding():
    torch.manual_seed(0)
    model = FactorizedEntropyModel(channels=4)
    model.update(force=True)
    z = torch.randn(30, 4) * 4
    z[0, 0] = 500.0  # forces an escape symbol
    string = model.compress(z)
    z_hat = model.decompress(string, 30)
    assert torch.allclose(z_hat, model.quantize(z, "round"))
    coded_bits = 8 * len(string)
    ideal = model.ideal_bits(z)
    assert coded_bits >= ideal * 0.95  # a coder cannot beat its own model by much
