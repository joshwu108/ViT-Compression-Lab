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
    # The bins tile [-60.5, 60.5], so their sum plus the mass outside that interval is exactly 1.
    # (At initialisation the density is wide: about 0.5% of the mass lies beyond +-60.)
    edges = torch.tensor([-60.5, 60.5]).repeat(3, 1, 1)  # [C, 1, 2]
    cdf = torch.sigmoid(model._logits_cumulative(edges, stop_gradient=True))[:, 0, :]
    tails = cdf[:, 0] + (1 - cdf[:, 1])
    totals = likelihood.sum(0)
    assert torch.allclose(totals + tails, torch.ones(3), atol=1e-5)
    assert torch.all(tails < 0.01)


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
