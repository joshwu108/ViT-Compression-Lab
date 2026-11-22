import torch

from vit_compression_lab.entropic.diagnostics import FeatureErrorStats


def test_pooled_metrics_do_not_depend_on_batching_and_mean_predictor_has_fvu_one():
    torch.manual_seed(0)
    target = torch.randn(50, 6) * torch.tensor([1, 1, 1, 1, 1, 30.0]) + torch.tensor([0, 0, 0, 0, 0, 100.0])
    mean, std = target.mean(0), target.std(0, unbiased=False)
    recon = target + torch.randn_like(target) * 0.1

    whole = FeatureErrorStats(mean, std)
    whole.add(target, recon)
    parts = FeatureErrorStats(mean, std)
    for a, b in ((0, 7), (7, 50)):  # unequal batches
        parts.add(target[a:b], recon[a:b])
    r, q = whole.result(channel=5), parts.result(channel=5)
    assert abs(r["raw_mse"] - (recon - target).pow(2).mean().item()) < 1e-6
    assert all(abs(r[k] - q[k]) < 1e-9 for k in ("raw_mse", "relative_l2", "cosine", "fvu_standardized"))

    # The training-mean predictor looks good on raw cosine but explains no variance.
    baseline = FeatureErrorStats(mean, std)
    baseline.add(target, mean.expand_as(target))
    b = baseline.result(channel=5)
    assert b["cosine"] > 0.9
    assert abs(b["fvu_raw"] - 1) < 1e-6 and abs(b["fvu_standardized"] - 1) < 1e-6
    assert abs(b["standardized_mse"] - 1) < 1e-6
    assert b["ch5"]["error_share"] > 0.9
