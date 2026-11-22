"""Dataset-level reconstruction diagnostics for the Entropic Student.

``FeatureErrorStats`` adds up squared errors over every token it is given and
only divides at the end, so the result is pooled over the whole dataset and
does not depend on how tokens were batched.

Raw MSE and cosine are dominated by the offset every token shares (merger
channel 3994 in particular): a predictor that outputs the training mean for
every token already scores well on them. The centered and standardized
numbers remove that offset using TRAINING-set channel mean/std:

    fvu_raw           sum err^2 / sum (h - mean)^2             energy weighted
    fvu_standardized  sum (err/std)^2 / sum ((h - mean)/std)^2  every channel equal

FVU = fraction of variance unexplained: 0 is perfect, 1 is what the
training-mean predictor gets on the training set, above 1 is worse than it.
"""

import torch
import torch.nn.functional as F


class FeatureErrorStats:
    def __init__(self, mean, std):
        self.mean = mean.double()
        self.std = std.double().clamp_min(1e-6)
        zeros = torch.zeros_like(self.mean)
        self.err_sq, self.tgt_sq, self.ctr_sq = zeros.clone(), zeros.clone(), zeros.clone()
        self.tokens = 0
        self.cos = self.cos_centered = self.cos_standardized = 0.0

    @torch.no_grad()
    def add(self, target, recon):
        """target, recon: [tokens, channels] in the teacher's feature scale."""
        t, r = target.double(), recon.double()
        self.tokens += t.shape[0]
        self.err_sq += (r - t).pow(2).sum(0)
        self.tgt_sq += t.pow(2).sum(0)
        self.ctr_sq += (t - self.mean).pow(2).sum(0)
        tc, rc = t - self.mean, r - self.mean
        self.cos += F.cosine_similarity(t, r, dim=-1).sum().item()
        self.cos_centered += F.cosine_similarity(tc, rc, dim=-1).sum().item()
        self.cos_standardized += F.cosine_similarity(tc / self.std, rc / self.std, dim=-1).sum().item()

    def result(self, channel=None):
        """Pooled metrics. ``channel`` splits the error into that channel and all the others."""
        n, c = self.tokens, self.err_sq.numel()
        var = self.std.pow(2)
        err, ctr = self.err_sq.sum().item(), self.ctr_sq.sum().item()
        out = {
            "tokens": n,
            "raw_mse": err / (n * c),
            "relative_l2": (err / self.tgt_sq.sum().item()) ** 0.5,
            "cosine": self.cos / n,
            "standardized_mse": (self.err_sq / var).sum().item() / (n * c),
            "fvu_raw": err / ctr,
            "fvu_standardized": (self.err_sq / var).sum().item() / (self.ctr_sq / var).sum().item(),
            "cosine_centered": self.cos_centered / n,
            "cosine_standardized": self.cos_standardized / n,
        }
        if channel is not None:
            e, k = self.err_sq[channel].item(), self.ctr_sq[channel].item()
            out[f"ch{channel}"] = {
                "mse": e / n, "error_share": e / err, "fvu": e / k,
                "target_energy_share": self.tgt_sq[channel].item() / self.tgt_sq.sum().item(),
            }
            out["other_channels"] = {"mse": (err - e) / (n * (c - 1)), "fvu_raw": (err - e) / (ctr - k)}
        return out

    def per_channel(self):
        """Per-channel arrays (CPU float tensors) for plots and tables."""
        n = self.tokens
        return {"mse": (self.err_sq / n).float().cpu(), "target_energy": (self.tgt_sq / n).float().cpu(),
                "fvu": (self.err_sq / self.ctr_sq.clamp_min(1e-12)).float().cpu()}
