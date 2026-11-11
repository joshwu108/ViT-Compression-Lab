"""Fully factorised learned prior (Ballé et al. 2018, Appendix 6.1).

Each latent channel gets its own non-parametric cumulative distribution built
from small monotone layers; positions share the channel's density. This is the
``EntropyBottleneck`` used by the Entropic Student paper (via CompressAI),
re-implemented in plain PyTorch because CompressAI is not installed here.

Channel-last convention: ``z`` is ``[..., C]``; internally ``[C, 1, L]``.

Training:   z_tilde = z + U(-1/2, 1/2)   (or straight-through rounding)
Inference:  z_q = round(z - median) + median per channel
Rate:       -log2 p(z) summed over positions and channels
Coding:     update() builds integer CDF tables; compress()/decompress() use rans.py.
"""

import math

import torch
from torch import nn

from . import rans
from .gdn import lower_bound


class FactorizedEntropyModel(nn.Module):
    def __init__(
        self,
        channels,
        filters=(3, 3, 3),
        init_scale=10.0,
        tail_mass=1e-9,
        likelihood_bound=1e-9,
    ):
        super().__init__()
        self.channels = int(channels)
        self.filters = tuple(int(f) for f in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)
        self.likelihood_bound = float(likelihood_bound)

        widths = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))
        self._matrices = nn.ParameterList()
        self._biases = nn.ParameterList()
        self._factors = nn.ParameterList()
        for i in range(len(self.filters) + 1):
            init = math.log(math.expm1(1 / scale / widths[i + 1]))
            matrix = torch.full((self.channels, widths[i + 1], widths[i]), init)
            self._matrices.append(nn.Parameter(matrix))
            bias = torch.empty(self.channels, widths[i + 1], 1).uniform_(-0.5, 0.5)
            self._biases.append(nn.Parameter(bias))
            if i < len(self.filters):
                self._factors.append(nn.Parameter(torch.zeros(self.channels, widths[i + 1], 1)))

        quantiles = torch.tensor([-self.init_scale, 0.0, self.init_scale])
        self.quantiles = nn.Parameter(quantiles.repeat(self.channels, 1, 1))
        target = math.log(2 / self.tail_mass - 1)
        self.register_buffer("target", torch.tensor([-target, 0.0, target]))

        # Integer tables for the range coder, filled by update().
        self.register_buffer("_quantized_cdf", torch.zeros(0, dtype=torch.int32))
        self.register_buffer("_offset", torch.zeros(0, dtype=torch.int32))
        self.register_buffer("_cdf_length", torch.zeros(0, dtype=torch.int32))

    # ------------------------------------------------------------- density

    def _logits_cumulative(self, inputs, stop_gradient):
        logits = inputs
        for i in range(len(self.filters) + 1):
            matrix = torch.nn.functional.softplus(self._matrices[i])
            bias = self._biases[i]
            if stop_gradient:
                matrix, bias = matrix.detach(), bias.detach()
            logits = torch.matmul(matrix, logits) + bias
            if i < len(self.filters):
                factor = torch.tanh(self._factors[i])
                if stop_gradient:
                    factor = factor.detach()
                logits = logits + factor * torch.tanh(logits)
        return logits

    def _likelihood(self, inputs):
        lower = self._logits_cumulative(inputs - 0.5, stop_gradient=False)
        upper = self._logits_cumulative(inputs + 0.5, stop_gradient=False)
        sign = -torch.sign(lower + upper).detach()
        likelihood = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
        return likelihood

    def medians(self):
        return self.quantiles[:, 0, 1].detach()

    def aux_loss(self):
        """Pushes the learned quantiles to the tail-mass targets (CompressAI)."""
        logits = self._logits_cumulative(self.quantiles, stop_gradient=True)
        return torch.abs(logits - self.target).sum()

    # ---------------------------------------------------------- quantisation

    def quantize(self, z, mode):
        """mode: 'noise' | 'ste' | 'round' | 'symbols'. Channel-last input."""
        if mode == "noise":
            return z + torch.empty_like(z).uniform_(-0.5, 0.5)
        medians = self.medians().to(z.dtype)
        rounded = torch.round(z - medians)
        if mode == "symbols":
            return rounded.to(torch.int32)
        if mode == "round":
            return rounded + medians
        if mode == "ste":
            return z + (rounded + medians - z).detach()
        raise ValueError(f"Unknown quantization mode {mode!r}")

    def dequantize(self, symbols):
        return symbols.to(self.medians().dtype) + self.medians()

    def forward(self, z, mode=None):
        """Returns (z_hat, likelihood) with the same channel-last shape as z."""
        if mode is None:
            mode = "noise" if self.training else "round"
        z_hat = self.quantize(z, mode)
        flat = z_hat.reshape(-1, self.channels).t().unsqueeze(1)  # [C, 1, L]
        likelihood = self._likelihood(flat)
        likelihood = lower_bound(likelihood, self.likelihood_bound)
        likelihood = likelihood.squeeze(1).t().reshape(z_hat.shape)
        return z_hat, likelihood

    def bits(self, likelihood):
        return -torch.log2(likelihood).sum()

    # ----------------------------------------------------------- coding

    @torch.no_grad()
    def update(self, force=False):
        """Build integer CDF tables from the current density. Returns True if built."""
        if self._offset.numel() > 0 and not force:
            return False
        medians = self.quantiles[:, 0, 1]
        minima = torch.clamp(torch.ceil(medians - self.quantiles[:, 0, 0]).int(), min=0)
        maxima = torch.clamp(torch.ceil(self.quantiles[:, 0, 2] - medians).int(), min=0)
        pmf_start = medians - minima
        pmf_length = maxima + minima + 1
        max_length = int(pmf_length.max().item())
        device = pmf_start.device
        samples = torch.arange(max_length, device=device)[None, None, :] + pmf_start[:, None, None]
        pmf = self._likelihood(samples)[:, 0, :]
        lower = self._logits_cumulative(samples[:, :, :1] - 0.5, stop_gradient=True)
        upper = self._logits_cumulative(samples[:, :, -1:] + 0.5, stop_gradient=True)
        tail = (torch.sigmoid(lower) + torch.sigmoid(-upper))[:, 0, 0]

        cdfs = []
        for c in range(self.channels):
            length = int(pmf_length[c].item())
            cdfs.append(rans.pmf_to_quantized_cdf(pmf[c, :length].tolist(), tail[c].item()))
        width = max(len(cdf) for cdf in cdfs)
        table = torch.zeros(self.channels, width, dtype=torch.int32)
        for c, cdf in enumerate(cdfs):
            table[c, : len(cdf)] = torch.tensor(cdf, dtype=torch.int32)
        self._quantized_cdf = table.to(device)
        self._cdf_length = (pmf_length + 2).to(torch.int32)
        self._offset = (-minima).to(torch.int32)
        return True

    def _tables(self):
        if self._offset.numel() == 0:
            raise RuntimeError("Call update() before compress()/decompress()")
        cdfs = self._quantized_cdf.cpu().tolist()
        lengths = self._cdf_length.cpu().tolist()
        offsets = self._offset.cpu().tolist()
        return cdfs, lengths, offsets

    def compress(self, z):
        """z: [L, C] float (channel-last). Returns one bytes object."""
        if z.dim() != 2 or z.shape[1] != self.channels:
            raise ValueError(f"Expected [L, {self.channels}] latents, got {tuple(z.shape)}")
        symbols = self.quantize(z, "symbols").cpu().reshape(-1).tolist()
        indexes = list(range(self.channels)) * z.shape[0]
        cdfs, lengths, offsets = self._tables()
        return rans.RansEncoder().encode_with_indexes(symbols, indexes, cdfs, lengths, offsets)

    def decompress(self, string, num_positions, device="cpu"):
        """Decode one string to [num_positions, C] floats on ``device`` (CPU by default,
        so the host-side decode and the device transfer can be timed separately)."""
        cdfs, lengths, offsets = self._tables()
        indexes = list(range(self.channels)) * int(num_positions)
        values = rans.RansDecoder().decode_with_indexes(string, indexes, cdfs, lengths, offsets)
        symbols = torch.tensor(values, dtype=torch.int32).reshape(int(num_positions), self.channels)
        medians = self.medians().detach().float().cpu()
        z_hat = symbols.float() + medians
        return z_hat.to(device) if device is not None else z_hat

    @torch.no_grad()
    def ideal_bits(self, z):
        """-log2 p of the rounded latents: the entropy coder's lower bound."""
        _, likelihood = self.forward(z, mode="round")
        return self.bits(likelihood).item()
