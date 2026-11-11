"""Generalized Divisive Normalization over the channel (last) dimension.

Ballé, Laparra & Simoncelli (2016):  y_i = x_i / sqrt(beta_i + sum_j gamma_ij x_j^2).
The inverse form (IGDN) multiplies instead of dividing and is used in decoders.
Parameters use the square-root reparametrisation with a lower bound whose
gradient passes through when it would move the value back inside the bound,
as in CompressAI. The layer acts on tensors of shape [..., C], which suits both
packed token sequences and flattened grids.
"""

import torch
from torch import nn


class _LowerBound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return torch.max(x, bound)

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        pass_through = (x >= bound) | (grad_output < 0)
        return pass_through.type_as(grad_output) * grad_output, None


def lower_bound(x, bound):
    bound = torch.as_tensor(bound, dtype=x.dtype, device=x.device)
    return _LowerBound.apply(x, bound)


class NonNegativeParametrizer(nn.Module):
    """value = reparam^2 - pedestal, with reparam >= sqrt(minimum + pedestal)."""

    def __init__(self, minimum=0.0, reparam_offset=2**-18):
        super().__init__()
        self.pedestal = reparam_offset**2
        self.bound = (minimum + self.pedestal) ** 0.5

    def init(self, value):
        return torch.sqrt(torch.clamp(value + self.pedestal, min=self.pedestal))

    def forward(self, reparam):
        return lower_bound(reparam, self.bound) ** 2 - self.pedestal


class GDN(nn.Module):
    def __init__(self, channels, inverse=False, beta_min=1e-6, gamma_init=0.1):
        super().__init__()
        self.inverse = inverse
        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        self.gamma_reparam = NonNegativeParametrizer()
        self.beta = nn.Parameter(self.beta_reparam.init(torch.ones(channels)))
        self.gamma = nn.Parameter(
            self.gamma_reparam.init(gamma_init * torch.eye(channels))
        )

    def forward(self, x):
        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        norm = torch.nn.functional.linear(x.pow(2), gamma, beta)
        norm = torch.sqrt(norm) if self.inverse else torch.rsqrt(norm)
        return x * norm

    def extra_repr(self):
        return f"channels={self.beta.numel()}, inverse={self.inverse}"
