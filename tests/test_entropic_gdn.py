import torch

from vit_compression_lab.entropic.gdn import GDN, lower_bound


def test_gdn_and_inverse_are_mutual_inverses_at_init():
    torch.manual_seed(0)
    x = torch.randn(10, 6)
    gdn, igdn = GDN(6), GDN(6, inverse=True)
    y = gdn(x)
    # With identical parameters igdn(gdn(x)) is not exactly x, but scales must be reciprocal.
    scale_forward = (y / x).abs()
    scale_inverse = (igdn(x) / x).abs()
    assert torch.allclose(scale_forward * scale_inverse, torch.ones_like(x), atol=1e-5)


def test_lower_bound_gradient_passes_when_pushing_inside():
    x = torch.tensor([0.0, 2.0], requires_grad=True)
    y = lower_bound(x, 1.0)
    y.sum().backward()
    # Below the bound, a gradient that would increase x (negative grad_output means decrease)...
    assert x.grad.tolist() == [0.0, 1.0]
    x2 = torch.tensor([0.0], requires_grad=True)
    (-lower_bound(x2, 1.0)).sum().backward()
    assert x2.grad.tolist() == [-1.0]  # grad_output < 0 passes through
