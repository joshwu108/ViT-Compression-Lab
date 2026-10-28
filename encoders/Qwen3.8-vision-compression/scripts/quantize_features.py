import torch

# load in saved tensor from experiment 1
features = torch.load(
    "outputs/features.pt",
    weights_only=True,
    map_location="cpu",
)

# input validation
if not isinstance(features, torch.Tensor):
    raise TypeError("Expected a tensor")

if features.ndim != 2 or features.numel() == 0:
    raise ValueError("Expected nonempty [tokens, feature_wdith] tensor")

if not features.is_floating_point():
    raise TypeError("Expected floating-point features")

if not torch.isfinite(features).all().item():
    raise ValueError("Features contain NaN or infinity")

# read output
raw_bytes = features.numel() * features.element_size()
print("Shape:", tuple(features.shape))
print("Dtype:", features.dtype)
print("Raw tensor bytes:", raw_bytes)



