import torch
import statistics
import time

torch.set_num_threads(16) # how many CPU threads can be use
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

x = features.float() # convert from BF16 to FP32 for calculations
max_abs = x.abs().amax(dim=1, keepdim=True)
scale = max_abs / 127 # creates scale mapping to INT8
scale = torch.where(max_abs == 0, torch.ones_like(scale), scale) #assign 1 to zero tokens

print("Calculation shape:", tuple(x.shape))
print("Scale shape:", tuple(scale.shape))

# Each token's scale broadcasts across its 5,120 feature values.
q = torch.round(x / scale)
q = q.clamp(-127, 127).to(torch.int8)

# Reconstruct in FP32, then restore the encoder's original dtype.
reconstructed_fp32 = q.float() * scale
reconstructed = reconstructed_fp32.to(features.dtype)

# Raw storage required for the INT8 values and FP32 scales.
quantized_bytes = (
    q.numel() * q.element_size()
    + scale.numel() * scale.element_size()
)

# Compare the original features with the restored-dtype reconstruction.
error = reconstructed.float() - x
mae = error.abs().mean().item()
max_error = error.abs().max().item()
mse = error.square().mean().item()

source_norm = torch.linalg.vector_norm(x).item()
error_norm = torch.linalg.vector_norm(error).item()
relative_l2 = error_norm / source_norm if source_norm > 0 else 0.0

assert q.shape == features.shape
assert q.dtype == torch.int8
assert scale.shape == (features.shape[0], 1)
assert torch.isfinite(reconstructed).all().item()

'''
print("Quantized shape:", tuple(q.shape))
print("Quantized dtype:", q.dtype)
print("INT8 values + scales bytes:", quantized_bytes)
print(
    "Raw compression ratio (original / quantized):",
    f"{raw_bytes / quantized_bytes:.3f}x",
)
print("Mean absolute error:", mae)
print("Maximum absolute error:", max_error)
print("Mean squared error:", mse)
print("Relative L2 error:", relative_l2)
'''

def quantize(features):
    x = features.float()
    max_abs = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(
        max_abs == 0,
        torch.ones_like(max_abs),
        max_abs / 127,
    )
    q = (x / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def dequantize(q, scale, dtype):
    return (q.float() * scale).to(dtype)


with torch.inference_mode():
    # Warm up before measuring.
    for _ in range(10):
        q, scale = quantize(features)
        restored = dequantize(q, scale, features.dtype)

    quantization_times = []
    reconstruction_times = []

    for _ in range(100):
        start = time.perf_counter()
        q, scale = quantize(features)
        quantization_times.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        restored = dequantize(q, scale, features.dtype)
        reconstruction_times.append((time.perf_counter() - start) * 1000)

print("CPU threads:", torch.get_num_threads())
print("Median quantization ms:", statistics.median(quantization_times))
print("Median reconstruction ms:", statistics.median(reconstruction_times))