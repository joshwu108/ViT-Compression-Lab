import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
)
from PIL import Image

checkpoint = "Qwen/Qwen3.8-27B"

config = AutoConfig.from_pretrained(checkpoint)
print(f"config:{config}")
processor = AutoProcessor.from_pretrained(checkpoint)

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    checkpoint,
    dtype=torch.bfloat16,
    device_map="auto",
)

model.eval()
vision = model.model.visual
image = Image.open("src/images/test1.jpg")
inputs = processor.image_processor(
    images=image,
    return_tensors="pt",
)

# find numeric + type of patch projections
device = vision.patch_embed.proj.weight.device
dtype = vision.patch_embed.proj.weight.dtype

pixel_values = inputs["pixel_values"].to(
    device=device,
    dtype=dtype,
)
grid_thw = inputs["image_grid_thw"].to(device=device)

print("Image grid [time, height, width]:", inputs["image_grid_thw"].tolist())

warmup_runs = 5
measurement_runs = 1000
latencies_ms = []

#profiling the timing through the vision transformer blocks
with torch.cuda.device(device), torch.inference_mode():

    # warm up the gpu
    for _ in range(warmup_runs):
        output = vision(pixel_values, grid_thw=grid_thw)

    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(measurement_runs):
        start.record()
        output = vision(pixel_values, grid_thw=grid_thw)
        end.record()
        end.synchronize()
        latencies_ms.append(start.elapsed_time(end))


latencies = torch.tensor(latencies_ms, dtype=torch.float64, device="cpu")
median_ms, p95_ms, p99_ms = torch.quantile(
    latencies, torch.tensor([0.5, 0.95, 0.99], dtype=torch.float64)
).tolist()

features = output.pooler_output

print(f"Encoder latency ({measurement_runs} runs, {warmup_runs} warmups):")
print(f"  Mean:   {latencies.mean().item():.3f} ms")
print(f"  Median: {median_ms:.3f} ms")
print(f"  p95:    {p95_ms:.3f} ms")
print(f"  p99:    {p99_ms:.3f} ms")
print("Feature shape:", features.shape)
print("Feature dtype:", features.dtype)
print("Feature bytes:", features.numel() * features.element_size())