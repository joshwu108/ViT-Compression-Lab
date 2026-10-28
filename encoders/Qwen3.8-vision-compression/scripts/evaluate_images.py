"""Local image pilot: compare original and reconstructed visual features."""

import argparse
import io
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


def quantize(features):
    x = features.float()
    max_abs = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / 127)
    return (x / scale).round().clamp(-127, 127).to(torch.int8), scale


def reconstruct(payload):
    dtype = getattr(torch, payload["original_dtype"])
    return (payload["q"].float() * payload["scale"]).to(dtype)


def serialize(value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def timing_summary(values):
    return {
        "median_ms": statistics.median(values),
        "p95_ms": torch.tensor(values, dtype=torch.float64).quantile(0.95).item(),
    }


def measure_features(features, runs, warmups):
    # All compression and reconstruction timings below operate on CPU.
    for _ in range(warmups):
        q, scale = quantize(features)
        reconstruct({"q": q, "scale": scale, "original_dtype": str(features.dtype).split('.')[-1]})
        
    quantization_times, reconstruction_times = [], []
    dtype_name = str(features.dtype).split('.')[-1]
    for _ in range(runs):
        start = time.perf_counter()
        q, scale = quantize(features)
        quantization_times.append((time.perf_counter() - start) * 1000)
        payload = {"q": q, "scale": scale, "original_dtype": dtype_name}
        start = time.perf_counter()
        reconstruct(payload)
        reconstruction_times.append((time.perf_counter() - start) * 1000)

    # Build the payload explicitly after the timing loop so it is always defined.
    q, scale = quantize(features)
    payload = {"q": q, "scale": scale, "original_dtype": dtype_name}
    encoded = serialize(payload)
    loaded = torch.load(io.BytesIO(encoded), weights_only=True, map_location="cpu")
    restored = reconstruct(loaded)
    if not torch.equal(restored, reconstruct(payload)):
        raise ValueError("Serialized reconstruction differs from in-memory reconstruction")
    x = features.float()
    error = restored.float() - x
    norm = torch.linalg.vector_norm(x).item()
    original_bytes = features.numel() * features.element_size()
    compressed_bytes = q.numel() * q.element_size() + scale.numel() * scale.element_size()
    original_serialized = serialize(features)
    metrics = {
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "original_bytes": original_bytes,
        "int8_plus_scales_bytes": compressed_bytes,
        "raw_ratio_original_over_quantized": original_bytes / compressed_bytes,
        "original_serialized_bytes": len(original_serialized),
        "quantized_serialized_bytes": len(encoded),
        "serialized_ratio_original_over_quantized": len(original_serialized) / len(encoded),
        "mae": error.abs().mean().item(),
        "maximum_absolute_error": error.abs().max().item(),
        "mse": error.square().mean().item(),
        "relative_l2": torch.linalg.vector_norm(error).item() / norm if norm else 0.0,
        "cpu_quantization": timing_summary(quantization_times),
        "cpu_reconstruction": timing_summary(reconstruction_times),
    }
    return metrics, encoded, restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if min(args.threads, args.runs, args.max_new_tokens) < 1 or args.warmups < 0:
        parser.error("Threads, runs, and token limit must be positive; warmups nonnegative")
    rows = json.loads(args.manifest.read_text())
    if not isinstance(rows, list) or not rows:
        parser.error("Manifest must be a nonempty JSON list")
    for row in rows:
        for key in ("image", "question", "reference", "category"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                parser.error(f"Every entry needs a nonempty {key} string")
        row["image"] = str((args.manifest.resolve().parent / row["image"]).resolve())
        if not Path(row["image"]).is_file():
            parser.error(f"Image does not exist: {row['image']}")
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new experiment directory")
    torch.set_num_threads(args.threads)
    # Cached files only: this command never downloads model assets.
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, device_map="auto", local_files_only=True
    ).eval()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    visual = model.model.visual
    results = []
    for index, row in enumerate(rows):
        with Image.open(row["image"]) as source:
            image = source.convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": row["question"]},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        captured = []

        def capture(module, arguments, output):
            captured.append(output.pooler_output.detach().cpu())

        with torch.inference_mode():
            handle = visual.register_forward_hook(capture)
            try:
                baseline_ids = model.generate(
                    **inputs, do_sample=False, max_new_tokens=args.max_new_tokens
                )
            finally:
                handle.remove()
            if len(captured) != 1:
                raise ValueError("Expected one vision call for a single-image prompt")
            features = captured[0]
            if (features.ndim != 2 or features.numel() == 0
                    or not features.is_floating_point()
                    or not torch.isfinite(features).all().item()):
                raise ValueError("Invalid captured visual features")
            metrics, encoded, restored = measure_features(features, args.runs, args.warmups)
            replacements = []

            def replace(module, arguments, output):
                if output.pooler_output.shape != restored.shape:
                    raise ValueError("Replacement feature shape mismatch")
                output.pooler_output = restored.to(
                    device=output.pooler_output.device, dtype=output.pooler_output.dtype
                )
                replacements.append(True)
                return output

            handle = visual.register_forward_hook(replace)
            try:
                compressed_ids = model.generate(
                    **inputs, do_sample=False, max_new_tokens=args.max_new_tokens
                )
            finally:
                handle.remove()
            if len(replacements) != 1:
                raise ValueError("Expected exactly one feature replacement")
        prompt_length = inputs["input_ids"].shape[1]
        grid = inputs["image_grid_thw"].cpu().tolist()
        patch_size = model.config.vision_config.patch_size
        record = {
            **row, **metrics,
            "original_resolution_wh": list(image.size),
            "image_grid_thw": grid,
            "processed_resolution_hw": [grid[0][1] * patch_size, grid[0][2] * patch_size],
            "baseline_answer": processor.batch_decode(
                baseline_ids[:, prompt_length:], skip_special_tokens=True
            )[0],
            "reconstructed_answer": processor.batch_decode(
                compressed_ids[:, prompt_length:], skip_special_tokens=True
            )[0],
            "baseline_correct": None,
            "reconstructed_correct": None,
        }
        folder = args.output_dir / f"image-{index:03d}"
        folder.mkdir()
        torch.save(features, folder / "features.pt")
        (folder / "quantized.pt").write_bytes(encoded)
        torch.save(restored, folder / "reconstructed.pt")
        (folder / "report.json").write_text(json.dumps(record, indent=2) + "\n")
        results.append(record)
        print(f"{index + 1}/{len(rows)}: {Path(row['image']).name}, tokens={features.shape[0]}")
        # Save partial progress after each image.
        (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    groups = {}
    for result in results:
        groups.setdefault(str(result["shape"][0]), []).append(result)
    summary = {
        "settings": {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()},
        "torch_version": torch.__version__,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "processor_settings": processor.image_processor.to_dict(),
        "cpu": platform.processor(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "timing_scope": "CPU quantization/reconstruction only; excludes serialization, GPU copies, encoder, generation, network",
        "quality_scoring": "Review paired answers against references; null correctness fields are unscored",
        "by_token_count": {
            count: {
                "images": len(group),
                "mean_relative_l2": statistics.mean(r["relative_l2"] for r in group),
                "median_image_quantization_median_ms": statistics.median(
                    r["cpu_quantization"]["median_ms"] for r in group
                ),
            } for count, group in groups.items()
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
