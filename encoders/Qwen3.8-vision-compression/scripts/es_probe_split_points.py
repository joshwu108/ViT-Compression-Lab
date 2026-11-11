"""Per-block statistics and noise sensitivity of the frozen vision tower.

Helps choose the split depth k before training anything. For a few images it
reports, per block: activation width, per-token RMS, the outlier ratio
max|x| / RMS (Experiment 3 found ~61x at the merger), the share of energy in
the largest channel, and the empirical entropy of uniformly quantised values
(bits/value at step = delta * RMS) as a training-free bitrate proxy. It then
injects Gaussian noise at each block, runs the frozen suffix to the merger and
measures how much the LLM-facing features move (relative L2, cosine).

    python encoders/Qwen3.8-vision-compression/scripts/es_probe_split_points.py \
        data/vqav2-pilot/subset.jsonl data/textvqa-pilot/subset.jsonl --limit 12 \
        --output outputs/qwen3.8/entropic-student/probe.json

Vision tower only; no LLM, no download.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from vit_compression_lab.entropic.config import DEFAULTS  # noqa: E402
from vit_compression_lab.entropic.metrics import reconstruction_metrics  # noqa: E402
from vit_compression_lab.entropic.qwen.data import load_processor, preprocess_image, read_manifest  # noqa: E402
from vit_compression_lab.entropic.qwen.vision_tower import TeacherSplit, load_vision_tower  # noqa: E402


def uniform_quantization_bits(x, delta):
    """Mean empirical entropy (bits/value) of round(x / (delta * rms)) per channel."""
    x = x.float()
    step = delta * x.pow(2).mean().sqrt().clamp_min(1e-8)
    q = torch.round(x / step)
    bits = []
    for c in range(0, q.shape[1], max(1, q.shape[1] // 64)):  # subsample channels for speed
        _, counts = torch.unique(q[:, c], return_counts=True)
        p = counts.float() / counts.sum()
        bits.append(-(p * p.log2()).sum().item())
    return statistics.mean(bits)


def activation_stats(x):
    x = x.float()
    rms = x.pow(2).mean().sqrt()
    channel_max = x.abs().amax(0)
    top = int(channel_max.argmax())
    energy = x.pow(2).sum(0)
    return {
        "tokens": x.shape[0], "width": x.shape[1], "rms": rms.item(),
        "max_abs": x.abs().max().item(), "outlier_ratio_max_over_rms": (x.abs().max() / rms).item(),
        "top_channel": top, "top_channel_energy_share": (energy[top] / energy.sum()).item(),
        "per_token_norm_cv": (x.norm(dim=1).std() / x.norm(dim=1).mean()).item(),
        "uniform_q_bits_delta0.5": uniform_quantization_bits(x, 0.5),
        "uniform_q_bits_delta1.0": uniform_quantization_bits(x, 1.0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--depths", default="0,4,8,12,16,20,24,27")
    parser.add_argument("--noise-levels", default="0.05,0.1,0.2,0.4")
    parser.add_argument("--checkpoint", default=DEFAULTS["teacher"]["checkpoint"])
    parser.add_argument("--revision", default=DEFAULTS["teacher"]["revision"])
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    depths = [int(d) for d in args.depths.split(",")]
    noise_levels = [float(s) for s in args.noise_levels.split(",")]
    rows = read_manifest(args.manifests, limit=args.limit)
    processor = load_processor(args.checkpoint, args.revision)
    split = TeacherSplit(load_vision_tower(args.checkpoint, args.revision, device=args.device))
    points = [f"block_{d}" for d in depths] + ["merger"]
    torch.manual_seed(args.seed)

    per_image = []
    for row in rows:
        pixel_values, grid_thw = preprocess_image(processor, row["image_path"])
        pixels = pixel_values.to(args.device, split.dtype)
        grid = grid_thw.to(args.device)
        with torch.inference_mode():
            geo = split.geometry(grid)
            acts = split.collect(pixels, grid, points)
            h = acts["merger"]
            record = {"id": row["id"], "grid_thw": grid_thw.tolist(), "points": {}}
            for point in points:
                entry = {"stats": activation_stats(acts[point])}
                if point != "merger":
                    depth = split.depth_of(point)
                    entry["noise_sensitivity"] = {}
                    a = acts[point]
                    rms = a.float().pow(2).mean().sqrt()
                    for sigma in noise_levels:
                        noisy = a + (sigma * rms * torch.randn_like(a.float())).to(a.dtype)
                        h_noisy = split.suffix(noisy, grid, depth, geo)
                        entry["noise_sensitivity"][str(sigma)] = reconstruction_metrics(h, h_noisy)
                record["points"][point] = entry
        per_image.append(record)
        print(f"probed {row['id']}")

    summary = {"checkpoint": args.checkpoint, "revision": args.revision, "images": len(per_image),
               "depths": depths, "noise_levels": noise_levels, "median_over_images": {}}
    for point in points:
        agg = {"stats": {}}
        for key in per_image[0]["points"][point]["stats"]:
            agg["stats"][key] = statistics.median(r["points"][point]["stats"][key] for r in per_image)
        if point != "merger":
            agg["noise_sensitivity"] = {}
            for sigma in noise_levels:
                agg["noise_sensitivity"][str(sigma)] = {
                    key: statistics.median(r["points"][point]["noise_sensitivity"][str(sigma)][key]
                                           for r in per_image)
                    for key in ("relative_l2", "cosine_mean", "cosine_min")}
        summary["median_over_images"][point] = agg

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "per_image": per_image}, indent=2) + "\n")
    print("| point | width | outlier max/rms | top-channel energy | bits@0.5rms | relL2 @ noise " +
          " / ".join(map(str, noise_levels)) + " |")
    print("| --- | ---: | ---: | ---: | ---: | --- |")
    for point in points:
        s = summary["median_over_images"][point]["stats"]
        ns = summary["median_over_images"][point].get("noise_sensitivity", {})
        rel = " / ".join(f"{ns[str(x)]['relative_l2']:.3f}" for x in noise_levels) if ns else "-"
        print(f"| {point} | {s['width']} | {s['outlier_ratio_max_over_rms']:.1f} | "
              f"{s['top_channel_energy_share']:.3f} | {s['uniform_q_bits_delta0.5']:.2f} | {rel} |")


if __name__ == "__main__":
    main()
