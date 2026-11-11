"""Cache frozen-teacher activations at chosen split points (vision tower only).

Loads just the Qwen3.5 vision tower from the local checkpoint cache (no LLM,
no download), runs each manifest image once, and stores pixel values, grid and
the requested activations as one safetensors file per example. Also
accumulates per-channel mean/std per point for the standardised distortion.

    python encoders/Qwen3.8-vision-compression/scripts/es_extract_teacher_features.py \
        data/train/manifest.jsonl --points block_16 merger \
        --cache-dir outputs/qwen3.8/teacher-features/train

Points: pixels, block_0 (stem) .. block_27, merger. Activations are stored in
the teacher dtype (BF16). Nothing is downloaded.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from vit_compression_lab.entropic.checkpoint import hardware_info, software_info  # noqa: E402
from vit_compression_lab.entropic.config import DEFAULTS  # noqa: E402
from vit_compression_lab.entropic.qwen.data import (  # noqa: E402
    FeatureCache,
    load_processor,
    preprocess_image,
    read_manifest,
)
from vit_compression_lab.entropic.qwen.vision_tower import TeacherSplit, load_vision_tower  # noqa: E402
from vit_compression_lab.run_checkpoint import write_json  # noqa: E402


class ChannelStats:
    def __init__(self):
        self.sum, self.sumsq, self.count = {}, {}, {}

    def add(self, point, x):
        x = x.double()
        self.sum[point] = self.sum.get(point, 0) + x.sum(0)
        self.sumsq[point] = self.sumsq.get(point, 0) + x.pow(2).sum(0)
        self.count[point] = self.count.get(point, 0) + x.shape[0]

    def result(self):
        out = {}
        for point in self.sum:
            mean = self.sum[point] / self.count[point]
            var = self.sumsq[point] / self.count[point] - mean.pow(2)
            out[point] = {"mean": mean.float().cpu(), "std": var.clamp_min(0).sqrt().float().cpu(),
                          "count": int(self.count[point])}
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--points", nargs="+", default=["merger"])
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default=DEFAULTS["teacher"]["checkpoint"])
    parser.add_argument("--revision", default=DEFAULTS["teacher"]["revision"])
    parser.add_argument("--device", default="cuda:1", help="GPU 0 is shared on this host")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-pixels", action="store_true", help="Do not store pixel_values (saves ~7 MB/image)")
    parser.add_argument("--verify-depths", nargs="*", type=int, default=[],
                        help="Check suffix(prefix(d)) against the full forward on the first image")
    args = parser.parse_args()

    rows = read_manifest(args.manifests, limit=args.limit)
    processor = load_processor(args.checkpoint, args.revision)
    visual = load_vision_tower(args.checkpoint, args.revision, device=args.device)
    split = TeacherSplit(visual)
    for point in args.points:
        split.depth_of(point)  # validates names
    cache = FeatureCache(args.cache_dir)
    stats = ChannelStats()

    info = {
        "checkpoint": args.checkpoint, "revision": args.revision, "points": args.points,
        "manifests": [str(p) for p in args.manifests], "examples": len(rows),
        "teacher_dtype": str(split.dtype), "software": software_info(),
        "hardware": hardware_info(args.device), "started_at": time.time(), "verification": [],
    }
    done = 0
    for index, row in enumerate(rows, 1):
        if cache.has(row["id"]) and not args.overwrite:
            continue
        pixel_values, grid_thw = preprocess_image(processor, row["image_path"])
        pixels = pixel_values.to(args.device, split.dtype)
        grid = grid_thw.to(args.device)
        if index == 1 and args.verify_depths:
            for depth in args.verify_depths:
                info["verification"].append(split.verify(pixels, grid, depth))
            print("verification:", info["verification"])
        with torch.inference_mode():
            acts = split.collect(pixels, grid, args.points)
        tensors = {"grid_thw": grid_thw}
        if not args.no_pixels:
            tensors["pixel_values"] = pixels.cpu()
        for point, tensor in acts.items():
            tensors[point] = tensor.cpu()
            stats.add(point, tensor)
        cache.save(row["id"], tensors, {"id": row["id"], "image_path": row["image_path"]})
        done += 1
        if done % 50 == 0 or index == len(rows):
            print(f"{index}/{len(rows)} cached ({done} new)")

    if done:
        result = stats.result()
        torch.save(result, args.cache_dir / "channel_stats.pt")
        info["channel_stats"] = {p: {"count": v["count"], "std_max": v["std"].max().item(),
                                     "std_median": v["std"].median().item()} for p, v in result.items()}
    info["finished_at"] = time.time()
    write_json(args.cache_dir / "extraction_info.json", info)
    print(json.dumps({k: v for k, v in info.items() if k != "verification"}, indent=2, default=str))


if __name__ == "__main__":
    main()
