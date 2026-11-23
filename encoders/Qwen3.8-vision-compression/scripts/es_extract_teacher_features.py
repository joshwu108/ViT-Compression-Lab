"""Cache frozen-teacher activations at chosen split points.

Loads just the Qwen3.5 vision tower from the local checkpoint cache, 
runs each manifest image once, and stores pixel values, grid and
the requested activations as one safetensors file per example.

    python encoders/Qwen3.8-vision-compression/scripts/es_extract_teacher_features.py \
        data/train/manifest.jsonl data/train/val_manifest.jsonl --points merger --no-pixels \
        --cache-dir outputs/qwen3.8/teacher-features/cache \
        --stats-manifests data/train/manifest.jsonl \
        --stats-output outputs/qwen3.8/teacher-features/stats/channel_stats.pt

Extraction is resumable: an existing entry is reused only if its stored
provenance (image hash, checkpoint revision, teacher dtype, preprocessing)
matches; anything else stops the run unless --replace-stale is given.

Per-channel mean/std (for the standardised distortion) are computed AFTER
caching, by streaming the completed cache over every id of --stats-manifests
(default: all manifests given). They therefore cover entries cached by earlier,
interrupted runs, and never include images outside those manifests: pass only
the training manifest so validation images cannot enter the statistics.
<stats-output>.json records the manifest hash and image/token counts.
Use --stats-only to compute statistics by running the teacher on every row
without storing activation files.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from vit_compression_lab.entropic.checkpoint import hardware_info, software_info  # noqa: E402
from vit_compression_lab.entropic.config import DEFAULTS, point_level  # noqa: E402
from vit_compression_lab.entropic.qwen.data import (  # noqa: E402
    FeatureCache,
    load_processor,
    preprocess_image,
    read_manifest,
)
from vit_compression_lab.entropic.qwen.vision_tower import (  # noqa: E402
    TeacherSplit,
    load_vision_tower,
)
from vit_compression_lab.run_checkpoint import append_record, sha256_file, write_json  # noqa: E402


class ChannelStats:
    """
    Calculates the mean and standard devition of each feature channel
    """
    def __init__(self):
        """
        self.sum - sum of observed values
        slef.sumsq - sum of squared values
        self.count - number of observed feature rows
        """
        self.sum, self.sumsq, self.count = {}, {}, {}

    def add(self, point, x):
        """add a tensor on to a specific key/point"""
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


def preprocessing_fingerprint(processor):
    """SHA-256 of the image processor's settings (None when no processor is available)."""
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return None
    return hashlib.sha256(json.dumps(image_processor.to_dict(), sort_keys=True, default=str).encode()).hexdigest()


def manifest_refs(paths):
    return [{"path": str(p), "sha256": sha256_file(p) if Path(p).is_file() else None} for p in paths]


def entry_provenance(row, checkpoint, revision, dtype, preprocessing):
    """What a cache entry was computed from; stored in the file and compared before reuse."""
    sha = (row.get("source") or {}).get("image_sha256")
    if sha is None and Path(row["image_path"]).is_file():
        sha = sha256_file(row["image_path"])
    return {"id": row["id"], "image_sha256": sha, "checkpoint": checkpoint, "revision": revision,
            "teacher_dtype": str(dtype), "preprocessing_sha256": preprocessing}


def entry_is_current(cache, provenance, points):
    """True if the cached file holds the points and was computed from the same inputs."""
    try:
        metadata, shapes = cache.describe(provenance["id"])
    except Exception:
        return False  # unreadable header
    return all(metadata.get(k) == v for k, v in provenance.items()) and all(
        k in shapes for k in ["grid_thw", *points])


def cache_channel_stats(cache, ids, points, device="cpu"):
    """Stream the cache over ``ids`` and return (ChannelStats, images, tokens per point).

    Every entry must exist, be finite and have the token count its grid implies.
    """
    stats, tokens = ChannelStats(), {p: 0 for p in points}
    for example_id in ids:
        tensors = cache.load(example_id, ["grid_thw", *points])
        t, h, w = tensors["grid_thw"][0].tolist()
        for point in points:
            x = tensors[point].to(device)
            expected = t * h * w // (4 if point_level(point) == "token" else 1)
            if x.shape[0] != expected:
                raise ValueError(f"{example_id}: {point} has {x.shape[0]} rows, grid implies {expected}")
            if not torch.isfinite(x).all():
                raise ValueError(f"{example_id}: non-finite values in {point}")
            stats.add(point, x)
            tokens[point] += x.shape[0]
    return stats, len(ids), tokens


def write_channel_stats(path, result, info):
    """channel_stats.pt ({point: mean/std/count}) plus a JSON sidecar describing its inputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(result, tmp)
    tmp.replace(path)
    info = {**info, "stats_file": str(path), "stats_sha256": sha256_file(path),
            "points": {p: {"tokens": v["count"], "channels": v["mean"].numel(),
                           "std_max": v["std"].max().item(), "std_median": v["std"].median().item(),
                           "std_min": v["std"].min().item()} for p, v in result.items()}}
    write_json(path.with_suffix(".json"), info)
    return info


def main():

    # set up parser + add arguments
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
    parser.add_argument("--stats-only", action="store_true",
                        help="Compute channel statistics over all manifest rows without storing features")
    parser.add_argument("--verify-depths", nargs="*", type=int, default=[],
                        help="Check suffix(prefix(d)) against the full forward on the first image")
    parser.add_argument("--replace-stale", action="store_true",
                        help="Re-extract cached entries whose provenance does not match (default: stop)")
    parser.add_argument("--stats-manifests", type=Path, nargs="+",
                        help="Manifests whose ids define the channel statistics (default: all manifests). "
                             "Pass the TRAINING manifest only.")
    parser.add_argument("--stats-output", type=Path, help="Default: <cache-dir>/channel_stats.pt")
    parser.add_argument("--no-stats", action="store_true", help="Only cache; leave statistics untouched")
    args = parser.parse_args()

    rows = read_manifest(args.manifests, limit=args.limit) # extract manifest data
    processor = load_processor(args.checkpoint, args.revision) #load model settings from transformers library
    preprocessing = preprocessing_fingerprint(processor)
    dtype = getattr(torch, DEFAULTS["teacher"]["dtype"])
    cache = FeatureCache(args.cache_dir)
    stats_output = args.stats_output or args.cache_dir / "channel_stats.pt"

    # Decide what still needs the teacher before loading it.
    provenance = {row["id"]: entry_provenance(row, args.checkpoint, args.revision, dtype, preprocessing)
                  for row in rows}
    pending, reused, stale = [], 0, []
    for row in rows:
        if args.stats_only or args.overwrite or not cache.has(row["id"]):
            pending.append(row)
        elif entry_is_current(cache, provenance[row["id"]], args.points):
            reused += 1
        else:
            stale.append(row["id"])
            pending.append(row)
    if stale and not args.replace_stale:
        raise SystemExit(f"{len(stale)} cached entries were computed from different inputs or lack "
                         f"provenance (first: {stale[0]}); pass --replace-stale to re-extract them")

    info = {
        "checkpoint": args.checkpoint, "revision": args.revision, "points": args.points,
        "manifests": manifest_refs(args.manifests),
        "examples": len(rows), "teacher_dtype": str(dtype), "preprocessing_sha256": preprocessing,
        "image_processor": processor.image_processor.to_dict() if preprocessing else None,
        "stored_pixels": not args.no_pixels, "software": software_info(),
        "started_at": time.time(), "verification": [], "stats_only": args.stats_only,
        "reused": reused, "replaced_stale": len(stale), "failures": [],
    }
    done = tokens = 0
    if pending:
        visual = load_vision_tower(args.checkpoint, args.revision, device=args.device, dtype=dtype) # fetch vision model
        split = TeacherSplit(visual)
        for point in args.points:
            split.depth_of(point)  # validates names
        info["hardware"] = hardware_info(args.device)
    stats = ChannelStats()
    for index, row in enumerate(pending, 1):
        try:
            pixel_values, grid_thw = preprocess_image(processor, row["image_path"])
            pixels = pixel_values.to(args.device, split.dtype)
            grid = grid_thw.to(args.device)
            if index == 1 and args.verify_depths:
                for depth in args.verify_depths:
                    info["verification"].append(split.verify(pixels, grid, depth))
                print("verification:", info["verification"])
            with torch.inference_mode():
                acts = split.collect(pixels, grid, args.points)
            if not all(torch.isfinite(tensor).all() for tensor in acts.values()):
                raise ValueError("non-finite teacher activations")
        except Exception as error:  # recorded; the image stays uncached and is retried on the next run
            info["failures"].append({"id": row["id"], "error": f"{type(error).__name__}: {error}"})
            print(f"FAILED {row['id']}: {type(error).__name__}: {error}")
            continue
        if args.stats_only:
            for point, tensor in acts.items():
                stats.add(point, tensor)
        else:
            tensors = {"grid_thw": grid_thw}
            if not args.no_pixels:
                tensors["pixel_values"] = pixels.cpu()
            tensors.update({point: tensor.cpu() for point, tensor in acts.items()})
            cache.save(row["id"], tensors, {**provenance[row["id"]], "image_path": row["image_path"]})
        done += 1
        tokens += next(iter(acts.values())).shape[0]
        if done % 50 == 0 or index == len(pending):
            action = "measured" if args.stats_only else "cached"
            rate = done / (time.time() - info["started_at"])
            print(f"{index}/{len(pending)} {action} ({reused} reused, {rate:.2f} images/s)", flush=True)
    info["extracted"], info["extracted_rows_of_first_point"] = done, tokens
    info["extraction_seconds"] = time.time() - info["started_at"]

    if info["failures"]:
        print(f"{len(info['failures'])} images failed; statistics not written")
    elif not args.no_stats:
        stats_manifests = args.stats_manifests or args.manifests
        stats_info = {
            "checkpoint": args.checkpoint, "revision": args.revision, "teacher_dtype": str(dtype),
            "preprocessing_sha256": preprocessing,
            "manifests": manifest_refs(stats_manifests),
            "source": "teacher forward over every row (--stats-only)" if args.stats_only
            else f"streamed from the completed cache {args.cache_dir}",
        }
        if args.stats_only:
            stats_info["images"] = done
        else:
            # Every id of the statistics manifests, read back from the cache (not just this run's images).
            ids = [row["id"] for row in read_manifest(stats_manifests, limit=args.limit)]
            stats, stats_info["images"], _ = cache_channel_stats(cache, ids, args.points, args.device)
            stats_info["ids_sha256"] = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        if stats_info["images"]:
            info["channel_stats"] = write_channel_stats(stats_output, stats.result(), stats_info)
    info["finished_at"] = time.time()

    # save the outputs from the feature extraction
    write_json(args.cache_dir / "extraction_info.json", info)
    append_record(args.cache_dir / "extraction_log.jsonl", {k: v for k, v in info.items()
                                                            if k not in ("verification", "image_processor")})
    print(json.dumps({k: v for k, v in info.items() if k not in ("verification", "image_processor")},
                     indent=2, default=str))
    if info["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
