"""Dataset-level evaluation of a merger->merger student from the feature cache.

No Qwen model and no images are loaded: cached merger features go through the
student with rounded latents, one image at a time, and every image is actually
entropy coded (unless --coded-examples limits it).

    python encoders/Qwen3.8-vision-compression/scripts/es_eval_cached.py \
        data/es-abc-pilot5k/val_manifest.jsonl \
        --student-checkpoint runs/qwen3.8/es-abc-pilot5k/<run>/checkpoints/best.pt \
        --output outputs/qwen3.8/es-abc-pilot5k/evaluation/val500_best.json

Aggregation is stated per number:
  pooled     sums over all tokens of all images, divided once (FeatureErrorStats);
             pooled bits = total payload bits / total visual tokens
  per-image  one value per image, then the median over images
  loss       the training objective with rounded latents, mean over images
The training-mean predictor (every token = training channel mean) is reported
with the same pooled metrics as a reference.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from vit_compression_lab.entropic import payload as payload_lib  # noqa: E402
from vit_compression_lab.entropic.checkpoint import load_checkpoint, student_from_checkpoint  # noqa: E402
from vit_compression_lab.entropic.diagnostics import FeatureErrorStats  # noqa: E402
from vit_compression_lab.entropic.grid import counts_for_level, token_counts  # noqa: E402
from vit_compression_lab.entropic.losses import stage1_loss  # noqa: E402
from vit_compression_lab.entropic.qwen.data import FeatureCache, read_manifest  # noqa: E402
from vit_compression_lab.run_checkpoint import sha256_file, write_json  # noqa: E402


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, help="Default: the checkpoint config's data.feature_cache")
    parser.add_argument("--channel-stats", type=Path, help="Default: the checkpoint config's data.channel_stats")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--coded-examples", type=int, help="Entropy code only the first N images (default: all)")
    parser.add_argument("--diagnostic-channel", type=int, default=3994)
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = load_checkpoint(args.student_checkpoint, map_location=device)
    student, config = student_from_checkpoint(ckpt)
    student = student.to(device).eval()
    student.entropy.update(force=True)
    sp = config.split
    if (sp.encoder_input, sp.decoder_target) != ("merger", "merger"):
        raise SystemExit("es_eval_cached.py supports merger -> merger students only")
    cache = FeatureCache(args.feature_cache or config["data"]["feature_cache"])
    stats_path = args.channel_stats or config["data"]["channel_stats"]
    stats = torch.load(stats_path, weights_only=False)["merger"]
    mean, std = stats["mean"].to(device), stats["std"].to(device)
    rows = read_manifest(args.manifests, limit=args.limit)
    n_coded = len(rows) if args.coded_examples is None else args.coded_examples

    student_err, mean_err = FeatureErrorStats(mean, std), FeatureErrorStats(mean, std)
    records, coding_seconds, started = [], 0.0, time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for index, row in enumerate(rows):
        tensors = cache.load(row["id"], ["grid_thw", "merger"])
        grid, target = tensors["grid_thw"].to(device), tensors["merger"].to(device).float()
        n = sum(token_counts(grid))
        out = student(target, grid, quant_mode="round")
        out["deep"] = {}
        counts = {level: counts_for_level(grid, level) for level in ("patch", "token")}
        _, log = stage1_loss(out, {"merger": target}, counts, n, config, {"merger": std})
        student_err.add(target, out["recon"])
        mean_err.add(target, mean.expand_as(target))
        image = FeatureErrorStats(mean, std)
        image.add(target, out["recon"])
        per_image = image.result()
        record = {"id": row["id"], "tokens": n, "grid_thw": grid[0].tolist(),
                  "loss": float(log["loss"]), "distortion": float(log["distortion"]), "rate_nats": float(log["rate"]),
                  "estimated_bits": float(log["bits"]),
                  **{k: per_image[k] for k in ("standardized_mse", "fvu_standardized", "raw_mse", "relative_l2",
                                               "cosine", "cosine_centered", "cosine_standardized")}}
        if index < n_coded:
            tick = time.time()
            strings, latent_counts, _ = student.compress(target, grid)
            blob = payload_lib.pack(payload_lib.make_header(sp, grid.cpu(), latent_counts, student.latent_channels),
                                    strings)
            header, strings_back = payload_lib.unpack(blob)
            z_dec = torch.cat([student.entropy.decompress(s, k, device=device)
                               for s, k in zip(strings_back, header["latent_counts"])])
            recon_dec = student.decode(z_dec, grid)
            coding_seconds += time.time() - tick
            acct = payload_lib.accounting(blob, strings)
            record.update({"payload_bytes": len(blob), "string_bytes": acct["string_bytes"],
                           "container_overhead_bytes": acct["container_overhead_bytes"],
                           "latent_exact": torch.equal(out["z_hat"].to(z_dec.dtype), z_dec),
                           "recon_max_abs_diff": (recon_dec - out["recon"]).abs().max().item()})
        records.append(record)
        if (index + 1) % 50 == 0 or index + 1 == len(rows):
            print(f"{index + 1}/{len(rows)} images, {time.time() - started:.0f}s", flush=True)

    coded = [r for r in records if "payload_bytes" in r]
    tokens = sum(r["tokens"] for r in records)

    def median(key, items=records):
        return statistics.median(r[key] for r in items) if items else None

    summary = {
        "checkpoint": {"path": str(args.student_checkpoint), "sha256": sha256_file(args.student_checkpoint),
                       "step": ckpt["step"], "epoch": ckpt["epoch"], "name": config.name},
        "manifests": [{"path": str(p), "sha256": sha256_file(p)} for p in args.manifests],
        "channel_stats": {"path": str(stats_path), "sha256": sha256_file(stats_path)},
        "images": len(records), "tokens": tokens, "quantization": "round",
        "objective_mean_over_images": {k: sum(r[k] for r in records) / len(records)
                                       for k in ("loss", "distortion", "rate_nats")},
        "pooled": student_err.result(args.diagnostic_channel),
        "pooled_training_mean_predictor": mean_err.result(args.diagnostic_channel),
        "estimated_bits_per_visual_token_pooled": sum(r["estimated_bits"] for r in records) / tokens,
        "per_image_median": {k: median(k) for k in ("standardized_mse", "fvu_standardized", "raw_mse", "relative_l2",
                                                    "cosine", "cosine_standardized", "tokens")},
        "seconds": time.time() - started, "coding_seconds": coding_seconds,
        "gpu_peak_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }
    summary["per_image_median"]["estimated_bits_per_visual_token"] = statistics.median(
        r["estimated_bits"] / r["tokens"] for r in records)
    if coded:
        coded_tokens = sum(r["tokens"] for r in coded)
        payload_bits = 8 * sum(r["payload_bytes"] for r in coded)
        summary["coded"] = {
            "images": len(coded), "tokens": coded_tokens, "payload_bytes": payload_bits // 8,
            "bits_per_visual_token_pooled": payload_bits / coded_tokens,
            "string_bits_per_visual_token_pooled": 8 * sum(r["string_bytes"] for r in coded) / coded_tokens,
            "container_overhead_bits_per_visual_token_pooled":
                8 * sum(r["container_overhead_bytes"] for r in coded) / coded_tokens,
            "estimated_bits_per_visual_token_pooled_same_images":
                sum(r["estimated_bits"] for r in coded) / coded_tokens,
            "payload_over_estimated": payload_bits / sum(r["estimated_bits"] for r in coded),
            "bits_per_visual_token_per_image_median": statistics.median(
                8 * r["payload_bytes"] / r["tokens"] for r in coded),
            "all_latents_exact": all(r["latent_exact"] for r in coded),
            "recon_max_abs_diff": max(r["recon_max_abs_diff"] for r in coded),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {"summary": summary, "records": records})
    print(json.dumps(summary, indent=2))
    if coded and not summary["coded"]["all_latents_exact"]:
        raise SystemExit("entropy round trip failed")


if __name__ == "__main__":
    main()
