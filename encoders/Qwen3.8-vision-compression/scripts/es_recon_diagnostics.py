"""Reconstruction diagnostics on cached merger features: students vs. simple references.

For the training and validation manifests of a stage-1 config, every method
below is scored with the same pooled, dataset-level metrics
(vit_compression_lab/entropic/diagnostics.py), including the mean-centered and
standardized ones that a content-free predictor cannot do well on:

    zero            outputs 0 for every token
    train_mean      outputs the training-set mean vector for every token
    pca-K           best rank-K linear reconstruction (fit on standardized TRAINING
                    features, float coefficients, no quantization, no bitrate):
                    a reference for what K numbers per token can carry linearly
    int8, int4, ... the repository's fixed codecs (bits = serialized payload)
    NAME=CKPT       a trained student, rounded latents, actual rANS coding

    python encoders/Qwen3.8-vision-compression/scripts/es_recon_diagnostics.py \
        --config runs/qwen3.8/es-recon-study/<run>/config.json \
        --students original=runs/qwen3.8/es-recon-study/<run>/checkpoints/latest.pt \
        --output outputs/qwen3.8/es-recon-study/diagnostics.json

Only merger -> merger students are supported (no teacher is loaded). Channel
statistics and the PCA basis come from the training manifest only.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from vit_compression_lab.codecs import CODEC_NAMES, get_codec  # noqa: E402
from vit_compression_lab.entropic import payload as payload_lib  # noqa: E402
from vit_compression_lab.entropic.checkpoint import load_checkpoint, student_from_checkpoint  # noqa: E402
from vit_compression_lab.entropic.config import MERGER, StudentConfig  # noqa: E402
from vit_compression_lab.entropic.diagnostics import FeatureErrorStats  # noqa: E402
from vit_compression_lab.entropic.qwen.data import FeatureCache, read_manifest  # noqa: E402


def channel_summary(stats, top):
    """Per-channel fraction of variance unexplained: spread over channels and the highest-energy ones."""
    per = stats.per_channel()
    q = torch.quantile(per["fvu"], torch.tensor([0.1, 0.5, 0.9]))
    return {"fvu_p10": q[0].item(), "fvu_median": q[1].item(), "fvu_p90": q[2].item(),
            "channels_with_fvu_below_0.5": int((per["fvu"] < 0.5).sum()),
            "top_energy_channels": [{"channel": c, "mse": per["mse"][c].item(), "fvu": per["fvu"][c].item()}
                                    for c in top]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="Stage-1 config naming manifests, cache, stats")
    parser.add_argument("--students", nargs="*", default=[], metavar="NAME=CHECKPOINT")
    parser.add_argument("--codecs", nargs="*", default=["int8", "int4", "int4-g64", "int4-g32"], choices=CODEC_NAMES)
    parser.add_argument("--pca-ranks", nargs="*", type=int, default=[256, 512, 1024])
    parser.add_argument("--channel", type=int, default=3994)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} exists; choose a new name")

    device = torch.device(args.device)
    config = StudentConfig.load(args.config)
    data = config["data"]
    cache = FeatureCache(data["feature_cache"])
    stats_file = torch.load(data["channel_stats"], weights_only=False)[MERGER]
    mean, std = stats_file["mean"].to(device), stats_file["std"].to(device).clamp_min(1e-6)
    splits = {"train": read_manifest(data["train_manifests"], limit=data["train_limit"]),
              "val": read_manifest(data["val_manifests"], limit=data["val_limit"])}
    assert not {r["id"] for r in splits["train"]} & {r["id"] for r in splits["val"]}, "train/val overlap"
    feats = {name: [cache.load(r["id"], ["merger", "grid_thw"]) for r in rows] for name, rows in splits.items()}

    train = torch.cat([f["merger"] for f in feats["train"]]).to(device).float()
    top = torch.topk(train.pow(2).mean(0), 5).indices.tolist()
    # PCA of standardized training features (rows of vt are the principal directions).
    _, _, vt = torch.linalg.svd((train - mean) / std, full_matrices=False)

    methods = {"zero": lambda h, grid: (torch.zeros_like(h), 0.0),
               "train_mean": lambda h, grid: (mean.expand_as(h), 0.0)}
    for k in args.pca_ranks:
        def pca(h, grid, basis=vt[:k]):
            coeff = ((h - mean) / std) @ basis.T
            return mean + std * (coeff @ basis), None  # float coefficients: no bitrate
        methods[f"pca-{k}"] = pca
    for name in args.codecs:
        def fixed(h, grid, codec=get_codec(name)):
            q, scale = codec.quantize(h)
            payload = codec.serialize(q.cpu(), scale.cpu(), h.dtype, list(h.shape), grid.tolist())
            return codec.dequantize(q, scale, h.dtype), 8 * len(payload)
        methods[name] = fixed
    roundtrips = {}
    for item in args.students:
        name, _, path = item.partition("=")
        student, cfg = student_from_checkpoint(load_checkpoint(path, map_location=device))
        if cfg.split.as_dict() != {"encoder_input": MERGER, "decoder_target": MERGER}:
            parser.error(f"{name}: only merger -> merger students are supported here")
        student = student.to(device).eval()
        student.entropy.update(force=True)
        roundtrips[name] = []

        def learned(h, grid, student=student, cfg=cfg, log=roundtrips[name]):
            u = h.float()
            out = student(u, grid)
            if cfg["quantization"]["eval"] == "none":
                return out["recon"], None  # float-latent diagnostic: not codable
            strings, counts, _ = student.compress(u, grid)
            blob = payload_lib.pack(payload_lib.make_header(cfg.split, grid.cpu(), counts, student.latent_channels),
                                    strings)
            decoded = student.decompress(strings, grid, device=h.device)
            log.append(torch.equal(decoded, out["recon"]))
            return decoded, 8 * len(blob)
        methods[name] = learned

    rows = []
    with torch.inference_mode():
        for method, fn in methods.items():
            for split_name, items in feats.items():
                stats, bits, tokens = FeatureErrorStats(mean, std), 0.0, 0
                for item in items:
                    h = item["merger"].to(device)  # teacher dtype, as the codecs expect
                    recon, n_bits = fn(h, item["grid_thw"].to(device))
                    stats.add(h.float(), recon.float())
                    bits = None if n_bits is None or bits is None else bits + n_bits
                    tokens += h.shape[0]
                rows.append({"method": method, "split": split_name, "images": len(items),
                             "bits_per_visual_token": None if bits is None else bits / tokens,
                             **stats.result(args.channel), "channels": channel_summary(stats, top)})
                r = rows[-1]
                print(f"{method:>28} {split_name:5} bits/tok {str(r['bits_per_visual_token'])[:8]:>8} "
                      f"relL2 {r['relative_l2']:.4f} cos {r['cosine']:.4f} "
                      f"fvu_std {r['fvu_standardized']:.4f} cos_std {r['cosine_standardized']:.4f}")

    result = {
        "aggregation": "pooled over all tokens of each split; bits = payload bits / visual tokens",
        "config": str(args.config), "channel_stats": data["channel_stats"], "top_energy_channels": top,
        "student_checkpoints": dict(item.split("=", 1) for item in args.students),
        "student_decode_equals_forward": {k: all(v) for k, v in roundtrips.items() if v},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
