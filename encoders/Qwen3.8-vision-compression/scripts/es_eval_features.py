"""Feature-level evaluation of a trained Entropic Student without the LLM.

For each manifest image: reconstruction error at the decoder target and at the
merger (the LLM-facing features), ideal and actually coded bits per visual
token, entropy-coding overhead, exact round-trip check, and the same error and
size figures for the INT8/INT4 baseline codecs on the merger features.

    python encoders/Qwen3.8-vision-compression/scripts/es_eval_features.py \
        data/vqav2-pilot/subset.jsonl --limit 20 \
        --student-checkpoint runs/qwen3.8/entropic-student/<name>/checkpoints/best.pt \
        --output outputs/qwen3.8/entropic-student/<name>/features-eval.json

Vision tower only; nothing is downloaded.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from vit_compression_lab.codecs import CODEC_NAMES, get_codec  # noqa: E402
from vit_compression_lab.entropic import payload as payload_lib  # noqa: E402
from vit_compression_lab.entropic.checkpoint import load_checkpoint, student_from_checkpoint  # noqa: E402
from vit_compression_lab.entropic.config import MERGER  # noqa: E402
from vit_compression_lab.entropic.metrics import compression_metrics, reconstruction_metrics  # noqa: E402
from vit_compression_lab.entropic.qwen.data import load_processor, preprocess_image, read_manifest  # noqa: E402
from vit_compression_lab.entropic.qwen.vision_tower import TeacherSplit, load_vision_tower  # noqa: E402
from vit_compression_lab.int8_codec import raw_bytes  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--baseline-codecs", nargs="*", default=["int8", "int4-g64"], choices=CODEC_NAMES)
    parser.add_argument("--no-coding", action="store_true", help="Skip the slow Python rANS round trip")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = load_checkpoint(args.student_checkpoint, map_location=device)
    student, config = student_from_checkpoint(ckpt)
    student = student.to(device).eval()
    student.entropy.update(force=True)
    teacher = config["teacher"]
    processor = load_processor(teacher["checkpoint"], teacher["revision"])
    split = TeacherSplit(load_vision_tower(teacher["checkpoint"], teacher["revision"], device=device,
                                           dtype=getattr(torch, teacher["dtype"])))
    sp = config.split
    points = sorted({sp.encoder_input, sp.decoder_target, MERGER})
    rows = read_manifest(args.manifests, limit=args.limit)
    baselines = [get_codec(n) for n in args.baseline_codecs]

    records = []
    for row in rows:
        pixel_values, grid_thw = preprocess_image(processor, row["image_path"])
        grid = grid_thw.to(device)
        with torch.inference_mode():
            geo = split.geometry(grid)
            acts = split.collect(pixel_values.to(device, split.dtype), grid, points)
            h = acts[MERGER]
            u = acts[sp.encoder_input].float()
            out = student(u, grid, quant_mode="round")
            h_hat = split.suffix(out["recon"].to(split.dtype), grid, sp.target_depth(split.dims), geo)
            n_tokens = h.shape[0]
            ideal_bits = -torch.log2(out["likelihood"]).sum().item()
            record = {
                "id": row["id"], "grid_thw": grid_thw.tolist(), "visual_tokens": n_tokens,
                "target": reconstruction_metrics(acts[sp.decoder_target], out["recon"]),
                "merger": reconstruction_metrics(h, h_hat.to(h.dtype)),
                "ideal_bits_per_visual_token": ideal_bits / n_tokens,
                "latent_shape": [sum(out["latent_counts"]), student.latent_channels],
                "baselines": {},
            }
            if not args.no_coding:
                strings, counts, ideal = student.compress(u, grid)
                header = payload_lib.make_header(sp, grid.cpu(), counts, student.latent_channels)
                blob = payload_lib.pack(header, strings)
                z_round = student.entropy.quantize(student.encode(u, grid), "round").float().cpu()
                z_dec = torch.cat([student.entropy.decompress(s, n) for s, n in zip(strings, counts)])
                acct = payload_lib.accounting(blob, strings)
                record["coded"] = compression_metrics(len(blob), acct["container_overhead_bytes"], sum(ideal),
                                                      n_tokens, raw_bytes(h))
                record["coded"]["roundtrip_exact"] = torch.equal(z_round, z_dec)
            for codec in baselines:
                q, scale = codec.quantize(h)
                restored = codec.dequantize(q, scale, h.dtype)
                payload = codec.serialize(q.cpu(), scale.cpu(), h.dtype, list(h.shape), grid_thw.tolist())
                record["baselines"][codec.name] = {
                    "bits_per_visual_token": 8 * len(payload) / n_tokens,
                    "reconstruction": reconstruction_metrics(h, restored)}
        records.append(record)
        print(f"{row['id']}: merger relL2 {record['merger']['relative_l2']:.4f} "
              f"cos {record['merger']['cosine_mean']:.4f} ideal bits/tok {record['ideal_bits_per_visual_token']:.1f}")

    def med(path):
        values = []
        for r in records:
            v = r
            for key in path:
                v = v.get(key) if isinstance(v, dict) else None
                if v is None:
                    break
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                values.append(v)
        return statistics.median(values) if values else None

    summary = {
        "student": str(args.student_checkpoint), "split": sp.as_dict(), "images": len(records),
        "median": {
            "target_mse": med(("target", "mse")), "target_cosine": med(("target", "cosine_mean")),
            "merger_relative_l2": med(("merger", "relative_l2")), "merger_cosine": med(("merger", "cosine_mean")),
            "merger_mse": med(("merger", "mse")),
            "ideal_bits_per_visual_token": med(("ideal_bits_per_visual_token",)),
            "coded_bits_per_visual_token": med(("coded", "bits_per_visual_token")),
            "entropy_coding_overhead_ratio": med(("coded", "entropy_coding_overhead_ratio")),
            "baselines": {c.name: {"bits_per_visual_token": med(("baselines", c.name, "bits_per_visual_token")),
                                   "merger_relative_l2": med(("baselines", c.name, "reconstruction", "relative_l2"))}
                          for c in baselines},
        },
        "all_roundtrips_exact": all(r.get("coded", {}).get("roundtrip_exact", True) for r in records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "records": records}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
