"""Experiment table and convergence check for step-mode stage-1 runs (reads metrics.jsonl only).

    python encoders/Qwen3.8-vision-compression/scripts/es_recon_study_summary.py \
        runs/qwen3.8/es-recon-study/* --output-prefix outputs/qwen3.8/es-recon-study/experiment_table

Writes <prefix>.json and <prefix>.csv with one row per run: the last rounded
evaluation of the fixed training subset and of the validation set (pooled,
dataset-level metrics), plus the convergence verdict.

Convergence rule: compare the best training-set loss among the evaluations in
the last ``--tail`` fraction of the updates with the best one before it. A
relative improvement above ``--threshold`` means "still improving", otherwise
"plateau". "unstable" = the run aborted, or its last loss exceeds 1.5x its best.
"""

import argparse
import csv
import json
from pathlib import Path

COLUMNS = ["loss", "standardized_mse", "fvu_standardized", "cosine_standardized", "raw_mse", "cosine",
           "relative_l2", "fvu_raw", "estimated_bits_per_visual_token"]


def read_run(path):
    records = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines() if line.strip()]
    return ([r for r in records if r["phase"] == "eval"], [r for r in records if r["phase"] == "abort"])


def convergence(evals, tail, threshold):
    first, last = evals[0]["step"], evals[-1]["step"]
    cut = last - tail * (last - first)
    before = [e["train"]["loss"] for e in evals if e["step"] <= cut]
    after = [e["train"]["loss"] for e in evals if e["step"] > cut]
    if not before or not after:
        return {"verdict": "too few evaluations"}
    gain = (min(before) - min(after)) / min(before)
    best = min(before + after)
    verdict = "still improving" if gain > threshold else "plateau"
    if evals[-1]["train"]["loss"] > 1.5 * best:
        verdict = "unstable"
    return {"verdict": verdict, "tail_starts_after_step": cut, "best_before_tail": min(before),
            "best_in_tail": min(after), "relative_improvement_in_tail": gain}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--tail", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--channel", type=int, default=3994)
    args = parser.parse_args()

    rows = []
    for path in args.runs:
        if not (path / "metrics.jsonl").exists():
            continue
        evals, aborts = read_run(path)
        if not evals:
            continue
        config = json.loads((path / "config.json").read_text())
        info = json.loads((path / "run_info.json").read_text())
        last = evals[-1]
        check = convergence(evals, args.tail, args.threshold)
        if aborts:
            check["verdict"] = f"aborted: {aborts[-1]['reason']}"
        row = {
            "run": path.name, "updates": last["step"], "seed": config["training"]["seed"],
            "beta": config["rate"]["beta"], "latent_channels": config["encoder"]["latent_channels"],
            "decoder_output": config["decoder"]["output"], "bias_init": config["decoder"]["bias_init"],
            "quantization_train": config["quantization"]["train"], "quantization_eval": last["quantization"],
            "init_checkpoint": (info.get("init_checkpoint") or {}).get("path"),
            "convergence": check["verdict"],
            "tail_improvement": check.get("relative_improvement_in_tail"),
        }
        for split in ("train", "val"):
            ev = last.get(split)
            if ev is None:
                continue
            for key in COLUMNS:
                row[f"{split}/{key}"] = ev[key]
            ch = f"ch{args.channel}"
            row[f"{split}/{ch}_mse"] = ev[ch]["mse"]
            row[f"{split}/{ch}_fvu"] = ev[ch]["fvu"]
            row[f"{split}/{ch}_error_share"] = ev[ch]["error_share"]
            row[f"{split}/other_channels_mse"] = ev["other_channels"]["mse"]
            row[f"{split}/other_channels_fvu_raw"] = ev["other_channels"]["fvu_raw"]
            coded = ev.get("coded", {})
            row[f"{split}/coded_bits_per_visual_token"] = coded.get("bits_per_visual_token")
            row[f"{split}/roundtrip_exact"] = coded.get("roundtrip_exact")
        row["all_roundtrips_exact"] = all(e[s]["coded"]["roundtrip_exact"] for e in evals for s in ("train", "val")
                                          if "coded" in e.get(s, {}))
        row["paths"] = {"config": str(path / "config.json"), "metrics": str(path / "metrics.jsonl"),
                        "latest": str(path / "checkpoints/latest.pt"), "best": str(path / "checkpoints/best.pt")}
        rows.append(row)
        print(f"{row['run']:42} upd {row['updates']:6} {row['convergence']:16} "
              f"train std-mse {row['train/standardized_mse']:.4f} relL2 {row['train/relative_l2']:.3f} | "
              f"val std-mse {row.get('val/standardized_mse', float('nan')):.4f} "
              f"relL2 {row.get('val/relative_l2', float('nan')):.3f} "
              f"bits {row.get('val/coded_bits_per_visual_token') or float('nan'):.0f}")

    if args.output_prefix:
        args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
        rule = {"tail": args.tail, "threshold": args.threshold, "metric": "train loss, rounded, dataset-level"}
        args.output_prefix.with_suffix(".json").write_text(
            json.dumps({"convergence_rule": rule, "rows": rows}, indent=2) + "\n")
        flat = [{k: v for k, v in r.items() if k != "paths"} for r in rows]
        fields = list(dict.fromkeys(k for r in flat for k in r))
        with open(args.output_prefix.with_suffix(".csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat)


if __name__ == "__main__":
    main()
