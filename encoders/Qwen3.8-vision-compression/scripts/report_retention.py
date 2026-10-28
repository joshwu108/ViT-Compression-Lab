"""Retention of official VQA accuracy under INT8 compression, with bootstrap bounds.

    python encoders/Qwen3.8-vision-compression/scripts/report_retention.py \\
        outputs/qwen3.8/inject-eval/<run> [--replicates 10000] [--seed 20261028]

Reads <run>/scored.jsonl and <run>/summary.json written by score_results.py,
plus run_config.json and each manifest's metadata.json when present. Writes
<run>/retention.json (machine-readable) and <run>/report.md.

Primary metric: official VQA accuracy ("vqa_accuracy" in scored.jsonl).
retention = mean compressed score / mean original score. The decision uses a
paired bootstrap that resamples images (image_group) with replacement:
  PASS          5th percentile of retention >= threshold
  FAIL          95th percentile of retention < threshold
  INCONCLUSIVE  otherwise
  INCOMPLETE    the run has failed or missing examples; no PASS is declared
"""

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from vit_compression_lab.paired_stats import paired_bootstrap_retention  # noqa: E402

CAVEATS = [
    "Bootstrap bounds describe sampling uncertainty within the fixed evaluation "
    "set only. They do not account for how representative VQA v2 and TextVQA "
    "validation questions are of any deployment, nor for distribution shift.",
    "The evaluation size was fixed before the run. Examples were not added until "
    "the result passed; any expanded study should be planned and reported separately.",
    "The overall result applies to the fixed dataset mixture recorded under "
    "'mixture'; per-dataset and per-task results are reported separately.",
    "Both conditions use greedy decoding with short answers; longer or open-ended "
    "answers may be more sensitive to feature compression.",
    "Latency figures are local, in-process measurements, not network or serving "
    "benchmarks; see run_info.json for which timings were repeated.",
]


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def subset_stats(rows, args):
    stats = paired_bootstrap_retention(
        [r["original"]["vqa_accuracy"] for r in rows],
        [r["compressed"]["vqa_accuracy"] for r in rows],
        [r["image_group"] for r in rows],
        replicates=args.replicates, seed=args.seed, threshold=args.threshold)
    original_acc = statistics.mean(r["original"]["correct"] for r in rows)
    compressed_acc = statistics.mean(r["compressed"]["correct"] for r in rows)
    stats["secondary"] = {
        "accuracy": {"original": original_acc, "compressed": compressed_acc,
                     "difference_pp": 100 * (compressed_acc - original_acc)},
        "answers_changed": sum(r["answer_changed"] for r in rows),
        "original_correct_compressed_wrong": sum(
            r["original"]["correct"] and not r["compressed"]["correct"] for r in rows),
        "original_wrong_compressed_correct": sum(
            not r["original"]["correct"] and r["compressed"]["correct"] for r in rows),
    }
    cers = [(r["original"]["cer"], r["compressed"]["cer"]) for r in rows
            if r["original"]["cer"] is not None]
    if cers:
        stats["secondary"]["mean_cer"] = {"original": statistics.mean(c[0] for c in cers),
                                          "compressed": statistics.mean(c[1] for c in cers)}
    return stats


def manifest_metadata(run_dir):
    """metadata.json beside each manifest named in run_config.json, when present."""
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return {}
    found = {}
    for manifest in json.loads(config_path.read_text())["manifests"]:
        path = Path(manifest["path"]).parent / "metadata.json"
        if path.exists():
            metadata = json.loads(path.read_text())
            found[manifest["path"]] = {k: metadata.get(k) for k in (
                "allocation", "sampling", "exclusions", "manifest_sha256", "num_records",
                "created_at", "repo_commit")}
    return found


def build_retention(scored, summary, run_dir, args):
    if summary.get("complete") is None:
        raise SystemExit("summary.json lacks 'complete'; re-run score_results.py first")
    overall = subset_stats(scored, args)
    datasets = sorted({r["dataset"] for r in scored})
    by_dataset = {d: subset_stats([r for r in scored if r["dataset"] == d], args)
                  for d in datasets}
    tasks = sorted({(r["dataset"], r["task_type"]) for r in scored})
    by_task = {f"{d}/{t}": subset_stats(
        [r for r in scored if (r["dataset"], r["task_type"]) == (d, t)], args) for d, t in tasks}
    counts = {d: by_dataset[d]["n_examples"] for d in datasets}
    medians = summary["medians_across_examples"]
    return {
        "metric": "official VQA accuracy (vqa_accuracy in scored.jsonl)",
        "retention_definition": "mean compressed VQA score / mean original VQA score",
        "threshold": args.threshold,
        "decision": overall["decision"] if summary["complete"] else "INCOMPLETE",
        "decision_on_completed_examples": overall["decision"],
        "complete": summary["complete"],
        "n_scored": len(scored),
        "expected_examples": summary.get("expected_examples"),
        "failures": summary.get("failures", []),
        "mixture": {"counts": counts,
                    "fractions": {d: n / len(scored) for d, n in counts.items()},
                    "note": "Fixed before evaluation; the overall result applies to this "
                            "mixture only."},
        "overall": overall,
        "by_dataset": by_dataset,
        "by_task": by_task,
        "secondary_from_summary": {
            "unparseable": {c: summary[c]["unparseable"] for c in ("original", "compressed")},
            "hit_token_limit": {c: summary[c]["hit_token_limit"] for c in ("original", "compressed")},
            "identity_checks_passed": summary["identity_checks_passed"],
            "compressed_next_token_argmax_changed": summary["compressed_next_token_argmax_changed"],
            "median_sizes": medians["sizes"],
            "median_timing_ms": medians["timing_median_ms"],
            "median_generation_ms": medians["generation_ms"],
            "median_reconstruction": medians["reconstruction"],
            "timing_modes": summary.get("timing_modes"),
        },
        "manifest_metadata": manifest_metadata(run_dir),
        "run_info": summary["run_info"],
        "caveats": CAVEATS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ------------------------------------------------------------------ markdown

def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def fmt(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def stats_row(name, s):
    secondary = s["secondary"]
    return (f"| {cell(name)} | {s['n_examples']} | {fmt(s['original_mean'])} | "
            f"{fmt(s['compressed_mean'])} | {s['difference_pp']:+.2f} | {fmt(s['retention'])} | "
            f"{fmt(s['retention_lower_95_one_sided'])} | {fmt(s['retention_upper_95_one_sided'])} | "
            f"{s['decision']} | {100 * secondary['accuracy']['original']:.1f}% | "
            f"{100 * secondary['accuracy']['compressed']:.1f}% |")


def write_report(retention, scored, path, max_changed):
    info = retention["run_info"]
    overall = retention["overall"]
    lines = [
        "# Experiment 2 large evaluation: INT8 feature retention",
        "",
        f"**Decision: {retention['decision']}** (threshold {retention['threshold']}; "
        f"PASS needs the one-sided 95% lower bound of retention ≥ {retention['threshold']}).",
        "",
        f"Retention of official VQA accuracy: **{fmt(overall['retention'])}** "
        f"(one-sided 95% lower bound {fmt(overall['retention_lower_95_one_sided'])}, "
        f"upper bound {fmt(overall['retention_upper_95_one_sided'])}; two-sided 95% CI "
        f"{overall['retention_ci95_two_sided']}). Original {fmt(overall['original_mean'])}, "
        f"compressed {fmt(overall['compressed_mean'])}, difference "
        f"{overall['difference_pp']:+.2f} pp (95% CI {overall['difference_pp_ci95']}).",
        "",
    ]
    if not retention["complete"]:
        lines += [f"**The run is incomplete**: {retention['n_scored']} of "
                  f"{retention['expected_examples']} examples scored, "
                  f"{len(retention['failures'])} failed. No PASS is declared; the decision "
                  f"on completed examples alone would be "
                  f"{retention['decision_on_completed_examples']}.", ""]
    lines += [
        "## Setup", "",
        "| Item | Value |", "| --- | --- |",
        f"| Model | `{info['checkpoint']}` revision `{info['checkpoint_revision']}`, "
        f"{info['model_dtype']} |",
        f"| Software | Python {info['software']['python']}, torch {info['software']['torch']}, "
        f"transformers {info['software']['transformers']} |",
        f"| Hardware | {info['hardware']['gpu']} ({info['hardware']['device']}), "
        f"{info['hardware']['torch_cpu_threads']} torch CPU threads |",
        f"| Prompt | question + \"{info['prompt']['answer_instruction']}\", "
        f"enable_thinking={info['prompt']['enable_thinking']} |",
        f"| Generation | {cell(json.dumps(info['generation_config']))} |",
        f"| Timing collected | {cell(json.dumps(info.get('timing_collection')))} |",
        f"| Bootstrap | {overall['bootstrap']['replicates']} replicates, seed "
        f"{overall['bootstrap']['seed']}, clusters = image_group, numpy "
        f"{overall['bootstrap']['numpy']} |",
        "",
        "## Dataset mixture", "",
        "| Dataset | Questions | Fraction |", "| --- | ---: | ---: |",
    ]
    for dataset, n in retention["mixture"]["counts"].items():
        lines.append(f"| {dataset} | {n} | {100 * retention['mixture']['fractions'][dataset]:.1f}% |")
    lines += ["", retention["mixture"]["note"], ""]
    for manifest, metadata in retention["manifest_metadata"].items():
        lines.append(f"Manifest `{manifest}`: allocation {cell(json.dumps(metadata['allocation']))}; "
                     f"seed {metadata['sampling']['seed'] if metadata['sampling'] else 'n/a'}; "
                     f"sha256 `{metadata['manifest_sha256']}`.")
    header = ("| Subset | n | Original VQA | Compressed VQA | Diff pp | Retention | "
              "Lower 95% | Upper 95% | Decision | Orig. acc. | Comp. acc. |")
    divider = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |"
    lines += ["", "## Primary results (official VQA accuracy)", "", header, divider,
              stats_row("overall", overall)]
    lines += [stats_row(d, s) for d, s in retention["by_dataset"].items()]
    lines += [stats_row(t, s) for t, s in retention["by_task"].items()]
    lines += ["", "Per-subset decisions are informative only; the primary decision is the "
              "overall row. 'Orig. acc.' and 'Comp. acc.' are the pilot's binary primary rule.",
              ""]
    secondary = retention["secondary_from_summary"]
    sizes, timing = secondary["median_sizes"], secondary["median_timing_ms"]
    lines += [
        "## Secondary metrics", "",
        "| Metric | Original | Compressed |", "| --- | ---: | ---: |",
        f"| Answers changed | — | {overall['secondary']['answers_changed']} |",
        f"| Correct → wrong / wrong → correct | — | "
        f"{overall['secondary']['original_correct_compressed_wrong']} / "
        f"{overall['secondary']['original_wrong_compressed_correct']} |",
        f"| Unparseable answers | {secondary['unparseable']['original']} | "
        f"{secondary['unparseable']['compressed']} |",
        f"| Hit token limit | {secondary['hit_token_limit']['original']} | "
        f"{secondary['hit_token_limit']['compressed']} |",
    ]
    for dataset, s in retention["by_dataset"].items():
        if "mean_cer" in s["secondary"]:
            lines.append(f"| Mean CER ({dataset}) | {s['secondary']['mean_cer']['original']:.4f} | "
                         f"{s['secondary']['mean_cer']['compressed']:.4f} |")
    lines += [
        f"| Median generation ms | {secondary['median_generation_ms']['original']:.1f} | "
        f"{secondary['median_generation_ms']['compressed']:.1f} |",
        "",
        f"Identity checks passed: {secondary['identity_checks_passed']} of "
        f"{retention['n_scored']}. Prefill next-token argmax changed under compression for "
        f"{secondary['compressed_next_token_argmax_changed']} examples.",
        "",
        f"Median payload: original {sizes['original_raw_bytes']:,.0f} bytes raw "
        f"({sizes['original_serialized_bytes']:,.0f} serialized) vs. compressed "
        f"{sizes['compressed_raw_bytes']:,.0f} bytes raw "
        f"({sizes['compressed_serialized_bytes']:,.0f} serialized); raw ratio "
        f"{sizes['raw_ratio']:.3f}×, serialized ratio {sizes['serialized_ratio']:.3f}×. "
        f"Median relative L2 reconstruction error "
        f"{secondary['median_reconstruction']['relative_l2']:.4f}.",
        "",
        "Median per-stage latency (ms, medians across examples; timing modes "
        f"{secondary['timing_modes']}): " + ", ".join(f"{k} {v:.3f}" for k, v in timing.items()),
        "",
    ]
    if retention["failures"]:
        lines += ["## Failures", "", "| Id | Attempts | Last error |", "| --- | ---: | --- |"]
        lines += [f"| `{f['id']}` | {f['attempts']} | {cell(f['last_error_type'])}: "
                  f"{cell(f['last_message'])[:120]} |" for f in retention["failures"]]
        lines.append("")
    changed = [r for r in scored if r["answer_changed"]]
    lines += [f"## Changed answers ({len(changed)} of {len(scored)}"
              + (f", first {max_changed} shown" if len(changed) > max_changed else "") + ")", ""]
    if changed:
        lines += ["| Id | Question | Reference | Original (VQA) | Compressed (VQA) |",
                  "| --- | --- | --- | --- | --- |"]
        for r in changed[:max_changed]:
            lines.append(
                f"| `{r['id']}` | {cell(r['question'])[:80]} | {cell(r['expected_answer'])} | "
                f"{cell(r['original']['answer'])} ({r['original']['vqa_accuracy']:.1f}) | "
                f"{cell(r['compressed']['answer'])} ({r['compressed']['vqa_accuracy']:.1f}) |")
    lines += ["", "## Caveats", ""] + [f"- {c}" for c in CAVEATS] + [""]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--max-changed-examples", type=int, default=50)
    args = parser.parse_args()
    if args.replicates < 10_000:
        parser.error("Use at least 10,000 bootstrap replicates")

    scored = load_jsonl(args.run_dir / "scored.jsonl")
    summary = json.loads((args.run_dir / "summary.json").read_text())
    retention = build_retention(scored, summary, args.run_dir, args)
    (args.run_dir / "retention.json").write_text(json.dumps(retention, indent=2) + "\n")
    write_report(retention, scored, args.run_dir / "report.md", args.max_changed_examples)
    overall = retention["overall"]
    print(f"decision={retention['decision']} retention={fmt(overall['retention'])} "
          f"lower={fmt(overall['retention_lower_95_one_sided'])} "
          f"upper={fmt(overall['retention_upper_95_one_sided'])} n={retention['n_scored']} "
          f"complete={retention['complete']}")


if __name__ == "__main__":
    main()
