"""Score eval_mmstar.py results using MMStar's official answer-prefix rule.

Source: MMStar-Benchmark/MMStar, eval/vlmeval/evaluate/mmstar.py.
This is multiple-choice accuracy, not VQAv2/TextVQA annotator consensus.
Partial evaluations are labelled explicitly; failed questions are never
silently counted as a completed 1,500-question benchmark.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from vit_compression_lab.paired_stats import paired_bootstrap_retention  # noqa: E402
from vit_compression_lab.run_checkpoint import write_json  # noqa: E402


def option_letter(prediction):
    """Mirror the official evaluator, including its permissive first-letter rule."""
    text = str(prediction).lower().strip().replace("\n", " ")
    if not text:
        return None
    if text[0] in "abcd":
        return text[0].upper()
    for prefix in ("(", "option ", "the answer is "):
        if text.startswith(prefix) and len(text) > len(prefix):
            letter = text[len(prefix)]
            return letter.upper() if letter in "abcd" else None
    return None


def accuracy(rows, condition):
    def group(key):
        return {
            name: {
                "n": len(selected),
                "accuracy": statistics.mean(
                    row[condition]["correct"] for row in selected
                ),
            }
            for name in sorted({row[key] for row in rows})
            if (selected := [row for row in rows if row[key] == name])
        }

    return {
        "n": len(rows),
        "accuracy": statistics.mean(row[condition]["correct"] for row in rows),
        "unparseable": sum(row[condition]["option"] is None for row in rows),
        "hit_token_limit": sum(row[condition]["hit_token_limit"] for row in rows),
        "by_category": group("category"),
        "by_l2_category": group("l2_category"),
    }


def score_run(run_dir):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "run_config.json").read_text())
    records = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("No completed results to score")
    ids = [row["id"] for row in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate result ids")
    expected = set(config["examples"]["ids"])
    if set(ids) - expected:
        raise ValueError(
            "Results contain ids outside the configured evaluation selection"
        )
    pipeline = config["pipeline"]
    if pipeline not in ("b", "c", "teacher"):
        raise ValueError(f"Unknown pipeline {pipeline!r}")
    conditions = ("original",) if pipeline == "teacher" else ("original", "compressed")
    scored = []
    for record in records:
        row = {
            key: record[key]
            for key in (
                "id",
                "image_group",
                "category",
                "l2_category",
                "expected_answer",
            )
        }
        for condition in conditions:
            result = record[condition]
            option = option_letter(result["answer"])
            row[condition] = {
                "answer": result["answer"],
                "option": option,
                "correct": option == record["expected_answer"].upper(),
                "hit_token_limit": result["generation"]["hit_token_limit"],
            }
        scored.append(row)
    complete = set(ids) == expected
    missing = sorted(expected - set(ids))
    summary = {
        "dataset": "MMStar",
        "pipeline": pipeline,
        "scoring": "official answer-prefix multiple-choice accuracy",
        "scoring_source": "https://github.com/MMStar-Benchmark/MMStar/blob/main/"
        "eval/vlmeval/evaluate/mmstar.py",
        "n_examples": len(records),
        "expected_examples": len(expected),
        "complete": complete,
        "full_benchmark": complete and len(expected) == 1500,
        "missing_ids": missing,
        "identity_checks_passed": sum(
            record["identity_check"]["passed"] for record in records
        ),
        "original": accuracy(scored, "original"),
        "note": "Qwen chat template, thinking disabled, deterministic generation; "
        "a subset score is a smoke check, not the full MMStar benchmark. "
        "The official prefix parser is permissive; raw answers are retained for inspection.",
    }
    if "compressed" in conditions:
        summary["compressed"] = accuracy(scored, "compressed")
        summary["paired_comparison"] = paired_bootstrap_retention(
            [int(row["original"]["correct"]) for row in scored],
            [int(row["compressed"]["correct"]) for row in scored],
            groups=[row["image_group"] for row in scored],
            seed=2026,
        )
        summary["roundtrip_checks_passed"] = sum(
            record["learned"]["roundtrip_exact"] for record in records
        )
        summary["actual_bits_per_visual_token_pooled"] = sum(
            record["sizes"]["coded_bits"] for record in records
        ) / sum(record["feature_shape"][0] for record in records)
        summary["reconstruction_medians"] = {
            key: statistics.median(record["reconstruction"][key] for record in records)
            for key in ("mse", "relative_l2", "cosine_mean")
        }
        summary[f"teacher_correct_{pipeline}_wrong"] = [
            row["id"]
            for row in scored
            if row["original"]["correct"] and not row["compressed"]["correct"]
        ]
        summary[f"teacher_wrong_{pipeline}_correct"] = [
            row["id"]
            for row in scored
            if not row["original"]["correct"] and row["compressed"]["correct"]
        ]
    (run_dir / "scored.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in scored)
    )
    write_json(run_dir / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(score_run(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
