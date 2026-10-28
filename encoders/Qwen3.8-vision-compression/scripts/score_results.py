"""Score inject_eval.py answers against manifest references; write a summary.

    python encoders/Qwen3.8-vision-compression/scripts/score_results.py \
        outputs/qwen3.8/inject-eval/<run>

Reads <run>/results.jsonl and <run>/run_info.json; writes <run>/scored.jsonl
and <run>/summary.json. Baseline answers are never used as references.

Primary scores (normalize(): lowercase, trim, collapse whitespace, remove
surrounding quotes and one trailing period; other punctuation and digits kept):
  yes_no   - first word must be yes/no; compared with expected_answer only
  counting - exactly one integer (digits or zero..twenty); equals expected_answer
  general  - normalized exact match with expected_answer or any accepted_answer
  ocr      - CER against expected_answer (both normalized); "correct" means a
             normalized exact match with expected_answer or any accepted_answer
Secondary: official VQA accuracy (scripts/vqa_accuracy.py), per row's "scoring",
applied to normalize(answer) rather than the raw answer.

Results are also grouped by dataset. Ids listed in failures.jsonl that never
completed are reported as failures, and "complete" is false whenever any
example failed or is missing from run_config.json's expected count.
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from vqa_accuracy import SCORERS as VQA_SCORERS  # noqa: E402

CONDITIONS = ("original", "compressed")
NUMBER_WORDS = {word: n for n, word in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}


def normalize(text):
    text = " ".join(text.lower().split())
    text = text.strip("\"'“”‘’ ")
    return text[:-1].rstrip() if text.endswith(".") else text


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def parse_count(text):
    found = re.findall(r"\b\d+\b|\b[a-z]+\b", text)
    numbers = {int(t) if t.isdigit() else NUMBER_WORDS[t]
               for t in found if t.isdigit() or t in NUMBER_WORDS}
    return numbers.pop() if len(numbers) == 1 else None


def score(row, answer):
    prediction = normalize(answer)
    expected = normalize(row["expected_answer"])
    accepted = {expected} | {normalize(a) for a in row.get("accepted_answers") or []}
    result = {"answer": answer, "normalized": prediction, "parsed": None,
              "unparseable": False, "cer": None}
    task = row["task_type"]
    if task == "ocr":  # an empty answer has CER 1.0
        result["cer"] = edit_distance(prediction, expected) / len(expected)
    if not prediction:
        result["unparseable"] = True
    elif task == "yes_no":
        first = re.split(r"[\s,.!]", prediction)[0]
        if first in ("yes", "no"):
            result["parsed"] = first
        else:
            result["unparseable"] = True
    elif task == "counting":
        result["parsed"] = parse_count(prediction)
        result["unparseable"] = result["parsed"] is None

    if result["unparseable"]:
        result["correct"] = False
    elif task == "yes_no":
        result["correct"] = result["parsed"] == expected
    elif task == "counting":
        result["correct"] = str(result["parsed"]) == expected
    else:
        result["correct"] = prediction in accepted
    vqa = VQA_SCORERS.get(row.get("scoring"))
    # Lowercased/trimmed input: the official scorer skips normalization when
    # all annotators agree, so "Grass." would otherwise score 0 against "grass".
    result["vqa_accuracy"] = vqa(prediction, row["accepted_answers"]) if vqa else None
    return result


def group_summary(scored, condition, key):
    groups = {}
    for name in sorted({r[key] for r in scored}):
        values = [r[condition] for r in scored if r[key] == name]
        groups[name] = {
            "n": len(values),
            "accuracy": statistics.mean(v["correct"] for v in values),
            "vqa_accuracy": statistics.mean(v["vqa_accuracy"] for v in values),
        }
        cers = [v["cer"] for v in values if v["cer"] is not None]
        if cers:
            groups[name]["mean_cer"] = statistics.mean(cers)
    return groups


def condition_summary(scored, condition):
    values = [r[condition] for r in scored]
    return {
        "n": len(values),
        "accuracy": statistics.mean(v["correct"] for v in values),
        "vqa_accuracy": statistics.mean(v["vqa_accuracy"] for v in values),
        "unparseable": sum(v["unparseable"] for v in values),
        "hit_token_limit": sum(r["generation"][condition]["hit_token_limit"] for r in scored),
        "by_dataset": group_summary(scored, condition, "dataset"),
        "by_task": group_summary(scored, condition, "task_type"),
    }


def pending_failures(run_dir, completed_ids):
    """Ids in failures.jsonl that never completed, with their attempt history."""
    path = run_dir / "failures.jsonl"
    if not path.exists():
        return []
    attempts = {}
    for line in path.read_text().splitlines():
        if line.strip():
            failure = json.loads(line)
            entry = attempts.setdefault(failure["id"], {"id": failure["id"], "attempts": 0})
            entry["attempts"] += 1
            entry["last_error_type"] = failure["error_type"]
            entry["last_message"] = failure["message"]
    return [entry for id_, entry in sorted(attempts.items()) if id_ not in completed_ids]


def median_of(records, *keys):
    values = []
    for record in records:
        for key in keys:
            record = record[key]
        values.append(record)
    return statistics.median(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in
               (args.run_dir / "results.jsonl").read_text().splitlines() if line.strip()]
    if not records:
        raise SystemExit("results.jsonl has no completed examples")
    if len({r["id"] for r in records}) != len(records):
        raise SystemExit("results.jsonl has duplicate ids; inspect before scoring")
    run_info = json.loads((args.run_dir / "run_info.json").read_text())
    config_path = args.run_dir / "run_config.json"
    expected = (json.loads(config_path.read_text())["examples"]["count"]
                if config_path.exists() else None)

    scored = []
    for record in records:
        row = {
            "id": record["id"], "task_type": record["task_type"],
            # Pilot results predate these fields; derive them from the id.
            "dataset": record.get("dataset") or record["id"].split("-", 1)[0],
            "image_group": record.get("image_group") or record["id"],
            "question": record["question"], "expected_answer": record["expected_answer"],
            "accepted_answers": record["accepted_answers"],
            "generation": {c: record[c]["generation"] for c in CONDITIONS},
        }
        for condition in CONDITIONS:
            row[condition] = score(record, record[condition]["answer"])
        row["answer_changed"] = record["original"]["answer"] != record["compressed"]["answer"]
        scored.append(row)
    with (args.run_dir / "scored.jsonl").open("w") as f:
        for row in scored:
            f.write(json.dumps(row) + "\n")

    original = condition_summary(scored, "original")
    compressed = condition_summary(scored, "compressed")
    timing_keys = list(records[0]["timing"])
    failures = pending_failures(args.run_dir, {r["id"] for r in scored})
    summary = {
        "run_info": run_info,
        "scoring_rules": __doc__.split("Primary scores", 1)[1].strip(),
        "n_examples": len(scored),
        "expected_examples": expected,
        "n_failed": len(failures),
        "failures": failures,
        "complete": not failures and (expected is None or expected == len(scored)),
        "timing_modes": sorted({r.get("timing_mode", "repeated") for r in records}),
        "original": original,
        "compressed": compressed,
        "accuracy_difference_pp": 100 * (compressed["accuracy"] - original["accuracy"]),
        "vqa_accuracy_difference_pp":
            100 * (compressed["vqa_accuracy"] - original["vqa_accuracy"]),
        "original_correct_compressed_wrong": [
            r["id"] for r in scored if r["original"]["correct"] and not r["compressed"]["correct"]],
        "original_wrong_compressed_correct": [
            r["id"] for r in scored if not r["original"]["correct"] and r["compressed"]["correct"]],
        "answers_changed": [r["id"] for r in scored if r["answer_changed"]],
        "identity_checks_passed": sum(r["identity_check"]["passed"] for r in records),
        "compressed_next_token_argmax_changed": sum(
            r["compressed_next_token_argmax_changed"] for r in records),
        "medians_across_examples": {
            "feature_tokens": median_of(records, "feature_shape", 0),
            "reconstruction": {k: median_of(records, "reconstruction", k)
                               for k in records[0]["reconstruction"]},
            "sizes": {k: median_of(records, "sizes", k) for k in records[0]["sizes"]
                      if k != "cpu_gpu_quantization_identical"},
            "timing_median_ms": {k: median_of(records, "timing", k, "median_ms")
                                 for k in timing_keys},
            "generation_ms": {c: median_of(records, c, "generation", "median_ms")
                              for c in CONDITIONS},
            "output_tokens": {c: median_of(records, c, "generation", "output_tokens")
                              for c in CONDITIONS},
        },
        "cpu_gpu_quantization_identical": sum(
            r["sizes"]["cpu_gpu_quantization_identical"] for r in records),
    }
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "run_info"}, indent=2))


if __name__ == "__main__":
    main()
