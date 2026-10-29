"""score_results.py then report_retention.py on a synthetic run directory (no model)."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = (
    Path(__file__).resolve().parents[1]
    / "encoders"
    / "Qwen3.8-vision-compression"
    / "scripts"
)


def record(i, dataset, task, answer_original, answer_compressed, expected, accepted):
    generation = {
        "median_ms": 500.0,
        "p95_ms": 500.0,
        "min_ms": 500.0,
        "runs": 1,
        "output_tokens": 2,
        "hit_token_limit": False,
    }
    return {
        "id": f"{dataset}-val-{i}",
        "dataset": dataset,
        "image_group": f"{dataset}:{i}",
        "task_type": task,
        "question": f"Question {i}?",
        "expected_answer": expected,
        "accepted_answers": accepted,
        "scoring": "vqav2_vqa_accuracy"
        if dataset == "vqav2"
        else "textvqa_vqa_accuracy",
        "image_path": "x.jpg",
        "original_resolution_wh": [10, 10],
        "image_grid_thw": [[1, 2, 2]],
        "prompt_tokens": 20,
        "feature_shape": [4, 8],
        "feature_dtype": "torch.bfloat16",
        "identity_check": {"passed": True},
        "compressed_prefill_max_abs_logit_diff": 1.0,
        "compressed_next_token_argmax_changed": False,
        "reconstruction": {
            "mae": 0.1,
            "max_abs_error": 0.5,
            "mse": 0.01,
            "relative_l2": 0.14,
        },
        "sizes": {
            "original_raw_bytes": 64,
            "quantized_raw_bytes": 32,
            "scale_raw_bytes": 16,
            "compressed_raw_bytes": 48,
            "original_serialized_bytes": 100,
            "compressed_serialized_bytes": 60,
            "cpu_gpu_quantization_identical": False,
            "raw_ratio": 64 / 48,
            "serialized_ratio": 100 / 60,
        },
        "timing": {
            "cpu_preprocess": {"median_ms": 8.0},
            "gpu_quantize": {"median_ms": 0.1},
        },
        "original": {"answer": answer_original, "generation": generation},
        "compressed": {"answer": answer_compressed, "generation": generation},
        "timing_mode": "minimal",
    }


def make_run(
    tmp_path,
    n_vqav2=30,
    n_textvqa=20,
    degrade_every=None,
    expected_count=None,
    failed_ids=(),
):
    records = []
    for i in range(n_vqav2):
        task = "yes_no" if i % 2 else "general"
        expected = "yes" if task == "yes_no" else "grass"
        compressed = "no" if degrade_every and i % degrade_every == 0 else expected
        records.append(
            record(i, "vqav2", task, expected, compressed, expected, [expected] * 10)
        )
    for i in range(n_textvqa):
        compressed = "wrong" if degrade_every and i % degrade_every == 0 else "sony"
        records.append(
            record(
                i,
                "textvqa",
                "ocr",
                "sony",
                compressed,
                "sony",
                ["sony"] * 8 + ["samsung"] * 2,
            )
        )
    (tmp_path / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )
    (tmp_path / "run_info.json").write_text(
        json.dumps(
            {
                "checkpoint": "Qwen/x",
                "checkpoint_revision": "abc",
                "model_dtype": "torch.bfloat16",
                "software": {"python": "3", "torch": "2", "transformers": "5"},
                "hardware": {
                    "gpu": "A100",
                    "device": "cuda:1",
                    "torch_cpu_threads": 16,
                },
                "prompt": {
                    "answer_instruction": "Answer briefly.",
                    "enable_thinking": False,
                },
                "generation_config": {"do_sample": False, "max_new_tokens": 32},
                "timing_collection": {"mode": "minimal"},
            }
        )
    )
    (tmp_path / "run_config.json").write_text(
        json.dumps(
            {
                "examples": {"count": expected_count or len(records)},
                "manifests": [
                    {"path": str(tmp_path / "manifest.jsonl"), "sha256": "0"}
                ],
            }
        )
    )
    for failed in failed_ids:
        with open(tmp_path / "failures.jsonl", "a") as f:
            f.write(
                json.dumps(
                    {
                        "id": failed,
                        "error_type": "RuntimeError",
                        "message": "boom",
                        "traceback": "",
                        "time": "t",
                    }
                )
                + "\n"
            )
    return records


def run(script, run_dir, *extra):
    subprocess.run(
        [sys.executable, str(SCRIPTS / script), str(run_dir), *extra],
        check=True,
        capture_output=True,
        text=True,
    )


def test_identical_answers_pass_with_per_dataset_breakdown(tmp_path):
    make_run(tmp_path)
    run("score_results.py", tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["complete"] is True and summary["n_failed"] == 0
    assert set(summary["original"]["by_dataset"]) == {"vqav2", "textvqa"}
    assert summary["original"]["by_dataset"]["textvqa"]["mean_cer"] == 0.0
    run("report_retention.py", tmp_path, "--replicates", "10000", "--seed", "1")
    retention = json.loads((tmp_path / "retention.json").read_text())
    assert retention["decision"] == "PASS" and retention["overall"]["retention"] == 1.0
    assert set(retention["by_task"]) == {"vqav2/general", "vqav2/yes_no", "textvqa/ocr"}
    assert retention["mixture"]["counts"] == {"vqav2": 30, "textvqa": 20}
    assert retention["overall"]["bootstrap"]["seed"] == 1
    report = (tmp_path / "report.md").read_text()
    assert "Decision: PASS" in report and "| overall | 50 |" in report


def test_degraded_answers_fail_and_list_changes(tmp_path):
    make_run(tmp_path, n_vqav2=200, n_textvqa=100, degrade_every=4)
    run("score_results.py", tmp_path)
    run("report_retention.py", tmp_path)
    retention = json.loads((tmp_path / "retention.json").read_text())
    assert retention["decision"] == "FAIL"
    assert retention["overall"]["secondary"]["answers_changed"] == 75
    assert retention["overall"]["secondary"]["original_correct_compressed_wrong"] == 75
    report = (tmp_path / "report.md").read_text()
    assert "## Changed answers (75 of 300, first 50 shown)" in report


def test_failed_or_missing_examples_make_the_run_incomplete(tmp_path):
    make_run(
        tmp_path,
        expected_count=52,
        failed_ids=["vqav2-val-999", "vqav2-val-999", "vqav2-val-0"],
    )
    run("score_results.py", tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    # vqav2-val-0 completed later, so only the never-completed id counts as failed.
    assert summary["complete"] is False and summary["n_failed"] == 1
    assert summary["failures"][0] == {
        "id": "vqav2-val-999",
        "attempts": 2,
        "last_error_type": "RuntimeError",
        "last_message": "boom",
    }
    run("report_retention.py", tmp_path)
    retention = json.loads((tmp_path / "retention.json").read_text())
    assert retention["decision"] == "INCOMPLETE"
    assert retention["decision_on_completed_examples"] == "PASS"
    assert "The run is incomplete" in (tmp_path / "report.md").read_text()


def test_report_rejects_too_few_replicates(tmp_path):
    make_run(tmp_path)
    run("score_results.py", tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "report_retention.py"),
            str(tmp_path),
            "--replicates",
            "100",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "10,000" in result.stderr


def test_duplicate_ids_in_results_are_rejected(tmp_path):
    records = make_run(tmp_path)
    with open(tmp_path / "results.jsonl", "a") as f:
        f.write(json.dumps(records[0]) + "\n")
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "score_results.py"), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "duplicate" in result.stderr
