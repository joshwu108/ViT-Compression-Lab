"""Evaluate uncompressed Qwen alone, or Option B/C paired with uncompressed Qwen.

Reuses es_eval.py's checked feature injection and actual rANS round trip.
No training or model downloads occur. Results are resumable, and are scored
with score_mmstar.py rather than the TextVQA/VQAv2 consensus scorer.
"""

import argparse
import fcntl
import json
import sys
import time
import traceback
from pathlib import Path

import torch
from PIL import Image
from transformers import GenerationConfig, Qwen3_5ForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from es_eval import evaluate_example as evaluate_student  # noqa: E402
from score_mmstar import score_run  # noqa: E402

from vit_compression_lab.entropic.checkpoint import (  # noqa: E402
    load_checkpoint,
    student_from_checkpoint,
)
from vit_compression_lab.entropic.qwen.injection import load_inject_eval  # noqa: E402
from vit_compression_lab.entropic.qwen.vision_tower import TeacherSplit  # noqa: E402
from vit_compression_lab.run_checkpoint import (  # noqa: E402
    append_record,
    check_resume,
    sha256_file,
    sha256_text,
    write_json,
)

DEFAULT_STUDENT = Path(
    "runs/qwen3.8/es-abc-pilot5k/abc5k_c_merger_tokenmlp_beta0.08_s0/"
    "checkpoints/step-010000-best.pt"
)
ANSWER_INSTRUCTION = (
    "Answer with only the correct option letter (A, B, C, or D). Do not explain."
)


def evaluate_teacher(ie, row, model, processor, generation_config, device):
    with Image.open(row["image_path"]) as source:
        image = source.convert("RGB")
    inputs = {
        key: value.to(device)
        for key, value in ie.build_inputs(processor, image, row["question"]).items()
    }
    with torch.inference_mode():
        features, _, identity = ie.check_identity(model, inputs, generation_config)
        if not identity["passed"]:
            raise RuntimeError(f"Identity check failed: {identity}")
        eos = generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        # Ordinary pixel -> vision tower -> merger -> LLM generation.
        ids, timing = ie.timed_generation(
            model, dict(inputs), generation_config, device, eos_ids
        )
    return {
        **{
            key: row[key]
            for key in (
                "id",
                "dataset",
                "image_group",
                "question",
                "image_path",
                "task_type",
                "expected_answer",
                "scoring",
            )
        },
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "image_grid_thw": inputs["image_grid_thw"].tolist(),
        "identity_check": identity,
        "original": {
            "answer": processor.decode(ids, skip_special_tokens=True),
            "generation": timing,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/mmstar/manifest.jsonl")
    )
    parser.add_argument("--pipeline", choices=("b", "c", "teacher"), default="c")
    parser.add_argument(
        "--student-checkpoint",
        type=Path,
        help="Default for --pipeline c: the archived step-010000-best.pt; required for b",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--limit", type=int, help="Smoke check only; omit for all 1,500 questions"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.8-27B")
    parser.add_argument(
        "--revision", default="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--max-failures", type=int, default=3)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_new_tokens < 1 or args.max_failures < 1:
        parser.error("--max-new-tokens and --max-failures must be positive")
    if args.pipeline == "b" and args.student_checkpoint is None:
        parser.error("--pipeline b needs an explicit --student-checkpoint")
    if args.student_checkpoint is None:
        args.student_checkpoint = DEFAULT_STUDENT
    if args.output_dir.exists() != args.resume:
        parser.error("Use a new output directory, or --resume with an existing run")
    metadata = json.loads((args.manifest.parent / "metadata.json").read_text())
    if metadata["manifest_sha256"] != sha256_file(args.manifest):
        parser.error("Manifest differs from its prepared metadata")
    for checked in metadata["overlap_check"]["checked_manifests"]:
        if sha256_file(checked["path"]) != checked["sha256"]:
            parser.error(
                "Training/validation manifest changed; repeat the dataset overlap audit"
            )
    if metadata["overlap_check"]["overlapping_questions"]:
        parser.error(
            "Dataset overlaps the audited train/validation images; inspect metadata.json"
        )

    ie = load_inject_eval()
    # Both conditions use this instruction through the same existing chat-template helper.
    ie.ANSWER_INSTRUCTION = ANSWER_INSTRUCTION
    rows = ie.load_manifests([args.manifest], None)
    if not rows or any(row.get("dataset") != "mmstar" for row in rows):
        parser.error("Expected a nonempty MMStar manifest")
    rows = rows[: args.limit] if args.limit is not None else rows
    # The shared evaluator's timing interface; avoid its repeated benchmark loops.
    args.manifests = [args.manifest]
    args.runs, args.warmups, args.encoder_runs, args.entropy_runs = 1, 0, 1, 1
    args.minimal_timing, args.baseline_codec = True, None
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    student = split = student_info = None
    if args.pipeline != "teacher":
        ckpt = load_checkpoint(args.student_checkpoint)
        student, config = student_from_checkpoint(ckpt)
        # C: merger -> merger. B: block k -> block m, then Qwen's frozen blocks and merger.
        if config.split.option != args.pipeline.upper():
            parser.error(
                f"--pipeline {args.pipeline} was given an Option "
                f"{config.split.option} checkpoint ({config.split.as_dict()})"
            )
        if (config["teacher"]["checkpoint"], config["teacher"]["revision"]) != (
            args.checkpoint,
            args.revision,
        ):
            parser.error("Student and teacher checkpoint/revision differ")
        audited = {
            Path(item["path"]).resolve()
            for item in metadata["overlap_check"]["checked_manifests"]
        }
        if any(
            Path(path).resolve() not in audited
            for path in config["data"]["train_manifests"]
        ):
            parser.error(
                "Student training manifests were not included in the dataset overlap audit"
            )
        student = student.to(device).eval()
        student.entropy.update(force=True)
        student_info = {
            "checkpoint": str(args.student_checkpoint),
            "sha256": sha256_file(args.student_checkpoint),
            "step": ckpt["step"],
            "epoch": ckpt["epoch"],
            "config": config.raw,
        }
    processor = ie.AutoProcessor.from_pretrained(
        args.checkpoint, revision=args.revision, local_files_only=True
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.checkpoint,
        revision=args.revision,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": device},
    ).eval()
    if student is not None:
        split = TeacherSplit(model.model.visual)
    defaults = model.generation_config
    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=defaults.eos_token_id,
        pad_token_id=defaults.pad_token_id,
        bos_token_id=defaults.bos_token_id,
    )
    info = ie.describe_run(args, model, processor, generation_config, device)
    info["student"] = student_info
    fingerprint = {
        "pipeline": args.pipeline,
        "checkpoint": args.checkpoint,
        "revision": args.revision,
        "student": student_info,
        "prompt": info["prompt"],
        "generation_config": info["generation_config"],
        "image_processor": info["image_processor"],
        "software": info["software"],
        "attention_backend": info["attention_backend"],
        "gpu": info["hardware"]["gpu"],
        "manifest": {"path": str(args.manifest), "sha256": sha256_file(args.manifest)},
        "examples": {
            "count": len(rows),
            "ids": [row["id"] for row in rows],
            "ids_sha256": sha256_text("\n".join(row["id"] for row in rows)),
        },
    }
    if not args.resume:
        args.output_dir.mkdir(parents=True)
    with (args.output_dir / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            completed, _ = check_resume(args.output_dir, fingerprint)
        else:
            completed = set()
            write_json(args.output_dir / "run_config.json", fingerprint)
            write_json(args.output_dir / "run_info.json", info)
        if completed - {row["id"] for row in rows}:
            raise ValueError("Results contain ids outside this evaluation selection")
        append_record(
            args.output_dir / "sessions.jsonl",
            {
                "started_at": ie.now(),
                "resume": args.resume,
                "completed_before": len(completed),
            },
        )
        failures, started, done = 0, time.time(), len(completed)
        print(f"{done} completed; {len(rows) - done} pending", flush=True)
        for row in rows:
            if row["id"] in completed:
                continue
            tick = time.time()
            try:
                if student is None:
                    record = evaluate_teacher(
                        ie, row, model, processor, generation_config, device
                    )
                else:
                    record, _ = evaluate_student(
                        ie,
                        row,
                        model,
                        processor,
                        student,
                        split,
                        generation_config,
                        device,
                        args,
                        None,
                    )
                    if not record["learned"]["roundtrip_exact"]:
                        raise RuntimeError(
                            "Entropy-coded latents failed the exact round trip"
                        )
            except Exception as error:
                failures += 1
                append_record(
                    args.output_dir / "failures.jsonl",
                    {
                        "id": row["id"],
                        "time": ie.now(),
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                )
                print(f"FAILED {row['id']}: {error}", flush=True)
                torch.cuda.empty_cache()
                if failures >= args.max_failures:
                    raise SystemExit(
                        "Stopped after repeated failures; inspect failures.jsonl and resume"
                    )
                continue
            record.update(
                {
                    "category": row["category"],
                    "l2_category": row["l2_category"],
                    "elapsed_s": time.time() - tick,
                }
            )
            append_record(args.output_dir / "results.jsonl", record)
            done += 1
            write_json(
                args.output_dir / "progress.json",
                {
                    "completed": done,
                    "examples": len(rows),
                    "remaining": len(rows) - done,
                    "failed_this_session": failures,
                    "updated_at": ie.now(),
                    "session_seconds": time.time() - started,
                    "last_id": row["id"],
                },
            )
            print(
                f"{done}/{len(rows)} {row['id']} {record['elapsed_s']:.1f}s "
                f"teacher={record['original']['answer']!r} "
                f"compressed={record.get('compressed', {}).get('answer')!r}",
                flush=True,
            )
        summary = score_run(args.output_dir)
        print(json.dumps(summary, indent=2))
        if not summary["complete"]:
            raise SystemExit(
                "Evaluation is incomplete; resume to retry missing examples"
            )


if __name__ == "__main__":
    main()
