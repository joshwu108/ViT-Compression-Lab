"""End-to-end evaluation of a trained Entropic Student against the unmodified model.

Same protocol, record schema and output files as inject_eval.py (identity
check, timed stages, greedy generation for both conditions, resume with
--resume), so score_results.py and report_retention.py run unchanged and the
result sits next to the INT8 / INT4-g64 runs. Per example the learned pipeline is

    client:  pixels -> frozen prefix (to the split) -> student encoder -> round -> rANS
    wire:    payload = header(grid_thw, split, latent shape) + strings
    server:  rANS decode -> student decoder -> frozen suffix (to the merger) -> LLM

    python encoders/Qwen3.8-vision-compression/scripts/es_eval.py \
        data/vqav2-pilot/subset.jsonl data/textvqa-pilot/subset.jsonl \
        --student-checkpoint runs/qwen3.8/entropic-student/<name>/checkpoints/best.pt \
        --output-dir outputs/qwen3.8/inject-eval/pilot70-es-<name>-001

--baseline-codec int4-g64 additionally measures that codec's bytes and
reconstruction error on the same features (no extra generation). The
pure-Python entropy coder is timed with --entropy-runs repeats (default 1);
its latency is reported separately and is not representative of a C coder.
Nothing is downloaded; the model is read from the local cache only.
"""

import argparse
import fcntl
import sys
import time
import traceback
from pathlib import Path

import torch
from PIL import Image
from transformers import GenerationConfig, Qwen3_5ForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from vit_compression_lab.codecs import CODEC_NAMES, get_codec  # noqa: E402
from vit_compression_lab.entropic import payload as payload_lib  # noqa: E402
from vit_compression_lab.entropic.checkpoint import load_checkpoint, student_from_checkpoint  # noqa: E402
from vit_compression_lab.entropic.metrics import compression_metrics, reconstruction_metrics  # noqa: E402
from vit_compression_lab.entropic.qwen.injection import load_inject_eval  # noqa: E402
from vit_compression_lab.entropic.qwen.vision_tower import TeacherSplit  # noqa: E402
from vit_compression_lab.int8_codec import raw_bytes  # noqa: E402
from vit_compression_lab.run_checkpoint import (  # noqa: E402
    ResumeError,
    append_record,
    check_resume,
    sha256_file,
    sha256_text,
    write_json,
)


def learned_round_trip(ie, student, split, inputs, features, device, args):
    """Client -> payload -> server, each stage timed. Returns restored features and diagnostics."""
    grid = inputs["image_grid_thw"]
    pixels = inputs["pixel_values"].to(split.dtype)
    sp = student.split
    in_depth, tgt_depth = sp.input_depth(split.dims), sp.target_depth(split.dims)
    geo = split.geometry(grid)
    timings = {}

    u, timings["gpu_client_prefix"] = ie.gpu_time(
        lambda: split.prefix(pixels, grid, in_depth, geo), device, args.runs, args.warmups)
    u = u.float()
    z, timings["gpu_student_encode"] = ie.gpu_time(
        lambda: student.encode(u, grid), device, args.runs, args.warmups)
    counts = student.latent_counts(grid)
    parts = list(torch.split(z, counts))

    def entropy_encode():
        strings = [student.entropy.compress(p) for p in parts]
        header = payload_lib.make_header(sp, grid.cpu(), counts, student.latent_channels,
                                         dtype=str(features.dtype).removeprefix("torch."),
                                         encoder_type=student.encoder.family)
        return payload_lib.pack(header, strings), strings

    (blob, strings), timings["cpu_entropy_encode"] = ie.cpu_time(entropy_encode, args.entropy_runs, 0)

    def entropy_decode():
        header, strings2 = payload_lib.unpack(blob)
        return torch.cat([student.entropy.decompress(s, n, device="cpu")
                          for s, n in zip(strings2, header["latent_counts"])])

    z_hat_cpu, timings["cpu_entropy_decode"] = ie.cpu_time(entropy_decode, args.entropy_runs, 0)
    z_hat, timings["transfer_h2d_latent"] = ie.gpu_time(
        lambda: z_hat_cpu.to(device), device, args.runs, args.warmups)
    recon, timings["gpu_student_decode"] = ie.gpu_time(
        lambda: student.decode(z_hat, grid), device, args.runs, args.warmups)
    h_hat, timings["gpu_server_suffix"] = ie.gpu_time(
        lambda: split.suffix(recon.to(split.dtype), grid, tgt_depth, geo), device, args.runs, args.warmups)
    restored = h_hat.to(features.dtype)
    if restored.shape != features.shape:
        raise RuntimeError(f"Reconstruction {tuple(restored.shape)} != features {tuple(features.shape)}")

    z_round = student.entropy.quantize(z, "round").float().cpu()
    ideal_bits = sum(student.entropy.ideal_bits(p) for p in parts)
    acct = payload_lib.accounting(blob, strings)
    sizes = compression_metrics(len(blob), acct["container_overhead_bytes"], ideal_bits, features.shape[0],
                                raw_bytes(features), len(ie.serialize_original(features.cpu())))
    sizes["compressed_raw_bytes"] = len(blob)
    sizes["latent_positions"] = sum(counts)
    sizes["latent_channels"] = student.latent_channels
    diagnostics = {
        "roundtrip_exact": torch.equal(z_round, z_hat_cpu.float()),
        "latent_shape": [sum(counts), student.latent_channels],
        "payload_version": payload_lib.unpack(blob)[0]["version"],
    }
    return restored, blob, timings, sizes, diagnostics


def baseline_codec_metrics(codec, features, grid_thw):
    q, scale = codec.quantize(features)
    restored = codec.dequantize(q, scale, features.dtype)
    payload = codec.serialize(q.cpu(), scale.cpu(), features.dtype, list(features.shape), grid_thw)
    return {"codec": codec.name, "compressed_serialized_bytes": len(payload),
            "bits_per_visual_token": 8 * len(payload) / features.shape[0],
            "reconstruction": reconstruction_metrics(features, restored)}


def evaluate_example(ie, row, model, processor, student, split, generation_config, device, args, baseline):
    with Image.open(row["image_path"]) as source:
        image = source.convert("RGB")
    inputs_cpu, preprocess_timing = ie.cpu_time(
        lambda: ie.build_inputs(processor, image, row["question"]), args.runs, args.warmups)
    inputs, h2d_timing = ie.gpu_time(
        lambda: {k: v.to(device) for k, v in inputs_cpu.items()}, device, args.runs, args.warmups)
    grid_thw = inputs["image_grid_thw"].tolist()

    with torch.inference_mode():
        features, original_logits, identity = ie.check_identity(model, inputs, generation_config)
        if not identity["passed"]:
            raise RuntimeError(f"Identity injection failed: {identity}")
        _, encoder_timing = ie.gpu_time(
            lambda: ie.encode_image(model, inputs), device, args.encoder_runs, args.warmups)
        restored, blob, codec_timings, sizes, diagnostics = learned_round_trip(
            ie, student, split, inputs, features, device, args)
        if torch.equal(restored, features):
            raise RuntimeError("Reconstruction equals the original; nothing to compare")
        compressed_logits = ie.injected_prefill(model, inputs, restored)
        eos = generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        original_ids, original_generation = ie.timed_generation(
            model, ie.injected_inputs(inputs, features), generation_config, device, eos_ids)
        compressed_ids, compressed_generation = ie.timed_generation(
            model, ie.injected_inputs(inputs, restored), generation_config, device, eos_ids)
        baseline_record = baseline_codec_metrics(baseline, features, grid_thw) if baseline else None

    last = slice(-1, None)
    record = {
        "id": row["id"],
        "dataset": row.get("dataset") or row["id"].split("-", 1)[0],
        "image_group": row.get("image_group") or row["id"],
        "task_type": row["task_type"],
        "question": row["question"],
        "expected_answer": row["expected_answer"],
        "accepted_answers": row.get("accepted_answers"),
        "scoring": row.get("scoring"),
        "image_path": row["image_path"],
        "original_resolution_wh": list(image.size),
        "image_grid_thw": grid_thw,
        "prompt_tokens": inputs["input_ids"].shape[1],
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "compression": f"learned:{student.config.name}",
        "identity_check": identity,
        "compressed_prefill_max_abs_logit_diff":
            (compressed_logits - original_logits).abs().max().item(),
        "compressed_next_token_argmax_changed": not torch.equal(
            compressed_logits[:, last].argmax(-1), original_logits[:, last].argmax(-1)),
        "reconstruction": reconstruction_metrics(features, restored),
        "sizes": sizes,
        "timing": {
            "cpu_preprocess": preprocess_timing,
            "transfer_h2d_processor_inputs": h2d_timing,
            "gpu_encoder_through_merger": encoder_timing,
            **codec_timings,
        },
        "learned": {**diagnostics, "split": student.split.as_dict(), "option": student.split.option,
                    "baseline": baseline_record},
        "original": {"answer": processor.decode(original_ids, skip_special_tokens=True),
                     "generation": original_generation},
        "compressed": {"answer": processor.decode(compressed_ids, skip_special_tokens=True),
                       "generation": compressed_generation},
    }
    return record, blob


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--revision", default="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
    parser.add_argument("--device", default="cuda:1", help="GPU 0 is shared on this host")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--encoder-runs", type=int, default=20)
    parser.add_argument("--entropy-runs", type=int, default=1, help="Repeats for the Python rANS stages")
    parser.add_argument("--minimal-timing", action="store_true")
    parser.add_argument("--skip-payloads", action="store_true")
    parser.add_argument("--baseline-codec", choices=CODEC_NAMES)
    parser.add_argument("--max-failures", type=int, default=10)
    args = parser.parse_args()

    ie = load_inject_eval()
    rows = ie.load_manifests(args.manifests, args.ids)
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.minimal_timing:
        args.runs, args.warmups, args.encoder_runs = 1, 0, 1
    if args.output_dir.exists() and not args.resume:
        parser.error("Output directory already exists; choose a new one or pass --resume")
    if args.resume and not args.output_dir.exists():
        parser.error("Nothing to resume: the output directory does not exist")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    processor = ie.AutoProcessor.from_pretrained(args.checkpoint, revision=args.revision, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.checkpoint, revision=args.revision, local_files_only=True,
        dtype=torch.bfloat16, device_map={"": device}).eval()
    ckpt = load_checkpoint(args.student_checkpoint, map_location=device)
    student, config = student_from_checkpoint(ckpt)
    student = student.to(device).eval()
    student.entropy.update(force=True)
    split = TeacherSplit(model.model.visual)
    if config["teacher"]["revision"] != args.revision:
        raise SystemExit("Student was trained against a different teacher revision")
    baseline = get_codec(args.baseline_codec) if args.baseline_codec else None

    defaults = model.generation_config
    generation_config = GenerationConfig(
        do_sample=False, max_new_tokens=args.max_new_tokens,
        eos_token_id=defaults.eos_token_id, pad_token_id=defaults.pad_token_id,
        bos_token_id=defaults.bos_token_id)

    run_info = ie.describe_run(args, model, processor, generation_config, device)
    run_info["measurement_notes"]["learned_pipeline"] = (
        "client prefix -> student encode -> round -> Python rANS; payload bytes include the header; "
        "server rANS decode -> student decode -> frozen suffix -> merger features injected via mm_encoder_outputs")
    run_info["student"] = {"checkpoint": str(args.student_checkpoint), "config": config.raw,
                           "parameters": student.parameter_counts(), "epoch": ckpt["epoch"], "step": ckpt["step"]}
    fingerprint = {
        "checkpoint": args.checkpoint, "revision": args.revision,
        "model_config_commit": getattr(model.config, "_commit_hash", None),
        "model_dtype": str(model.dtype), "gpu": torch.cuda.get_device_properties(device).name,
        "attention_backend": run_info["attention_backend"],
        "prompt": {"answer_instruction": ie.ANSWER_INSTRUCTION, "enable_thinking": False},
        "generation_config": generation_config.to_diff_dict(),
        "image_processor": processor.image_processor.to_dict(),
        "compression": {"name": f"learned:{config.name}",
                        "student_checkpoint_sha256": sha256_file(args.student_checkpoint),
                        "student_config": config.raw, "baseline_codec": args.baseline_codec},
        "manifests": [{"path": str(p), "sha256": sha256_file(p)} for p in args.manifests],
        "examples": {"count": len(rows), "ids_sha256": sha256_text("\n".join(r["id"] for r in rows))},
        "software": run_info["software"],
    }
    if args.resume:
        try:
            completed, repaired = check_resume(args.output_dir, fingerprint)
        except ResumeError as error:
            raise SystemExit(f"Cannot resume: {error}")
        if repaired:
            print("Discarded a partial last line of results.jsonl")
    else:
        completed = set()
        args.output_dir.mkdir(parents=True)
        write_json(args.output_dir / "run_config.json", fingerprint)
        write_json(args.output_dir / "run_info.json", run_info)
    lock = open(args.output_dir / ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another session is already writing to this output directory")
    if not args.skip_payloads:
        (args.output_dir / "payloads").mkdir(exist_ok=True)
    session = {"started_at": ie.now(), "resume": args.resume, "completed_before": len(completed),
               "timing_collection": run_info["timing_collection"], "hardware": run_info["hardware"],
               "software": run_info["software"]}
    append_record(args.output_dir / "sessions.jsonl", session)

    pending = [row for row in rows if row["id"] not in completed]
    print(f"{len(completed)} completed, {len(pending)} pending of {len(rows)} examples")
    failures, session_start = 0, time.time()
    for index, row in enumerate(pending, 1):
        example_start = time.time()
        try:
            record, blob = evaluate_example(ie, row, model, processor, student, split, generation_config,
                                            device, args, baseline)
        except Exception as error:
            failures += 1
            append_record(args.output_dir / "failures.jsonl", {
                "id": row["id"], "error_type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc(), "time": ie.now()})
            print(f"FAILED {row['id']}: {type(error).__name__}: {error}")
            torch.cuda.empty_cache()
            if failures >= args.max_failures:
                raise SystemExit(f"Stopping after {failures} failures; see failures.jsonl")
            continue
        record["elapsed_s"] = time.time() - example_start
        record["timing_mode"] = run_info["timing_collection"]["mode"]
        if not args.skip_payloads:
            (args.output_dir / "payloads" / f"{row['id']}.bin").write_bytes(blob)
        append_record(args.output_dir / "results.jsonl", record)
        done = len(completed) + index - failures
        elapsed = time.time() - session_start
        write_json(args.output_dir / "progress.json", {
            "examples": len(rows), "completed": done, "remaining": len(rows) - done,
            "failed_this_session": failures, "session_started_at": session["started_at"],
            "updated_at": ie.now(), "session_seconds": elapsed,
            "seconds_per_example_this_session": elapsed / index,
            "eta_seconds": elapsed / index * (len(rows) - done), "last_id": row["id"]})
        print(f"{done}/{len(rows)} {row['id']} {record['elapsed_s']:.1f}s "
              f"bits/tok={record['sizes']['bits_per_visual_token']:.0f} "
              f"original={record['original']['answer']!r} compressed={record['compressed']['answer']!r}")


if __name__ == "__main__":
    main()
