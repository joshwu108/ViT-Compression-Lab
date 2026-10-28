"""Feed original or reconstructed Qwen visual features into the same Qwen model.

Injection uses the installed model's own `mm_encoder_outputs` argument
(Qwen3_5Model.forward): precomputed merged features replace the image
placeholder embeddings, and no `pixel_values` are passed, so the vision
encoder cannot run again. Ordinary `generate()` uses this same argument
internally after encoding the pixels itself.

One example, from the repository root:

    python encoders/Qwen3.8-vision-compression/scripts/inject_eval.py \
        data/vqav2-pilot/subset.jsonl --ids vqav2-val-386189004 \
        --output-dir outputs/qwen3.8/inject-eval/one-example

Several manifests can be given at once. Score the results afterwards with
score_results.py. Long runs append one line per example to results.jsonl and
can be continued with --resume, which refuses to continue if the model
revision, prompt, generation settings, compression codec, or manifests differ.
Failed examples are recorded in failures.jsonl and retried on resume.
"""

import argparse
import fcntl
import io
import json
import platform
import statistics
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from PIL import Image
from transformers import (
    AutoProcessor,
    GenerationConfig,
    Qwen3_5ForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from vit_compression_lab.int8_codec import (  # noqa: E402
    dequantize,
    deserialize,
    quantize,
    raw_bytes,
    reconstruction_errors,
    serialize,
)
from vit_compression_lab.run_checkpoint import (  # noqa: E402
    ResumeError,
    append_record,
    check_resume,
    sha256_file,
    sha256_text,
    write_json,
)

ANSWER_INSTRUCTION = (
    "Answer with only the requested word, number, or option letter. Do not explain."
)

def load_manifests(paths, ids):
    rows = []
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                row["image_path"] = str((path.parent / row["image_path"]).resolve())
                rows.append(row)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate ids across manifests")
    if ids:
        wanted = set(ids)
        rows = [row for row in rows if row["id"] in wanted]
        missing = wanted - {row["id"] for row in rows}
        if missing:
            raise ValueError(f"Ids not in manifest: {sorted(missing)}")
    for row in rows:
        for key in ("id", "image_path", "question", "task_type", "expected_answer"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{row.get('id')}: needs a nonempty {key} string")
        if not Path(row["image_path"]).is_file():
            raise ValueError(f"Image does not exist: {row['image_path']}")
    return rows


# ---------------------------------------------------------------- timing helpers

def summarize(times_ms):
    ordered = sorted(times_ms)
    return {
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))],
        "min_ms": ordered[0],
        "runs": len(ordered),
    }


def cpu_time(fn, runs, warmups):
    """CPU wall-clock timing for CPU-only operations."""
    for _ in range(warmups):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - start) * 1000)
    return result, summarize(times)


def gpu_time(fn, device, runs, warmups):
    """CUDA events on `device`'s current stream; covers GPU work and copies."""
    stream = torch.cuda.current_stream(device)
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize(device)
    times = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        result = fn()
        end.record(stream)
        end.synchronize()
        times.append(start.elapsed_time(end))
    return result, summarize(times)


# ------------------------------------------------------------- model interface

def build_inputs(processor, image, question):
    """CPU preprocessing: chat template, tokenization, and image resizing/patching."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": f"{question}\n{ANSWER_INSTRUCTION}"},
    ]}]
    return processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", enable_thinking=False,
    )


def encode_image(model, inputs):
    """Vision encoder through `visual.merger`. Returns merged features [tokens, 5120]."""
    output = model.model.get_image_features(
        inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True
    )
    if len(output.pooler_output) != 1:
        raise ValueError("Expected exactly one image per prompt")
    return output.pooler_output[0]


def injected_inputs(inputs, features):
    """Same text inputs and grid metadata, but precomputed features instead of pixels."""
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "image_grid_thw": inputs["image_grid_thw"],
        "mm_token_type_ids": inputs["mm_token_type_ids"],
        # A fresh dict each call: generate() writes into this argument.
        "mm_encoder_outputs": {"image": BaseModelOutputWithPooling(pooler_output=(features,))},
    }


@contextmanager
def record_encoder_calls(model):
    """Temporarily record every vision-encoder output; the hook is always removed."""
    calls = []

    def hook(module, arguments, output):
        calls.append(output.pooler_output.detach().clone())

    handle = model.model.visual.register_forward_hook(hook)
    try:
        yield calls
    finally:
        handle.remove()


@contextmanager
def record_decoder_image_rows(model, input_ids):
    """Temporarily record the image-placeholder rows of embeddings entering the decoder."""
    image_rows = input_ids[0] == model.config.image_token_id
    calls = []

    def hook(module, args, kwargs):
        calls.append(kwargs["inputs_embeds"][0][image_rows].detach().clone())

    handle = model.model.language_model.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield calls
    finally:
        handle.remove()


def prefill_logits(model, kwargs):
    model.model.rope_deltas = None  # no state carried over from earlier calls
    return model(**kwargs, use_cache=False).logits.float()


def generate(model, kwargs, generation_config):
    model.model.rope_deltas = None  # fresh cache is created inside each generate()
    output = model.generate(**kwargs, generation_config=generation_config)
    return output[0, kwargs["input_ids"].shape[1]:]


def injected_prefill(model, inputs, features):
    """Prefill logits with injected features, proving the decoder saw exactly them."""
    with record_encoder_calls(model) as encoder_calls, \
            record_decoder_image_rows(model, inputs["input_ids"]) as decoder_rows:
        logits = prefill_logits(model, injected_inputs(inputs, features))
    if encoder_calls:
        raise RuntimeError("Vision encoder ran during an injected forward pass")
    if len(decoder_rows) != 1 or not torch.equal(decoder_rows[0], features):
        raise RuntimeError("Decoder image embeddings differ from the injected features")
    return logits


# --------------------------------------------------------------------- phases

def check_identity(model, inputs, generation_config):
    """Phase 2: original features injected by us must reproduce ordinary execution."""
    features = encode_image(model, inputs)
    with record_encoder_calls(model) as calls:
        reference_ids = generate(model, dict(inputs), generation_config)
    if len(calls) != 1 or not torch.equal(calls[0], features):
        raise RuntimeError("Ordinary generate() used different visual features")

    identity_ids = generate(model, injected_inputs(inputs, features), generation_config)
    reference_logits = prefill_logits(model, dict(inputs))
    rerun_logits = prefill_logits(model, dict(inputs))
    identity_logits = injected_prefill(model, inputs, features)

    rerun_diff = (rerun_logits - reference_logits).abs().max().item()
    identity_diff = (identity_logits - reference_logits).abs().max().item()
    report = {
        "reference_rerun_max_abs_logit_diff": rerun_diff,
        "identity_max_abs_logit_diff": identity_diff,
        "tolerance": "identity diff <= reference rerun diff (run-to-run noise)",
        "same_next_token_argmax": torch.equal(
            identity_logits[:, -1].argmax(-1), reference_logits[:, -1].argmax(-1)
        ),
        "same_generated_ids": torch.equal(identity_ids, reference_ids),
    }
    report["passed"] = (identity_diff <= rerun_diff and report["same_next_token_argmax"]
                        and report["same_generated_ids"])
    return features, reference_logits, report


def measure_codec(features, grid_thw, device, runs, warmups):
    """Phases 3-4: INT8 round trip, each stage timed separately. Returns restored features."""
    timings = {}
    # GPU quantization; timing includes the BF16 -> FP32 cast inside quantize().
    (q, scale), timings["gpu_quantize"] = gpu_time(
        lambda: quantize(features), device, runs, warmups)
    (q_cpu, scale_cpu), timings["transfer_d2h_int8_and_scales"] = gpu_time(
        lambda: (q.cpu(), scale.cpu()), device, runs, warmups)
    payload, timings["cpu_serialize"] = cpu_time(
        lambda: serialize(q_cpu, scale_cpu, features.dtype, grid_thw), runs, warmups)
    (q2, scale2, dtype, shape, grid2), timings["cpu_deserialize"] = cpu_time(
        lambda: deserialize(payload), runs, warmups)
    (q_gpu, scale_gpu), timings["transfer_h2d_int8_and_scales"] = gpu_time(
        lambda: (q2.to(device), scale2.to(device)), device, runs, warmups)
    restored, timings["gpu_reconstruct"] = gpu_time(
        lambda: dequantize(q_gpu, scale_gpu, dtype), device, runs, warmups)

    # CPU variant, for continuity with experiment1 (original BF16 copied to CPU first).
    features_cpu, timings["transfer_d2h_original_bf16"] = gpu_time(
        lambda: features.cpu(), device, runs, warmups)
    (q_c, scale_c), timings["cpu_quantize"] = cpu_time(
        lambda: quantize(features_cpu), runs, warmups)
    _, timings["cpu_reconstruct"] = cpu_time(
        lambda: dequantize(q_c, scale_c, features.dtype), runs, warmups)

    if shape != list(features.shape) or dtype != features.dtype or grid2 != grid_thw:
        raise RuntimeError("Deserialized metadata does not match the original features")
    if restored.shape != features.shape or restored.dtype != features.dtype:
        raise RuntimeError("Reconstruction has the wrong shape or dtype")
    if not torch.equal(restored, dequantize(q, scale, features.dtype)):
        raise RuntimeError("Serialized round trip changed the reconstruction")

    original_serialized = len(serialize_original(features_cpu))
    sizes = {
        "original_raw_bytes": raw_bytes(features),
        "int8_raw_bytes": raw_bytes(q),
        "scale_raw_bytes": raw_bytes(scale),
        "compressed_raw_bytes": raw_bytes(q) + raw_bytes(scale),
        "original_serialized_bytes": original_serialized,
        "compressed_serialized_bytes": len(payload),
        "cpu_gpu_quantization_identical":
            torch.equal(q_c, q.cpu()) and torch.equal(scale_c, scale.cpu()),
    }
    sizes["raw_ratio"] = sizes["original_raw_bytes"] / sizes["compressed_raw_bytes"]
    sizes["serialized_ratio"] = original_serialized / len(payload)
    return restored, payload, timings, sizes


def serialize_original(features_cpu):
    """Original BF16 tensor through the same torch.save container, for a fair byte count."""
    buffer = io.BytesIO()
    torch.save(features_cpu.contiguous(), buffer)
    return buffer.getvalue()


def timed_generation(model, kwargs, generation_config, device, eos_ids):
    (new_ids,), timing = gpu_time(
        lambda: (generate(model, kwargs, generation_config),), device, runs=1, warmups=0)
    hit_limit = not any(token in eos_ids for token in new_ids.tolist())
    return new_ids, {**timing, "output_tokens": new_ids.numel(), "hit_token_limit": hit_limit}


def evaluate_example(row, model, processor, generation_config, device, args):
    with Image.open(row["image_path"]) as source:
        image = source.convert("RGB")
    inputs_cpu, preprocess_timing = cpu_time(
        lambda: build_inputs(processor, image, row["question"]), args.runs, args.warmups)
    # Not BatchFeature.to(): it moves tensors in place, so repeats would time no-ops.
    inputs, h2d_timing = gpu_time(
        lambda: {k: v.to(device) for k, v in inputs_cpu.items()}, device,
        args.runs, args.warmups)
    grid_thw = inputs["image_grid_thw"].tolist()

    with torch.inference_mode():
        features, original_logits, identity = check_identity(model, inputs, generation_config)
        if not identity["passed"]:
            raise RuntimeError(f"Identity injection failed: {identity}")

        _, encoder_timing = gpu_time(
            lambda: encode_image(model, inputs), device, args.encoder_runs, args.warmups)
        restored, payload, codec_timings, sizes = measure_codec(
            features, grid_thw, device, args.runs, args.warmups)

        # The compressed path must consume exactly `restored`, not the original features.
        if torch.equal(restored, features):
            raise RuntimeError("Reconstruction equals the original; nothing to compare")
        compressed_logits = injected_prefill(model, inputs, restored)

        eos = generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        original_ids, original_generation = timed_generation(
            model, injected_inputs(inputs, features), generation_config, device, eos_ids)
        compressed_ids, compressed_generation = timed_generation(
            model, injected_inputs(inputs, restored), generation_config, device, eos_ids)

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
        "identity_check": identity,
        "compressed_prefill_max_abs_logit_diff":
            (compressed_logits - original_logits).abs().max().item(),
        "compressed_next_token_argmax_changed": not torch.equal(
            compressed_logits[:, last].argmax(-1), original_logits[:, last].argmax(-1)),
        "reconstruction": reconstruction_errors(features, restored),
        "sizes": sizes,
        "timing": {
            "cpu_preprocess": preprocess_timing,
            "transfer_h2d_processor_inputs": h2d_timing,
            "gpu_encoder_through_merger": encoder_timing,
            **codec_timings,
        },
        "original": {"answer": processor.decode(original_ids, skip_special_tokens=True),
                     "generation": original_generation},
        "compressed": {"answer": processor.decode(compressed_ids, skip_special_tokens=True),
                       "generation": compressed_generation},
    }
    return record, payload


def describe_run(args, model, processor, generation_config, device):
    """Everything needed to reproduce or interpret the run."""
    import importlib.util

    properties = torch.cuda.get_device_properties(device)
    return {
        "settings": {k: [str(p) for p in v] if k == "manifests" else
                     str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "checkpoint": args.checkpoint,
        # Loaded with local_files_only from this pinned revision's cached snapshot.
        "checkpoint_revision": args.revision,
        "checkpoint_snapshot": getattr(model.config, "_name_or_path", None),
        "model_dtype": str(model.dtype),
        "attention_backend": {
            "language_model_full_attention": model.config.text_config._attn_implementation,
            "vision": model.config.vision_config._attn_implementation,
            "linear_attention_fast_kernels_installed": {
                name: importlib.util.find_spec(name) is not None
                for name in ("fla", "causal_conv1d")
            },
        },
        "prompt": {"answer_instruction": ANSWER_INSTRUCTION, "enable_thinking": False},
        "generation_config": generation_config.to_diff_dict(),
        "image_processor": processor.image_processor.to_dict(),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "transformers": transformers.__version__,
        },
        "hardware": {
            "gpu": properties.name,
            "gpu_uuid": str(properties.uuid),
            "gpu_memory_bytes": properties.total_memory,
            "device": str(device),
            "cpu": platform.processor() or platform.machine(),
            "torch_cpu_threads": torch.get_num_threads(),
        },
        "timing_collection": {
            "mode": "minimal" if args.minimal_timing else "repeated",
            "runs": args.runs, "warmups": args.warmups, "encoder_runs": args.encoder_runs,
            "generation_runs": 1,
        },
        "measurement_notes": {
            "encoder_boundary": "get_image_features: patch embed -> 27 ViT blocks -> "
                                "visual.merger output (before per-image split and scatter)",
            "quantization_site": "GPU (cuda), timing includes BF16->FP32 cast; "
                                 "CPU variant also measured",
            "gpu_timing": "CUDA events on the device's current stream",
            "cpu_timing": "time.perf_counter wall clock",
            "serialization": "torch.save/torch.load to an in-memory buffer; not network",
            "generation": "single run per condition, both via injected features "
                          "(encoder excluded), fresh cache per call",
        },
    }


def run_fingerprint(args, model, processor, generation_config, device, rows, run_info):
    """Settings that must match for --resume. Timing and output options are excluded."""
    codec_path = REPO_ROOT / "src" / "vit_compression_lab" / "int8_codec.py"
    return {
        "checkpoint": args.checkpoint,
        "revision": args.revision,
        "model_config_commit": getattr(model.config, "_commit_hash", None),
        "model_dtype": str(model.dtype),
        "gpu": torch.cuda.get_device_properties(device).name,
        "attention_backend": run_info["attention_backend"],
        "prompt": {"answer_instruction": ANSWER_INSTRUCTION, "enable_thinking": False},
        "generation_config": generation_config.to_diff_dict(),
        "image_processor": processor.image_processor.to_dict(),
        "compression": {"scheme": "symmetric per-token INT8, FP32 scales (int8_codec.py)",
                        "codec_sha256": sha256_file(codec_path)},
        "manifests": [{"path": str(p), "sha256": sha256_file(p)} for p in args.manifests],
        "examples": {"count": len(rows),
                     "ids_sha256": sha256_text("\n".join(r["id"] for r in rows))},
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
    }


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifests", type=Path, nargs="+", help="JSONL manifest(s)")
    parser.add_argument("--ids", nargs="*", help="Only these manifest ids")
    parser.add_argument("--limit", type=int, help="Only the first N selected examples (smoke tests)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interrupted run in --output-dir; settings must match")
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--revision", default="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
    parser.add_argument("--device", default="cuda:1", help="GPU 0 is shared on this host")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--threads", type=int, default=16, help="CPU threads for torch")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50, help="Repeats for codec/preprocess")
    parser.add_argument("--encoder-runs", type=int, default=20)
    parser.add_argument("--minimal-timing", action="store_true",
                        help="runs=1, warmups=0, encoder-runs=1: no repeated latency "
                             "measurement; quality checks are unchanged")
    parser.add_argument("--skip-payloads", action="store_true",
                        help="Do not save payloads/<id>.pt (about 3.4 MB per example)")
    parser.add_argument("--max-failures", type=int, default=10,
                        help="Stop after this many failed examples in one session")
    args = parser.parse_args()

    rows = load_manifests(args.manifests, args.ids)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        rows = rows[:args.limit]
    if args.minimal_timing:
        args.runs, args.warmups, args.encoder_runs = 1, 0, 1
    if args.output_dir.exists() and not args.resume:
        parser.error("Output directory already exists; choose a new one or pass --resume")
    if args.resume and not args.output_dir.exists():
        parser.error("Nothing to resume: the output directory does not exist")
    torch.set_num_threads(args.threads)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    # Cached files only: this never downloads model assets.
    processor = AutoProcessor.from_pretrained(
        args.checkpoint, revision=args.revision, local_files_only=True
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.checkpoint, revision=args.revision, local_files_only=True,
        dtype=torch.bfloat16, device_map={"": device},
    ).eval()

    # The checkpoint defaults to sampling; both conditions use this greedy config.
    defaults = model.generation_config
    generation_config = GenerationConfig(
        do_sample=False, max_new_tokens=args.max_new_tokens,
        eos_token_id=defaults.eos_token_id, pad_token_id=defaults.pad_token_id,
        bos_token_id=defaults.bos_token_id,
    )

    run_info = describe_run(args, model, processor, generation_config, device)
    fingerprint = run_fingerprint(args, model, processor, generation_config, device, rows, run_info)
    if args.resume:
        try:
            completed, repaired = check_resume(args.output_dir, fingerprint)
        except ResumeError as error:
            raise SystemExit(f"Cannot resume: {error}")
        if repaired:
            print("Discarded a partial last line of results.jsonl left by an interrupted write")
    else:
        completed = set()
        args.output_dir.mkdir(parents=True)
        write_json(args.output_dir / "run_config.json", fingerprint)
        write_json(args.output_dir / "run_info.json", run_info)
    lock = open(args.output_dir / ".lock", "w")  # held until the process exits
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another session is already writing to this output directory")
    if not args.skip_payloads:
        (args.output_dir / "payloads").mkdir(exist_ok=True)
    session = {"started_at": now(), "resume": args.resume, "completed_before": len(completed),
               "timing_collection": run_info["timing_collection"],
               "hardware": run_info["hardware"], "software": run_info["software"]}
    append_record(args.output_dir / "sessions.jsonl", session)

    pending = [row for row in rows if row["id"] not in completed]
    print(f"{len(completed)} completed, {len(pending)} pending of {len(rows)} examples")
    failures = 0
    session_start = time.time()
    for index, row in enumerate(pending, 1):
        example_start = time.time()
        try:
            record, payload = evaluate_example(row, model, processor, generation_config, device, args)
        except Exception as error:  # recorded; the example stays pending for a resume
            failures += 1
            append_record(args.output_dir / "failures.jsonl", {
                "id": row["id"], "error_type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc(), "time": now()})
            print(f"FAILED {row['id']}: {type(error).__name__}: {error}")
            torch.cuda.empty_cache()
            if failures >= args.max_failures:
                raise SystemExit(f"Stopping after {failures} failures; see failures.jsonl, "
                                 "fix the cause, then rerun with --resume")
            continue
        record["elapsed_s"] = time.time() - example_start
        record["timing_mode"] = run_info["timing_collection"]["mode"]
        if not args.skip_payloads:
            (args.output_dir / "payloads" / f"{row['id']}.pt").write_bytes(payload)
        append_record(args.output_dir / "results.jsonl", record)
        done = len(completed) + index - failures
        elapsed = time.time() - session_start
        write_json(args.output_dir / "progress.json", {
            "examples": len(rows), "completed": done, "remaining": len(rows) - done,
            "failed_this_session": failures, "session_started_at": session["started_at"],
            "updated_at": now(), "session_seconds": elapsed,
            "seconds_per_example_this_session": elapsed / index,
            "eta_seconds": elapsed / index * (len(rows) - done), "last_id": row["id"]})
        print(f"{done}/{len(rows)} {row['id']} {record['elapsed_s']:.1f}s "
              f"original={record['original']['answer']!r} "
              f"compressed={record['compressed']['answer']!r}")


if __name__ == "__main__":
    main()
