# Entropic Student: how the pipeline is meant to be run

Documentation only. **None of these commands has been executed.** They are the
order in which the pieces are intended to be used once training images are
available and a GPU is free. All commands run from the repository root with the
existing virtual environment active, read the model from the local Hugging Face
cache (`local_files_only=True`), and never download anything.

Design: `docs/entropic-student-plan.md`. Configs: `configs/qwen3_8/entropic_student/`.

## 0. Unit tests (never run in the authoring session)

```sh
python -m pytest tests/test_entropic_*.py
python -m ruff check src/vit_compression_lab/entropic tests
```

## 1. Index training images (local files only)

The repository contains no training images. Point the tool at a directory you
already have; it refuses any image that appears in `data/eval-3000` or the pilots.

```sh
python scripts/prepare_image_manifest.py /path/to/images \
    --output data/train-images/manifest.jsonl --val-fraction 0.05
```

Writes `manifest.jsonl`, `val_manifest.jsonl`, `metadata.json` (with the list of refused images).

## 2. Verify the teacher split and probe the blocks (vision tower only)

```sh
# per-block statistics, outlier ratio, uniform-quantisation entropy, noise sensitivity
python encoders/Qwen3.8-vision-compression/scripts/es_probe_split_points.py \
    data/vqav2-pilot/subset.jsonl data/textvqa-pilot/subset.jsonl --limit 12 \
    --output outputs/qwen3.8/entropic-student/probe.json
```

Use the printed table to choose the Option B depth `k`. The probe also exercises
`TeacherSplit.suffix(prefix(...))`; the extraction step below can additionally
assert it against the unmodified forward pass with `--verify-depths`.

## 3. Cache teacher activations and channel statistics

```sh
python encoders/Qwen3.8-vision-compression/scripts/es_extract_teacher_features.py \
    data/train-images/manifest.jsonl data/train-images/val_manifest.jsonl \
    --points block_8 block_16 block_27 merger \
    --cache-dir outputs/qwen3.8/teacher-features/train-images \
    --verify-depths 0 8 16 27
```

Check `extraction_info.json` -> `verification`: `max_abs_diff` must be within
the run-to-run noise seen in `identity_check` of the inject_eval runs (0.0 on
the pilot). Disk: ~3 MB per image per point in BF16 plus ~7 MB of pixel values
(`--no-pixels` to omit them).

## 4. Stage 1: rate–distortion training

```sh
# Option C (recommended first): no model needed, trains from the cache
python encoders/Qwen3.8-vision-compression/scripts/es_train_stage1.py \
    --config configs/qwen3_8/entropic_student/c_merger_tokenmlp_beta0.08.json \
    --runs-dir runs/qwen3.8/entropic-student --cache-only --device cuda:1

# smoke test first: 50 images, 2 epochs
python encoders/Qwen3.8-vision-compression/scripts/es_train_stage1.py \
    --config configs/qwen3_8/entropic_student/c_merger_tokenmlp_beta0.08.json \
    --runs-dir runs/qwen3.8/entropic-student-smoke --cache-only --limit 50 --epochs 2

# Option B (teacher vision tower loaded for deep supervision through the frozen suffix)
python encoders/Qwen3.8-vision-compression/scripts/es_train_stage1.py \
    --config configs/qwen3_8/entropic_student/b_block16_windowmlp_beta0.08.json \
    --runs-dir runs/qwen3.8/entropic-student --device cuda:1

# beta sweep, as in the paper
for b in 0.02 0.08 0.32; do
  python encoders/Qwen3.8-vision-compression/scripts/es_train_stage1.py \
      --config configs/qwen3_8/entropic_student/c_merger_tokenmlp_beta$b.json --cache-only
done
```

Each run writes `runs/<name>/metrics.jsonl` (train and validation: loss, MSE,
cosine, bits per visual token; coded bytes, overhead ratio and exact round-trip
flag on a few images per epoch) and `checkpoints/{latest,best,epoch-XXX}.pt`.
Resume with `--resume`.

## 5. Feature-level evaluation (no LLM)

```sh
python encoders/Qwen3.8-vision-compression/scripts/es_eval_features.py \
    data/vqav2-pilot/subset.jsonl data/textvqa-pilot/subset.jsonl \
    --student-checkpoint runs/qwen3.8/entropic-student/c_merger_tokenmlp_beta0.08/checkpoints/best.pt \
    --baseline-codecs int8 int4-g64 \
    --output outputs/qwen3.8/entropic-student/c_merger_tokenmlp_beta0.08/features-eval.json
```

Reports median merger relative L2 / cosine, ideal and coded bits per visual
token, entropy-coding overhead, exact round trip, and the INT8 / INT4-g64
numbers on the same features (INT4-g64 reference: 21,760 bits per token,
relative L2 0.105 measured in Experiment 3).

## 6. End-to-end evaluation against the unmodified model

Same protocol and outputs as `inject_eval.py`; the existing scoring and
retention scripts run unchanged.

```sh
# 70-question pilot gate
python encoders/Qwen3.8-vision-compression/scripts/es_eval.py \
    data/vqav2-pilot/subset.jsonl data/textvqa-pilot/subset.jsonl \
    --student-checkpoint runs/qwen3.8/entropic-student/c_merger_tokenmlp_beta0.08/checkpoints/best.pt \
    --baseline-codec int4-g64 \
    --output-dir outputs/qwen3.8/inject-eval/pilot70-es-c-beta0.08-001
python encoders/Qwen3.8-vision-compression/scripts/score_results.py \
    outputs/qwen3.8/inject-eval/pilot70-es-c-beta0.08-001
python encoders/Qwen3.8-vision-compression/scripts/report_retention.py \
    outputs/qwen3.8/inject-eval/pilot70-es-c-beta0.08-001

# pre-registered 3,000-question evaluation (same manifest as Experiments 2-3)
python encoders/Qwen3.8-vision-compression/scripts/es_eval.py \
    data/eval-3000/manifest.jsonl --minimal-timing --skip-payloads \
    --student-checkpoint runs/qwen3.8/entropic-student/c_merger_tokenmlp_beta0.08/checkpoints/best.pt \
    --output-dir outputs/qwen3.8/inject-eval/eval3000-es-c-beta0.08-001
```

Per-example records add a `learned` block (split, latent shape, round-trip
flag, baseline codec bytes and error) and stage timings
(`gpu_client_prefix`, `gpu_student_encode`, `cpu_entropy_encode`,
`cpu_entropy_decode`, `transfer_h2d_latent`, `gpu_student_decode`,
`gpu_server_suffix`). The Python rANS stages are slow and are timed with
`--entropy-runs` repeats (default 1); treat them as byte counts, not latency.

## 7. Stage 2: task-level distillation (needs an idle 80 GB GPU)

```sh
python encoders/Qwen3.8-vision-compression/scripts/es_train_stage2.py \
    --student-checkpoint runs/qwen3.8/entropic-student/c_merger_tokenmlp_beta0.08/checkpoints/best.pt \
    --manifests data/train-images/vqa_manifest.jsonl \
    --runs-dir runs/qwen3.8/entropic-student --device cuda:1
```

Requires a manifest with questions (any VQA-style training set; not the
evaluation manifests). Then repeat steps 5–6 with the Stage 2 checkpoint.

## 8. Video (not exercised)

The payload header carries `modality` and per-video `grid_thw`;
`injection.injected_kwargs(..., modality="video")` passes `video_grid_thw`;
timestamps are text tokens produced by the processor and need no handling.
`data.preprocess_video_frames()` wraps the checkpoint's `Qwen3VLVideoProcessor`
but frame decoding, a video QA manifest format and a temporal benchmark do not
exist in this repository yet. Treat every video path as unverified.

## What to look at if something fails

| Symptom | First check |
| --- | --- |
| `load_vision_tower` key errors | `model.safetensors.index.json` prefix `model.visual.`; transformers version 5.21.0 |
| `verify()` max_abs_diff far above the identity-check noise | `TeacherSplit.geometry()` vs `Qwen3_5VisionModel.forward` in the installed transformers |
| Bits per token do not fall with smaller beta | `rate.reduction` / `distortion.reduction` scale; aux loss; try `quantization.train: ste` |
| `roundtrip_exact` false | `FactorizedEntropyModel.update(force=True)` was not called after the last optimiser step |
| Stage 2 OOM | `grad_checkpointing`, shorter `max_new_tokens`, a free GPU |
