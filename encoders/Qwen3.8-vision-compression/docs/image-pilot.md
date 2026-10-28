# Local image compression pilot

The existing single-image scripts remain unchanged. `scripts/evaluate_images.py`
in this encoder directory adds paired answer generation and CPU compression
measurements. It requires the already-cached checkpoint and processor; it never
downloads assets. Running it loads the model and uses the GPU.

## Prepare inputs

Copy `configs/image-pilot.example.json` to a local manifest, then replace the
reference with an answer based on inspecting the image yourself. Start with one
image. Add 20–50 diverse local images after debugging: documents/text, clutter,
small objects, people/actions, low-light/blur, and varied sizes/aspect ratios.
Each entry requires image, category, question, and reference strings. Image paths
are relative to the manifest. Keep private datasets/manifests under ignored
`data/`; do not commit sensitive images.

## Run explicitly when ready

From the repository root, with the virtual environment activated:

```sh
python encoders/Qwen3.8-vision-compression/scripts/evaluate_images.py \
  configs/image-pilot.example.json --output-dir outputs/image-pilot-001
```

Choose a fresh output directory on each run; existing directories are refused.
Defaults: 16 CPU threads, 10 warmups, 100 timing samples, deterministic generation
with a 128-token output limit. Use `--runs`, `--threads`, `--warmups`, or
`--max-new-tokens` to change them. Keep settings fixed across the pilot.

## What it compares

For each image/question, the script runs generation with original merged visual
features and captures those features. It quantizes the CPU features to symmetric
per-token INT8 with FP32 scales, serializes and reloads that payload, and restores
the original dtype. A temporary vision forward hook inserts that reconstruction
before Qwen splits and inserts visual embeddings into the language-model input.
The second generation uses the same prompt and deterministic generation settings.
The encoder still executes in that second run: this is a quality test, not an
optimized serving pipeline or an end-to-end latency benchmark.

Per-image folders save original features, the quantized payload, reconstructed
features, and a JSON report. `results.json` contains all paired answers and
metrics; `summary.json` records settings and groups images by merged token count.
Reports include original/processed resolution, grid, token count, raw/serialized
sizes, explicitly named compression ratios, reconstruction errors, and median/p95
CPU quantization and reconstruction times. CPU timing excludes model execution,
serialization, GPU copies, and network transfer. Repeated timing uses the same
cached tensor; it is not a cold-cache client benchmark.

## Review answers

In a separate copy of `results.json`, mark `baseline_correct` and
`reconstructed_correct` true/false against your references. Null means unscored,
not incorrect. Define acceptable answers before looking at outputs. Report the
accuracy of both paths, correct-to-incorrect regressions, and incorrect-to-correct
changes. Equivalent wording can be correct. Numerical errors alone do not measure
understanding. Check truncation if either answer reaches the token limit.

Compare results within token-count groups. Do not interpret a larger image's
longer compression time as necessarily worse implementation performance.

Short video evaluation, deployment/network timing, and automated quality scoring
are follow-up work. Image results cannot establish video quality.
