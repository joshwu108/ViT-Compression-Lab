# ViT-compression-lab

A learning-focused research repository for compression of vision encoder
features. Qwen3.8 is the first encoder under study; other encoders can have
independent experiments alongside it.

## Layout

```text
encoders/
  Qwen3.8-vision-compression/
    scripts/experiment.py          # Vision encoder latency and feature export
    scripts/quantize_features.py   # Saved feature loading and validation
    scripts/inject_eval.py         # Original vs INT8 features through the same model
    scripts/score_results.py       # Answer scoring (primary rules + official VQA accuracy)
    scripts/report_retention.py    # Retention, bootstrap bounds, decision, Markdown report
    docs/experiment1.md            # Baseline results
    docs/experiment2.md            # 70-question pilot
    docs/experiment2-large-eval.md # 3,000-question retention evaluation
    docs/experiment3.md            # INT4: outlier channels and group scaling
src/
  vit_compression_lab/             # INT8/INT4 codecs, sampling, paired statistics, checkpointing
  images/test1.jpg                 # Existing Qwen test image
scripts/                          # Dataset preparation and VQA accuracy scoring
configs/                          # Experiment settings
tests/                           # Package checks
outputs/                          # Local generated artifacts (ignored)
```

Keep encoder-specific scripts and notes together under `encoders/<encoder>/`.
Introduce shared functions only when experiments need them; no common model
interface or training framework is required.

## Running the current experiments

Run from the repository root with your existing virtual environment activated:

```sh
python encoders/Qwen3.8-vision-compression/scripts/experiment.py
python encoders/Qwen3.8-vision-compression/scripts/quantize_features.py
```

The encoder script loads the model, benchmarks the vision encoder, and exports
`outputs/features.pt`. The quantization script currently loads and validates
that tensor; quantization is still being developed. These scripts use paths
relative to the repository root.

## Environment and checks

Use Python 3.11 or newer. Research dependencies are listed in `requirements.txt`;
choose versions compatible with your hardware. Package runtime dependencies are
kept empty so adding another encoder does not impose its dependencies on others.

For a fresh development environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run available checks with:

```sh
python -m pytest
python -m ruff check src/vit_compression_lab tests
python -m ruff format --check src/vit_compression_lab tests
```

## Research artifacts

Keep weights in an external model cache. Generated tensors, checkpoints, and
reports belong in ignored directories such as `outputs/`, `checkpoints/`, or
`runs/`. For future runs, use encoder-specific output directories such as
`outputs/<encoder>/<experiment>/` to avoid collisions. Existing scripts retain
`outputs/features.pt` until that change is made explicitly.

Record model revisions, dependency versions, preprocessing, hardware, dtype,
and attention backend alongside results. Numerical reconstruction error and
downstream answer quality should be evaluated separately.
