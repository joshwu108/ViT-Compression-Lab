# qwen-vision-compression

A learning-focused research repository for studying client-side Qwen vision
encoders for real-time video understanding, with future exploration of feature
compression, fine-tuning, and teacher–student distillation.

This scaffold contains no model implementation, pretrained weights, model or
processor loading code, inference code, or training code. No checkpoint or GPU
setup has been tested.

## Directory map

```text
.
├── README.md
├── AGENTS.md
├── pyproject.toml
├── .gitignore
├── configs/README.md
├── scripts/README.md
├── src/qwen_vision_compression/__init__.py
└── tests/test_package.py
```

## Setup

Use Python 3.11 or newer:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Runtime dependencies are intentionally empty. Install compatible research
dependencies after choosing your execution hardware. If local architecture
changes become necessary, a separate Transformers checkout can later be installed
in editable mode using `python -m pip install -e /path/to/your/transformers-checkout`;
replace that example path with the checkout's actual location.

## Checks

After installation:

```sh
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

The single smoke test checks that the package imports.

## Learning sequence

Manually implement loading → inspect feature shapes → measure an unmodified
baseline → explore compression → fine-tune or distill.

Keep downloaded pretrained weights in an external model cache, outside this
repository. Put generated checkpoints and measurements in ignored local
directories such as `checkpoints/` and `outputs/`. Root-level `data/`, `runs/`,
and `artifacts/` are also ignored.

When experiments begin, record the model revision, dependency versions,
preprocessing settings, hardware, dtype, and attention backend so results can be
understood and reproduced.
