# Entropic Student configs (Qwen3.8-27B)

One JSON file per Stage 1 run; the `stage2` block of the same file configures
Stage 2. Unspecified keys take the defaults in
`src/vit_compression_lab/entropic/config.py` (`DEFAULTS`). Files are named
`<option>_<input>_<encoder>_beta<beta>.json`.

| File | Option | Client runs | Latent per | Notes |
| --- | --- | --- | --- | --- |
| `c_merger_tokenmlp_beta{0.02,0.08,0.32}.json` | C | full ViT + merger | LLM token (5120 -> 256) | recommended first run; trains from the feature cache alone |
| `c_merger_gridconv_beta0.08.json` | C | full ViT + merger | LLM token | paper-style conv + GDN on the token grid |
| `b_block{8,16,27}_windowmlp_beta0.08.json` | B | stem + k blocks | LLM token (4 x 1152 -> 192) | deep supervision at the merger through the frozen suffix |
| `a_block0_attention_to_block8_beta0.08.json` | A | stem only + 2 attention blocks | LLM token | student replaces blocks 1-8 |
| `a_pixels_linear_to_block8_beta0.08.json` | A | nothing pretrained | LLM token | student stem from raw patches |

Key fields:

- `split.encoder_input` / `split.decoder_target`: `pixels`, `block_0`..`block_27`, `merger`.
- `encoder.type`: `token_mlp`, `window_mlp` (patch-level input only), `grid_conv` (`num_down` stride-2 stages).
- `encoder.latent_channels`: bottleneck width `C_z`; bits scale roughly with it.
- `distortion.type`: `mse` (paper) or `standardized_mse` (per-channel teacher std from `data.channel_stats`,
  written by `es_extract_teacher_features.py`); `reduction`: `sum_per_image` (paper) or `mean_per_element`.
- `deep_supervision`: extra distortion terms propagated through the frozen teacher to deeper points.
- `rate.beta`: the paper's beta (sum reductions, natural log). Sweep 0.01-2.56 as the authors did.
- `quantization.train`: `noise` (paper) or `ste` (straight-through rounding, ablation).
- `data.*`: manifests from `scripts/prepare_image_manifest.py`; `feature_cache` from
  `es_extract_teacher_features.py` (needed with `--cache-only`).
- `training.max_patches_per_batch`: dynamic batch size in ViT patches (~1200 per median image).

Paths are relative to the repository root. No data referenced here exists yet:
the training image directory must be supplied and indexed first.

## A/B/C pilot on broader data

`pilot5k/` contains three new Stage 1 configurations (not trained):

| Config | Split | Student encoder |
| --- | --- | --- |
| `a_pixels_merger_beta0.08.json` | pixels -> merger: full vision replacement | learned patch stem + 2 attention blocks + window MLP |
| `b_block8_windowmlp_beta0.08.json` | block 8 -> block 8, then frozen Qwen suffix | window MLP |
| `c_merger_tokenmlp_beta0.08.json` | merger -> merger | token MLP |

Suggested dataset: sample 2,750 unique images from COCO **train2014** and 2,750
from the TextVQA **training annotation IDs**, then split the 5,500-image pool
into 5,000 training and 500 held-out images. Stage 1 needs images, not questions
or answers. Keep the existing pilot/eval-3000 manifests reserved for evaluation.
The TextVQA image archive includes validation images too; select training IDs
before indexing. Official sources: [COCO/VQA images](https://visualqa.org/download.html),
[TextVQA](https://textvqa.org/dataset/).

After obtaining that image-only pool under `data/es-abc-pilot5k/images`, run from
the repository root (these commands require a working CUDA GPU):

```sh
.venv/bin/python scripts/prepare_image_manifest.py data/es-abc-pilot5k/images \
  --output data/es-abc-pilot5k/manifest.jsonl --val-fraction 0.09090909090909091 --seed 2026

.venv/bin/python encoders/Qwen3.8-vision-compression/scripts/es_extract_teacher_features.py \
  data/es-abc-pilot5k/manifest.jsonl --points block_8 merger --stats-only \
  --cache-dir outputs/qwen3.8/es-abc-pilot5k/stats --device cuda:1 --verify-depths 8
```

Confirm the manifests contain the expected 5,000/500 images after duplicate and
evaluation-image exclusions. Statistics must be computed from training only.
`--stats-only` writes channel statistics and extraction metadata, not cached
activations. All three configs set `feature_cache=null`, so teacher features are
recomputed during training. This trades GPU time for lower disk use.

The configs share 256 latent channels, beta 0.08, normalized decoder output,
an 8,192-patch batch budget, seed 0, and an initial 5,000-update budget.
They evaluate 32 fixed training images and up to 500 held-out images every 500
updates, and retain latest/best checkpoints without epoch or milestone copies.
Learning-rate milestones are disabled for this first controlled pilot.
Check for images skipped by the patch budget; all runs should use the same
eligible image set. Normalized output is a conditioning choice, not a proven
improvement to answer accuracy.

First measure a 100-update calibration run per config. For example, for B:

```sh
.venv/bin/python encoders/Qwen3.8-vision-compression/scripts/es_train_stage1.py \
  --config configs/qwen3_8/entropic_student/pilot5k/b_block8_windowmlp_beta0.08.json \
  --runs-dir runs/qwen3.8/es-abc-pilot5k --run-name calibration_b --device cuda:1 \
  --diagnostic-channel 0 --coded-examples 1 \
  --set training.max_steps=100 --set training.eval_every_steps=100 \
  --set data.train_eval_limit=8 --set data.val_limit=32
```

B reconstructs 1,152-channel block features: **pass `--diagnostic-channel 0`**;
the default channel 3994 is valid for A/C merger targets, but invalid for B.
Use distinct calibration run names for A/C. After inspecting timing and finite
losses, start fresh pilot runs using the commands below (no `--cache-only`):

```sh
.venv/bin/python encoders/Qwen3.8-vision-compression/scripts/es_train_stage1.py \
  --config configs/qwen3_8/entropic_student/pilot5k/a_pixels_merger_beta0.08.json \
  --runs-dir runs/qwen3.8/es-abc-pilot5k --device cuda:1 --coded-examples 4

.venv/bin/python encoders/Qwen3.8-vision-compression/scripts/es_train_stage1.py \
  --config configs/qwen3_8/entropic_student/pilot5k/b_block8_windowmlp_beta0.08.json \
  --runs-dir runs/qwen3.8/es-abc-pilot5k --device cuda:1 --coded-examples 4 --diagnostic-channel 0

.venv/bin/python encoders/Qwen3.8-vision-compression/scripts/es_train_stage1.py \
  --config configs/qwen3_8/entropic_student/pilot5k/c_merger_tokenmlp_beta0.08.json \
  --runs-dir runs/qwen3.8/es-abc-pilot5k --device cuda:1 --coded-examples 4
```

5,000 updates are a pilot budget, not a convergence guarantee, particularly for
full-replacement A. At 0.5/1/2 seconds per update, optimization alone takes
42/83/167 minutes per run; add extraction, evaluation, and checkpoint time.
Measure each architecture separately: B differentiates through the frozen
remaining vision blocks for its merger loss. All GPU timings are unmeasured
for these configs. Previous C timings used cached features and do not transfer
directly to these uncached runs.

Compare merger reconstruction, actual bits/visual token, client compute, and
downstream answers on identical evaluation examples. Equal beta/latent width
does not guarantee equal bitrate. Total training losses are not comparable
across these setups because B has an extra reconstruction term.
`es_eval_features.py` and `es_eval.py` support the configured splits. The old
`es_recon_diagnostics.py` supports merger->merger students only; the existing
study plots assume channel 3994 and need adaptation for B. Summarize B separately
with `es_recon_study_summary.py --channel 0`.
