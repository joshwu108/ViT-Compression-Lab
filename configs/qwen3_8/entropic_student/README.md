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

## Pilot on broader data (`pilot5k/`): Option C is the only active experiment

`pilot5k/` holds three Stage 1 configurations. **Only C is being run.** The A and B
files are kept unchanged for future work and have not been trained.

| Config | Split | Student encoder | Status |
| --- | --- | --- | --- |
| `c_merger_tokenmlp_beta0.08.json` | merger -> merger | token MLP | **active**: trains from the feature cache alone |
| `a_pixels_merger_beta0.08.json` | pixels -> merger: full vision replacement | learned patch stem + 2 attention blocks + window MLP | not run |
| `b_block8_windowmlp_beta0.08.json` | block 8 -> block 8, then frozen Qwen suffix | window MLP | not run |

On this SSH host, the new pilot's storage lives under
`/mnt/nfs/joshua/ViT-compression-lab/`. The following local directory links
point to the corresponding paths on NFS:

```text
data/es-abc-pilot5k                -> /mnt/nfs/joshua/ViT-compression-lab/data/es-abc-pilot5k
outputs/qwen3.8/es-abc-pilot5k      -> /mnt/nfs/joshua/ViT-compression-lab/outputs/qwen3.8/es-abc-pilot5k
runs/qwen3.8/es-abc-pilot5k         -> /mnt/nfs/joshua/ViT-compression-lab/runs/qwen3.8/es-abc-pilot5k
```

Thus the relative paths in the configs and commands below already use NFS.
These links are local setup in Git-ignored directories; another checkout needs
equivalent storage setup. Existing experiments and the base-model cache retain
their current locations. To inspect the destination, run
`readlink -f data/es-abc-pilot5k`. A small create/write/sync/read/rename/delete
check passed through all three links. Available filesystem capacity does not
establish an individual quota; a `quota` tool is not installed on this host.

### Data

`scripts/prepare_es_train_pool.py` downloads only the selected images: 2,750 from
COCO **train2014** and 2,750 from the TextVQA **training annotation IDs** (seed
2026). Images whose source id, file SHA-256 or decoded-pixel SHA-256 appears in
`data/eval-3000`, the two pilots or `data/es-stage1-smoke`, and duplicates, are
rejected and replaced by the next candidate, so the counts stay exact. It splits
by image, stratified by source:

| File | Images |
| --- | --- |
| `manifest.jsonl` | 5,000 training (2,500 + 2,500) |
| `val_manifest.jsonl` | 500 validation (250 + 250) |
| `val_fast_manifest.jsonl` | fixed 64-image subset of validation (32 + 32) |
| `train_monitor_manifest.jsonl` | the first 32 training rows (16 + 16), which `data.train_eval_limit=32` evaluates |

```sh
.venv/bin/python scripts/prepare_es_train_pool.py --output-dir data/es-abc-pilot5k
```

### Feature cache and training-only statistics

C needs only merger features and `grid_thw`, so the cache stores no pixels
(`--no-pixels`). One command caches training and validation images and then
computes the channel mean/std by streaming the completed cache over the ids of
`--stats-manifests` only. Rerunning it resumes; cached entries are reused only if
their stored image hash, model revision, dtype and preprocessing hash match.

```sh
.venv/bin/python encoders/Qwen3.8-vision-compression/scripts/es_extract_teacher_features.py \
  data/es-abc-pilot5k/manifest.jsonl data/es-abc-pilot5k/val_manifest.jsonl \
  --points merger --no-pixels --device cuda:1 \
  --cache-dir outputs/qwen3.8/es-abc-pilot5k/cache \
  --stats-manifests data/es-abc-pilot5k/manifest.jsonl \
  --stats-output outputs/qwen3.8/es-abc-pilot5k/stats/channel_stats.pt
```

`stats/channel_stats.json` records the training manifest hash and the image and
token counts behind the statistics.

### Training

The C config uses 256 latent channels, beta 0.08, normalized decoder output, an
8,192-patch batch budget, learning rate 0.001, seed 0, no learning-rate
milestones, and 5,000 optimizer updates. Two fields set the validation cadence:

- `training.eval_every_steps=500`: the 32 training-monitor images and the 64
  images of `data.val_fast_manifests`; writes `latest.pt`.
- `training.full_eval_every_steps=1000` (and the last update): the full 500
  validation images too. Only these evaluations can replace `best.pt`, which is
  the checkpoint with the lowest full-validation loss with rounded latents.

With `--cache-only` the trainer loads neither the Qwen vision tower nor the image
processor; batch sizes come from the cached grids. An image above the whole patch
budget forms a batch of its own instead of being dropped.

```sh
# 100-update calibration in its own run directory
.venv/bin/python encoders/Qwen3.8-vision-compression/scripts/es_train_stage1.py \
  --config configs/qwen3_8/entropic_student/pilot5k/c_merger_tokenmlp_beta0.08.json \
  --runs-dir runs/qwen3.8/es-abc-pilot5k --run-name c_calibration100_s0 --device cuda:1 \
  --cache-only --coded-examples 4 \
  --set training.max_steps=100 --set training.eval_every_steps=50 --set training.full_eval_every_steps=100

# the pilot run (add --resume to continue it from checkpoints/latest.pt)
.venv/bin/python encoders/Qwen3.8-vision-compression/scripts/es_train_stage1.py \
  --config configs/qwen3_8/entropic_student/pilot5k/c_merger_tokenmlp_beta0.08.json \
  --runs-dir runs/qwen3.8/es-abc-pilot5k --device cuda:1 --cache-only --coded-examples 4

tail -f runs/qwen3.8/es-abc-pilot5k/abc5k_c_merger_tokenmlp_beta0.08_s0/metrics.jsonl
```

5,000 updates are a pilot budget, not a convergence guarantee, and not 5,000
images: `metrics.jsonl` records `images_seen` and `tokens_seen`.

### Evaluation

```sh
# full-validation reconstruction and actual coded size, from the cache (no Qwen model)
.venv/bin/python encoders/Qwen3.8-vision-compression/scripts/es_eval_cached.py \
  data/es-abc-pilot5k/val_manifest.jsonl \
  --student-checkpoint runs/qwen3.8/es-abc-pilot5k/abc5k_c_merger_tokenmlp_beta0.08_s0/checkpoints/best.pt \
  --output outputs/qwen3.8/es-abc-pilot5k/evaluation/val500_best.json
```

Downstream answers on the fixed 70-question pilot use `es_eval.py`, then
`score_results.py` and `report_retention.py`, exactly as for the INT8 / INT4-g64
runs. Results are in `outputs/qwen3.8/es-abc-pilot5k/REPORT.md`.

### Notes for A and B (not run)

A and B have `feature_cache=null` and recompute teacher features during
training (no `--cache-only`); B reconstructs 1,152-channel block features, so pass
`--diagnostic-channel 0` (the default channel 3994 is valid only for merger
targets). Total training losses are not comparable across the setups because B
has an extra reconstruction term, and equal beta/latent width does not guarantee
equal bitrate. All A/B timings are unmeasured.
