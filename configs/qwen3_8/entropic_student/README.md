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
