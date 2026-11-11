# Entropic Student for Qwen3.8 visual features: research notes and implementation plan

Status: design document written before any training or inference was run. Every
number marked *measured* comes from Experiments 1–3 in this repository; every
number marked *estimate* is arithmetic on the model configuration. Items marked
**UNVERIFIED** must be checked by running the listed command before they are
relied upon. Nothing in this document was executed.

---

## 1. Research summary: Matsubara et al., WACV 2022

Source: *Supervised Compression for Resource-Constrained Edge Computing Systems*
(Matsubara, Yang, Levorato, Mandt), arXiv 2108.11898 / WACV 2022. The official
code is `github.com/yoshitomo-matsubara/supervised-compression` (torchdistill +
CompressAI); the extended version is `sc2-benchmark` (`sc2bench`). The CVF PDF
returned HTTP 403 during this session, so the arXiv version was used; the
training recipe below was cross-checked against the authors' released YAML
config `bottleneck_resnet50-b24ch_igdn-beta0.08_from_resnet50.yaml`.

### 1.1 Architecture (Sec. 3.1)

* Teacher: ResNet-50, `x -> h -> y`, where `h` is the output of the **first
  residual stage (`layer1`)**. The student replaces *only* the stem and
  `layer1` ("all the layers before its second residual block"). `layer2–4`,
  pooling and the classifier are copied from the teacher.
* Student encoder `f_theta`: convolutions with GDN (generalized divisive
  normalisation). In the released code (`FPBasedResNetBottleneck`): Conv5x5/s2
  -> GDN -> Conv5x5/s2 -> GDN -> Conv2x2, channels 3 -> 96 -> 48 -> **24**
  bottleneck channels. Mobile-side parameter count 0.14 M (their Table 1).
* Student decoder `g_phi`: Conv2x2 -> IGDN -> Conv2x2 -> IGDN -> Conv2x2,
  24 -> 512 -> 256 -> 256 channels, output shape equal to `layer1`'s output.
* Bottleneck `z = f_theta(x)`, quantised to integers, entropy coded.

### 1.2 Objective (Sec. 3.2)

Eq. (1): `L(x,h) = -E_{q(z|x)}[log p_phi(h|z)] - beta * E[log p_psi(z)]`, with
`q(z|x)` a unit-width box centred on `f_theta(x)` and `p(h|z) = N(h; g_phi(z), I)`,
which gives the training form (their Eq. 2, the Stage 1 loss requested here):

```
L_1 = 1/2 * || h - g_phi(f_theta(x) + eps) ||_2^2  -  beta * log p_psi(f_theta(x) + eps),
eps ~ Uniform(-1/2, 1/2)
```

* Rate term: `p_psi(z)` is the **fully factorised learned prior** of Ballé et
  al. 2018 (one non-parametric cumulative per latent channel, shared over
  spatial positions). No hyperprior is used in the Entropic Student.
* Quantisation: additive uniform noise in training; **rounding** at inference.
  Entropy coding uses the CompressAI range coder with the prior's CDF tables
  (`EntropyBottleneck.update()`), including an escape symbol for out-of-range
  values.
* The released config implements the distortion as `MSELoss(reduction="sum")`
  on **`layer1` through `layer4` outputs** (teacher vs student, factor 1.0
  each) plus `BppLoss(reduction="sum", factor=beta)`. So in practice the
  distortion is applied at the replaced layer *and* propagated through the
  frozen downstream blocks (`layer2–4` are frozen in Stage 1). This "deep
  supervision" detail matters for the ViT adaptation and is kept here.
* beta sweep in the released configs: 0.01, 0.02, 0.04, 0.08, 0.16, 0.32,
  0.64, 1.28, 2.56 (sum reductions, natural log).

### 1.3 Two-stage training (Sec. 3.3, Appendix E.2)

| Stage | Trainable | Frozen | Loss | Optimiser |
| --- | --- | --- | --- | --- |
| 1 | encoder, decoder, prior | teacher; student `layer2–4` | Eq. 2 (+ MSE on later frozen layers) | Adam 1e-3, 10 epochs, x0.1 at 5, 8; batch 64 |
| 2 | decoder, `layer2–4`, classifier | **encoder and entropy bottleneck** (so the bitrate is fixed and the encoder can be shared across tasks) | `alpha * CE(y_hat, y) + (1-alpha) * tau^2 * KL(o_T, o_S)`, alpha=0.5, tau=1 | SGD 1e-3, momentum 0.9, wd 5e-4, batch 128, x0.1 at 5 |

Once the encoder is frozen, Stage 2 uses rounding (not noise) for `z`.
Detection/segmentation reuse the Stage 1 encoder and fine-tune decoder + heads.

### 1.4 Results and limitations relevant here

* ImageNet: best supervised rate–distortion curve against CR+BQ (channel
  reduction + 8-bit quantisation), Ballé factorised prior, Minnen mean-scale
  hyperprior and JPEG/WebP/BPG input compression. Removing the teacher
  (end-to-end training from the task loss) was "significantly worse".
* Encoder 10–40x smaller than neural image codecs; latency reduction 47–62 %
  on a 37.5 kbit/s LoRa link (Table 2).
* Limitations stated or implied: gains on detection are smaller; end-to-end
  training from an early stochastic bottleneck is hard to optimise; one
  operating point for latency; no energy measurements.

### 1.5 What transfers to a ViT + LLM, and what does not

| Paper (CNN classifier) | Qwen3.8 (ViT -> merger -> LLM) | Consequence |
| --- | --- | --- |
| Replace stem + `layer1`; later stages frozen | Replace the stem + first `k` ViT blocks; blocks `k+1..27` + merger frozen on the server | Same shape of idea, but the ViT has no spatial downsampling between blocks, so every block's activation is `[N_patches, 1152]` — the student must do all the dimensionality reduction |
| 2-D conv + GDN over a feature map | Tokens are a ragged packed sequence ordered in 2x2 merge blocks | Need token<->grid reshaping; per-token MLPs or window MLPs are the natural ViT analogues; 2-D conv is kept as an option |
| `h` is a 256-channel feature map with ordinary dynamics | Merger output has one channel ~61x the token RMS (*measured*, Experiment 3) | Plain MSE at the merger is dominated by channel 3994; a channel-standardised distortion is offered as an explicit deviation from the paper |
| Stage 2 = CE + KD on 1000-way logits | Downstream is autoregressive text generation | Stage 2 becomes token-level KL between next-token distributions with original vs reconstructed visual tokens (teacher-forced), LLM frozen |
| Teacher is cheap to run | Teacher LLM is 27 B parameters (~54 GB BF16) | Stage 1 must not need the LLM: only the 0.44 B-parameter vision tower is loaded (one safetensors shard) |
| Fixed 224x224 input | Dynamic resolution, 300 visual tokens median, up to 16 MP | Variable-length packing, dynamic batching by patch budget |
| Position comes from conv structure | Absolute pos-embed (bilinear 48x48 table) + 2-D axial RoPE | Server recomputes RoPE/cu_seqlens from `grid_thw` (3 ints), which is transmitted anyway; the pos-embed is part of the stem |

---

## 2. Infrastructure assessment (verified from the repository and the installed packages)

| Item | Finding | Where verified |
| --- | --- | --- |
| Checkpoint | `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, cached locally, 18 shards | `inject_eval.py`, `~/.cache/huggingface/hub` |
| Architecture class | `Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5` (transformers 5.21.0). **"Qwen3.8" is the checkpoint name; the architecture is the Qwen3.5 family, not Qwen3-VL.** | `config.json`, `modeling_qwen3_5.py` |
| Vision encoder | 27 blocks, hidden 1152, 16 heads (head_dim 72), MLP 4304 with `gelu_pytorch_tanh`, pre-LN blocks, 2-D axial RoPE, learned 48x48 position table bilinearly resampled, patch 16, temporal patch 2, Conv3d patch embed `[3,2,16,16] -> 1152` | `Qwen3_5VisionModel` |
| Merger | `LayerNorm(1152)` per patch -> view `[N/4, 4608]` -> `Linear(4608,4608)` -> GELU -> `Linear(4608,5120)`; output `[N/4, 5120]` | `Qwen3_5VisionPatchMerger`, `out_hidden_size: 5120` |
| DeepStack | **Not used.** `deepstack_visual_indexes: []`; zero occurrences of "deepstack" in `modeling_qwen3_5.py` (31 in `modeling_qwen3_vl.py`) | `config.json`, grep |
| Patch order | `(t, h/2, w/2, 2, 2)`: consecutive groups of 4 patches form one 2x2 merge window | `image_processing_qwen2_vl.py`, `get_vision_position_ids` |
| Injection path | `mm_encoder_outputs={"image": BaseModelOutputWithPooling(pooler_output=(features,))}`; `Qwen3_5Model.forward` `masked_scatter`s them into the `<image_pad>` rows; identity checks already exist | `inject_eval.py`, `modeling_qwen3_5.py` |
| Position metadata | `image_grid_thw` + `mm_token_type_ids` -> M-RoPE position ids via `get_rope_index`; nothing else is derived from pixels | `compute_3d_position_ids` |
| Video | `Qwen3VLVideoProcessor` (fps 2 default, temporal patch 2); timestamps are inserted as **text** `<X.X seconds>` tokens around each frame's `<vision_start>...<vision_end>`; `get_video_features` is literally `get_image_features`; `video_grid_thw` rows are split per frame for RoPE | `video_preprocessor_config.json`, `processing_qwen3_vl.py`, `get_rope_index` |
| LLM | 64 layers, hidden 5120, Gated DeltaNet linear attention with full attention every 4th layer, `fla`/`causal_conv1d` kernels installed | `config.json`, `run_info.json` |
| Existing codecs | `int8`, `int4`, `int4-g64`, `int4-g32` in `src/vit_compression_lab/{int8,int4}_codec.py`, registry `codecs.py` | — |
| Existing eval | `inject_eval.py` (identity check, codec round trip, timing, generation) -> `score_results.py` -> `report_retention.py` (paired cluster bootstrap). Results schema in `results.jsonl` | — |
| Data | `data/eval-3000/manifest.jsonl` (2000 VQA v2 + 1000 TextVQA, frozen, pre-registered), two 70-question pilots. **No training images and no video data are present.** | `data/` |
| Hardware | 2x A100 80 GB PCIe; at inspection time both GPUs had ~58 GB in use by other processes (shared host) | `nvidia-smi` |
| Software | Python 3.13.5, torch 2.14.1+cu130, transformers 5.21.0, safetensors 0.8.0, numpy 2.5.4. **No CompressAI / constriction / torchac** installed, and installation is out of scope for this task | site-packages |
| Vision weights location | All 333 `model.visual.*` tensors are in `model-00001-of-00018.safetensors` (3.97 GB), so the vision tower can be loaded alone | `model.safetensors.index.json` |

*Measured* baselines to beat (Experiments 2–3, 2,999 questions): BF16 median
payload 3.07 MB (300 tokens x 5120 x 2 B); INT8 1.54 MB, retention 0.9984;
INT4-g64 0.82 MB (4.25 bits/value, 21,760 bits per visual token), retention
0.9998. Codec cost ≈2.5 ms. Encoder-through-merger ≈20 ms per image on the A100.

---

## 3. Split-architecture analysis

Notation: `N` = patches (median 1200, *measured* median 300 merged tokens x 4),
`M = N/4` visual tokens seen by the LLM. Per-block ViT cost *estimate*:
≈15.2 M parameters per block (attention 4x1152^2, MLP 2x1152x4304), ≈410 M for
27 blocks, merger ≈44.6 M, patch embed 1.8 M, pos table 2.65 M.

### Option A — full student replacement (closest to the paper's spirit)

Client: stem (patch embed + pos table, optionally copied from the teacher, 4.4 M
params) -> lightweight student body -> compression encoder -> `z`. Server:
decoder -> reconstruction of the output of teacher block `m` -> frozen blocks
`m+1..27` -> merger -> LLM.

* Client compute: tiny (no pretrained ViT blocks). Server compute: `27-m` blocks.
* Transmission: set entirely by the bottleneck.
* Risk: the student must reproduce the function of `m` pretrained blocks of a
  410 M-parameter SigLIP-style encoder from pixels. In the paper the replaced
  part was the first 1/4 of a ResNet-50 at full resolution; here the analogue
  is `m` small. Taking `m = 27` or `m = merger` (student replaces the *whole*
  encoder, "lightweight student approximates the final embedding") is judged
  **unlikely to preserve LLM behaviour** without a student of comparable size,
  and is therefore implemented but not recommended as the first run.
* Implemented as `encoder_input: "pixels"` or `"block_0"` (stem output) with
  `decoder_target: "block_m"`.

### Option B — intermediate ViT split

Client: stem -> blocks `1..k` -> compression encoder. Server: decoder -> blocks
`k+1..27` -> merger -> LLM. Candidate `k`: early (4–8), middle (13–16), final (27).

* Activation at every block is `[N, 1152]` BF16 = 2.76 MB per median image,
  i.e. 90 % of the merged-feature size: **moving the split inside the ViT does
  not by itself reduce bytes**; all savings must come from the learned
  bottleneck (channel reduction + 2x2 window merging + entropy coding).
* Client compute grows linearly with `k` (`k/27` of the ≈20 ms encoder time);
  server compute shrinks correspondingly.
* Trainability: the decoder's error is filtered by `27-k` frozen blocks before
  it reaches the LLM. Deep supervision at the merger output is cheap (ViT only).
  Intuition from the paper and from the Experiment 3 outlier finding: earlier
  activations are less structured (no outlier channel yet? **UNVERIFIED**, see
  the probe script), so plain MSE is a more faithful target there, while later
  splits leave fewer frozen blocks to "repair" reconstruction error.
* Recommendation: sweep `k ∈ {8, 16, 27}` using `es_probe_split_points.py`
  first (noise-sensitivity per block, per-block outlier statistics), then train
  the best two.

### Option C — post-merger learned compression

Client: full frozen ViT + merger -> compression encoder on `[M, 5120]`. Server:
decoder -> `[M, 5120]` -> LLM. Lowest risk: it is the learned analogue of the
INT8/INT4 codecs already evaluated and plugs into the existing injection and
scoring pipeline unchanged.

* Client compute: full encoder (≈20 ms) + small MLP. Server compute: small MLP.
* Transmission: `M x C_z` integers entropy coded; a 5120 -> 256 bottleneck at
  ~3 bits/symbol would be ≈770 bits per visual token vs 21,760 for INT4-g64
  (*estimate*, to be tested).
* Distortion target is exactly the quantity the LLM consumes, so Stage 1's
  MSE is directly comparable to the `reconstruction` numbers already recorded
  for INT8/INT4. The outlier channel 3994 requires the standardised-MSE option
  or the LLM will be the only judge of whether MSE weighted it sensibly.
* Can be trained from a **disk cache of merger outputs with no model loaded at
  all** (3 MB/image BF16; 10 k images ≈ 30 GB).

### Comparison

| Criterion | A (pixels/stem -> block m) | B (block k -> block k) | C (merger -> merger) |
| --- | --- | --- | --- |
| Client FLOPs | minimal | k/27 of encoder | full encoder |
| Server FLOPs | (27-m)/27 encoder + LLM | (27-k)/27 encoder + LLM | LLM only |
| Raw activation at split | 1536-d patches | 1152-d patches | 5120-d tokens |
| Supervision target | block m output (+ merger) | block k output (+ merger) | merger output |
| Trainable | student stem/body, enc, dec, prior | enc, dec, prior | enc, dec, prior |
| Spatial info | student must learn position | preserved (RoPE recomputed on server) | preserved |
| Temporal (video) | frames = extra `t` rows; same path | same | same |
| Changes to Qwen path | server must re-enter the ViT at block m (new code, `TeacherSplit.suffix`) | same | none (existing injection) |
| Feasibility now | needs training images; ViT-only GPU footprint | same | same, or no GPU for Stage 1 with a feature cache |
| Risk | high | medium | low |

### Recommended first implementation

1. **Option C first** (lowest risk, directly comparable to INT8/INT4, cheapest
   to train). It answers whether learned RD compression beats 21,760 bits per
   token at retention ≥ 0.99.
2. **Option B with k from the probe** second; it is the paper's structure and
   the only one that reduces client compute while keeping pretrained weights.
3. Option A only if B shows that early-block targets are easy to match.

All three share one code path (`SplitSpec(encoder_input, decoder_target)`),
so the choice is a config change, not a rewrite.

---

## 4. Modules and interfaces

Package `src/vit_compression_lab/entropic/` (model-agnostic) and
`src/vit_compression_lab/entropic/qwen/` (Qwen3.5-specific). Scripts live in
`encoders/Qwen3.8-vision-compression/scripts/es_*.py`. Configs in
`configs/qwen3_8/entropic_student/`.

| Module | Purpose |
| --- | --- |
| `entropic/config.py` | `SplitSpec`, `StudentConfig`, JSON loading and validation |
| `entropic/gdn.py` | GDN / IGDN over the channel (last) dimension, Ballé reparametrisation |
| `entropic/entropy_model.py` | `FactorizedEntropyModel` (Ballé 2018 non-parametric factorised prior): likelihood, noise/round quantisation, quantile aux loss, CDF tables, `compress`/`decompress` |
| `entropic/rans.py` | Pure-Python rANS coder with escape + bypass bits (interface mirrors CompressAI so a C backend can replace it) |
| `entropic/grid.py` | Packed-token <-> per-image grid reshaping using `grid_thw` and the 2x2 merge order |
| `entropic/bottleneck.py` | Student encoders/decoders (`token_mlp`, `window_mlp`, `grid_conv`, optional tiny attention body) and `EntropicStudent` (encoder -> entropy model -> decoder) |
| `entropic/losses.py` | Distortion variants, rate in nats/bits, Stage 1 RD loss, Stage 2 token-KL loss |
| `entropic/payload.py` | Bitstream container: header (version, split, grid_thw, modality, latent shape, channel count) + coded strings; byte accounting |
| `entropic/metrics.py` | Compression, reconstruction (MSE, cosine, relative L2) and overhead metrics |
| `entropic/checkpoint.py` | Save/load student + entropy tables + config; run directory layout; JSONL metric logging (reuses `run_checkpoint.py`) |
| `entropic/qwen/vision_tower.py` | Load **only** the vision tower from the cached shard; `TeacherSplit` with `prefix`, `suffix`, `full`, `verify` |
| `entropic/qwen/data.py` | Image/video preprocessing with the checkpoint's processor, training manifests, dynamic batching by patch budget, optional feature cache |
| `entropic/qwen/injection.py` | Access to the reusable functions of `inject_eval.py` (loaded by path, no copy) and teacher-forced inputs for Stage 2 |

### Data flow and tensor shapes (one image, grid `[1, h, w]`, `N = h*w`, `M = N/4`)

```
pixel_values          [N, 1536]              (processor; 3*2*16*16)
stem                  [N, 1152]              (patch_embed + resampled pos table)   = block_0
block k output        [N, 1152]              k = 1..27
merger output  h      [M, 5120]              injected into <image_pad> rows

encoder input  u      [N_in, d_in]           d_in ∈ {1536, 1152, 5120}
latent         z      [L, C_z]               L = N_in (token_mlp) | N_in/4 (window_mlp) | ceil(H/s)*ceil(W/s) per frame (grid_conv)
quantised      z_q    [L, C_z] int32         rounding around learned medians
bitstream             header + C_z-channel rANS string
decoder output ĥ_tgt  [N_tgt, d_tgt]         d_tgt ∈ {1152, 5120}
server suffix  ĥ      [M, 5120]              blocks m+1..27 + merger (identity when target is the merger)
```

Transmitted metadata: `grid_thw` (3 x int32 per image/video), modality flag,
`C_z`, latent length. RoPE, `cu_seqlens`, position-table resampling and M-RoPE
position ids are all recomputed from `grid_thw` on the server, exactly as the
unmodified model does. Video timestamps live in `input_ids` as text and are
never part of the visual payload.

### Bottleneck architecture (default for Option C)

`token_mlp`: `LayerNorm(5120) -> Linear(5120, 1024) -> GDN -> Linear(1024, C_z)`
with `C_z ∈ {128, 256, 512}`; decoder `Linear(C_z, 1024) -> IGDN -> Linear(1024, 5120)`.
For Options A/B at patch level the default is `window_mlp`: the 4 patches of a
merge window are concatenated (`[N/4, 4*1152]`) before the same MLP, so that
`L = M` and the latent grid equals the LLM token grid. `grid_conv` reproduces
the paper's Conv5x5/s2 + GDN stack on the per-frame token grid for ablations.

### Entropy model and quantisation

Factorised prior per latent channel (filters `(3,3,3)`, init scale 10, tail
mass 1e-9), exactly Ballé 2018 Appendix 6.1 / CompressAI `EntropyBottleneck`.
Training: `z̃ = z + U(-1/2, 1/2)`; rate = `-Σ log p(z̃)`. Inference: `z_q =
round(z - median) + median` per channel, CDF tables built by `update()`, rANS
coding with an escape symbol and 4-bit bypass chunks for out-of-range values.
Reported both as *ideal bits* (`-Σ log2 p(z_q)`) and *coded bytes*; their ratio
is the entropy-coding overhead requested in Phase 5. A straight-through
rounding option exists for ablation. **The pure-Python coder is for
correctness and byte counts, not latency**; the plan records its runtime
separately from the GPU stages.

### Losses

Stage 1 (per image, averaged over the batch; `|·|` sums over elements):

```
L_1 = 1/2 * || h_tgt - ĥ_tgt ||^2            (distortion at the decoder target)
    + Σ_j w_j * 1/2 * || h_j - ĥ_j ||^2      (optional deep supervision through frozen blocks, e.g. merger output)
    - beta * Σ log p_psi(z̃)                  (rate, nats)
    + aux quantile loss (separate optimiser, as in CompressAI)
```

Distortion variants (`distortion.type`): `mse` (paper), `standardized_mse`
(per-channel teacher std from a statistics pass, needed because of channel
3994), `mse_plus_cosine`. beta grid as in the paper, rescaled per element
count (`beta_per: image|token|element`) so that configs are comparable across
split points with different `N_in`.

Stage 2 (adaptation of Eq. 7): encoder and entropy model frozen (rounding),
decoder (optionally also a private copy of the server suffix blocks) trainable,
LLM frozen with gradient checkpointing:

```
L_2 = alpha * CE(student logits, reference answer tokens)        (optional, alpha default 0)
    + (1 - alpha) * tau^2 * KL( softmax(teacher logits / tau) || softmax(student logits / tau) )
```

over the answer-token positions, teacher-forced on the teacher's own greedy
answer (cached) or the reference. Teacher logits come from the uncompressed
`h`, student logits from `ĥ`, both through the same frozen LLM via
`mm_encoder_outputs`. This is the multimodal analogue of the paper's
classification KD and is explicitly a deviation.

### Datasets and feature loading

* Training images: **none exist in the repository.** `scripts/prepare_image_manifest.py`
  writes a manifest from a local image directory and refuses images present in
  `data/eval-3000` (by SHA-256 and by path), so evaluation data is never used
  for training. Downloading is out of scope.
* Manifest format = existing JSONL (`id`, `image_path`, optional `question`,
  `expected_answer`, ...), so eval manifests also work as held-out validation.
* `es_extract_teacher_features.py` caches per-image `pixel_values`, `grid_thw`,
  and the activations at the configured points as safetensors files
  (`outputs/qwen3.8/teacher-features/<set>/<id>.safetensors`).
* Dynamic batching by total patches (`max_patches_per_batch`), packed sequences.

### Evaluation and baselines

`es_eval.py` reproduces `inject_eval.py`'s record schema (same `identity_check`,
`reconstruction`, `sizes`, `timing`, `original`/`compressed` blocks) so that
`score_results.py` and `report_retention.py` run unchanged and the result sits
next to `eval3000-001` (INT8) and `eval3000-int4g64-001` (INT4-g64).
Additional per-example fields: `learned` (bits per visual token, ideal vs coded
bits, header bytes, latent shape, stage timings for prefix / encoder /
entropy-encode / entropy-decode / decoder / suffix, cosine similarity, and the
same metrics for an optional quantisation baseline codec on the same features).
`es_eval_features.py` gives the ViT-only subset of those metrics (no LLM).
`es_probe_split_points.py` reports per-block statistics and noise sensitivity
to choose `k`.

### Configuration and checkpoints

JSON configs (the repository's existing choice; no framework). Each run writes
`runs/<name>/config.json`, `run_info.json` (software, hardware, checkpoint
revision, teacher split, config SHA), `metrics.jsonl`, `checkpoints/latest.pt`
and `checkpoints/best.pt` holding `{"config", "state_dict", "entropy_tables",
"optimizer", "step", "epoch"}`. `runs/` and `outputs/` are already ignored by Git.

### Hardware requirements and bottlenecks

| Stage | Needs | Estimate |
| --- | --- | --- |
| Feature extraction / probe | vision tower only, BF16 ≈0.9 GB + activations | fits beside the shared host's current load |
| Stage 1, Option C from cache | **no model**; CPU or any GPU | disk: 3 MB/image BF16 |
| Stage 1, Options A/B online | vision tower BF16 + student FP32 + backward through frozen suffix blocks | a few GB at 16 k patches per batch; **UNVERIFIED** |
| Stage 2 | full 27 B LLM BF16 (≈54 GB) + activations with gradient checkpointing, gradients only w.r.t. visual embeddings | needs an otherwise idle 80 GB GPU; **not feasible while ~58 GB is in use by others** |
| Eval (`es_eval.py`) | same as `inject_eval.py` (full model) | ≈2.2 s per example *measured* for the INT4 run |

Bottlenecks: pure-Python rANS (seconds for 300 x 256 symbols, acceptable for
byte counts only); backward through the LLM in Stage 2; disk for feature caches.

---

## 5. Implementation sequence

1. Shared modules (`gdn`, `entropy_model`, `rans`, `grid`, `bottleneck`,
   `losses`, `payload`, `metrics`, `checkpoint`, `config`) with unit tests
   (**written, not run**).
2. Qwen-specific `vision_tower.py` (`TeacherSplit`) + `data.py` + `injection.py`.
3. Scripts: probe, extract, Stage 1, Stage 2, eval, features-only eval,
   manifest preparation.
4. Configs for Options C, B (k = 8, 16, 27) and A, plus a Stage 2 config.
5. Documentation of the future execution order (`entropic-student-usage.md`).

## 6. Verification steps that must run before any result is trusted

1. `python -m pytest tests/test_entropic_*.py` — never executed in this session.
2. `TeacherSplit.verify(...)` on a few images: `suffix(prefix(x, 27))` must
   equal `visual(x).pooler_output` within kernel noise (compare with the
   `identity_check` noise floor from `inject_eval.py`).
3. `es_probe_split_points.py` on the 70-question pilot images: confirm whether
   the outlier channel exists inside the ViT blocks and how merged-feature
   error grows with noise injected at each block.
4. rANS round trip on real latents after `update()` (`es_eval_features.py
   --check-roundtrip`).
5. Stage 1 smoke run on ≤ 50 images for 2 epochs (`--limit 50 --epochs 2`),
   checking that bits/token falls and MSE falls as beta decreases.

## 7. Unresolved risks and assumptions

* **No training data in the repository.** Any image set must be supplied; the
  manifest tool only indexes local files.
* The per-block noise sensitivity, the existence of the outlier channel inside
  the ViT, and the achievable bits/token at retention ≥ 0.99 are all unknown
  until the probe and Stage 1 run.
* Stage 2 memory on a shared 80 GB GPU is not available at inspection time.
* The pure-Python rANS coder is slow; a C backend (CompressAI/constriction) is
  the intended replacement when latency, not bytes, is the question.
* Video: the preprocessing path and payload header carry `video_grid_thw` and
  modality, but no video dataset, video benchmark, or video chat-template
  loading has been exercised; `transformers` needs a video backend
  (`torchvision`/`av`/`decord`) that was not checked.
* `output_hidden_states` capture in transformers 5.21 was *not* used for
  intermediate activations because its semantics (`tie_last_hidden_states`)
  are opaque; `TeacherSplit` re-implements the block loop explicitly and must
  be verified against the full forward (step 6.2).
* Fine-tuning the server-side suffix blocks in Stage 2 changes shared
  pretrained weights; it is off by default and only applies to a private copy.
