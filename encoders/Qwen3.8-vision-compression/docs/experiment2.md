# Experiment 2: INT8 visual features fed back into Qwen

How does symmetric per-token INT8 compression of Qwen's merged visual features
affect payload size, processing latency, and downstream answer quality?

Original and reconstructed features are both consumed by the **same** Qwen
language model. (Claude's API cannot consume Qwen feature tensors.)

## Summary

On 70 VQA v2 and TextVQA questions, INT8 reconstruction halved the feature
payload (2.00×) and left primary accuracy unchanged: 87.1% for both conditions,
with no question flipping between correct and wrong in either direction. The
official VQA accuracy fell by 0.71 percentage points. Three of 70 answers
changed text. Relative L2 reconstruction error was 13.7–14.3% on every image.
This is a small pilot; it cannot detect small accuracy effects.

## Setup

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3.8-27B` (`Qwen3_5ForConditionalGeneration`), revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, loaded from the local cache only |
| Weights / features dtype | `torch.bfloat16` |
| Attention backend | SDPA (vision and full-attention layers); linear-attention layers use the PyTorch reference implementation (`fla`, `causal_conv1d` not installed) |
| Software | Python 3.13.5, torch 2.14.1+cu130, transformers 5.21.0 |
| Hardware | NVIDIA A100 80GB PCIe (`cuda:1`); 16 torch CPU threads. GPU 0 was busy with another user's process throughout |
| Preprocessing | Qwen processor defaults: patch 16, merge 2, mean/std 0.5, bicubic resize, pixels between 65,536 and 16,777,216 |
| Prompt | Chat template with `enable_thinking=False`; question followed by "Answer with only the requested word, number, or option letter. Do not explain." |
| Generation | Greedy (`do_sample=False`), `max_new_tokens=32`, identical for both conditions. The checkpoint's default sampling settings are overridden |
| Timing repeats | 5 warm-ups, then 50 runs (codec, preprocessing, transfers) or 20 runs (encoder); medians reported |
| Data | `data/vqav2-pilot` (30: 10 counting, 10 yes/no, 10 general) and `data/textvqa-pilot` (40 OCR), one question per image |

## Method

### Feature injection

The installed model accepts precomputed visual features through
`mm_encoder_outputs` (`Qwen3_5Model.forward`); ordinary `generate()` uses this
same argument internally after encoding the pixels. Each condition calls
`generate()` with the processor's `input_ids`, `attention_mask`,
`image_grid_thw`, and `mm_token_type_ids`, plus the injected features, and no
`pixel_values`. The vision encoder therefore cannot run again. No installed
Transformers files were edited.

**Encoder boundary:** `get_image_features`: patch embedding → 27 ViT blocks →
`visual.merger`. The merger output (`[tokens, 5120]`) is what gets compressed.

### Verification (every example)

1. Features from `get_image_features` are bit-identical to those ordinary
   `generate()` computes internally (same token order, dtype, grid).
2. **Identity check:** injecting the original features gives prefill logits
   whose maximum absolute difference from ordinary execution must not exceed the
   run-to-run difference of ordinary execution. Greedy token ids must match.
   Result: **70/70 passed, maximum difference 0.0 (bit-exact)**.
3. During injected passes, a temporary hook confirms the encoder never runs. A
   second temporary hook confirms that the image rows entering the decoder equal
   the injected tensor exactly. For the compressed path this is the
   deserialized reconstruction, which differs from the original. Both hooks are
   removed in `finally`, and `rope_deltas` is reset before every call so no
   state carries over.

### Compression

Symmetric per-token INT8 with FP32 scales (`scale = max|x| / 127`; all-zero
tokens get scale 1), reused unchanged from `quantize_features.py`. The main path:

GPU quantize → copy INT8 + scales to CPU → `torch.save` to an in-memory buffer →
`torch.load` → copy to GPU → GPU reconstruct to BF16.

The payload stores INT8 values, scales, dtype, shape, and `image_grid_thw`. A CPU
quantization variant was also timed for comparison with Experiment 1.

## Payload size

Medians over 70 images (160–1024 visual tokens; median 656).

| Quantity | Original BF16 | INT8 + FP32 scales | Ratio |
| --- | ---: | ---: | ---: |
| Raw tensor bytes | 6,717,440 | 3,361,344 (scales: 2,624) | 1.998× |
| Serialized payload bytes (`torch.save`) | 6,719,017 | 3,363,237 | 1.998× |

The ratio is essentially fixed: 2 bytes → 1 byte per value, plus 4 bytes per
token for the scale.

## Latency

Medians across the 70 per-image medians. CUDA events time GPU work and copies;
`time.perf_counter` times CPU work. All measurements are local and in-process.

| Stage | Where | Median ms | Range of per-image medians |
| --- | --- | ---: | ---: |
| Preprocessing (chat template, tokenize, resize/patch) | CPU wall | 8.07 | — |
| Vision encoder through merger | GPU | 34.16 | 17.4–57.8 |
| Quantize (incl. BF16→FP32 cast) | GPU | 0.148 | 0.114–0.225 |
| Copy INT8 + scales GPU→CPU | transfer | 0.318 | — |
| Serialize | CPU wall | 1.327 | 0.44–3.80 |
| Deserialize | CPU wall | 0.570 | — |
| Copy INT8 + scales CPU→GPU | transfer | 0.278 | — |
| Reconstruct to BF16 | GPU | 0.053 | 0.032–0.081 |
| *CPU variant:* copy BF16 GPU→CPU | transfer | 0.559 | — |
| *CPU variant:* quantize | CPU wall | 1.584 | — |
| *CPU variant:* reconstruct | CPU wall | 0.514 | — |

Per-image encoder p95 had a median of 35.4 ms. Encoder time grows with token
count, so compare images of similar size.

**Not valid in this run:** the processor-input CPU→GPU transfer was reported as
0.015 ms. `BatchFeature.to()` moves tensors in place, so after the first warm-up
the timed calls copied nothing. The script has been fixed to build a new GPU
copy each time; the next run will measure this correctly. No other timing used
that call.

### Generation

Single run per condition, after the identity check's generations, with both conditions using the
injected path. The encoder is excluded and a fresh cache is created per call.

| Condition | Median ms | Range ms | Output tokens (median, range) | Hit 32-token limit |
| --- | ---: | ---: | --- | ---: |
| Original features | 562.7 | 243.6–1164.1 | 2 (2–8) | 0 |
| Reconstructed features | 590.6 | 242.8–1147.3 | 2 (2–8) | 0 |

Generation was not repeated, so the 28 ms difference in medians should not be
read as an effect of compression. Absolute times are inflated by the
linear-attention layers' reference implementation. Serialization here is local
byte packing, not network transfer, and nothing in this experiment
demonstrates real-time performance.

## Reconstruction error (diagnostics)

| Metric | Median over images | Range |
| --- | ---: | ---: |
| Mean absolute error | 0.1276 | — |
| Maximum absolute error | 0.8125 | up to 1.25 |
| Mean squared error | 0.0287 | — |
| Relative L2 error | 0.1412 | 0.1372–0.1429 |
| Max abs prefill-logit change vs. original | 28.4 | up to 38.7 |

Next-token argmax at the end of the prompt changed for 2 of 70 examples. These
numbers describe tensor fidelity, not task quality.

CPU and GPU quantization were **not** bit-identical on any image. A handful of
values differ by ±1 at exact rounding ties, because `max|x| / 127` differs in the
last bit between devices (on random data, 104 of 1.5M values). Both are valid
quantizations; this experiment uses the GPU result throughout.

## Answer quality

### Scoring rules

Before scoring, answers are normalized: lowercase, trim, collapse whitespace,
and remove surrounding quotes and one trailing period. Other punctuation and
digits are kept. Baseline answers are never used as references.

| Task | Primary rule |
| --- | --- |
| yes/no | First word must be "yes"/"no" (else unparseable); compared with `expected_answer` only, because some accepted lists contain both |
| counting | Exactly one integer (digits or zero–twenty) is required (else unparseable); must equal `expected_answer` |
| general | Exact match with `expected_answer` or any `accepted_answers` entry |
| OCR | Character error rate vs. `expected_answer`. "Correct" = exact match with `expected_answer` or any accepted answer |

The secondary metric is official VQA accuracy (`scripts/vqa_accuracy.py`),
applied to the normalized answer. Without normalization the official scorer gives
"Grass." 0 against ten "grass" answers. Because `accepted_answers` includes
single-annotator answers, the primary rule for general and OCR questions is
lenient, while the counting rule is strict (majority answer only). VQA accuracy
weights annotator agreement.

### Results

| Task (n) | Original accuracy | Compressed accuracy | Original VQA acc. | Compressed VQA acc. | Mean CER (orig → comp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Yes/no (10) | 100% | 100% | 1.000 | 1.000 | — |
| Counting (10) | 60% | 60% | 0.850 | 0.790 | — |
| General (10) | 90% | 90% | 0.690 | 0.690 | — |
| OCR (40) | 90% | 90% | 0.835 | 0.838 | 0.174 → 0.176 |
| **All (70)** | **87.1%** | **87.1%** | **0.840** | **0.833** | — |

- Accuracy difference (compressed − original): **0.0 percentage points** primary,
  **−0.71 pp** VQA accuracy.
- Original correct → compressed wrong: **0**. Original wrong → compressed correct: **0**.
- Unparseable answers: 0 in both conditions. Answers hitting the token limit: 0.

### Answers that changed

| Id | Question | Reference | Original | Compressed |
| --- | --- | --- | --- | --- |
| `vqav2-val-495443000` | How many batches of bananas are there? | 8 (annotators: 7, 8) | 7 (wrong; VQA 0.6) | 6 (wrong; VQA 0.0) |
| `textvqa-val-37186` | How much is the compact space? | $229.95 | 229.95 (correct; CER 0.14) | $229.95 (correct; CER 0.0) |
| `textvqa-val-35084` | What is the time right now? | 2:57 | 14:56 (wrong; CER 0.75) | 11:14 (wrong; CER 1.0) |

All three changes fall on counting and fine-grained reading: one moved away
from the annotators, one moved closer, and one was wrong both ways. Raw answers
for all examples are in `scored.jsonl`.

## Limitations

- 70 questions, one per image, with one run per condition. A 0 pp difference
  rules out only large effects. Three of 70 answers changed, so a larger set
  is needed to estimate the effect size.
- Greedy decoding with short answers; longer, open-ended answers may be more
  sensitive.
- The counting reference is the majority answer only. For example, "6" for
  `vqav2-val-117125000` scores wrong under the primary rule but 1.0 VQA accuracy.
- Timings come from a shared host whose other GPU was fully loaded. Generation
  latency is single-run.

## Reproduce

From the repository root (writes to ignored `outputs/`; output directory must be new):

```sh
# One example
.venv/bin/python encoders/Qwen3.8-vision-compression/scripts/inject_eval.py \
  data/vqav2-pilot/subset.jsonl --ids vqav2-val-386189004 \
  --output-dir outputs/qwen3.8/inject-eval/one-example

# Full pilot, then scoring
.venv/bin/python encoders/Qwen3.8-vision-compression/scripts/inject_eval.py \
  data/vqav2-pilot/subset.jsonl data/textvqa-pilot/subset.jsonl \
  --output-dir outputs/qwen3.8/inject-eval/pilot70-001
.venv/bin/python encoders/Qwen3.8-vision-compression/scripts/score_results.py \
  outputs/qwen3.8/inject-eval/pilot70-001
```

Outputs: `results.jsonl` (per-example measurements and raw answers),
`scored.jsonl` (per-condition scores), `summary.json` (aggregates plus
`run_info`: versions, hardware, dtype, attention backend, preprocessing,
generation settings), `run_info.json`, and `payloads/<id>.pt` (serialized INT8
payloads).

Results above are from run `pilot70-001`, 2026-10-28.
