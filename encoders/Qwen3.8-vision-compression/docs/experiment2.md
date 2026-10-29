# Experiment 2: trying INT8 compression on 70 questions

**Joshua Wu** · Experiment 2 (pilot) · October 28–29, 2026

I wanted to see whether I could cut the size of Qwen's visual features in half
without noticeably changing its answers. I started with 70 questions, comparing
the original features with an INT8-compressed version fed into the same model.

The first run got 61 out of 70 answers right in both cases. The more detailed
VQA score dropped by 0.71 percentage points, and three answers changed. The
payload was about half the size. That was encouraging, but 70 questions weren't
enough to tell whether the accuracy loss would stay below my target.

I also repeated the pilot with faster attention kernels before moving on to the
[3,000-question follow-up](experiment2-large-eval.md). The details below cover
both pilot runs, including the timing issue I found along the way.

## What I wanted to find out

Qwen3.8's vision encoder produces one BF16 vector of width 5,120 per merged
visual token, roughly 10 KB per token and several megabytes per image. If the
encoder and the language model were to run on different machines, that tensor
is what would cross the wire. I wanted to check three things about compressing each token to INT8:

1. How much smaller is the payload?
2. What does the codec add to end-to-end latency?
3. Does the language model answer differently when it sees the reconstructed
   features instead of the originals?

Both conditions use the same checkpoint, prompt, and greedy decoding; the
original features are injected through the same code path as the compressed
ones, so any difference is attributable to the codec rather than to the
injection mechanism.

## How I tested it

### Feeding the features back into Qwen

`Qwen3_5Model.forward` accepts precomputed visual features through the
`mm_encoder_outputs` argument; ordinary `generate()` uses this same argument
internally after encoding pixels. I call `generate()` with the processor's
`input_ids`, `attention_mask`, `image_grid_thw`, and `mm_token_type_ids`, plus
the injected features and no `pixel_values`, so the vision encoder cannot run a
second time. No installed Transformers source was modified.

The encoder boundary is `get_image_features`: patch embedding, 27 ViT blocks,
then `visual.merger`. The merger output, shape `[tokens, 5120]`, is the tensor
that is compressed.

### Compressing and rebuilding the features

Symmetric per-token INT8 with FP32 scales, `scale = max|x| / 127` (all-zero
tokens get scale 1), reused unchanged from `quantize_features.py`. The main path
is: GPU quantize → copy INT8 and scales to CPU → `torch.save` into an in-memory
buffer → `torch.load` → copy to GPU → reconstruct to BF16 on GPU. The payload
holds the INT8 values, scales, dtype, shape, and `image_grid_thw`. A CPU
quantization variant was timed for comparison with Experiment 1.

### Checks I ran on every example

1. Features from `get_image_features` are bit-identical to those ordinary
   `generate()` computes internally (same token order, dtype, grid).
2. **Identity check.** Injecting the *original* features must reproduce ordinary
   execution: the maximum absolute difference in prefill logits may not exceed
   the run-to-run difference of ordinary execution, and greedy token ids must
   match. Result: 70/70 passed with a maximum difference of 0.0 (bit-exact).
3. During injected passes a temporary forward hook confirms the encoder never
   runs, and a second hook confirms that the image rows entering the decoder
   equal the injected tensor exactly (for the compressed path, the deserialized
   reconstruction). Hooks are removed in `finally`; `rope_deltas` is reset before
   every call so no state leaks between calls.

### How I scored the answers

Answers are normalized before scoring: lowercase, trim, collapse whitespace,
strip surrounding quotes and one trailing period. Other punctuation and digits
are kept. Model answers from either condition are never used as references.

| Task | Primary (binary) rule |
| --- | --- |
| yes/no | first word must be "yes" or "no" (else unparseable); compared with `expected_answer` only, since some accepted lists contain both |
| counting | exactly one integer (digits or zero–twenty) required (else unparseable); must equal `expected_answer` |
| general | exact match with `expected_answer` or any `accepted_answers` entry |
| OCR | character error rate (CER) against `expected_answer`; "correct" = exact match with `expected_answer` or any accepted answer |

The secondary metric is official VQA accuracy (`scripts/vqa_accuracy.py`,
leave-one-annotator-out) on the normalized answer. Without normalization the
official scorer gives "Grass." a score of 0 against ten annotators who wrote
"grass". Because `accepted_answers` includes single-annotator answers, the
binary rule is lenient for general and OCR questions and strict for counting
(majority answer only); VQA accuracy weights annotator agreement.

## What I ran it on

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3.8-27B` (`Qwen3_5ForConditionalGeneration`), revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, loaded from the local cache |
| Dtype | `torch.bfloat16` for weights and features |
| Attention | SDPA for vision and full-attention layers. Run `pilot70-001`: linear-attention layers on the PyTorch reference implementation. Run `pilot70-002-fla`: `flash-linear-attention==0.6.0` and `causal-conv1d==1.7.0` kernels (the rerun below) |
| Software | Python 3.13.5, torch 2.14.1+cu130, transformers 5.21.0 |
| Hardware | NVIDIA A100 80GB PCIe (`cuda:1`), 16 torch CPU threads; a shared host whose other GPU was loaded throughout |
| Preprocessing | Qwen processor defaults: patch 16, merge 2, mean/std 0.5, bicubic resize, 65,536–16,777,216 pixels |
| Prompt | chat template, `enable_thinking=False`; question followed by "Answer with only the requested word, number, or option letter. Do not explain." |
| Generation | greedy (`do_sample=False`), `max_new_tokens=32`, identical for both conditions; the checkpoint's default sampling settings are overridden |
| Timing | 5 warm-ups, then 50 runs (codec, preprocessing, transfers) or 20 runs (encoder); medians reported |
| Data | `data/vqav2-pilot` (30 questions: 10 counting, 10 yes/no, 10 general) and `data/textvqa-pilot` (40 OCR); one question per image |

## What happened in the first run (`pilot70-001`)

### How much smaller the features got

Medians over 70 images (160–1,024 visual tokens, median 656).

| Quantity | Original BF16 | INT8 + FP32 scales | Ratio |
| --- | ---: | ---: | ---: |
| Raw tensor bytes | 6,717,440 | 3,361,344 (scales: 2,624) | 1.998× |
| Serialized bytes (`torch.save`) | 6,719,017 | 3,363,237 | 1.998× |

The ratio is fixed by construction: 2 bytes become 1 byte per value, plus 4
bytes per token for the scale.

### How long each step took

Medians across the 70 per-image medians. CUDA events time GPU work and copies;
`time.perf_counter` times CPU work. All measurements are in-process on one host.

| Stage | Where | Median ms | Range of per-image medians |
| --- | --- | ---: | ---: |
| Preprocessing (chat template, tokenize, resize/patch) | CPU | 8.07 | — |
| Vision encoder through merger | GPU | 34.16 | 17.4–57.8 |
| Quantize (incl. BF16→FP32 cast) | GPU | 0.148 | 0.114–0.225 |
| Copy INT8 + scales GPU→CPU | transfer | 0.318 | — |
| Serialize | CPU | 1.327 | 0.44–3.80 |
| Deserialize | CPU | 0.570 | — |
| Copy INT8 + scales CPU→GPU | transfer | 0.278 | — |
| Reconstruct to BF16 | GPU | 0.053 | 0.032–0.081 |
| *CPU variant:* copy BF16 GPU→CPU | transfer | 0.559 | — |
| *CPU variant:* quantize | CPU | 1.584 | — |
| *CPU variant:* reconstruct | CPU | 0.514 | — |

The per-image encoder p95 had a median of 35.4 ms. Encoder time grows with
token count, so comparisons should be made between images of similar size.

**A timing issue I found:** In this run the processor-input CPU→GPU copy was reported
as 0.015 ms, which is not a real measurement: `BatchFeature.to()` moves tensors
in place, so after the first warm-up the timed calls copied nothing. The script
now builds a fresh GPU copy on each timed call; the faster-kernel timing section below reports the corrected
value. No other timing used that call.

**Generation.** One run per condition, after the identity check, both through
the injected path; the encoder is excluded and a fresh KV cache is created per
call.

| Condition | Median ms | Range ms | Output tokens (median, range) | Hit 32-token limit |
| --- | ---: | ---: | --- | ---: |
| Original features | 562.7 | 243.6–1164.1 | 2 (2–8) | 0 |
| Reconstructed features | 590.6 | 242.8–1147.3 | 2 (2–8) | 0 |

Generation was not repeated, so the 28 ms gap in medians is within single-run
noise and should not be read as a codec effect. Absolute times are inflated by
the linear-attention reference implementation (see the faster-kernel timing section below).

### How much the feature values changed

| Metric | Median over images | Range |
| --- | ---: | ---: |
| Mean absolute error | 0.1276 | — |
| Maximum absolute error | 0.8125 | up to 1.25 |
| Mean squared error | 0.0287 | — |
| Relative L2 error | 0.1412 | 0.1372–0.1429 |
| Max abs prefill-logit change vs. original | 28.4 | up to 38.7 |

The next-token argmax at the end of the prompt changed for 2 of 70 examples.
These figures describe tensor fidelity, not task quality: a 14% relative L2
error at the feature level coexists with unchanged accuracy below.

CPU and GPU quantization were not bit-identical on any image. A handful of
values differ by ±1 at exact rounding ties because `max|x| / 127` differs in
the last bit between devices (104 of 1.5 M values on random data). Both are
valid quantizations; all results use the GPU path.

### Original vs. compressed answers

| Task (n) | Original acc. | Compressed acc. | Original VQA | Compressed VQA | Mean CER (orig → comp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Yes/no (10) | 100% | 100% | 1.000 | 1.000 | — |
| Counting (10) | 60% | 60% | 0.850 | 0.790 | — |
| General (10) | 90% | 90% | 0.690 | 0.690 | — |
| OCR (40) | 90% | 90% | 0.835 | 0.838 | 0.174 → 0.176 |
| **All (70)** | **87.1%** | **87.1%** | **0.840** | **0.833** | — |

Compressed − original: 0.0 pp under the binary rule, −0.71 pp VQA accuracy.
Correct → wrong: 0. Wrong → correct: 0. Unparseable answers: 0 in both
conditions. Answers hitting the token limit: 0.

### The three answers that changed

| Id | Question | Reference | Original | Compressed |
| --- | --- | --- | --- | --- |
| `vqav2-val-495443000` | How many batches of bananas are there? | 8 (annotators: 7, 8) | 7 (wrong; VQA 0.6) | 6 (wrong; VQA 0.0) |
| `textvqa-val-37186` | How much is the compact space? | $229.95 | 229.95 (correct; CER 0.14) | $229.95 (correct; CER 0.0) |
| `textvqa-val-35084` | What is the time right now? | 2:57 | 14:56 (wrong; CER 0.75) | 11:14 (wrong; CER 1.0) |

All three fall on counting and fine-grained reading: one moved away from the
annotators, one moved closer, one was wrong both ways. Raw answers for every
example are in `scored.jsonl`.

## Trying the same questions with faster attention kernels

Qwen3.8 is a hybrid model whose linear-attention layers fall back to a slow
PyTorch reference implementation unless `flash-linear-attention` (Triton) and
`causal-conv1d` (CUDA) are installed. I installed both (now in
`requirements.txt`) and repeated the 70 questions with every other setting
unchanged, for two reasons: to make the 3,000-question evaluation affordable,
and to measure how much the kernel choice alone perturbs greedy answers.
Results from the two kernel sets are not bit-comparable, so they are reported
separately rather than merged.

### Checking the new run

Identity checks passed 70/70 with a maximum logit difference of 0.0; the
run-to-run difference of ordinary execution was also 0.0, so the fast kernels
are deterministic on this hardware.

### Comparing the answers again

| Task (n) | Original acc. | Compressed acc. | Original VQA | Compressed VQA | Mean CER (orig → comp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Yes/no (10) | 100% | 100% | 1.000 | 1.000 | — |
| Counting (10) | 60% | 70% | 0.850 | 0.880 | — |
| General (10) | 80% | 90% | 0.660 | 0.690 | — |
| OCR (40) | 90% | 90% | 0.835 | 0.835 | 0.174 → 0.180 |
| **All (70)** | **85.7%** | **88.6%** | **0.836** | **0.844** | — |

Compressed − original: +2.86 pp binary, +0.86 pp VQA accuracy. Retention of VQA
accuracy 1.010 with one-sided 95% bootstrap bounds 0.981–1.046, i.e.
inconclusive against a 0.99 threshold, as expected at n = 70. Correct → wrong:
0. Wrong → correct: 2 (`vqav2-val-434060011`, `vqav2-val-417115000`).
Unparseable: 0. Token-limit hits: 0.

Changed answers:

| Id | Question | Reference | Original | Compressed |
| --- | --- | --- | --- | --- |
| `vqav2-val-495443000` | How many batches of bananas are there? | 8 | 7 (wrong; VQA 0.6) | 6 (wrong; VQA 0.0) |
| `vqav2-val-434060011` | How many animals are in the picture? | 30 | 20 (wrong; VQA 0.0) | 30 (correct; VQA 0.9) |
| `vqav2-val-417115000` | What kind of bus is shown? | passenger | 17 (wrong; VQA 0.0) | city (wrong; VQA 0.3) |
| `textvqa-val-35701` | Who wrote this book? | brian k. vaughan | brian vaughan (VQA 0.3) | BRIAN VAUGHAN (VQA 0.3) |
| `textvqa-val-35084` | What is the time right now? | 2:57 | 14:56 (wrong) | 11:14 (wrong) |

### What changed just from switching kernels

Holding the condition fixed and comparing `pilot70-001` with `pilot70-002-fla`,
the kernels changed 1 of 70 original answers and 3 of 70 compressed answers, all
on questions that were already borderline:

| Id | Condition | Reference kernels | Fast kernels | Dataset answer |
| --- | --- | --- | --- | --- |
| `vqav2-val-417115000` | original | city (VQA 0.3) | 17 (VQA 0.0) | passenger |
| `vqav2-val-434060011` | compressed | 20 (VQA 0.0) | 30 (VQA 0.9) | 30 |
| `textvqa-val-37186` | compressed | $229.95 (VQA 1.0) | 229.95 (VQA 0.9) | $229.95 |
| `textvqa-val-35701` | compressed | brian vaughan (VQA 0.3) | BRIAN VAUGHAN (VQA 0.3) | brian k. vaughan |

These are floating-point rounding differences between kernel implementations
flipping near-tied greedy choices, not an effect of compression. They put the
kernel-choice noise floor on this set at roughly 1–3 answers in 70, which is the
same order as the codec's own 3 changed answers. This shows that kernel choice can also change answers on this small set. It
doesn’t erase the paired compression differences, but it makes me cautious
about generalising from just a few changed answers.

### How much faster it ran

Medians across the 70 per-image medians, same timing method as the first run. Codec,
transfer, and encoder stages are unchanged; generation is about 1.6× faster.

| Stage | Reference kernels (ms) | Fast kernels (ms) |
| --- | ---: | ---: |
| Preprocessing (CPU) | 8.07 | 5.20 |
| Processor inputs CPU→GPU (corrected measurement) | 0.015 (invalid) | 1.07 |
| Vision encoder through merger (GPU) | 34.16 | 34.23 |
| Quantize (GPU) | 0.148 | 0.132 |
| Copy INT8 + scales GPU→CPU | 0.318 | 0.317 |
| Serialize (CPU) | 1.327 | 1.263 |
| Deserialize (CPU) | 0.570 | 0.520 |
| Copy INT8 + scales CPU→GPU | 0.278 | 0.267 |
| Reconstruct to BF16 (GPU) | 0.053 | 0.052 |
| *CPU variant:* quantize | 1.584 | 0.824 |
| *CPU variant:* reconstruct | 0.514 | 0.365 |
| Generation, original features | 562.7 | 355.4 |
| Generation, compressed features | 590.6 | 350.8 |

Generation with the fast kernels ranged 187–745 ms (single run per condition,
median 2 output tokens). Wall time per example, including the identity check and
all repeated timing, was 4.1 s at steady state; the first examples after a cold
start are much slower while Triton compiles kernels.

## What I take from the pilot

The compression did what I expected on payload size. The 2× payload reduction is deterministic, the
end-to-end codec cost (quantize, two copies, serialize, deserialize,
reconstruct) is under 3 ms per image on the GPU path, under a tenth of the
encoder's 34 ms and under 1% of generation. The identity check being bit-exact
on all 70 examples means the injection path itself introduces no error;
everything observed between conditions is the codec.

On answer quality the pilot is informative mainly about what it cannot show. A
14% relative L2 error in the features produced zero flips under the binary
rule and a 0.7 pp VQA change, while swapping attention kernels produced a
perturbation of the same size. Distinguishing a real sub-1 pp effect from this
noise needs a much larger paired sample, so I followed up with a larger test whose size and rules I fixed in advance.
That run is in [the larger evaluation report](experiment2-large-eval.md).

## What this pilot can’t tell me

- 70 questions, one per image, one generation per condition. A 0 pp difference
  rules out only large effects.
- Greedy decoding with short answers; longer open-ended answers may be more
  sensitive to feature perturbation.
- The counting reference is the majority answer only; e.g. "6" for
  `vqav2-val-117125000` is wrong under the binary rule but scores 1.0 VQA.
- Timings come from a shared host. Generation latency is single-run.
  Serialization is local byte packing, not a network transfer; nothing here
  demonstrates real-time performance over a link.

## Commands and output files

From the repository root (writes to ignored `outputs/`; the output directory
must not already exist):

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
payloads). `pilot70-002-fla` was produced with the same command after
installing the packages in `requirements.txt`; it was run in two sessions
(`sessions.jsonl`) using `--resume`.

Runs: `pilot70-001` on 2026-10-28, `pilot70-002-fla` on 2026-10-29.
