# INT4 Compression of Qwen3.8 Visual Features: Outlier Channels and Group Scaling

**Joshua Wu** · Experiment 3 · October 29, 2026

## Abstract

Experiment 2 showed that per-token INT8 compression of Qwen3.8-27B's merged
visual features is free on the accuracy axis at 2× compression. This experiment
pushes to INT4. Per-token INT4 fails outright: on the 70-question pilot,
official VQA accuracy falls from 0.836 to 0.180 (retention 0.215), because the
features carry one outlier channel whose magnitude is about 61× the per-token
RMS, so a per-token 4-bit grid rounds 99.9% of values to zero. Scaling per group
of 64 channels instead confines the outlier to its own group and, at 4.25 bits
per value (3.76× over BF16), restores the pilot to retention 0.995. On the
pre-registered 3,000-question evaluation of Experiment 2, INT4 with group-64
scales retains 0.9998 of original VQA accuracy (one-sided 95% lower bound
0.9962; difference −0.02 pp, 95% CI −0.36 to +0.33 pp) on the 2,999 examples
that completed, with 17 correct→wrong against 22 wrong→correct flips and a
slightly lower OCR character error rate. The compressed payload falls from a
median 3.07 MB to 0.82 MB, and the codec costs about 2.5 ms per image. The
outlier-channel structure, not bit width, is the constraint on quantizing this
interface.

## 1. Motivation

Experiment 2 established that symmetric per-token INT8 compression of
Qwen3.8-27B's merged visual features halves the payload with no measurable loss
in official VQA accuracy (retention 0.9984, one-sided 95% lower bound 0.9944 on
2,999 questions). A 14% relative L2 error in the features left answers
essentially unchanged, which suggested the vision-to-language interface
tolerates considerably more compression. INT4 is the natural next rung: 4×
over BF16, and the same per-token symmetric scheme with one fewer nibble.

This experiment asks whether INT4 retains 99% of VQA accuracy under the same
pre-registered protocol as Experiment 2: same 3,000-question manifest, same
injection path and identity checks, same prompt and greedy decoding, same
scoring, same paired cluster bootstrap and decision rule. Only the codec
changes. The 70-question pilot is used as a cheap gate before each full run.

## 2. Codecs

Both codecs quantize the merger output (`[tokens, 5120]`, BF16) to the
symmetric range [−7, 7], pack two values per byte, and store FP16 scales; the
stored scale is the one used for quantization, so encoder and decoder agree
exactly. They are selected with `--compression` in `inject_eval.py`, and the
run fingerprint records the codec name, description, and source hash.

| Codec | Scale granularity | Bits per value | Ratio vs BF16 |
| --- | --- | ---: | ---: |
| `int8` (Experiment 2) | one FP32 scale per token | 8.006 | 1.998× |
| `int4` | one FP16 scale per token | 4.003 | 3.997× |
| `int4-g64` | one FP16 scale per group of 64 channels | 4.25 | 3.765× |
| `int4-g32` | one FP16 scale per group of 32 channels | 4.5 | 3.556× |

Implementation: `src/vit_compression_lab/int4_codec.py` (quantize, pack,
unpack, dequantize, serialize, deserialize) and a small registry in
`codecs.py` that gives all codecs one interface. Unit tests cover nibble
packing of every level, the half-step error bound per token and per group,
all-zero tokens, odd widths, group sizes that do not divide the width,
serialization round trips, and the 2× size relation between INT4 and INT8.

## 3. Pilot results (70 questions, fast kernels)

The pilot set, settings, and timing method are those of `experiment2.md` §5
(`pilot70-002-fla`). In every run below the identity check passed 70/70 with a
maximum logit difference of 0.0, and the original-condition answers were
identical to those of `pilot70-002-fla` on all 70 questions, so differences
between runs are entirely due to the compressed condition.

### 3.1 Per-token INT4 collapses

| Task (n) | Original acc. | INT4 acc. | Original VQA | INT4 VQA | Mean CER (orig → INT4) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Yes/no (10) | 100% | 60% | 1.000 | 0.630 | — |
| Counting (10) | 60% | 20% | 0.850 | 0.200 | — |
| General (10) | 80% | 20% | 0.660 | 0.130 | — |
| OCR (40) | 90% | 7.5% | 0.835 | 0.075 | 0.174 → 0.886 |
| **All (70)** | **85.7%** | **18.6%** | **0.836** | **0.180** | — |

Retention 0.2154 (one-sided 95% bounds 0.133–0.303): **FAIL**. 63 of 70
answers changed; 47 went from correct to wrong and none the other way; 1
answer was unparseable. The prefill next-token argmax changed on 58 of 70
examples. Median relative L2 reconstruction error was 0.475 (INT8: 0.141),
with a maximum absolute error of 11.6 per image. Typical failures: counts
become 0 or 1, OCR answers become unrelated strings, yes/no answers flip. The
3,000-question run was not started.

### 3.2 Why: one outlier channel

Inspecting the features behind the INT8 pilot payloads (12 images) explains
the collapse:

- Every token's largest-magnitude value sits in the same channel, index 3994,
  on every image. Its per-image maximum is 94× the median channel's maximum;
  the next largest channels (4349, 3986) are 13× and 11×.
- The per-token maximum is about 61× the token's RMS.
- Under per-token INT4 the step is max/7 ≈ 9 RMS, so 99.93% of values round
  to zero. The payload carries essentially one channel.

The same structure is why per-token INT8 already shows a 14% relative error:
its step is max/127 ≈ 0.5 RMS, so most values occupy only a few levels. The
language model tolerates that; it does not tolerate losing everything but
the outlier.

Simulated relative L2 error of alternative schemes on the same 12 images
(computed against the INT8-reconstructed features, so figures are relative to
that proxy rather than to the BF16 originals):

| Scheme | Relative L2 (median) |
| --- | ---: |
| INT6 per token | 0.422 |
| INT5 per token | 0.469 |
| INT4 per token | 0.492 |
| INT4, group 128 | 0.129 |
| INT4, group 64 | 0.099 |
| INT4, group 32 | 0.083 |
| INT4 per token, channel 3994 sent in FP16 | 0.326 |
| INT4 per token, top 16 channels sent in FP16 | 0.072 |
| INT4, group 64, top 16 channels sent in FP16 | 0.056 |

Per-token scaling fails at any width below 8 bits because of the outlier.
Grouping channels confines the outlier's influence to its own group: with
groups of 64 the simulated error falls below INT8's measured 0.141, at a cost
of 0.25 extra bits per value. Group scaling is simpler than holding channels
out and needs no model-specific channel list, so it was chosen for the next
rung.

### 3.3 INT4 with group-64 scales holds

| Task (n) | Original acc. | INT4-g64 acc. | Original VQA | INT4-g64 VQA | Mean CER (orig → g64) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Yes/no (10) | 100% | 100% | 1.000 | 1.000 | — |
| Counting (10) | 60% | 60% | 0.850 | 0.790 | — |
| General (10) | 80% | 90% | 0.660 | 0.690 | — |
| OCR (40) | 90% | 90% | 0.835 | 0.835 | 0.174 → 0.174 |
| **All (70)** | **85.7%** | **87.1%** | **0.836** | **0.831** | — |

Retention 0.9949 (one-sided 95% bounds 0.974–1.011): INCONCLUSIVE at n = 70,
exactly as INT8 was on this pilot (0.9915 and 1.010 in its two runs). Two of
70 answers changed, both on questions the INT8 codec also perturbed:

| Id | Question | Reference | Original | INT4-g64 |
| --- | --- | --- | --- | --- |
| `vqav2-val-495443000` | How many batches of bananas are there? | 8 | 7 (VQA 0.6) | 6 (VQA 0.0) |
| `vqav2-val-417115000` | What kind of bus is shown? | passenger | 17 (VQA 0.0) | city (VQA 0.3) |

Correct → wrong: 0. Wrong → correct: 1. Next-token argmax changed on 2 of 70.
Median relative L2 error 0.106 (below INT8's 0.141), maximum absolute error
11.6 per image (the outlier channel's own group has a coarse step). The
compressed answers differ from the INT8 pilot's on 3 of 70 questions, within
the 1–3 kernel-noise floor measured in `experiment2.md` §5.3. The pilot
cleared the gate and the 3,000-question run was started.

### 3.4 Payload and latency (pilot medians, 656 tokens)

| Quantity | INT8 | INT4 | INT4-g64 |
| --- | ---: | ---: | ---: |
| Quantized values, bytes | 3,358,720 | 1,679,360 | 1,679,360 |
| Scales, bytes | 2,624 | 1,312 | 104,960 |
| Serialized payload, bytes | 3,363,237 | 1,682,565 | 1,786,213 |
| Serialized ratio vs BF16 (6,719,017) | 1.998× | 3.993× | 3.762× |

| Stage (ms, medians of per-image medians) | INT8 | INT4 | INT4-g64 |
| --- | ---: | ---: | ---: |
| Quantize (GPU, incl. packing) | 0.132 | 0.256 | 0.272 |
| Copy to CPU | 0.317 | 0.173 | 0.174 |
| Serialize (CPU) | 1.263 | 0.743 | 0.767 |
| Deserialize (CPU) | 0.520 | 0.443 | 0.442 |
| Copy to GPU | 0.267 | 0.175 | 0.178 |
| Reconstruct (GPU, incl. unpacking) | 0.052 | 0.127 | 0.128 |
| Codec total | 2.55 | 1.92 | 1.96 |
| Vision encoder through merger | 34.2 | 35.6 | 34.8 |
| Generation, original / compressed | 355 / 351 | 362 / 357 | 359 / 355 |

Packing doubles the quantize and reconstruct kernels' time but halves the
bytes moved and serialized, so the INT4 codecs are slightly cheaper end to end.
All codec costs remain under 2% of generation.

## 4. Main evaluation: INT4-g64 on 3,000 questions

`eval3000-int4g64-001` ran on the frozen Experiment 2 manifest with the same
flags as `eval3000-001` (`--minimal-timing --skip-payloads`), on the same A100,
from 04:50 to 06:42 UTC in one session (≈2.2 s per example). As in Experiment 2,
`textvqa-val-35389` (4608×3456 px) failed with CUDA out-of-memory; it is
excluded and the run is reported at n = 2,999. Because both runs evaluate the
identical example set, the INT8 and INT4-g64 results are directly comparable,
and the original condition's 2,999 answers are bit-for-bit identical between the
two runs.

### 4.1 Primary result

| Subset | n | Original VQA | INT4-g64 VQA | Diff pp | Retention | Lower 95% | Upper 95% | Rule outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **Overall** | **2999** | **0.8156** | **0.8155** | **−0.02** | **0.9998** | **0.9962** | **1.0034** | **PASS** |
| vqav2 | 2000 | 0.8247 | 0.8213 | −0.34 | 0.9959 | 0.9925 | 0.9993 | PASS |
| textvqa | 999 | 0.7976 | 0.8038 | +0.62 | 1.0078 | 0.9996 | 1.0162 | PASS |
| vqav2 / yes_no | 749 | 0.9326 | 0.9339 | +0.13 | 1.0014 | 1.0000 | 1.0043 | PASS |
| vqav2 / general | 1043 | 0.7429 | 0.7379 | −0.50 | 0.9933 | 0.9871 | 0.9991 | INCONCLUSIVE |
| vqav2 / counting | 208 | 0.8462 | 0.8341 | −1.20 | 0.9858 | 0.9696 | 1.0006 | INCONCLUSIVE |

Two-sided 95% CI of the overall difference: [−0.36, +0.33] pp; of retention:
[0.9956, 1.0041]. The formal status is INCOMPLETE at 2,999 of 3,000, as in
Experiment 2; the one missing example can move retention by at most ≈0.0004,
so the hypothesis is supported with the exclusion stated.

Side by side with INT8 on the same 2,999 examples:

| Codec | Bits/value | Payload ratio | Compressed VQA | Diff pp [95% CI] | Retention | Lower 95% |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| INT8 per token | 8.006 | 1.998× | 0.8143 | −0.13 [−0.52, +0.25] | 0.9984 | 0.9944 |
| INT4 group 64 | 4.25 | 3.765× | 0.8155 | −0.02 [−0.36, +0.33] | 0.9998 | 0.9962 |

The two codecs' compressed answers differ on 225 examples (99 after
whitespace/case normalisation) and their VQA scores differ on 61; the net
difference between them is +0.12 pp in INT4-g64's favour, which is noise.

### 4.2 Subsets

TextVQA, the task most dependent on fine spatial detail, is unaffected
(+0.62 pp, CI includes zero). VQA v2 shows a small loss (−0.34 pp, CI
[−0.68, −0.01]) that just excludes zero, concentrated in counting (−1.20 pp on
208 questions, CI [−2.93, +0.24], 4 correct→wrong and 0 wrong→correct) and
general questions (−0.50 pp). Counting is the subset where INT8 also changed
answers in the pilot and where greedy choices between adjacent integers are
closest; it is the first place to look if the codec is pushed further.

### 4.3 Secondary metrics

| Metric | Original | INT4-g64 |
| --- | ---: | ---: |
| Binary accuracy (pilot primary rule) | 83.5% | 83.7% |
| Answers changed (raw string) | — | 202 of 2999 |
| … differing after whitespace/case normalisation | — | 87 |
| … with a different VQA score | — | 47 |
| Correct → wrong / wrong → correct (binary rule) | — | 17 / 22 |
| Mean CER, TextVQA | 0.2364 | 0.2337 |
| Unparseable answers / token-limit hits | 2 / 1 | 2 / 1 |
| Identity checks passed | 2999 of 2999 | — |
| Prefill next-token argmax changed | — | 180 |
| Median generation time | 279.8 ms | 277.3 ms |

Generation is slower in absolute terms than in the INT8 run (233.6 ms) because
the shared host was more loaded; the two conditions within a run remain equal.
Illustrative score-changing answers (full list in the run's `report.md`):

| Id | Question | Reference | Original | INT4-g64 |
| --- | --- | --- | --- | --- |
| `vqav2-val-102912001` | How many elephants are there? | 2 | 2 (1.0) | 3 (0.3) |
| `vqav2-val-390134002` | How many tines are on the fork? | 3 | 3 (1.0) | 4 (0.0) |
| `vqav2-val-18982001` | Is the wall painted or papered? | papered | papered (1.0) | painted (0.3) |
| `vqav2-val-571823001` | What is the dog stepping on? | shoe | blanket (0.0) | shoes (0.6) |
| `vqav2-val-204853002` | Whose birthday is it? | girl's | woman (0.0) | girl (0.6) |
| `vqav2-val-314108005` | What type of photo is this? | black and white | black and white (1.0) | B (0.0) |

### 4.4 Reconstruction, payload, and latency

Medians over 2,999 images (median 300 tokens).

| Reconstruction | INT8 | INT4-g64 |
| --- | ---: | ---: |
| Mean absolute error | 0.1283 | 0.0689 |
| Maximum absolute error | 0.8203 | 11.44 |
| Mean squared error | 0.0287 | 0.0159 |
| Relative L2 error | 0.1413 | 0.1050 |

INT4-g64 has lower average error than INT8 but a far larger maximum, which sits
in the outlier channel's group (step = max/7 there). The language model is
evidently indifferent to that channel's precision.

| Serialized payload | Min | Median | p90 | Max | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original (BF16) | 0.72 MB | 3.07 MB | 7.87 MB | 49.2 MB | 4.37 MB |
| INT8 | 0.36 MB | 1.54 MB | 3.94 MB | 24.6 MB | 2.19 MB |
| INT4-g64 | 0.19 MB | 0.82 MB | 2.09 MB | 13.1 MB | 1.16 MB |

Median raw bytes 3,072,000 → 816,000 (768,000 values + 48,000 scales),
3.765×; serialized 3.758×.

| Stage (ms, median) | INT8 run | INT4-g64 run |
| --- | ---: | ---: |
| Vision encoder through merger | 20.29 | 21.03 |
| Quantize (GPU) | 0.20 | 0.42 |
| Copy to CPU | 0.22 | 0.18 |
| Serialize | 0.96 | 0.84 |
| Deserialize | 0.66 | 0.66 |
| Copy to GPU | 0.25 | 0.17 |
| Reconstruct (GPU) | 0.06 | 0.21 |
| Codec total | 2.35 | 2.48 |

Estimated transfer time from the measured payload sizes (bytes ÷ bandwidth,
no protocol overhead), median image:

| Link | BF16 | INT8 | INT4-g64 | Saved vs BF16 (median / p90 image) |
| --- | ---: | ---: | ---: | ---: |
| 50 Mbps | 492 ms | 246 ms | 131 ms | 361 / 924 ms |
| 100 Mbps | 246 ms | 123 ms | 65 ms | 180 / 462 ms |
| 1 Gbps | 24.6 ms | 12.3 ms | 6.5 ms | 18.0 / 46.2 ms |
| 10 Gbps | 2.5 ms | 1.2 ms | 0.7 ms | 1.8 / 4.6 ms |

## 5. Discussion

The headline is that bit width was never the binding constraint. The merger
output of this model is dominated by a single channel roughly two orders of
magnitude larger than the rest, and any scheme that shares one scale between
that channel and the others spends almost all of its levels on it. Per-token
INT8 survives this only because 127 levels leave a few for everything else;
per-token INT4 does not survive it at all. Give the outlier its own scale
and INT4 is as accurate as INT8 by every measure used here, at 3.76× instead
of 2×. The same diagnosis predicts that per-token INT6 and INT5 would also
fail (simulated error 0.42 and 0.47), so there was no intermediate rung
worth running.

Two caveats temper the result. First, the VQA v2 subset's loss (−0.34 pp)
is borderline significant, and counting carries most of it. The overall
mixture passes with margin, but a deployment weighted toward counting or
fine-grained reasoning should treat INT4-g64 as costing a few tenths of a
point rather than nothing. Second, the simulation in §3.2 suggests that
holding a handful of channels out in FP16 could bring INT4 near 4× with even
lower error; that remains to try, as do token reduction and entropy coding,
which attack the token count and the value distribution rather than the bit
width and are the only routes to an order of magnitude.

For the question this series set out to answer, the position is now: INT8
per token and INT4 with group-64 scales both retain ≥ 99% of VQA accuracy on
a 3,000-question paired test; the payload is 2.2 MB and 1.2 MB per image on
average respectively; and the codec adds about 2.5 ms either way.

## 6. Limitations

- One model. The outlier-channel structure is a property of this checkpoint's
  merger output; other encoders may need different group sizes or none.
- Group size was chosen from a simulation on 12 pilot images rather than
  swept on the full set; `int4-g32` is implemented but was not run.
- The simulated errors in §3.2 are measured against the INT8 reconstruction,
  not the BF16 originals, and are therefore slightly optimistic.
- As in Experiment 2: greedy short answers, one evaluation mixture, no real
  network hop, and images above roughly 16 megapixels cannot be evaluated on
  one 80 GB GPU.

## Appendix A. Reproduction

From the repository root on the GPU host:

```sh
# 70-question pilot for a codec (full repeated timing), then scoring
python encoders/Qwen3.8-vision-compression/scripts/inject_eval.py \
  data/vqav2-pilot/subset.jsonl data/textvqa-pilot/subset.jsonl \
  --compression int4-g64 --output-dir outputs/qwen3.8/inject-eval/pilot70-int4g64-001
python encoders/Qwen3.8-vision-compression/scripts/score_results.py \
  outputs/qwen3.8/inject-eval/pilot70-int4g64-001
python encoders/Qwen3.8-vision-compression/scripts/report_retention.py \
  outputs/qwen3.8/inject-eval/pilot70-int4g64-001

# 3,000-question evaluation on the Experiment 2 manifest (resume with --resume)
python encoders/Qwen3.8-vision-compression/scripts/inject_eval.py \
  data/eval-3000/manifest.jsonl --minimal-timing --skip-payloads --compression int4-g64 \
  --output-dir outputs/qwen3.8/inject-eval/eval3000-int4g64-001
python encoders/Qwen3.8-vision-compression/scripts/score_results.py \
  outputs/qwen3.8/inject-eval/eval3000-int4g64-001
python encoders/Qwen3.8-vision-compression/scripts/report_retention.py \
  outputs/qwen3.8/inject-eval/eval3000-int4g64-001
```

Runs: `pilot70-int4-001` and `pilot70-int4g64-001` (2026-10-29, 04:33 and
04:43 UTC), `eval3000-int4g64-001` (2026-10-29, 04:50–06:42 UTC). Output files
are as listed in `experiment2-large-eval.md`, Appendix C.
