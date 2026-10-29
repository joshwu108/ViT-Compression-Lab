# Experiment 2: checking accuracy on 3,000 questions

**Joshua Wu** · Experiment 2 (main evaluation) · October 29, 2026

The [70-question pilot](experiment2.md) looked promising: the features were
about half the size, and the answers barely changed. I wanted a bigger test
before saying that the accuracy loss was small enough to accept.

For this run I chose a target of **99% retention of the original VQA accuracy**.
That means keeping at least 99% of the baseline score, rather than allowing a
one-percentage-point drop. I fixed the sample size, dataset mix, seed, and
scoring rules before running the evaluation.

I selected 3,000 questions: 2,000 from VQA v2 and 1,000 from TextVQA. Of those,
2,999 finished. On the completed questions, the original score was **81.56%**
and the compressed score was **81.43%**, a drop of about **0.13 percentage
points**. That works out to **99.84% retention**, with a one-sided 95% lower
bound of **99.44%**.

Those completed results clear my 99% target. There is one important catch:
one image ran out of GPU memory even on retry. The full run is therefore
**INCOMPLETE** under the rule I set beforehand, rather than an official PASS.
I keep that distinction throughout this report.

## How I set up the larger test

### What stayed the same as the pilot

The injection, verification, codec, prompt, generation settings, and scoring
rules are those of the pilot, unchanged:

- Features from `get_image_features` (patch embedding, 27 ViT blocks, merger;
  shape `[tokens, 5120]`, BF16) are either injected as-is (original) or
  quantized per token to INT8 with FP32 scales, serialized, deserialized, and
  reconstructed to BF16 (compressed). Both are injected through
  `mm_encoder_outputs` with no `pixel_values`, so the encoder cannot re-run.
- Every example first passes an identity check: injecting the original
  features must reproduce ordinary generation bit-exactly in prefill logits and
  greedy token ids. Hooks confirm the encoder did not run and that the decoder
  saw exactly the injected tensor.
- Prompt: chat template with `enable_thinking=False`, the question, then
  "Answer with only the requested word, number, or option letter. Do not
  explain." Generation: greedy, `max_new_tokens=32`.

The only infrastructure difference from the pilot's first run is that the
linear-attention layers use the `flash-linear-attention` and `causal-conv1d`
kernels; their effect was checked in the pilot’s faster-kernel rerun (
identity checks still bit-exact, 1–3 answers in 70 perturbed). Each example is
timed once rather than with repeated runs, and serialized payloads are not
written to disk (their sizes are recorded).

### Where the questions came from

`scripts/prepare_eval_manifest.py` draws the set with a recorded seed and writes
`data/eval-3000/manifest.jsonl` and `metadata.json`.

| Item | Value |
| --- | --- |
| VQA v2 | val2014, v2.0; 2,000 questions sampled uniformly from the 214,354 validation questions (natural task mix), one per image |
| TextVQA | v0.5.1 val; 1,000 images sampled from 3,166, one question per image |
| Seed / allocation | 2026; `--vqav2 2000 --textvqa 1000` |
| Exclusions | the 30 VQA v2 and 40 TextVQA pilot images |
| Task labels | VQA v2: "how many…" → counting, answer_type yes/no → yes_no, else general (pilot rule); TextVQA: ocr |
| Realised mix | counting 208, general 1,043, yes_no 749, ocr 1,000 |
| Mixture | 2/3 VQA v2, 1/3 TextVQA, fixed before the run; the overall result applies to this mixture only |
| Preserved | dataset question text, all 10 annotator answers (duplicates kept), majority answer, question and image ids, other question ids on the image, image URLs, image SHA-256 and dimensions |
| Manifest SHA-256 | `98b217bdaf19272bd55765051e077df864823566343c6ba2158e1e892c4d8c16` |
| Licences | VQA v2 annotations CC BY 4.0 (COCO images under Flickr terms); TextVQA annotations CC BY 4.0 (Open Images images CC BY 2.0). Details in `metadata.json` |

Annotation files were read from the already-verified pilot `raw/` directories;
only the 3,000 selected images were downloaded (≈575 MB, one request each).

### How I decided whether it met the target

- **Primary metric:** official VQA accuracy (`scripts/vqa_accuracy.py`,
  leave-one-annotator-out), applied to the normalized answer exactly as in the
  pilot.
- **Retention** = mean compressed VQA score / mean original VQA score on the
  same examples.
- **Uncertainty:** paired percentile bootstrap, 10,000 replicates, seed
  20261028. Each replicate resamples images with replacement and carries both
  conditions' scores together, so the pairing is preserved. With one question
  per image the clusters coincide with the examples.
- **Decision** from the one-sided 95% bounds (5th and 95th percentiles of the
  replicate retentions): PASS if the lower bound ≥ 0.99, FAIL if the upper
  bound < 0.99, otherwise INCONCLUSIVE. Any failed or missing example makes the
  run INCOMPLETE and no PASS is declared automatically.
- **Secondary metrics** (unchanged pilot rules): binary task accuracy, OCR CER,
  answer changes, correct↔wrong flips, unparseable answers, token-limit hits,
  payload sizes, reconstruction error, latency.

**Why I started with 3,000 questions:** On the pilot, retention was 0.9915 with a one-sided lower
bound of 0.971. Scaling by √(70/3000) ≈ 0.15 predicts a half-width of roughly
0.003–0.005 at n = 3,000, enough to resolve a 1 pp question unless the true
retention sits very close to 0.99.

### Keeping track of the run

The evaluation runs as an append-only, resumable job. Each completed example is
written as one fsync'd line of `results.jsonl`; a per-run fingerprint
(`run_config.json`: checkpoint and revision, dtype, GPU, attention backend,
prompt, generation config, image-processor settings, codec source hash,
manifest hashes, ordered example-id hash, software versions) is checked on
resume so that a continued run cannot silently mix settings. Per-example
failures are recorded with tracebacks in `failures.jsonl`, do not abort the run,
and are retried on resume. These mechanisms were tested on synthetic data and
in smoke runs before the evaluation.

## What I ran it on

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, `torch.bfloat16` |
| Software | Python 3.13.5, torch 2.14.1+cu130, transformers 5.21.0, numpy 2.5.4; `flash-linear-attention` 0.6.0, `causal-conv1d` 1.7.0 |
| Hardware | NVIDIA A100 80GB PCIe (`cuda:1`), 16 torch CPU threads, shared host |
| Run | `eval3000-001`: main pass 2026-10-29 01:16–03:03 UTC (2,998 examples, ≈1.9 s per example at steady state), one resume at 03:34 UTC retrying 2 failed examples |
| Timing | one measurement per stage per example (`--minimal-timing`); payloads not written (`--skip-payloads`) |

## What I found

### Did it keep 99% of the VQA score?

| Subset | n | Original VQA | Compressed VQA | Diff pp | Retention | Lower 95% | Upper 95% | Rule outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **Overall** | **2999** | **0.8156** | **0.8143** | **−0.13** | **0.9984** | **0.9944** | **1.0023** | **PASS** |
| vqav2 | 2000 | 0.8247 | 0.8222 | −0.25 | 0.9970 | 0.9929 | 1.0012 | PASS |
| textvqa | 999 | 0.7976 | 0.7985 | +0.09 | 1.0011 | 0.9929 | 1.0093 | PASS |
| vqav2 / yes_no | 749 | 0.9326 | 0.9318 | −0.08 | 0.9991 | 0.9941 | 1.0042 | PASS |
| vqav2 / general | 1043 | 0.7429 | 0.7387 | −0.41 | 0.9945 | 0.9878 | 1.0008 | INCONCLUSIVE |
| vqav2 / counting | 208 | 0.8462 | 0.8462 | +0.00 | 1.0000 | 0.9815 | 1.0188 | INCONCLUSIVE |

The overall difference is −0.13 pp with a two-sided 95% CI of [−0.52, +0.25] pp;
the two-sided CI for retention is [0.9936, 1.0030]. The overall row is the
planned test. Per-subset rows are descriptive: the general and counting
subsets are too small on their own to resolve a 1 pp question, and their point
estimates (0.9945 and 1.0000) are consistent with the overall picture.

**Why the run still counts as incomplete:** The run completed 2,999 of 3,000 examples (described below), so by the
planned rule the tool reports INCOMPLETE rather than PASS. On the
completed examples the rule gives PASS. A single example changes the mean VQA
score of either condition by at most 1/3000, so the retention estimate can move
by at most ≈0.0004 whichever way the missing example would have scored, and
the lower bound would remain above 0.99. I treat this as encouraging evidence on the completed questions, while
keeping the full run’s INCOMPLETE status.

### What else changed in the answers

| Metric | Original | Compressed |
| --- | ---: | ---: |
| Binary accuracy (pilot primary rule) | 83.5% | 83.4% |
| Answers changed (raw string) | — | 195 of 2999 |
| … differing after whitespace/case normalisation | — | 99 |
| … with a different VQA score | — | 57 |
| Correct → wrong / wrong → correct (binary rule) | — | 26 / 22 |
| Mean CER, TextVQA | 0.2364 | 0.2369 |
| Unparseable answers / token-limit hits | 2 / 1 | 2 / 1 |
| Identity checks passed | 2999 of 2999 | — |
| Prefill next-token argmax changed | — | 167 |
| Median generation time | 233.6 ms | 231.9 ms |

The raw change count follows the pilot's rule and includes answers that differ
only in a leading space or capitalisation (e.g. " yes" vs "yes"); fewer than a
third of the changes, 57, alter the VQA score, and those split almost evenly in
direction. Illustrative substantive changes (full list in the run's
`report.md`):

| Id | Question | Reference | Original | Compressed |
| --- | --- | --- | --- | --- |
| `vqav2-val-198661000` | What time is it? | 11:10 | 11:10 (1.0) | 1:10 (0.0) |
| `vqav2-val-390134002` | How many tines are on the fork? | 3 | 3 (1.0) | 4 (0.0) |
| `vqav2-val-153834007` | How many different colored cows do you see? | 2 | 2 (1.0) | 1 (0.0) |
| `vqav2-val-197745000` | Are both giraffes mature? | yes | no (0.3) | yes (1.0) |
| `vqav2-val-412355000` | What is the bird sitting on? | bird feeder | feeder (1.0) | bird feeder (1.0) |

As in the pilot, the changes concentrate on fine-grained reading and counting,
where greedy choices are near-tied.

### Feature differences and file sizes

Medians over 2,999 images (70–4,800 visual tokens, median 300, p90 768).

| Reconstruction | Median |
| --- | ---: |
| Mean absolute error | 0.1283 |
| Maximum absolute error | 0.8203 |
| Mean squared error | 0.0287 |
| Relative L2 error | 0.1413 |

| Serialized payload | Min | Median | p90 | Max | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original (BF16) | 0.72 MB | 3.07 MB | 7.87 MB | 49.2 MB | 4.37 MB |
| Compressed (INT8 + scales) | 0.36 MB | 1.54 MB | 3.94 MB | 24.6 MB | 2.19 MB |

Raw ratio 1.998×, serialized 1.997×; ≈10.2 KB per token original, ≈5.1 KB
compressed. Reconstruction error matches the pilot (relative L2 0.1412 there),
showing similar reconstruction error across these two samples.

### How long it took

Medians across examples, single measurement each.

| Stage | Where | Median ms |
| --- | --- | ---: |
| Preprocessing | CPU | 6.15 |
| Processor inputs CPU→GPU | transfer | 1.22 |
| Vision encoder through merger | GPU | 20.29 |
| Quantize | GPU | 0.20 |
| Copy INT8 + scales GPU→CPU | transfer | 0.22 |
| Serialize | CPU | 0.96 |
| Deserialize | CPU | 0.66 |
| Copy INT8 + scales CPU→GPU | transfer | 0.25 |
| Reconstruct to BF16 | GPU | 0.06 |
| Generation, original / compressed | GPU | 233.6 / 231.9 |

The encoder median is lower than the pilot's 34 ms because this set has fewer
tokens per image (median 300 vs 656). The full codec path costs ≈2.4 ms, about
1% of generation.

### What the smaller payload could save over a network

This experiment did not measure a network hop: encoder, codec, and decoder ran
on one host. What it does fix is the byte count that would cross a link, so
transfer time can be estimated as bytes ÷ bandwidth from the measured sizes.

| Link | Original (median) | Compressed (median) | Saved (median) | Saved (p90 image) |
| --- | ---: | ---: | ---: | ---: |
| 50 Mbps | 492 ms | 246 ms | 246 ms | 629 ms |
| 100 Mbps | 246 ms | 123 ms | 123 ms | 314 ms |
| 1 Gbps | 24.6 ms | 12.3 ms | 12.3 ms | 31.4 ms |
| 10 Gbps | 2.5 ms | 1.2 ms | 1.2 ms | 3.1 ms |

These ignore protocol overhead, round-trip latency, and transport-level
compression. In these estimates, the savings at 50–100 Mbps are much larger than the
compression cost. At 1 Gbps the saving is about 12 ms; at 10 Gbps it is only
about 1.2 ms, less than the roughly 2.4 ms compression path. A real measurement would need the encoder and
decoder on separate hosts.

## The image that didn’t finish

`textvqa-val-35389` (TextVQA image `c01928d2bc823bae`, 4608×3456 px, 5.6 MB)
failed with CUDA out-of-memory in the main pass and again on a fresh retry with
no other process on the GPU: the model and resident activations occupied
≈73 GB and the prefill for this image requested a further 14.4 GB. This is a
capacity limit of one A100 80GB for a 16-megapixel input under the processor's
default pixel budget, not a transient. A second example, `textvqa-val-36759`,
failed in the main pass for the same reason but succeeded on retry.

Evaluating the excluded example would require either capping the image
processor's maximum pixels for oversized inputs, which changes the model's
inputs and falls outside the planned protocol, or sharding the model
across two GPUs. I left the protocol unchanged and report n = 2,999 with the
exclusion stated; the results above discuss its possible effect on the point estimate.

## How I read these results

On the 2,999 completed questions, the lower retention bound was 99.44%,
which is above my 99% target. The measured VQA loss was small: about 0.13
percentage points. Compression made some answers worse and some better;
under the binary rule, 26 went from correct to wrong and 22 went the other way.

That makes INT8 worth considering for this setup: the payload is about half
the size, and the compression path adds roughly 2.4 ms. It doesn't mean the
answers are identical, or that every task meets the target on its own. The
general and counting subsets still have too much uncertainty to say that.

I also need to keep the unfinished image visible. The completed results are
encouraging, but the full planned run remains incomplete. Network savings are
estimates too; I haven’t tested the encoder and decoder on separate hosts.

## What I still can’t claim

- One model, one codec, greedy decoding with short answers. Longer generations
  or sampling could be more sensitive to feature noise.
- The bootstrap quantifies sampling uncertainty within this evaluation set. It
  says nothing about distribution shift from VQA v2 and TextVQA validation
  questions to deployment inputs.
- The fixed 2/3–1/3 mixture determines the overall number; per-dataset rows
  are reported but were not separately powered.
- Images above roughly 16 megapixels cannot be evaluated on a single 80 GB GPU
  with the default processor settings (described below).
- Network transfer times are computed from payload sizes, not measured.
- Greedy answers are sensitive to kernel implementation at the 1–3 in 70 level
  (pilot’s kernel comparison); the paired design cancels this for the retention estimate only
  insofar as both conditions share the same kernels, which they do.

## Commands to run it again

All commands run from the repository root on the GPU host with the virtual
environment active.

1. Build the manifest (dry run first; the real run downloads ≈575 MB of images
   and refuses to overwrite an existing manifest):

```sh
python scripts/prepare_eval_manifest.py --dry-run
python scripts/prepare_eval_manifest.py
```

2. Smoke test on the first 5 manifest examples:

```sh
python encoders/Qwen3.8-vision-compression/scripts/inject_eval.py \
  data/eval-3000/manifest.jsonl --limit 5 --minimal-timing --skip-payloads \
  --output-dir outputs/qwen3.8/inject-eval/eval3000-smoke-001
```

3. Full evaluation, and resuming after an interruption (same arguments plus
   `--resume`):

```sh
nohup python encoders/Qwen3.8-vision-compression/scripts/inject_eval.py \
  data/eval-3000/manifest.jsonl --minimal-timing --skip-payloads \
  --output-dir outputs/qwen3.8/inject-eval/eval3000-001 \
  > outputs/qwen3.8/inject-eval/logs/eval3000-001.log 2>&1 &

nohup python encoders/Qwen3.8-vision-compression/scripts/inject_eval.py \
  data/eval-3000/manifest.jsonl --minimal-timing --skip-payloads \
  --output-dir outputs/qwen3.8/inject-eval/eval3000-001 --resume \
  >> outputs/qwen3.8/inject-eval/logs/eval3000-001.log 2>&1 &
```

Omit `--minimal-timing` to collect the pilot's repeated latency measurements
(50 codec/preprocess runs and 20 encoder runs per example). `progress.json`
reports completed count and ETA.

4. Scoring and statistical report:

```sh
python encoders/Qwen3.8-vision-compression/scripts/score_results.py \
  outputs/qwen3.8/inject-eval/eval3000-001
python encoders/Qwen3.8-vision-compression/scripts/report_retention.py \
  outputs/qwen3.8/inject-eval/eval3000-001
```

`report_retention.py` enforces ≥ 10,000 replicates and defaults to seed
20261028 and threshold 0.99.

## Code used for this run

The evaluation reuses the pilot's `inject_eval.py` and `score_results.py` with
additions for scale: resumable append-only checkpointing with a settings
fingerprint and lock file, per-example failure capture with retry, `--limit`,
`--minimal-timing`, and `--skip-payloads` options, and per-dataset reporting.
`report_retention.py` implements the bootstrap and decision rule. Shared code
lives in `src/vit_compression_lab/` (`eval_sampling.py`, `paired_stats.py`,
`run_checkpoint.py`) with synthetic-data tests in `tests/`. The pilot rescored
with the updated scorer reproduces its original −0.714 pp VQA difference.

## Where to find the outputs

| Path | Contents |
| --- | --- |
| `data/eval-3000/manifest.jsonl`, `metadata.json` | frozen evaluation set and sampling record |
| `<run>/run_config.json`, `run_info.json` | settings fingerprint; model, software, hardware, prompt, generation, timing collection |
| `<run>/results.jsonl` | per example: both answers, identity check, sizes, reconstruction error, timings |
| `<run>/failures.jsonl`, `progress.json`, `sessions.jsonl` | failures with tracebacks, progress, sessions |
| `<run>/scored.jsonl`, `summary.json` | per-example scores and aggregates |
| `<run>/retention.json`, `report.md` | retention, bounds, decision, subset tables, all changed answers |
