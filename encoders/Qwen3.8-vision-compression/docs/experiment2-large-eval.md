# Experiment 2, large evaluation: does INT8 keep ≥ 99% of VQA accuracy?

The 70-question pilot in `experiment2.md` could not detect small effects. This
evaluation reuses the pilot's injection, verification, codec, prompts,
generation settings, and scoring rules on a fixed, seeded set of 3,000
questions and answers one pre-registered question:

> Do INT8-compressed visual features retain at least 99% of the original
> features' official VQA accuracy?

**Status (2026-10-28):** implementation, tests, and smoke test done. The
evaluation manifest has not been frozen and the full run has not started;
both need explicit approval (see "Before the full run").

## Decision rule

- **Primary metric:** official VQA accuracy (`scripts/vqa_accuracy.py`,
  applied to the normalized answer, exactly as the pilot's secondary metric).
- **Retention** = mean compressed VQA score / mean original VQA score, on the
  same examples with identical prompts and greedy generation settings.
- **Uncertainty:** paired percentile bootstrap, 10,000 replicates, seed
  20261028. Each replicate resamples images (`image_group`) with replacement
  and carries both conditions' scores together. With one question per image
  the clusters are the examples; clustering matters only if several questions
  share an image.
- **Decision** from one-sided 95% bounds (5th and 95th percentiles):
  `PASS` if the lower bound ≥ 0.99; `FAIL` if the upper bound < 0.99;
  `INCONCLUSIVE` otherwise. A zero original mean makes retention `UNDEFINED`.
  Any failed or missing example makes the run `INCOMPLETE`; no PASS is declared
  on partial results.
- **Secondary metrics** (unchanged pilot rules): binary task accuracy, OCR
  character error rate, answer changes, correct↔wrong flips, unparseable
  answers, token-limit hits, payload sizes, reconstruction error, latency.
- The evaluation size is fixed at 3,000. Examples are not added until the
  result passes. An expanded study, if wanted, is planned and reported
  separately.
- The bootstrap describes sampling uncertainty within this evaluation set. It
  says nothing about how representative VQA v2 and TextVQA validation questions
  are of deployment inputs, or about distribution shift.

Illustration of power: on the 70-question pilot, retention is 0.9915 with a
one-sided lower bound of 0.971 (`report_retention.py` on a rescored copy). At
3,000 questions the bound tightens by roughly √(70/3000) ≈ 0.15×, to about
±0.003–0.005 around the estimate. If true retention is close to 0.99 the result
can still be INCONCLUSIVE; 3,000 is an initial size, not a power guarantee.

## Evaluation set

`scripts/prepare_eval_manifest.py` selects questions with a recorded seed and
writes `data/eval-3000/manifest.jsonl` plus `metadata.json`.

| Item | Value |
| --- | --- |
| VQA v2 | val2014, version 2.0; 2,000 questions sampled uniformly from the 214,354 validation questions (natural task mix), one per image |
| TextVQA | v0.5.1 val; 1,000 images sampled from 3,166, one question per image |
| Seed | 2026 (`--seed`); allocation `--vqav2 2000 --textvqa 1000` |
| Exclusions | the 30 VQA v2 and 40 TextVQA pilot images (`data/*-pilot/subset.jsonl`) |
| Task labels | VQA v2: `how many…` → counting, answer_type yes/no → yes_no, else general (pilot rule); TextVQA: ocr |
| Dry-run mix | counting 208, general 1,043, yes_no 749, ocr 1,000 |
| Mixture | 2/3 VQA v2, 1/3 TextVQA, fixed before the run; the overall result applies to this mixture only |
| Preserved | dataset question text, all 10 annotator answers (duplicates kept), majority answer, question/image ids, `other_question_ids_on_image`, image URLs, image SHA-256 and dimensions |
| Sources | VQA v2 annotations CC BY 4.0 (COCO images: Flickr terms); TextVQA annotations CC BY 4.0 (Open Images images CC BY 2.0, per-image verification advised). Full notes in `metadata.json` |

Official annotation files are read from the pilot `raw/` directories (already
present, checksum verified); only the selected images are downloaded, one
request each. `manifest.jsonl` is never overwritten; delete the directory
deliberately to resample.

## What changed from the pilot

Unchanged: feature injection through `mm_encoder_outputs`, the per-example
identity check (bit-exact in the smoke test), the encoder-did-not-run and
decoder-saw-the-injected-tensor hooks, the INT8 codec, prompt, greedy
generation with 32 new tokens, and every scoring rule in `score_results.py`.

Changed in `inject_eval.py`:

- `--resume` continues an existing output directory. `run_config.json` stores
  the checkpoint and revision, model dtype and GPU name, prompt, generation
  config, image-processor settings, codec source hash, manifest hashes, the
  ordered example-id hash, and software versions. A resume whose settings
  differ in any of these is rejected. Completed ids are read from
  `results.jsonl` (one fsync'd line per example); a partial last line from an
  interrupted write is discarded and reported; corruption anywhere else, or
  duplicate ids, is an error. A lock file prevents two sessions from writing
  the same output directory.
- Per-example failures (including a failed identity check, which previously
  aborted the whole run) are written to `failures.jsonl` with a traceback and
  retried on resume. The run stops after `--max-failures` (default 10)
  failures in one session.
- `--limit N` for smoke tests, `--minimal-timing` (runs=1, warmups=0,
  encoder-runs=1; quality checks unchanged; recorded in `run_info.json`
  `timing_collection` and per record as `timing_mode`), `--skip-payloads`
  (payloads are ~3.4 MB per example; sizes are still recorded).
- `progress.json` (completed, remaining, seconds per example, ETA) and
  `sessions.jsonl` (one line per start or resume) are written. Progress is
  printed as one line per example instead of the full record.
- Records carry `dataset` and `image_group` from the manifest.

`score_results.py` additionally reports `by_dataset`, pending failures,
`expected_examples`, `complete`, and `timing_modes`. Pilot results still score
identically (`vqa_accuracy_difference_pp` = −0.714 on `pilot70-001`).

New: `report_retention.py` (bootstrap, decision, `retention.json`,
`report.md`), `src/vit_compression_lab/{eval_sampling,paired_stats,run_checkpoint}.py`,
and tests in `tests/` using synthetic data only.

## Commands

All from the repository root on the GPU host, with the virtual environment.

1. Prepare the manifest (dry run first; the real run downloads ~575 MB):

```sh
python scripts/prepare_eval_manifest.py --dry-run
python scripts/prepare_eval_manifest.py          # needs approval: downloads images
```

2. Smoke test (first 5 manifest examples, no repeated timing):

```sh
python encoders/Qwen3.8-vision-compression/scripts/inject_eval.py \
  data/eval-3000/manifest.jsonl --limit 5 --minimal-timing --skip-payloads \
  --output-dir outputs/qwen3.8/inject-eval/eval3000-smoke-001
python encoders/Qwen3.8-vision-compression/scripts/score_results.py \
  outputs/qwen3.8/inject-eval/eval3000-smoke-001
python encoders/Qwen3.8-vision-compression/scripts/report_retention.py \
  outputs/qwen3.8/inject-eval/eval3000-smoke-001
```

3. Full evaluation, and resuming it after an interruption (same arguments plus
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

Drop `--minimal-timing` to keep the pilot's repeated latency measurement
(50 codec/preprocess runs and 20 encoder runs per example). Watch
`progress.json` for the ETA.

4. Statistical report:

```sh
python encoders/Qwen3.8-vision-compression/scripts/score_results.py \
  outputs/qwen3.8/inject-eval/eval3000-001
python encoders/Qwen3.8-vision-compression/scripts/report_retention.py \
  outputs/qwen3.8/inject-eval/eval3000-001
```

## Outputs (all under ignored directories)

| Path | Contents |
| --- | --- |
| `data/eval-3000/manifest.jsonl`, `metadata.json` | frozen evaluation set and sampling record |
| `<run>/run_config.json` | resume fingerprint |
| `<run>/run_info.json` | model, software, hardware, prompt, generation, timing collection |
| `<run>/results.jsonl` | per example: both answers, identity check, sizes, errors, timings |
| `<run>/failures.jsonl`, `progress.json`, `sessions.jsonl` | failures with tracebacks, progress/ETA, sessions |
| `<run>/scored.jsonl`, `summary.json` | per-example scores and aggregates |
| `<run>/retention.json`, `report.md` | retention, bounds, decision, side-by-side tables, changed answers |

## Before the full run

Needs approval, with the numbers below from the dry run and the smoke test.

| Item | Estimate |
| --- | --- |
| Image download | ≈575 MB for 3,000 images (HEAD sample of 40 per dataset); ~165 KB per COCO image, ~307 KB per TextVQA image |
| Disk | images ≈0.6 GB; results/scores ≈30 MB; payloads ≈10 GB if `--skip-payloads` is omitted; 94 GB free on the host |
| GPU | one A100 80GB with ≥58 GB free for the whole run (bf16 weights ≈51 GB); both GPUs are shared and memory use by other users fluctuates |
| Runtime | 6–10 s per example (pilot 6.2 s/example with repeated timing; smoke test 6.5–12.6 s with minimal timing, warm-up included): about 5–8.5 h for 3,000, plus ~1 min model load per start |

Smoke test on the pilot manifests (`smoke-resume-001/002`, 2026-10-28):
answers identical to `pilot70-001`, identity checks bit-exact, resume with a
changed example set or generation setting rejected, resumption from a
truncated `results.jsonl` completed the missing example without duplicates.
