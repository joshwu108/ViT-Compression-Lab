# MMStar evaluation

Run commands from `/home/joshua/ViT-compression-lab` using the existing `.venv`.
This evaluates frozen models; it does not train or download model weights.
Choose a free GPU before starting. The default is `cuda:1`.

## Data

The official [Lin-Chen/MMStar dataset](https://huggingface.co/datasets/Lin-Chen/MMStar)
is pinned to revision `bc98d668301da7b14f648724866e57302778ab27`.
The 59,559,119-byte TSV contains base64-encoded images; extraction preserves
those image bytes and needs only Python and Pillow. There are 1,500 questions,
250 in each of six categories, on 1,430 distinct decoded images.

```bash
cd /home/joshua/ViT-compression-lab
.venv/bin/python scripts/prepare_mmstar.py
```

Data lives at `/mnt/nfs/joshua/ViT-compression-lab/data/mmstar`, accessible
through the ignored `data/mmstar` symlink:

- `raw/MMStar.tsv`: original release, verified against its published SHA-256.
- `images/`: all extracted question images, including repeated benchmark images.
- `manifest.jsonl`: all 1,500 questions in original index order.
- `metadata.json`: revision, checksums, category counts, and overlap audit.

The exact-file and decoded-pixel audit found no overlap against the current
5,000-image training manifest or 500-image reconstruction-validation manifest.
This does not detect resized/recompressed near-duplicates or establish that
MMStar was absent from Qwen's pretraining. The evaluator refuses a changed
manifest or an audit with overlap. For a different student's training data,
repeat preparation with `--check-manifests` naming its training manifests.

## Option C and teacher together

The default student is the immutable `step-010000-best.pt` archive, not the
mutable `best.pt`. Override `--student-checkpoint` to evaluate another C model.
Both answers use identical examples, the Qwen chat template, disabled thinking,
the same option-letter instruction, greedy decoding, and at most 32 new tokens.

```text
Teacher: image -> original Qwen ViT + merger -> original features -> Qwen LLM
C:       image -> original Qwen ViT + merger -> student encoder -> round + rANS
         -> rANS decode -> student decoder -> reconstructed features -> Qwen LLM
```

Start with two questions, using a separate output directory:

```bash
.venv/bin/python encoders/Qwen3.8-vision-compression/scripts/eval_mmstar.py \
  --pipeline c --limit 2 --device cuda:1 \
  --output-dir outputs/qwen3.8/mmstar/c-step10000-smoke
```

Then evaluate all 1,500:

```bash
.venv/bin/python encoders/Qwen3.8-vision-compression/scripts/eval_mmstar.py \
  --pipeline c --device cuda:1 \
  --output-dir outputs/qwen3.8/mmstar/c-step10000
```

The paired command already generates uncompressed teacher answers. It performs
the existing identity-injection checks and exact latent round-trip checks,
and measures actual rANS payload bytes including the container header. Payloads
are measured in memory rather than saved as 1,500 individual binary files.
Timing uses one measurement per stage, not repeated latency benchmarking.

## Option B and teacher together

`--pipeline b` evaluates a block-k -> block-m student with the same examples,
prompt, generation settings, identity checks, round-trip checks and scoring.
The checkpoint must be given explicitly, and must be an Option B checkpoint:

```text
B: image -> Qwen stem + blocks 1-8 -> student encoder -> round + rANS
   -> rANS decode -> student decoder -> Qwen blocks 9-27 + merger -> Qwen LLM
```

```bash
.venv/bin/python encoders/Qwen3.8-vision-compression/scripts/eval_mmstar.py \
  --pipeline b --device cuda:1 \
  --student-checkpoint runs/qwen3.8/es-abc-pilot5k/<B-run>/checkpoints/step-<N>-best.pt \
  --output-dir outputs/qwen3.8/mmstar/b-step<N>
```

The summary's reconstruction metrics and bits per visual token are measured at
the merger features the LLM receives, the same location as for C.

## Teacher alone

This command runs ordinary image-to-answer Qwen generation without loading a
student. It retains the same identity checks and prompt/generation settings:

```bash
.venv/bin/python encoders/Qwen3.8-vision-compression/scripts/eval_mmstar.py \
  --pipeline teacher --device cuda:1 \
  --output-dir outputs/qwen3.8/mmstar/teacher
```

Add `--limit 2` and use `teacher-smoke` as the output directory to check it first.
A separate full teacher-only run is optional because the C command includes it.

## Results and resume

All output directories above resolve through `outputs/qwen3.8/mmstar` to
`/mnt/nfs/joshua/ViT-compression-lab/outputs/qwen3.8/mmstar`.
Check `progress.json` for completed/remaining counts. Resume a stopped run by
repeating its original command with `--resume`; keep the same checkpoint,
example selection, prompt, generation settings, and software. Failed examples
are retried. Model weights and optimizer states are not changed.

Each completed example is appended to `results.jsonl`. The evaluator writes
`scored.jsonl` and `summary.json` when its loop finishes. To score existing
completed records again:

```bash
.venv/bin/python encoders/Qwen3.8-vision-compression/scripts/score_mmstar.py \
  outputs/qwen3.8/mmstar/c-step10000
```

Scoring follows the official MMStar answer-prefix rule, including its permissive
handling of the first letter. This is multiple-choice accuracy, not VQAv2 or
TextVQA annotator-consensus accuracy. Raw answers remain available for inspection.
Summaries include category/subcategory accuracy, missing ids, identity checks,
and an explicit `full_benchmark` flag. C additionally reports paired bootstrap
confidence intervals clustered by decoded image, actual pooled bits per visual
token, median reconstruction metrics, and exact round-trip checks. A partial
result or two-question smoke check is not a full benchmark result.

Generated data, outputs, and existing student weights are already covered by
`.gitignore`. Preparation/evaluation performs no Git operations and creates no
new model weights.
