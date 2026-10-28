"""Prepare a small, reproducible TextVQA v0.5.1 validation subset.

Downloads only the 3.1 MB official validation annotation file and the selected
images (one HTTP request per image from the Open Images public bucket). It never
downloads the 7 GB TextVQA image archive. Run from the repository root:

    python scripts/prepare_textvqa_pilot.py --dry-run   # sizes only, no images
    python scripts/prepare_textvqa_pilot.py

Output (all under ignored data/):
    raw/TextVQA_0.5.1_val.json   official annotations, checksum verified
    images/<image_id>.jpg        selected images
    subset.jsonl                 one record per selected question
    metadata.json                source, revision, license, sampling settings
"""

import argparse
import hashlib
import json
import random
import subprocess
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ANNOTATIONS_URL = "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json"
# The server's ETag for this single-part upload is the file's MD5.
ANNOTATIONS_MD5 = "90f839369ad2177b4b76610438603a6a"
# TextVQA images are Open Images training images, addressed by image_id.
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/train/{image_id}.jpg"
MAX_IMAGE_BYTES = 20_000_000  # Refuse anything unexpectedly large.

LICENSES = {
    "annotations": "TextVQA questions/answers: CC BY 4.0 (https://textvqa.org)",
    "images": (
        "Open Images images are listed as CC BY 2.0; Open Images asks users to "
        "verify each image's license themselves. Image URLs are recorded per record."
    ),
}


def fetch(url, max_bytes):
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"{url} exceeds {max_bytes} bytes")
    return data


def remote_size(url):
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def load_annotations(raw_dir):
    path = raw_dir / "TextVQA_0.5.1_val.json"
    if not path.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fetch(ANNOTATIONS_URL, 10_000_000))
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    if md5 != ANNOTATIONS_MD5:
        raise ValueError(f"{path} has MD5 {md5}, expected {ANNOTATIONS_MD5}")
    return json.loads(path.read_text())


def select(questions, num_images, seed):
    """Pick distinct images at random, then one question per image."""
    by_image = defaultdict(list)
    for q in questions:
        by_image[q["image_id"]].append(q)
    # Sorting before shuffling makes the result independent of file order.
    rng = random.Random(seed)
    image_ids = sorted(by_image)
    rng.shuffle(image_ids)
    selected = []
    for image_id in image_ids[:num_images]:
        candidates = sorted(by_image[image_id], key=lambda q: q["question_id"])
        chosen = rng.choice(candidates)
        others = [q["question_id"] for q in candidates if q is not chosen]
        selected.append((chosen, others))
    return selected


def most_common_answer(answers):
    # Counter keeps first-seen order for ties, so this is deterministic.
    return Counter(answers).most_common(1)[0][0]


def git_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path("data/textvqa-pilot"))
    parser.add_argument("--num-images", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dry-run", action="store_true", help="report image sizes; download none"
    )
    args = parser.parse_args()

    annotations = load_annotations(args.output_dir / "raw")
    selected = select(annotations["data"], args.num_images, args.seed)

    sizes = [remote_size(IMAGE_URL.format(image_id=q["image_id"])) for q, _ in selected]
    print(f"{len(selected)} images, {sum(sizes) / 1e6:.1f} MB total, "
          f"largest {max(sizes) / 1e6:.1f} MB")
    if args.dry_run:
        return
    if max(sizes) > MAX_IMAGE_BYTES:
        raise SystemExit("An image is unexpectedly large; inspect before continuing.")

    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for q, other_question_ids in selected:
        url = IMAGE_URL.format(image_id=q["image_id"])
        image_path = image_dir / f"{q['image_id']}.jpg"
        if not image_path.exists():
            image_path.write_bytes(fetch(url, MAX_IMAGE_BYTES))
        data = image_path.read_bytes()
        with Image.open(image_path) as image:
            image_size = list(image.size)
        if image_size != [q["image_width"], q["image_height"]]:
            raise ValueError(f"{image_path} is {image_size}, annotations differ")

        records.append({
            "id": f"textvqa-val-{q['question_id']}",
            "image_path": image_path.relative_to(args.output_dir).as_posix(),
            "question": q["question"],
            "task_type": "ocr",
            # Convenience only. Score against all accepted_answers.
            "expected_answer": most_common_answer(q["answers"]),
            # All 10 human answers, in order, duplicates kept: frequency matters.
            "accepted_answers": q["answers"],
            "scoring": "textvqa_vqa_accuracy",
            "source": {
                "dataset": "TextVQA",
                "version": annotations["dataset_version"],
                "split": q["set_name"],
                "question_id": q["question_id"],
                "image_id": q["image_id"],
                "image_classes": q["image_classes"],
                "image_width": q["image_width"],
                "image_height": q["image_height"],
                "image_url": url,
                "image_sha256": hashlib.sha256(data).hexdigest(),
                "flickr_original_url": q["flickr_original_url"],
                "other_question_ids_on_image": other_question_ids,
            },
        })

    with open(args.output_dir / "subset.jsonl", "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    metadata = {
        "dataset": "TextVQA",
        "paper": "Singh et al., Towards VQA Models That Can Read, CVPR 2019",
        "version": annotations["dataset_version"],
        "split": annotations["dataset_type"],
        "annotations_url": ANNOTATIONS_URL,
        "annotations_md5": ANNOTATIONS_MD5,
        "image_url_template": IMAGE_URL,
        "licenses": LICENSES,
        "sampling": {
            "method": (
                "sort unique image_ids, shuffle with random.Random(seed), take the "
                "first num_images, then rng.choice one question per image "
                "(candidates sorted by question_id)"
            ),
            "seed": args.seed,
            "num_images": len(records),
            "split_questions": len(annotations["data"]),
            "split_images": len({q["image_id"] for q in annotations["data"]}),
        },
        "scoring": (
            "TextVQA/VQA accuracy as in MMF's TextVQAAccuracyEvaluator: normalize "
            "answers with EvalAIAnswerProcessor; score = mean over the 10 "
            "leave-one-annotator-out subsets of min(1, matches / 3). "
            "See scripts/vqa_accuracy.py."
        ),
        "answers": "human annotated, 10 independent annotators per question",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_commit": git_commit(),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {len(records)} records to {args.output_dir / 'subset.jsonl'}")


if __name__ == "__main__":
    main()
