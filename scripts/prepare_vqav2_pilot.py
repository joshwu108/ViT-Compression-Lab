"""Prepare a small, reproducible VQA v2 validation subset.

Downloads only the official validation question and annotation zips (3.5 MB and
10.5 MB) and the selected COCO val2014 images, one HTTP request per image. It
never downloads the 6.6 GB val2014 image archive. Run from the repository root:

    python scripts/prepare_vqav2_pilot.py --dry-run   # sizes only, no images
    python scripts/prepare_vqav2_pilot.py

Output (all under ignored data/):
    raw/*.zip                    official files, checksum verified, not extracted
    images/COCO_val2014_*.jpg    selected images
    subset.jsonl                 one record per selected question
    metadata.json                source, revision, license, sampling settings
"""

import argparse
import hashlib
import json
import random
import subprocess
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

BASE_URL = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/"
# Each server ETag is the file's MD5 (single-part uploads).
FILES = {
    "questions": (
        "v2_Questions_Val_mscoco.zip",
        "v2_OpenEnded_mscoco_val2014_questions.json",
        "b9d3a3147866ec65b7e000a4767cc8aa",
    ),
    "annotations": (
        "v2_Annotations_Val_mscoco.zip",
        "v2_mscoco_val2014_annotations.json",
        "4607184e3d7f057ed22f65ca55038e6b",
    ),
}
# COCO's image host serves plain HTTP only; integrity is recorded as SHA-256.
IMAGE_URL = "http://images.cocodataset.org/val2014/COCO_val2014_{image_id:012d}.jpg"
MAX_IMAGE_BYTES = 20_000_000  # Refuse anything unexpectedly large.
TASK_TYPES = ["counting", "yes_no", "general"]

LICENSES = {
    "annotations": "VQA v2 questions/answers: CC BY 4.0 (https://visualqa.org/terms.html)",
    "images": (
        "COCO does not own the images; use must follow the Flickr Terms of Use "
        "(https://cocodataset.org/#termsofuse). Image URLs are recorded per record."
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


def load_official(raw_dir, key):
    zip_name, member, expected_md5 = FILES[key]
    path = raw_dir / zip_name
    if not path.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fetch(BASE_URL + zip_name, 20_000_000))
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    if md5 != expected_md5:
        raise ValueError(f"{path} has MD5 {md5}, expected {expected_md5}")
    with zipfile.ZipFile(path) as archive:
        return json.load(archive.open(member))


def task_type(annotation):
    # VQA's answer_type "number" also covers times and printed numbers, so
    # counting is identified from the "how many" question types instead.
    if annotation["question_type"].startswith("how many"):
        return "counting"
    if annotation["answer_type"] == "yes/no":
        return "yes_no"
    return "general"


def select(annotations, per_task, seed):
    """Pick per_task questions per task type, each on a different image."""
    rng = random.Random(seed)
    # Sorting before shuffling makes the result independent of file order.
    shuffled = sorted(annotations, key=lambda a: a["question_id"])
    rng.shuffle(shuffled)
    selected = {t: [] for t in TASK_TYPES}
    used_images = set()
    for annotation in shuffled:
        chosen = selected[task_type(annotation)]
        if len(chosen) < per_task and annotation["image_id"] not in used_images:
            chosen.append(annotation)
            used_images.add(annotation["image_id"])
    return [a for t in TASK_TYPES for a in selected[t]]


def git_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path("data/vqav2-pilot"))
    parser.add_argument("--per-task", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dry-run", action="store_true", help="report image sizes; download none"
    )
    args = parser.parse_args()

    raw_dir = args.output_dir / "raw"
    questions = load_official(raw_dir, "questions")
    annotations = load_official(raw_dir, "annotations")
    question_text = {q["question_id"]: q["question"] for q in questions["questions"]}
    question_ids_by_image = defaultdict(list)
    for q in questions["questions"]:
        question_ids_by_image[q["image_id"]].append(q["question_id"])

    selected = select(annotations["annotations"], args.per_task, args.seed)

    sizes = [remote_size(IMAGE_URL.format(image_id=a["image_id"])) for a in selected]
    print(f"{len(selected)} images, {sum(sizes) / 1e6:.1f} MB total, "
          f"largest {max(sizes) / 1e6:.1f} MB")
    if args.dry_run:
        return
    if max(sizes) > MAX_IMAGE_BYTES:
        raise SystemExit("An image is unexpectedly large; inspect before continuing.")

    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for a in selected:
        url = IMAGE_URL.format(image_id=a["image_id"])
        image_path = image_dir / url.rsplit("/", 1)[1]
        if not image_path.exists():
            image_path.write_bytes(fetch(url, MAX_IMAGE_BYTES))
        data = image_path.read_bytes()
        with Image.open(image_path) as image:
            width, height = image.size

        records.append({
            "id": f"vqav2-val-{a['question_id']}",
            "image_path": image_path.relative_to(args.output_dir).as_posix(),
            "question": question_text[a["question_id"]],
            "task_type": task_type(a),
            # The dataset's own most-common answer. Score against all 10.
            "expected_answer": a["multiple_choice_answer"],
            # All 10 human answers, in order, duplicates kept: frequency matters.
            "accepted_answers": [ans["answer"] for ans in a["answers"]],
            "scoring": "vqav2_vqa_accuracy",
            "source": {
                "dataset": "VQA v2",
                "version": annotations["info"]["version"],
                "split": annotations["data_subtype"],
                "question_id": a["question_id"],
                "image_id": a["image_id"],
                "question_type": a["question_type"],
                "answer_type": a["answer_type"],
                # Includes each annotator's self-reported answer_confidence.
                "answers": a["answers"],
                "image_width": width,
                "image_height": height,
                "image_url": url,
                "image_sha256": hashlib.sha256(data).hexdigest(),
                "other_question_ids_on_image": [
                    qid
                    for qid in question_ids_by_image[a["image_id"]]
                    if qid != a["question_id"]
                ],
            },
        })

    with open(args.output_dir / "subset.jsonl", "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    metadata = {
        "dataset": "VQA v2",
        "paper": (
            "Goyal et al., Making the V in VQA Matter: Elevating the Role of Image "
            "Understanding in Visual Question Answering, CVPR 2017"
        ),
        "version": annotations["info"]["version"],
        "date_created": annotations["info"]["date_created"],
        "split": annotations["data_subtype"],
        "files": {
            key: {"url": BASE_URL + zip_name, "member": member, "md5": md5}
            for key, (zip_name, member, md5) in FILES.items()
        },
        "image_url_template": IMAGE_URL,
        "licenses": LICENSES,
        "sampling": {
            "method": (
                "sort annotations by question_id, shuffle with random.Random(seed), "
                "then walk the list taking a question when its task type still has "
                "room and its image is unused. task_type: question_type starting "
                "'how many' -> counting; else answer_type 'yes/no' -> yes_no; "
                "else general"
            ),
            "seed": args.seed,
            "per_task": args.per_task,
            "num_images": len(records),
            "split_questions": len(annotations["annotations"]),
            "split_images": len(question_ids_by_image),
        },
        "scoring": (
            "VQA accuracy as in the official VQAEval.evaluate(): normalize answers "
            "unless all 10 agree; score = mean over the 10 leave-one-annotator-out "
            "subsets of min(1, matches / 3). See scripts/vqa_accuracy.py."
        ),
        "answers": "human annotated, 10 independent annotators per question",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_commit": git_commit(),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {len(records)} records to {args.output_dir / 'subset.jsonl'}")


if __name__ == "__main__":
    main()
