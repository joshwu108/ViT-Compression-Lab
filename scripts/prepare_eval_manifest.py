"""Prepare the frozen, seeded VQA v2 + TextVQA evaluation manifest.

Reuses the pilot scripts' official-file loaders, image URL templates, licence
notes, and VQA v2 task-type rule. Pilot images are excluded. Run from the
repository root:

    python scripts/prepare_eval_manifest.py --dry-run   # counts and sizes only
    python scripts/prepare_eval_manifest.py             # download images, freeze

Output (all under ignored data/):
    images/vqav2/COCO_val2014_*.jpg, images/textvqa/<image_id>.jpg
    manifest.jsonl    one record per question, in a fixed order
    metadata.json     allocation, seed, exclusions, versions, licences, hashes

The official annotation files are read from the pilot raw/ directories by
default, so nothing is downloaded during --dry-run except a few HEAD requests
used to estimate the image download size. manifest.jsonl is never overwritten.
"""

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare_textvqa_pilot as textvqa  # noqa: E402
import prepare_vqav2_pilot as vqav2  # noqa: E402

from vit_compression_lab.eval_sampling import (  # noqa: E402
    excluded_images_from_manifests,
    select_textvqa,
    select_vqav2,
)

DEFAULT_EXCLUDE = ["data/vqav2-pilot/subset.jsonl", "data/textvqa-pilot/subset.jsonl"]


def vqav2_record(annotation, question_text, question_ids_by_image, image_path, url):
    a = annotation
    return {
        "id": f"vqav2-val-{a['question_id']}",
        "dataset": "vqav2",
        "image_group": f"vqav2:{a['image_id']}",
        "image_path": image_path,
        "question": question_text[a["question_id"]],
        "task_type": vqav2.task_type(a),
        "expected_answer": a["multiple_choice_answer"],
        "accepted_answers": [ans["answer"] for ans in a["answers"]],
        "scoring": "vqav2_vqa_accuracy",
        "source": {
            "dataset": "VQA v2",
            "split": "val2014",
            "question_id": a["question_id"],
            "image_id": a["image_id"],
            "question_type": a["question_type"],
            "answer_type": a["answer_type"],
            "answers": a["answers"],
            "image_url": url,
            "other_question_ids_on_image": [
                qid for qid in question_ids_by_image[a["image_id"]]
                if qid != a["question_id"]
            ],
        },
    }


def textvqa_record(question, other_question_ids, image_path, url):
    q = question
    return {
        "id": f"textvqa-val-{q['question_id']}",
        "dataset": "textvqa",
        "image_group": f"textvqa:{q['image_id']}",
        "image_path": image_path,
        "question": q["question"],
        "task_type": "ocr",
        "expected_answer": textvqa.most_common_answer(q["answers"]),
        "accepted_answers": q["answers"],
        "scoring": "textvqa_vqa_accuracy",
        "source": {
            "dataset": "TextVQA",
            "split": q["set_name"],
            "question_id": q["question_id"],
            "image_id": q["image_id"],
            "image_classes": q["image_classes"],
            "image_width": q["image_width"],
            "image_height": q["image_height"],
            "image_url": url,
            "flickr_original_url": q["flickr_original_url"],
            "other_question_ids_on_image": other_question_ids,
        },
    }


def select_records(args, excluded):
    """Build manifest records (without image hashes) in their final order."""
    questions = vqav2.load_official(args.vqav2_raw_dir, "questions")
    annotations = vqav2.load_official(args.vqav2_raw_dir, "annotations")
    question_text = {q["question_id"]: q["question"] for q in questions["questions"]}
    question_ids_by_image = defaultdict(list)
    for q in questions["questions"]:
        question_ids_by_image[q["image_id"]].append(q["question_id"])
    chosen = select_vqav2(annotations["annotations"], args.vqav2, args.seed,
                          excluded.get("VQA v2", set()))
    records = []
    for a in chosen:
        url = vqav2.IMAGE_URL.format(image_id=a["image_id"])
        path = f"images/vqav2/{url.rsplit('/', 1)[1]}"
        records.append(vqav2_record(a, question_text, question_ids_by_image, path, url))

    text_annotations = textvqa.load_annotations(args.textvqa_raw_dir)
    chosen = select_textvqa(text_annotations["data"], args.textvqa, args.seed,
                            excluded.get("TextVQA", set()))
    for q, others in chosen:
        url = textvqa.IMAGE_URL.format(image_id=q["image_id"])
        records.append(textvqa_record(q, others, f"images/textvqa/{q['image_id']}.jpg", url))
    versions = {"vqav2": annotations["info"]["version"],
                "textvqa": text_annotations["dataset_version"]}
    totals = {"vqav2": {"questions": len(annotations["annotations"]),
                        "images": len(question_ids_by_image)},
              "textvqa": {"questions": len(text_annotations["data"]),
                          "images": len({q["image_id"] for q in text_annotations["data"]})}}
    return records, versions, totals


def estimate_download(records, sample_size, seed):
    """HEAD a seeded sample per dataset and extrapolate the total image bytes."""
    estimate = {}
    rng = random.Random(seed)
    for dataset in ("vqav2", "textvqa"):
        rows = [r for r in records if r["dataset"] == dataset]
        sample = rng.sample(rows, min(sample_size, len(rows)))
        sizes = [vqav2.remote_size(r["source"]["image_url"]) for r in sample]
        mean = sum(sizes) / len(sizes) if sizes else 0
        estimate[dataset] = {"images": len(rows), "sampled": len(sizes),
                             "mean_bytes": mean, "max_sampled_bytes": max(sizes, default=0),
                             "estimated_total_bytes": mean * len(rows)}
    estimate["estimated_total_bytes"] = sum(
        estimate[d]["estimated_total_bytes"] for d in ("vqav2", "textvqa"))
    return estimate


def write_atomic(path, data):
    """Write bytes to a temporary file and rename, so a crash leaves no partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def download_images(records, output_dir):
    """Fetch missing images, then record hash and size for every record."""
    for index, record in enumerate(records, 1):
        path = output_dir / record["image_path"]
        if not path.exists():
            write_atomic(path, vqav2.fetch(record["source"]["image_url"], vqav2.MAX_IMAGE_BYTES))
        data = path.read_bytes()
        with Image.open(path) as image:
            image.load()  # decode fully; a truncated file fails here
            width, height = image.size
        source = record["source"]
        if record["dataset"] == "textvqa" and [width, height] != [
                source["image_width"], source["image_height"]]:
            raise ValueError(f"{path} is {[width, height]}; annotations differ")
        source["image_width"], source["image_height"] = width, height
        source["image_sha256"] = hashlib.sha256(data).hexdigest()
        source["image_bytes"] = len(data)
        if index % 100 == 0 or index == len(records):
            print(f"{index}/{len(records)} images present")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/eval-3000"))
    parser.add_argument("--vqav2", type=int, default=2000, help="VQA v2 questions")
    parser.add_argument("--textvqa", type=int, default=1000, help="TextVQA questions")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--vqav2-raw-dir", type=Path, default=Path("data/vqav2-pilot/raw"))
    parser.add_argument("--textvqa-raw-dir", type=Path, default=Path("data/textvqa-pilot/raw"))
    parser.add_argument("--exclude", type=Path, nargs="+", default=[Path(p) for p in DEFAULT_EXCLUDE],
                        help="Manifests whose images must not be reused (default: both pilots)")
    parser.add_argument("--size-sample", type=int, default=40,
                        help="Images per dataset to HEAD for the size estimate (0: skip)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Select and report; download nothing, write nothing")
    args = parser.parse_args()
    if min(args.vqav2, args.textvqa) < 0 or args.vqav2 + args.textvqa == 0:
        parser.error("Allocation must be nonnegative and nonzero in total")
    manifest_path = args.output_dir / "manifest.jsonl"
    if manifest_path.exists():
        parser.error(f"{manifest_path} already exists; the manifest is frozen")

    excluded = excluded_images_from_manifests(args.exclude)
    records, versions, totals = select_records(args, excluded)
    task_mix = Counter((r["dataset"], r["task_type"]) for r in records)
    print(f"Selected {len(records)} questions on {len({r['image_group'] for r in records})} images")
    for (dataset, task), count in sorted(task_mix.items()):
        print(f"  {dataset}/{task}: {count}")
    print("Excluded images:", {k: len(v) for k, v in excluded.items()})
    estimate = estimate_download(records, args.size_sample, args.seed) if args.size_sample else None
    if estimate:
        print(f"Estimated image download: {estimate['estimated_total_bytes'] / 1e6:.0f} MB "
              f"(HEAD sample of {args.size_sample} per dataset)")
    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    download_images(records, args.output_dir)
    write_atomic(manifest_path, "".join(json.dumps(r) + "\n" for r in records).encode())
    metadata = {
        "purpose": "Experiment 2 large evaluation: INT8 visual-feature retention",
        "datasets": {
            "vqav2": {"name": "VQA v2", "version": versions["vqav2"], "split": "val2014",
                      "paper": "Goyal et al., CVPR 2017", "files": {
                          key: {"url": vqav2.BASE_URL + zip_name, "member": member, "md5": md5}
                          for key, (zip_name, member, md5) in vqav2.FILES.items()},
                      "image_url_template": vqav2.IMAGE_URL, "licenses": vqav2.LICENSES,
                      "split_totals": totals["vqav2"], "scoring": "vqav2_vqa_accuracy"},
            "textvqa": {"name": "TextVQA", "version": versions["textvqa"], "split": "val",
                        "paper": "Singh et al., CVPR 2019",
                        "annotations_url": textvqa.ANNOTATIONS_URL,
                        "annotations_md5": textvqa.ANNOTATIONS_MD5,
                        "image_url_template": textvqa.IMAGE_URL, "licenses": textvqa.LICENSES,
                        "split_totals": totals["textvqa"], "scoring": "textvqa_vqa_accuracy"},
        },
        "allocation": {"vqav2": args.vqav2, "textvqa": args.textvqa,
                       "mixture_note": "Fixed before evaluation; overall results apply to "
                                       "this mixture, not to either benchmark alone."},
        "sampling": {
            "seed": args.seed,
            "vqav2": "sort annotations by question_id, shuffle with random.Random(seed), "
                     "walk the list taking a question when its image is unused and not "
                     "excluded (natural task mix, one question per image)",
            "textvqa": "sort unique image_ids minus exclusions, shuffle with "
                       "random.Random(seed), take the first N, rng.choice one question "
                       "per image (candidates sorted by question_id)",
            "task_type_rule": "VQA v2: question_type starting 'how many' -> counting; "
                              "else answer_type 'yes/no' -> yes_no; else general. "
                              "TextVQA: ocr",
            "task_mix": {f"{d}/{t}": n for (d, t), n in sorted(task_mix.items())},
            "one_question_per_image": True,
            "image_group_field": "image_group (dataset:image_id) for clustered statistics",
        },
        "exclusions": {"manifests": [str(p) for p in args.exclude],
                       "excluded_images": {k: sorted(map(str, v)) for k, v in excluded.items()}},
        "download_estimate": estimate,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "num_records": len(records),
        "answers": "human annotated, 10 annotators per question; accepted_answers keeps "
                   "duplicates because frequency matters for VQA accuracy",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_commit": vqav2.git_commit(),
    }
    write_atomic(args.output_dir / "metadata.json", (json.dumps(metadata, indent=2) + "\n").encode())
    print(f"Wrote {len(records)} records to {manifest_path}")


if __name__ == "__main__":
    main()
