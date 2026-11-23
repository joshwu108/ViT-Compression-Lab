"""Download a balanced, deduplicated image pool for Entropic Student training.

Selects COCO train2014 images and TextVQA *training-annotation* images with a
seeded shuffle, downloads only the selected images (one request per image),
and writes image-level train/validation manifests:

    python scripts/prepare_es_train_pool.py --output-dir data/es-abc-pilot5k

    raw/                         official metadata + download_log.jsonl
    images/coco_train2014/       selected COCO images
    images/textvqa_train/        selected TextVQA (Open Images) images
    manifest.jsonl               training rows
    val_manifest.jsonl           held-out rows
    val_fast_manifest.jsonl      fixed subset of val_manifest.jsonl
    train_monitor_manifest.jsonl fixed subset of manifest.jsonl
    metadata.json                counts, rejections, exclusions, hashes

Candidates are taken in shuffled order. A candidate is rejected (and the next
one takes its place) when its source id, file SHA-256 or decoded-pixel SHA-256
is already in an evaluation manifest or was already accepted, or when it
cannot be downloaded or decoded. Rerunning reuses files already on disk.

Manifests interleave the two sources, so the first 2k rows of each manifest
hold k images per source: the first rows of manifest.jsonl are the training
monitor subset and the first rows of val_manifest.jsonl are the fast subset.
"""

import argparse
import hashlib
import json
import random
import time
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

VQA_QUESTIONS_URL = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip"
VQA_QUESTIONS_MD5 = "6785d6b8b31c901ec7847e4c65e14bb9"  # the server's ETag (single-part upload)
VQA_QUESTIONS_MEMBER = "v2_OpenEnded_mscoco_train2014_questions.json"
TEXTVQA_TRAIN_URL = "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train.json"
TEXTVQA_TRAIN_BYTES = 21634937  # Content-Length; the multi-part ETag is not an MD5
COCO_IMAGE_URL = "http://images.cocodataset.org/train2014/COCO_train2014_{image_id:012d}.jpg"
OPEN_IMAGES_URL = "https://open-images-dataset.s3.amazonaws.com/train/{image_id}.jpg"
MAX_IMAGE_BYTES = 20_000_000
DEFAULT_EXCLUDES = ["data/eval-3000/manifest.jsonl", "data/vqav2-pilot/subset.jsonl",
                    "data/textvqa-pilot/subset.jsonl", "data/es-stage1-smoke/manifest.jsonl",
                    "data/es-stage1-smoke/val_manifest.jsonl"]
SOURCES = ("coco_train2014", "textvqa_train")


def fetch(url, max_bytes, attempts=3):
    """GET with bounded retries. Returns (bytes, attempts used)."""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ValueError(f"{url} exceeds {max_bytes} bytes")
            return data, attempt
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(2 * attempt)


def write_atomic(path, data):
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def image_hashes(path):
    """File SHA-256, SHA-256 of the decoded RGB pixels (catches re-encodings of
    identical pixels, not resized or recompressed copies), and (width, height)."""
    data = Path(path).read_bytes()
    with Image.open(path) as source:
        image = source.convert("RGB")
    pixels = hashlib.sha256(f"{image.size}".encode() + image.tobytes()).hexdigest()
    return hashlib.sha256(data).hexdigest(), pixels, image.size


def load_metadata(raw_dir):
    """Candidate lists per source from the official files (downloaded once)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / Path(VQA_QUESTIONS_URL).name
    if not zip_path.exists():
        write_atomic(zip_path, fetch(VQA_QUESTIONS_URL, 50_000_000)[0])
    if hashlib.md5(zip_path.read_bytes()).hexdigest() != VQA_QUESTIONS_MD5:
        raise ValueError(f"{zip_path}: MD5 differs from the official file")
    with zipfile.ZipFile(zip_path) as archive:
        questions = json.load(archive.open(VQA_QUESTIONS_MEMBER))
    if questions["data_subtype"] != "train2014":
        raise ValueError("VQA question file is not the train2014 split")
    coco = [{"source": "coco_train2014", "image_id": i, "url": COCO_IMAGE_URL.format(image_id=i),
             "file": f"COCO_train2014_{i:012d}.jpg", "id": f"coco-train2014-{i:012d}", "split": "train2014"}
            for i in sorted({q["image_id"] for q in questions["questions"]})]

    json_path = raw_dir / Path(TEXTVQA_TRAIN_URL).name
    if not json_path.exists():
        write_atomic(json_path, fetch(TEXTVQA_TRAIN_URL, 100_000_000)[0])
    if json_path.stat().st_size != TEXTVQA_TRAIN_BYTES:
        raise ValueError(f"{json_path}: size differs from the official file")
    textvqa_data = json.loads(json_path.read_text())
    if textvqa_data["dataset_type"] != "train":
        raise ValueError("TextVQA annotation file is not the train split")
    sizes = {}
    for q in textvqa_data["data"]:
        if q["set_name"] != "train":
            raise ValueError(f"TextVQA question {q['question_id']} is not a train question")
        sizes[q["image_id"]] = [q["image_width"], q["image_height"]]
    textvqa = [{"source": "textvqa_train", "image_id": i, "url": OPEN_IMAGES_URL.format(image_id=i),
                "file": f"{i}.jpg", "id": f"textvqa-train-{i}", "split": "train", "annotation_size": sizes[i]}
               for i in sorted(sizes)]
    files = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
             for p in (zip_path, json_path)}
    return {"coco_train2014": coco, "textvqa_train": textvqa}, files


def load_exclusions(manifests):
    """Source image ids, file hashes and decoded-pixel hashes of every evaluation image."""
    ids, shas, pixels, counts = set(), set(), set(), {}
    seen_paths = set()
    for manifest in map(Path, manifests):
        if not manifest.exists():
            counts[str(manifest)] = None
            continue
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        counts[str(manifest)] = len(rows)
        for row in rows:
            source = row.get("source") or {}
            if "image_id" in source:
                ids.add(str(source["image_id"]))
            if source.get("image_sha256"):
                shas.add(source["image_sha256"])
            path = (manifest.parent / row["image_path"]).resolve()
            if path in seen_paths or not path.exists():
                continue
            seen_paths.add(path)
            sha, pixel, _ = image_hashes(path)
            shas.add(sha)
            pixels.add(pixel)
    return {"ids": ids, "sha256": shas, "pixel_sha256": pixels, "rows_per_manifest": counts,
            "unique_image_files": len(seen_paths)}


def download_candidate(candidate, image_dir):
    """Download (or reuse) one image and hash it. Never raises; returns a status record."""
    path = image_dir / candidate["file"]
    record = {**candidate, "attempts": 0, "status": "ok"}
    try:
        if not path.exists():
            data, record["attempts"] = fetch(candidate["url"], MAX_IMAGE_BYTES)
            write_atomic(path, data)
            record["status"] = "downloaded"
        else:
            record["status"] = "reused"
    except Exception as error:
        return {**record, "status": "download_failed", "error": f"{type(error).__name__}: {error}"}
    try:
        record["image_sha256"], record["pixel_sha256"], size = image_hashes(path)
        record["image_width"], record["image_height"] = size
        record["bytes"] = path.stat().st_size
    except Exception as error:
        return {**record, "status": "unreadable", "error": f"{type(error).__name__}: {error}"}
    return record


def select_unique(candidates, download, target, excluded, seen, chunk=64, workers=16):
    """Accept the first ``target`` candidates (in order) that download, decode and are
    neither excluded nor duplicates; later candidates replace rejected ones.

    ``excluded``: {"ids", "sha256", "pixel_sha256"} of evaluation images.
    ``seen``: same keys, updated with accepted images (shared across sources).
    Returns (accepted records, rejected records).
    """
    accepted, rejected, position = [], [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while len(accepted) < target:
            # Only fetch as many as are still needed, so no extra image is downloaded.
            batch = candidates[position: position + min(chunk, target - len(accepted))]
            if not batch:
                raise RuntimeError(f"Ran out of candidates with {len(accepted)}/{target} accepted")
            position += len(batch)
            fetchable = [c for c in batch if str(c["image_id"]) not in excluded["ids"]]
            results = {c["id"]: r for c, r in zip(fetchable, pool.map(download, fetchable))}
            for candidate in batch:
                record = results.get(candidate["id"]) or {**candidate, "status": "excluded_id"}
                reason = None
                if record["status"] in ("excluded_id", "download_failed", "unreadable"):
                    reason = record["status"]
                elif record["image_sha256"] in excluded["sha256"] or record["pixel_sha256"] in excluded["pixel_sha256"]:
                    reason = "excluded_evaluation_image"
                elif record["image_sha256"] in seen["sha256"] or record["pixel_sha256"] in seen["pixel_sha256"]:
                    reason = "duplicate"
                if reason:
                    rejected.append({**record, "rejected": reason})
                    continue
                seen["sha256"].add(record["image_sha256"])
                seen["pixel_sha256"].add(record["pixel_sha256"])
                accepted.append(record)
    return accepted, rejected


def split_source(records, n_val, seed):
    """Seeded image-level split of one source: (train, val)."""
    order = list(records)
    random.Random(f"{seed}:split:{records[0]['source']}").shuffle(order)
    return order[n_val:], order[:n_val]


def interleave(by_source):
    rows = []
    for group in zip(*by_source):
        rows.extend(group)
    return rows


def manifest_row(record, split):
    source = {k: record[k] for k in ("image_id", "image_sha256", "pixel_sha256", "image_width", "image_height")}
    source.update({"dataset": record["source"], "dataset_split": record["split"], "image_url": record["url"],
                   "download_status": record["status"], "pool_split": split})
    if "annotation_size" in record:
        source["annotation_size_matches"] = record["annotation_size"] == [record["image_width"],
                                                                         record["image_height"]]
    return {"id": record["id"], "image_path": f"images/{record['source']}/{record['file']}", "source": source}


def write_jsonl(path, rows):
    tmp = path.with_name(path.name + ".part")
    tmp.write_text("".join(json.dumps(row) + "\n" for row in rows))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, help="Official metadata directory (default: <output-dir>/raw)")
    parser.add_argument("--per-source", type=int, default=2750)
    parser.add_argument("--val-per-source", type=int, default=250)
    parser.add_argument("--val-fast-per-source", type=int, default=32)
    parser.add_argument("--monitor-per-source", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--exclude-manifests", nargs="*", default=DEFAULT_EXCLUDES)
    args = parser.parse_args()

    out = args.output_dir
    raw_dir = args.raw_dir or out / "raw"
    (out / "raw").mkdir(parents=True, exist_ok=True)
    candidates, metadata_files = load_metadata(raw_dir)
    excluded = load_exclusions(args.exclude_manifests)
    print(f"exclusions: {len(excluded['ids'])} source ids, {len(excluded['sha256'])} file hashes, "
          f"{len(excluded['pixel_sha256'])} pixel hashes")

    seen = {"sha256": set(), "pixel_sha256": set()}
    accepted, rejected, log = {}, {}, []
    for source in SOURCES:
        pool = list(candidates[source])
        random.Random(f"{args.seed}:{source}").shuffle(pool)
        image_dir = out / "images" / source
        image_dir.mkdir(parents=True, exist_ok=True)
        accepted[source], rejected[source] = select_unique(
            pool, lambda c: download_candidate(c, image_dir), args.per_source, excluded, seen, workers=args.workers)
        for record in rejected[source]:  # images/ holds selected images only
            (image_dir / record["file"]).unlink(missing_ok=True)
        log += [{**r, "accepted": True} for r in accepted[source]] + [{**r, "accepted": False}
                                                                       for r in rejected[source]]
        print(f"{source}: {len(accepted[source])} accepted, {len(rejected[source])} rejected "
              f"{dict(Counter(r['rejected'] for r in rejected[source]))}")
    write_jsonl(out / "raw" / "download_log.jsonl", log)

    splits = {source: split_source(accepted[source], args.val_per_source, args.seed) for source in SOURCES}
    train = [manifest_row(r, "train") for r in interleave([splits[s][0] for s in SOURCES])]
    val = [manifest_row(r, "val") for r in interleave([splits[s][1] for s in SOURCES])]
    val_fast = val[: 2 * args.val_fast_per_source]
    monitor = train[: 2 * args.monitor_per_source]

    # Exact counts, source membership, uniqueness, and zero overlap before anything is written.
    n_train = args.per_source - args.val_per_source
    members = {s: {c["id"] for c in candidates[s]} for s in SOURCES}
    for name, rows, per_source in (("train", train, n_train), ("val", val, args.val_per_source),
                                   ("val_fast", val_fast, args.val_fast_per_source),
                                   ("monitor", monitor, args.monitor_per_source)):
        counts = Counter(r["source"]["dataset"] for r in rows)
        assert counts == {s: per_source for s in SOURCES}, (name, counts)
        assert all(r["id"] in members[r["source"]["dataset"]] for r in rows), name
    for key in ("id", "image_sha256", "pixel_sha256"):
        values = [r["id"] if key == "id" else r["source"][key] for r in train + val]
        assert len(set(values)) == len(values), f"duplicate {key}"
    for rows in (train, val):
        assert not {r["source"]["image_sha256"] for r in rows} & excluded["sha256"]
        assert not {r["source"]["pixel_sha256"] for r in rows} & excluded["pixel_sha256"]
        assert not {str(r["source"]["image_id"]) for r in rows} & excluded["ids"]
    assert not {r["id"] for r in train} & {r["id"] for r in val}
    assert all((out / r["image_path"]).is_file() for r in train + val)

    outputs = {"manifest.jsonl": train, "val_manifest.jsonl": val, "val_fast_manifest.jsonl": val_fast,
               "train_monitor_manifest.jsonl": monitor}
    for name, rows in outputs.items():
        write_jsonl(out / name, rows)
    metadata = {
        "purpose": "Entropic Student training/validation image pool (images only; no questions or answers)",
        "created_at": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
        "sources": {"coco_train2014": {"metadata_url": VQA_QUESTIONS_URL, "image_url_template": COCO_IMAGE_URL,
                                       "candidates": len(candidates["coco_train2014"])},
                    "textvqa_train": {"metadata_url": TEXTVQA_TRAIN_URL, "image_url_template": OPEN_IMAGES_URL,
                                      "candidates": len(candidates["textvqa_train"])}},
        "metadata_files": metadata_files,
        "sampling": "sort source image ids, shuffle with random.Random(f'{seed}:{source}'), accept in order; "
                    "split each source with random.Random(f'{seed}:split:{source}'), first val_per_source -> val",
        "counts": {name: dict(Counter(r["source"]["dataset"] for r in rows), total=len(rows))
                   for name, rows in outputs.items()},
        "rejected": {s: dict(Counter(r["rejected"] for r in rejected[s])) for s in SOURCES},
        "rejected_records": [{k: r.get(k) for k in ("id", "url", "rejected", "error")}
                             for s in SOURCES for r in rejected[s]],
        "download_status": dict(Counter(r["status"] for s in SOURCES for r in accepted[s])),
        "textvqa_annotation_size_mismatches": sum(
            1 for r in train + val if r["source"].get("annotation_size_matches") is False),
        "exclusions": {"manifests": excluded["rows_per_manifest"], "source_ids": len(excluded["ids"]),
                       "file_sha256": len(excluded["sha256"]), "pixel_sha256": len(excluded["pixel_sha256"]),
                       "unique_image_files": excluded["unique_image_files"]},
        "overlap_with_exclusions": 0, "train_val_overlap": 0,
        "manifest_sha256": {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in outputs},
        "image_bytes": sum(r["bytes"] for s in SOURCES for r in accepted[s]),
        "licenses": "COCO images follow the Flickr Terms of Use (cocodataset.org/#termsofuse); Open Images "
                    "images are listed as CC BY 2.0. Image URLs are recorded per row.",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: metadata[k] for k in ("counts", "rejected", "download_status")}, indent=2))


if __name__ == "__main__":
    main()
