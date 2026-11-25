"""Download the pinned official MMStar TSV and extract a local evaluation manifest.

Images remain byte-for-byte identical to the embedded images. No model weights
or optional dataset libraries are downloaded. Run from the repository root.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import sys
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

from PIL import Image

REVISION = "bc98d668301da7b14f648724866e57302778ab27"
TSV_SHA256 = "38a99f4a33743665e6990961a9a0072b07f7ac6a74d1555b43c4f82145377cb4"
TSV_URL = (
    f"https://huggingface.co/datasets/Lin-Chen/MMStar/resolve/{REVISION}/MMStar.tsv"
)
DEFAULT_OUTPUT = Path("/mnt/nfs/joshua/ViT-compression-lab/data/mmstar")
DEFAULT_CHECK = [
    Path("data/es-abc-pilot5k/manifest.jsonl"),
    Path("data/es-abc-pilot5k/val_manifest.jsonl"),
]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def download(path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".part")
        count = 0
        with urlopen(TSV_URL, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > 100_000_000:
                    raise ValueError("MMStar TSV exceeds the expected download size")
                output.write(chunk)
        if sha256(temporary.read_bytes()) != TSV_SHA256:
            raise ValueError("Downloaded TSV differs from the pinned official checksum")
        os.replace(temporary, path)
    if sha256(path.read_bytes()) != TSV_SHA256:
        raise ValueError(f"{path}: differs from the pinned official TSV checksum")


def reference_hashes(manifests):
    references, provenance = [], []
    for path in manifests:
        data = path.read_bytes()
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        provenance.append(
            {"path": str(path), "sha256": sha256(data), "rows": len(rows)}
        )
        for row in rows:
            source = row["source"]
            references.append(
                {
                    "manifest": str(path),
                    "id": row["id"],
                    "image_sha256": source["image_sha256"],
                    "pixel_sha256": source["pixel_sha256"],
                }
            )
    return references, provenance


def extract(tsv_path, output_dir, check_manifests, expected_count=1500):
    references, checked = reference_hashes(check_manifests)
    by_file, by_pixels = {}, {}
    for reference in references:
        by_file.setdefault(reference["image_sha256"], []).append(reference)
        by_pixels.setdefault(reference["pixel_sha256"], []).append(reference)
    csv.field_size_limit(sys.maxsize)
    with tsv_path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle, delimiter="\t"))
    if len(raw) != expected_count:
        raise ValueError(f"Expected {expected_count} questions, found {len(raw)}")
    if len({row["index"] for row in raw}) != len(raw):
        raise ValueError("Official TSV has duplicate indices")
    images = {row["index"]: row["image"] for row in raw}
    records, overlaps = [], []
    for row in sorted(raw, key=lambda item: int(item["index"])):
        index = int(row["index"])
        encoded = row["image"]
        # VLMEval datasets may refer to an earlier image by its row index.
        visited = set()
        while encoded in images:
            if encoded in visited:
                raise ValueError("Circular image reference in TSV")
            visited.add(encoded)
            encoded = images[encoded]
        data = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            extension = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}[image.format]
            rgb = image.convert("RGB")
            pixel_hash = sha256(f"{rgb.size}".encode() + rgb.tobytes())
            width, height = rgb.size
        file_hash = sha256(data)
        relative = f"images/{index:04d}.{extension}"
        path = output_dir / relative
        if path.exists() and sha256(path.read_bytes()) != file_hash:
            raise ValueError(f"Existing extracted image differs: {path}")
        if not path.exists():
            write_atomic(path, data)
        answer = row["answer"].strip().upper()
        if answer not in "ABCD" or len(answer) != 1:
            raise ValueError(f"Unexpected option for MMStar index {index}: {answer!r}")
        record = {
            "id": f"mmstar-val-{index:04d}",
            "dataset": "mmstar",
            "image_group": f"mmstar:{pixel_hash}",
            "image_path": relative,
            "question": row["question"],
            "task_type": "multiple_choice",
            "expected_answer": answer,
            "accepted_answers": [answer],
            "scoring": "mmstar_official_prefix",
            "category": row["category"],
            "l2_category": row["l2_category"],
            "source": {
                "dataset": "MMStar",
                "split": "val",
                "index": index,
                "bench": row["bench"],
                "revision": REVISION,
                "image_sha256": file_hash,
                "pixel_sha256": pixel_hash,
                "image_width": width,
                "image_height": height,
                "image_bytes": len(data),
            },
        }
        records.append(record)
        matches = {
            f"{match['manifest']}:{match['id']}": match
            for match in by_file.get(file_hash, []) + by_pixels.get(pixel_hash, [])
        }
        if matches:
            overlaps.append({"id": record["id"], "matches": list(matches.values())})
    manifest = "".join(json.dumps(record) + "\n" for record in records).encode()
    path = output_dir / "manifest.jsonl"
    if path.exists() and path.read_bytes() != manifest:
        raise ValueError("Existing manifest differs; choose a new output directory")
    write_atomic(path, manifest)
    metadata = {
        "dataset": "Lin-Chen/MMStar",
        "split": "val",
        "revision": REVISION,
        "tsv_url": TSV_URL,
        "tsv_sha256": sha256(tsv_path.read_bytes()),
        "manifest_sha256": sha256(manifest),
        "questions": len(records),
        "unique_image_files": len({r["source"]["image_sha256"] for r in records}),
        "unique_decoded_images": len({r["image_group"] for r in records}),
        "image_bytes": sum(r["source"]["image_bytes"] for r in records),
        "categories": dict(Counter(r["category"] for r in records)),
        "overlap_check": {
            "checked_manifests": checked,
            "overlapping_questions": len(overlaps),
            "matches": overlaps,
            "method": "exact file SHA-256 or decoded RGB pixels including dimensions; "
            "does not detect resized/recompressed near-duplicates or teacher pretraining overlap",
        },
        "scoring_source": "https://github.com/MMStar-Benchmark/MMStar/blob/main/"
        "eval/vlmeval/evaluate/mmstar.py",
    }
    write_atomic(
        output_dir / "metadata.json", (json.dumps(metadata, indent=2) + "\n").encode()
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check-manifests", type=Path, nargs="+", default=DEFAULT_CHECK
    )
    args = parser.parse_args()
    tsv = args.output_dir / "raw" / "MMStar.tsv"
    download(tsv)
    metadata = extract(tsv, args.output_dir, args.check_manifests)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
