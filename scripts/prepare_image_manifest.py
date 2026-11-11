"""Write a training/validation manifest from a local image directory (no downloads).

    python scripts/prepare_image_manifest.py /path/to/images \
        --output data/train-images/manifest.jsonl --val-fraction 0.05

Rows follow the repository manifest format (``id``, ``image_path`` relative to
the manifest, ``source`` with sha256 and size). Images whose SHA-256 or
resolved path appears in the evaluation manifests (``data/eval-3000`` and the
two pilots by default) are refused, so the pre-registered evaluation set can
never leak into training. A held-out validation manifest is split off with a
seeded shuffle.
"""

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_EXCLUDES = ["data/eval-3000/manifest.jsonl", "data/vqav2-pilot/subset.jsonl",
                    "data/textvqa-pilot/subset.jsonl"]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def excluded_hashes(manifests):
    hashes, paths = set(), set()
    for manifest in manifests:
        manifest = Path(manifest)
        if not manifest.exists():
            continue
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            image = (manifest.parent / row["image_path"]).resolve()
            paths.add(str(image))
            sha = (row.get("source") or {}).get("image_sha256")
            if sha is None and image.exists():
                sha = sha256_file(image)
            if sha:
                hashes.add(sha)
    return hashes, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id-prefix", default="train")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--exclude-manifests", nargs="*", default=DEFAULT_EXCLUDES)
    args = parser.parse_args()

    hashes, paths = excluded_hashes(args.exclude_manifests)
    files = sorted(p for p in args.image_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit:
        files = files[: args.limit]
    rows, refused = [], []
    for path in files:
        sha = sha256_file(path)
        if sha in hashes or str(path.resolve()) in paths:
            refused.append(str(path))
            continue
        try:
            with Image.open(path) as image:
                width, height = image.size
        except Exception as error:  # unreadable file: skip, record
            refused.append(f"{path} ({type(error).__name__})")
            continue
        rows.append({
            "id": f"{args.id_prefix}-{sha[:16]}",
            "image_path": str(path.resolve().relative_to(args.output.parent.resolve()))
            if path.resolve().is_relative_to(args.output.parent.resolve()) else str(path.resolve()),
            "source": {"image_sha256": sha, "image_width": width, "image_height": height,
                       "original_path": str(path.resolve())},
        })
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("Duplicate images (same SHA-256) in the directory; deduplicate first")

    random.Random(args.seed).shuffle(rows)
    n_val = int(round(len(rows) * args.val_fraction))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    val_path = args.output.with_name("val_" + args.output.name)
    with args.output.open("w") as f:
        for row in train_rows:
            f.write(json.dumps(row) + "\n")
    with val_path.open("w") as f:
        for row in val_rows:
            f.write(json.dumps(row) + "\n")
    metadata = {
        "purpose": "Entropic Student training images (local files only; nothing downloaded)",
        "created_at": datetime.now(timezone.utc).isoformat(), "image_dir": str(args.image_dir.resolve()),
        "train": len(train_rows), "val": len(val_rows), "refused_as_evaluation_images": refused,
        "excluded_manifests": args.exclude_manifests, "seed": args.seed, "val_fraction": args.val_fraction,
    }
    (args.output.parent / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{len(train_rows)} train, {len(val_rows)} val, {len(refused)} refused -> {args.output}, {val_path}")


if __name__ == "__main__":
    main()
