"""Manifests, preprocessing, packed batching and the on-disk teacher feature cache.

Manifests are the repository's JSONL format (``id``, ``image_path`` relative to
the manifest, optional ``question``/``expected_answer``/...). Training manifests
need only ``id`` and ``image_path``; ``scripts/prepare_image_manifest.py``
writes them from a local directory.

Batches are packed sequences, exactly like the model's own input: pixel values
and grids of several images are concatenated along the token axis and
``grid_thw`` has one row per image.
"""

import json
import math
import random
from pathlib import Path

import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import Dataset, Sampler
from transformers import AutoProcessor


def load_processor(checkpoint, revision):
    return AutoProcessor.from_pretrained(checkpoint, revision=revision, local_files_only=True)


def read_manifest(paths, limit=None, ids=None, require_question=False):
    rows = []
    for path in paths:
        path = Path(path)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" not in row or "image_path" not in row:
                raise ValueError(f"{path}: every row needs 'id' and 'image_path'")
            row["image_path"] = str((path.parent / row["image_path"]).resolve())
            if require_question and not row.get("question"):
                raise ValueError(f"{row['id']}: missing question")
            rows.append(row)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate ids across manifests")
    if ids:
        wanted = set(ids)
        rows = [r for r in rows if r["id"] in wanted]
    if limit is not None:
        rows = rows[:limit]
    for row in rows:
        if not Path(row["image_path"]).is_file():
            raise FileNotFoundError(row["image_path"])
    return rows


def preprocess_image(processor, image_path):
    """(pixel_values [N, 1536] float32, grid_thw [1, 3] int64) via the checkpoint's image processor."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    out = processor.image_processor(images=image, return_tensors="pt")
    return out["pixel_values"], out["image_grid_thw"]


def preprocess_video_frames(processor, frames, fps=2.0):
    """(pixel_values_videos, video_grid_thw) for a list of PIL frames.

    UNVERIFIED: the Qwen3VL video processor's keyword interface in transformers
    5.21 was not exercised in this repository. Frame decoding (a video backend)
    is the caller's responsibility; this function never reads a video file.
    """
    out = processor.video_processor(videos=[frames], fps=fps, return_tensors="pt")
    return out["pixel_values_videos"], out["video_grid_thw"]


def estimate_patch_grid(width, height, processor):
    """Qwen2-VL smart_resize, used only to budget batches before preprocessing."""
    ip = processor.image_processor
    factor = ip.patch_size * ip.merge_size
    size = ip.size
    min_pixels, max_pixels = size["shortest_edge"], size["longest_edge"]
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar // ip.patch_size, w_bar // ip.patch_size


def estimate_patch_count(row, processor):
    source = row.get("source") or {}
    if "image_width" in source and "image_height" in source:
        width, height = source["image_width"], source["image_height"]
    else:
        with Image.open(row["image_path"]) as image:
            width, height = image.size
    h, w = estimate_patch_grid(width, height, processor)
    return h * w


class FeatureCache:
    """One safetensors file per example: grid_thw, teacher activations, optionally pixel_values.

    Files are written to a temporary name and renamed, so an interrupted
    extraction never leaves a partial entry. Each file's metadata records what
    it was computed from (see es_extract_teacher_features.py).
    """

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, example_id):
        return self.root / f"{example_id}.safetensors"

    def has(self, example_id):
        return self.path(example_id).exists()

    def save(self, example_id, tensors, metadata=None):
        tensors = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
        meta = {k: json.dumps(v) for k, v in (metadata or {}).items()}
        tmp = self.path(example_id).with_suffix(".tmp")
        save_file(tensors, str(tmp), metadata=meta)
        tmp.replace(self.path(example_id))

    def load(self, example_id, keys=None):
        """Read only the requested tensors (all of them when ``keys`` is None)."""
        with safe_open(str(self.path(example_id)), framework="pt") as f:
            missing = [k for k in (keys or []) if k not in f.keys()]
            if missing:
                raise KeyError(f"{example_id}: cache lacks {missing}")
            return {k: f.get_tensor(k) for k in (keys if keys is not None else f.keys())}

    def describe(self, example_id):
        """(metadata dict, {tensor name: shape}) from the file header; no tensor data is read."""
        with safe_open(str(self.path(example_id)), framework="pt") as f:
            metadata = {k: json.loads(v) for k, v in (f.metadata() or {}).items()}
            return metadata, {k: list(f.get_slice(k).get_shape()) for k in f.keys()}

    def patch_count(self, example_id):
        """Patches of a cached example from its stored grid (the geometry actually extracted)."""
        t, h, w = self.load(example_id, ["grid_thw"])["grid_thw"][0].tolist()
        return t * h * w


class ImageManifestDataset(Dataset):
    """Returns grid_thw plus pixel_values and/or cached teacher activations.

    With ``load_pixels=False`` and ``cache_points`` set, items come from the
    feature cache alone (no pixel tensors, no image preprocessing, and
    ``processor`` may be None): the cache-only path of Option C.
    """

    def __init__(self, rows, processor, cache=None, cache_points=(), load_pixels=True):
        self.rows = rows
        self.processor = processor
        self.cache = cache
        self.cache_points = list(cache_points)
        self.load_pixels = load_pixels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        item = {"id": row["id"]}
        if self.cache is not None and self.cache.has(row["id"]):
            keys = ["grid_thw"] + self.cache_points + (["pixel_values"] if self.load_pixels else [])
            tensors = self.cache.load(row["id"], keys)
            item["grid_thw"] = tensors["grid_thw"]
            if self.load_pixels:
                item["pixel_values"] = tensors["pixel_values"]
            item["features"] = {p: tensors[p] for p in self.cache_points}
        else:
            if self.cache_points:
                raise FileNotFoundError(f"{row['id']} is not in the feature cache")
            item["pixel_values"], item["grid_thw"] = preprocess_image(self.processor, row["image_path"])
            item["features"] = {}
        return item


def collate_packed(items):
    """Packed batch; ``pixel_values`` is present only when every item carries it."""
    batch = {
        "ids": [it["id"] for it in items],
        "grid_thw": torch.cat([it["grid_thw"] for it in items], dim=0),
        "features": {},
    }
    if all("pixel_values" in it for it in items):
        batch["pixel_values"] = torch.cat([it["pixel_values"] for it in items], dim=0)
    for point in items[0]["features"]:
        batch["features"][point] = torch.cat([it["features"][point] for it in items], dim=0)
    return batch


class PatchBudgetBatchSampler(Sampler):
    """Greedy packing of examples until the summed patch count reaches the budget.

    An example larger than the whole budget is dropped when ``drop_oversize`` is
    true (counted in ``dropped``); otherwise it forms a batch of its own, so
    every example stays in the epoch (``oversize`` counts those).
    """

    def __init__(self, patch_counts, max_patches, shuffle=True, seed=0, drop_oversize=True):
        self.patch_counts = list(patch_counts)
        self.max_patches = int(max_patches)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.start_batch = 0  # skip this many batches of the next epoch (resume inside an epoch)
        self.indices = [i for i, n in enumerate(self.patch_counts) if not (drop_oversize and n > self.max_patches)]
        self.dropped = len(self.patch_counts) - len(self.indices)
        self.oversize = sum(1 for i in self.indices if self.patch_counts[i] > self.max_patches)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _batches(self):
        order = list(self.indices)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        batch, total = [], 0
        for i in order:
            n = self.patch_counts[i]
            if batch and total + n > self.max_patches:
                yield batch
                batch, total = [], 0
            batch.append(i)
            total += n
            if total > self.max_patches:  # a single oversize example: never packed with others
                yield batch
                batch, total = [], 0
        if batch:
            yield batch

    def __iter__(self):
        batches = list(self._batches())[self.start_batch:]
        self.start_batch = 0
        return iter(batches)

    def __len__(self):
        return sum(1 for _ in self._batches())
