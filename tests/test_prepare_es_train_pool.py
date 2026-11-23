import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_es_train_pool.py"
spec = importlib.util.spec_from_file_location("prepare_es_train_pool", SCRIPT)
pool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool)


def candidates(n, source="coco_train2014"):
    return [{"id": f"{source}-{i}", "image_id": i, "source": source, "file": f"{i}.jpg",
             "url": f"http://example/{i}", "split": "train"} for i in range(n)]


def fake_download(overrides):
    def download(candidate):
        record = {**candidate, "status": "downloaded", "image_sha256": f"sha{candidate['image_id']}",
                  "pixel_sha256": f"pix{candidate['image_id']}"}
        record.update(overrides.get(candidate["image_id"], {}))
        return record
    return download


def empty():
    return {"ids": set(), "sha256": set(), "pixel_sha256": set()}


def test_rejected_candidates_are_replaced_until_the_exact_count():
    overrides = {
        1: {"image_sha256": "sha0"},            # byte-identical duplicate of candidate 0
        2: {"pixel_sha256": "pix0"},            # same pixels, different encoding
        3: {"status": "download_failed"},
        4: {"status": "unreadable"},
        6: {"image_sha256": "eval-sha"},        # an evaluation image by file hash
        7: {"pixel_sha256": "eval-pix"},        # an evaluation image by decoded pixels
    }
    excluded = {"ids": {"5"}, "sha256": {"eval-sha"}, "pixel_sha256": {"eval-pix"}}
    accepted, rejected = pool.select_unique(candidates(20), fake_download(overrides), 5, excluded, empty(),
                                            chunk=3, workers=2)
    assert [r["image_id"] for r in accepted] == [0, 8, 9, 10, 11]
    assert {r["image_id"]: r["rejected"] for r in rejected} == {
        1: "duplicate", 2: "duplicate", 3: "download_failed", 4: "unreadable", 5: "excluded_id",
        6: "excluded_evaluation_image", 7: "excluded_evaluation_image"}


def test_selection_is_reproducible_and_never_fetches_more_than_needed():
    fetched = []

    def download(candidate):
        fetched.append(candidate["image_id"])
        return fake_download({})(candidate)

    first, _ = pool.select_unique(candidates(50), download, 7, empty(), empty(), chunk=4, workers=3)
    assert sorted(fetched) == list(range(7))
    second, _ = pool.select_unique(candidates(50), fake_download({}), 7, empty(), empty(), chunk=64, workers=1)
    assert [r["id"] for r in first] == [r["id"] for r in second]


def test_duplicates_across_sources_share_the_seen_set():
    seen = empty()
    pool.select_unique(candidates(3, "coco_train2014"), fake_download({}), 3, empty(), seen)
    accepted, rejected = pool.select_unique(candidates(6, "textvqa_train"), fake_download({}), 2, empty(), seen)
    assert [r["image_id"] for r in accepted] == [3, 4] and len(rejected) == 3


def test_running_out_of_candidates_is_an_error():
    with pytest.raises(RuntimeError):
        pool.select_unique(candidates(2), fake_download({}), 3, empty(), empty())


def test_split_and_interleave_are_balanced_and_disjoint():
    records = {s: candidates(10, s) for s in pool.SOURCES}
    splits = {s: pool.split_source(records[s], 2, seed=2026) for s in pool.SOURCES}
    assert splits == {s: pool.split_source(records[s], 2, seed=2026) for s in pool.SOURCES}
    for train, val in splits.values():
        assert len(train) == 8 and len(val) == 2
        assert not {r["id"] for r in train} & {r["id"] for r in val}
    rows = pool.interleave([splits[s][0] for s in pool.SOURCES])
    assert [r["source"] for r in rows[:4]] == list(pool.SOURCES) * 2 and len(rows) == 16
