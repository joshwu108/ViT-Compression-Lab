import importlib.util
import json
from pathlib import Path

import pytest
import torch

from vit_compression_lab.entropic.qwen.data import FeatureCache

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "encoders/Qwen3.8-vision-compression/scripts/es_extract_teacher_features.py"
)


@pytest.mark.parametrize(
    "stats_only,existing_cache", [(False, False), (True, False), (True, True)]
)
def test_extraction_statistics_with_and_without_cached_features(
    tmp_path, monkeypatch, stats_only, existing_cache
):
    spec = importlib.util.spec_from_file_location("extract_teacher_features", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [
        {"id": "first", "image_path": "first.jpg"},
        {"id": "second", "image_path": "second.jpg"},
    ]
    cache = FeatureCache(tmp_path)
    if existing_cache:
        cache.save("first", {"merger": torch.full((4, 3), 100.0)})

    class FakeSplit:
        dtype = torch.float32

        def __init__(self, visual):
            pass

        def depth_of(self, point):
            return 1

        def collect(self, pixels, grid, points):
            return {point: pixels.clone() for point in points}

    def preprocess(processor, path):
        value = 1.0 if path == "first.jpg" else 3.0
        return torch.full((4, 3), value), torch.tensor([[1, 4, 4]])  # 16 patches -> 4 merged tokens

    monkeypatch.setattr(module, "read_manifest", lambda *a, **kw: rows)
    monkeypatch.setattr(module, "load_processor", lambda *a, **kw: None)
    monkeypatch.setattr(module, "load_vision_tower", lambda *a, **kw: None)
    monkeypatch.setattr(module, "TeacherSplit", FakeSplit)
    monkeypatch.setattr(module, "preprocess_image", preprocess)
    monkeypatch.setattr(module, "hardware_info", lambda *a: {})
    monkeypatch.setattr(module, "software_info", lambda: {})
    args = [str(SCRIPT), "train.jsonl", "--cache-dir", str(tmp_path), "--device", "cpu"]
    if stats_only:
        args.append("--stats-only")
    monkeypatch.setattr(module.sys, "argv", args)
    module.main()

    stats = torch.load(tmp_path / "channel_stats.pt", weights_only=True)["merger"]
    assert stats["count"] == 8
    assert torch.equal(stats["mean"], torch.full((3,), 2.0))
    assert torch.equal(stats["std"], torch.ones(3))
    info = json.loads((tmp_path / "extraction_info.json").read_text())
    assert info["stats_only"] is stats_only
    files = list(tmp_path.glob("*.safetensors"))
    assert len(files) == (int(existing_cache) if stats_only else 2)
    if existing_cache:
        assert torch.equal(cache.load("first")["merger"], torch.full((4, 3), 100.0))


# ---------------------------------------------------------------------------
# Resumable caching, provenance, and training-only statistics from the cache.

VALUES = {"a": 1.0, "b": 3.0, "c": 5.0, "d": 7.0, "v": 100.0}


def load_script(monkeypatch, calls):
    spec = importlib.util.spec_from_file_location("extract_teacher_features", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeSplit:
        dtype = torch.bfloat16

        def __init__(self, visual):
            pass

        def depth_of(self, point):
            return 1

        def collect(self, pixels, grid, points):
            return {point: pixels.clone() for point in points}

    def preprocess(processor, path):
        name = Path(path).stem
        calls.append(name)
        if name == "boom":
            raise OSError("simulated crash")
        return torch.full((4, 3), VALUES[name]), torch.tensor([[1, 4, 4]])

    monkeypatch.setattr(module, "load_processor", lambda *a, **kw: None)
    monkeypatch.setattr(module, "load_vision_tower", lambda *a, **kw: calls.append("load_teacher"))
    monkeypatch.setattr(module, "TeacherSplit", FakeSplit)
    monkeypatch.setattr(module, "preprocess_image", preprocess)
    monkeypatch.setattr(module, "hardware_info", lambda *a: {})
    monkeypatch.setattr(module, "software_info", lambda: {})
    return module


def write_manifest(path, names, sha="sha"):
    for name in names:
        (path.parent / f"{name}.jpg").write_bytes(name.encode())
    path.write_text("".join(json.dumps({"id": n, "image_path": f"{n}.jpg",
                                        "source": {"image_sha256": f"{sha}-{n}"}}) + "\n" for n in names))
    return path


def run(module, monkeypatch, *args):
    monkeypatch.setattr(module.sys, "argv", [str(SCRIPT), *map(str, args), "--device", "cpu", "--no-pixels"])
    module.main()


def test_interrupted_extraction_resumes_and_statistics_cover_every_training_id(tmp_path, monkeypatch):
    calls = []
    module = load_script(monkeypatch, calls)
    cache_dir, stats = tmp_path / "cache", tmp_path / "stats" / "channel_stats.pt"
    train = write_manifest(tmp_path / "train.jsonl", ["a", "b", "boom", "d"])
    val = write_manifest(tmp_path / "val.jsonl", ["v"])
    common = ("--cache-dir", cache_dir, "--stats-manifests", train, "--stats-output", stats)

    # First run: one image fails. The others are cached, but no (incomplete) statistics are written.
    with pytest.raises(SystemExit):
        run(module, monkeypatch, train, val, *common)
    assert sorted(p.stem for p in cache_dir.glob("*.safetensors")) == ["a", "b", "d", "v"]
    assert not stats.exists() and not list(cache_dir.glob("*.tmp"))

    # Resume after the cause is fixed: only the missing image is extracted ...
    (tmp_path / "boom.jpg").rename(tmp_path / "c.jpg")
    train.write_text(train.read_text().replace("boom", "c"))
    calls.clear()
    run(module, monkeypatch, train, val, *common)
    assert calls == ["load_teacher", "c"]
    # ... and the statistics cover all four training ids (previously cached ones included), never "v".
    result = torch.load(stats, weights_only=True)["merger"]
    assert result["count"] == 16
    assert torch.allclose(result["mean"], torch.full((3,), 4.0))
    assert torch.allclose(result["std"], torch.full((3,), 5.0).sqrt())
    sidecar = json.loads(stats.with_suffix(".json").read_text())
    assert sidecar["images"] == 4 and sidecar["points"]["merger"]["tokens"] == 16
    assert sidecar["manifests"][0]["sha256"] == module.sha256_file(train)

    # The cache holds features and grid only, with provenance; no pixels.
    cache = FeatureCache(cache_dir)
    metadata, shapes = cache.describe("a")
    assert set(shapes) == {"grid_thw", "merger"} and metadata["image_sha256"] == "sha-a"
    assert cache.patch_count("a") == 16

    # A fully cached rerun loads no teacher and reproduces the same statistics.
    calls.clear()
    run(module, monkeypatch, train, val, *common)
    assert calls == [] and torch.equal(torch.load(stats, weights_only=True)["merger"]["mean"], result["mean"])

    # Extracting validation images alone leaves the training statistics untouched.
    before = stats.read_bytes()
    run(module, monkeypatch, val, "--cache-dir", cache_dir, "--no-stats")
    assert stats.read_bytes() == before
    assert not (cache_dir / "channel_stats.pt").exists()


def test_stale_cache_entries_are_refused_unless_replaced(tmp_path, monkeypatch):
    calls = []
    module = load_script(monkeypatch, calls)
    cache_dir = tmp_path / "cache"
    run(module, monkeypatch, write_manifest(tmp_path / "train.jsonl", ["a", "b"]), "--cache-dir", cache_dir)

    # Same id, different image content: the cached features belong to another image.
    changed = write_manifest(tmp_path / "train.jsonl", ["a", "b"], sha="other")
    with pytest.raises(SystemExit, match="2 cached entries"):
        run(module, monkeypatch, changed, "--cache-dir", cache_dir)
    with pytest.raises(SystemExit, match="2 cached entries"):  # another model revision
        run(module, monkeypatch, write_manifest(tmp_path / "train.jsonl", ["a", "b"]), "--cache-dir", cache_dir,
            "--revision", "different")
    calls.clear()
    changed = write_manifest(tmp_path / "train.jsonl", ["a", "b"], sha="other")
    run(module, monkeypatch, changed, "--cache-dir", cache_dir, "--replace-stale")
    assert calls == ["load_teacher", "a", "b"]
    assert FeatureCache(cache_dir).describe("a")[0]["image_sha256"] == "other-a"


def test_cache_statistics_reject_missing_or_non_finite_entries(tmp_path, monkeypatch):
    module = load_script(monkeypatch, [])
    cache = FeatureCache(tmp_path)
    grid = torch.tensor([[1, 4, 4]])
    cache.save("good", {"grid_thw": grid, "merger": torch.ones(4, 3)})
    cache.save("nan", {"grid_thw": grid, "merger": torch.full((4, 3), float("nan"))})
    cache.save("short", {"grid_thw": grid, "merger": torch.ones(3, 3)})
    assert module.cache_channel_stats(cache, ["good"], ["merger"])[1:] == (1, {"merger": 4})
    for bad in ("nan", "short", "absent"):
        with pytest.raises((ValueError, FileNotFoundError, OSError)):
            module.cache_channel_stats(cache, ["good", bad], ["merger"])
