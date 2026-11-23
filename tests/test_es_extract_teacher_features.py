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
        return torch.full((4, 3), value), torch.tensor([[1, 2, 2]])

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
