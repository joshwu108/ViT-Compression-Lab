"""Option B training from cached features with a (fake) frozen teacher suffix: CPU, no Qwen, no images."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from vit_compression_lab.entropic.bottleneck import EntropicStudent
from vit_compression_lab.entropic.config import StudentConfig, VisionDims, point_depth
from vit_compression_lab.entropic.qwen.data import FeatureCache

SCRIPT = Path(__file__).resolve().parents[1] / "encoders/Qwen3.8-vision-compression/scripts/es_train_stage1.py"
GRIDS = [(4, 4), (4, 8), (8, 8), (4, 4), (2, 4), (4, 8)]


class FakeSplit:
    """Stands in for TeacherSplit: one frozen linear layer plays blocks 9-27 and the merger."""

    def __init__(self, visual):
        self.visual, self.dims, self.dtype = visual, VisionDims(), torch.float32

    def depth_of(self, point):
        return point_depth(point, self.dims.depth)

    def geometry(self, grid_thw):
        return None

    def suffix(self, hidden, grid_thw, from_depth, geo=None):
        assert from_depth == 8
        return self.visual(hidden.reshape(-1, 4, hidden.shape[-1]).mean(1))


def fake_teacher():
    torch.manual_seed(1)
    visual = torch.nn.Linear(1152, 5120)
    for p in visual.parameters():
        p.requires_grad_(False)
    return visual


def load_trainer(monkeypatch):
    spec = importlib.util.spec_from_file_location("es_train_stage1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def forbidden(*args, **kwargs):
        raise AssertionError("cached-feature training must not load the image processor")

    monkeypatch.setattr(module, "load_processor", forbidden)
    monkeypatch.setattr(module, "load_vision_tower", lambda *a, **k: fake_teacher())
    monkeypatch.setattr(module, "TeacherSplit", FakeSplit)
    monkeypatch.setattr(module, "hardware_info", lambda *a: {})
    return module


@pytest.fixture
def workspace(tmp_path):
    torch.manual_seed(0)
    cache = FeatureCache(tmp_path / "cache")
    (tmp_path / "image.jpg").write_bytes(b"never decoded")
    teacher = fake_teacher()

    def rows(prefix, grids):
        out = []
        for i, (h, w) in enumerate(grids):
            block = torch.randn(h * w, 1152)
            merger = teacher(block.reshape(-1, 4, 1152).mean(1)).detach()
            cache.save(f"{prefix}-{i:02d}", {"grid_thw": torch.tensor([[1, h, w]]),
                                             "block_8": block.to(torch.bfloat16),
                                             "merger": merger.to(torch.bfloat16)})
            out.append({"id": f"{prefix}-{i:02d}", "image_path": "image.jpg"})
        return out

    def manifest(name, items):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in items))
        return str(tmp_path / name)

    train, val = rows("train", GRIDS), rows("val", GRIDS[:3])
    torch.save({"block_8": {"mean": torch.zeros(1152), "std": torch.ones(1152), "count": 1},
                "merger": {"mean": torch.zeros(5120), "std": torch.ones(5120), "count": 1}},
               tmp_path / "channel_stats.pt")
    config = {
        "name": "tiny-b",
        "split": {"encoder_input": "block_8", "decoder_target": "block_8"},
        "encoder": {"type": "window_mlp", "hidden": 16, "latent_channels": 4},
        "decoder": {"hidden": 16, "output": "normalized"},
        "distortion": {"type": "standardized_mse"},
        "deep_supervision": [{"point": "merger", "weight": 1.0, "type": "standardized_mse"}],
        "training": {"max_steps": 2, "eval_every_steps": 1, "full_eval_every_steps": 2, "milestones": [],
                     "max_patches_per_batch": 40, "num_workers": 0, "log_every": 1},
        "data": {"train_manifests": [manifest("train.jsonl", train)], "val_manifests": [manifest("val.jsonl", val)],
                 "val_fast_manifests": [manifest("val_fast.jsonl", val[:1])], "train_eval_limit": 2,
                 "feature_cache": str(tmp_path / "cache"), "channel_stats": str(tmp_path / "channel_stats.pt")},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


def run(module, monkeypatch, workspace, *flags):
    monkeypatch.setattr(module.sys, "argv", [
        str(SCRIPT), "--config", str(workspace / "config.json"), "--runs-dir", str(workspace / "runs"),
        "--device", "cpu", "--coded-examples", "1", *flags])
    module.main()
    path = workspace / "runs" / "tiny-b"
    return path, [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]


def test_cached_features_keep_the_frozen_suffix_and_default_diagnostic_channel_is_safe(workspace, monkeypatch):
    # No --diagnostic-channel: the default 3994 does not exist in a 1,152-channel block target.
    path, records = run(load_trainer(monkeypatch), monkeypatch, workspace, "--cached-features")
    info = json.loads((path / "run_info.json").read_text())
    assert info["teacher_loaded"] and info["cache_points"] == ["block_8", "merger"]
    assert info["teacher_trainable_parameters"] == 0
    assert info["batching"]["oversize_single_example_batches"] == 1 and info["batching"]["dropped"] == 0

    evals = [r for r in records if r["phase"] == "eval"]
    assert [r["step"] for r in evals] == [0, 1, 2] and [r["full_validation"] for r in evals] == [True, False, True]
    for record in evals:
        fast = record["val_fast"]
        assert "ch3994" not in fast and fast["coded"]["roundtrip_exact"]
        # the block target and the propagated merger features are reported separately
        assert fast["tokens"] == 16 and fast["deep"]["merger"]["tokens"] == 4
        assert fast["deep"]["merger"]["standardized_mse"] > 0
    steps = [r for r in records if r["phase"] == "train"]
    assert len(steps) == 2
    assert all(r["grad_norm/encoder"] > 0 and r["grad_norm/decoder"] > 0 and r["grad_norm/aux_quantiles"] > 0
               and "deep/merger/mse" in r for r in steps)
    assert records[-1]["phase"] == "done" and records[-1]["step"] == 2 and records[-1]["teacher_unchanged"] is True


def test_merger_error_alone_reaches_the_student_through_the_frozen_suffix(workspace, monkeypatch):
    module = load_trainer(monkeypatch)
    config = StudentConfig.load(workspace / "config.json")
    student = EntropicStudent(config)
    split = FakeSplit(fake_teacher())
    cache = FeatureCache(workspace / "cache")
    tensors = cache.load("train-01", ["grid_thw", "block_8", "merger"])
    batch = {"ids": ["train-01"], "grid_thw": tensors["grid_thw"],
             "features": {p: tensors[p] for p in ("block_8", "merger")}}
    stats = {"block_8": torch.ones(1152), "merger": torch.ones(5120)}
    points = module.needed_points(config)
    _, _, out, acts, _ = module.run_batch(student, split, batch, config, points, stats, torch.device("cpu"), True)

    propagated = out["deep"]["merger"]
    assert propagated.shape == acts["merger"].shape and propagated.requires_grad
    assert not acts["merger"].requires_grad and not acts["block_8"].requires_grad  # teacher targets are constants
    merger_only = (propagated - acts["merger"].float()).pow(2).sum()
    grads = torch.autograd.grad(merger_only, [student.encoder.net[-1].weight, student.decoder.net[-1].weight])
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    assert all(p.grad is None and not p.requires_grad for p in split.visual.parameters())


def test_cache_only_still_refuses_deep_supervision(workspace, monkeypatch):
    with pytest.raises(SystemExit):
        run(load_trainer(monkeypatch), monkeypatch, workspace, "--cache-only")
    with pytest.raises(SystemExit):
        run(load_trainer(monkeypatch), monkeypatch, workspace, "--cache-only", "--cached-features")
