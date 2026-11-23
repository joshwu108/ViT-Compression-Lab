"""Cache-only Stage 1 training on a synthetic feature cache (CPU, no Qwen model, no images decoded)."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from vit_compression_lab.entropic.checkpoint import load_checkpoint
from vit_compression_lab.entropic.qwen.data import (
    FeatureCache,
    ImageManifestDataset,
    PatchBudgetBatchSampler,
    collate_packed,
)

SCRIPT = Path(__file__).resolve().parents[1] / "encoders/Qwen3.8-vision-compression/scripts/es_train_stage1.py"
GRIDS = [(4, 4), (4, 8), (8, 8), (4, 4), (2, 4), (4, 8), (4, 4), (4, 12), (2, 4), (4, 4), (4, 8), (4, 4)]


@pytest.fixture
def workspace(tmp_path):
    """12 training + 6 validation examples; features and grid only. train-02 (64 patches) and train-07 (48) exceed the 40-patch budget."""
    torch.manual_seed(0)
    cache = FeatureCache(tmp_path / "cache")
    (tmp_path / "image.jpg").write_bytes(b"never decoded")

    def rows(prefix, grids):
        out = []
        for i, (h, w) in enumerate(grids):
            cache.save(f"{prefix}-{i:02d}", {"grid_thw": torch.tensor([[1, h, w]]),
                                             "merger": torch.randn(h * w // 4, 5120).to(torch.bfloat16)})
            out.append({"id": f"{prefix}-{i:02d}", "image_path": "image.jpg",
                        "source": {"image_width": 1, "image_height": 1}})
        return out

    def manifest(name, items):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in items))
        return str(tmp_path / name)

    train, val = rows("train", GRIDS), rows("val", GRIDS[:6])
    torch.save({"merger": {"mean": torch.zeros(5120), "std": torch.ones(5120), "count": 1}},
               tmp_path / "channel_stats.pt")
    config = {
        "name": "tiny",
        "encoder": {"hidden": 16, "latent_channels": 4}, "decoder": {"hidden": 16, "output": "normalized"},
        "distortion": {"type": "standardized_mse"},
        "training": {"max_steps": 6, "eval_every_steps": 2, "full_eval_every_steps": 4, "milestones": [],
                     "max_patches_per_batch": 40, "num_workers": 0, "log_every": 1},
        "data": {"train_manifests": [manifest("train.jsonl", train)], "val_manifests": [manifest("val.jsonl", val)],
                 "val_fast_manifests": [manifest("val_fast.jsonl", val[:2])], "train_eval_limit": 3,
                 "feature_cache": str(tmp_path / "cache"), "channel_stats": str(tmp_path / "channel_stats.pt")},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


def train(workspace, monkeypatch, run_name, *extra):
    spec = importlib.util.spec_from_file_location("es_train_stage1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def forbidden(*args, **kwargs):
        raise AssertionError("cache-only training must not load the teacher or the processor")

    monkeypatch.setattr(module, "load_vision_tower", forbidden)
    monkeypatch.setattr(module, "load_processor", forbidden)
    monkeypatch.setattr(module, "hardware_info", lambda *a: {})
    monkeypatch.setattr(module.sys, "argv", [
        str(SCRIPT), "--config", str(workspace / "config.json"), "--runs-dir", str(workspace / "runs"),
        "--run-name", run_name, "--device", "cpu", "--cache-only", "--coded-examples", "1",
        "--diagnostic-channel", "7", *extra])
    module.main()
    run = workspace / "runs" / run_name
    return run, [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]


def test_features_only_items_and_batches_have_no_pixels(workspace):
    rows = [{"id": f"train-{i:02d}"} for i in range(3)]
    dataset = ImageManifestDataset(rows, None, FeatureCache(workspace / "cache"), ["merger"], load_pixels=False)
    batch = collate_packed([dataset[0], dataset[1]])
    assert "pixel_values" not in batch and batch["grid_thw"].tolist() == [[1, 4, 4], [1, 4, 8]]
    assert batch["features"]["merger"].shape == (4 + 8, 5120)
    # the default still asks for pixels, so a pixel-less cache fails loudly rather than silently
    with pytest.raises(KeyError):
        ImageManifestDataset(rows, None, FeatureCache(workspace / "cache"), ["merger"])[0]


def test_oversize_example_gets_its_own_batch_instead_of_being_dropped():
    counts = [16, 64, 16, 8, 100, 16]
    kept = PatchBudgetBatchSampler(counts, 40, shuffle=False, drop_oversize=False)
    assert list(kept) == [[0], [1], [2, 3], [4], [5]] and kept.oversize == 2 and kept.dropped == 0
    assert sorted(i for batch in kept for i in batch) == list(range(6))
    dropped = PatchBudgetBatchSampler(counts, 40, shuffle=False)  # previous default behaviour is unchanged
    assert list(dropped) == [[0, 2, 3], [5]] and dropped.dropped == 2
    kept.start_batch = 3  # resume inside an epoch: skipped batches are never loaded
    assert list(kept) == [[4], [5]] and list(kept)[0] == [0]


def test_two_validation_cadences_and_best_checkpoint_from_full_validation_only(workspace, monkeypatch):
    run, records = train(workspace, monkeypatch, "cadence")
    evals = {r["step"]: r for r in records if r["phase"] == "eval"}
    assert sorted(evals) == [0, 2, 4, 6]
    assert [evals[s]["full_validation"] for s in (0, 2, 4, 6)] == [True, False, True, True]
    for step, record in evals.items():
        assert record["train"]["images"] == 3 and record["val_fast"]["images"] == 2
        assert ("val" in record) == record["full_validation"]
        assert record["val_fast"]["coded"]["roundtrip_exact"] and "ch7" in record["val_fast"]
    assert evals[4]["val"]["images"] == 6

    # best.pt: lowest FULL-validation loss among the saved full evaluations; latest.pt: the last step.
    full = {s: evals[s]["val"]["loss"] for s in (4, 6)}
    best = load_checkpoint(run / "checkpoints/best.pt")
    assert best["step"] == min(full, key=full.get) and best["extra"]["best"] == min(full.values())
    latest = load_checkpoint(run / "checkpoints/latest.pt")
    assert latest["step"] == 6 and latest["extra"]["best"] == min(full.values())
    assert sorted(p.name for p in (run / "checkpoints").iterdir()) == ["best.pt", "latest.pt"]

    # Every training image stays eligible: the two oversize images are single-example batches.
    info = json.loads((run / "run_info.json").read_text())
    assert info["teacher_loaded"] is False and info["cache_points"] == ["merger"]
    assert info["batching"]["oversize_single_example_batches"] == 2 and info["batching"]["dropped"] == 0
    assert info["val_fast_ids"] == ["val-00", "val-01"]
    steps = [r for r in records if r["phase"] == "train"]
    assert len(steps) == 6 and all(r["grad_norm"] > 0 and r["grad_norm/aux_quantiles"] > 0 for r in steps)
    done = records[-1]
    assert done["phase"] == "done" and done["images_seen"] == sum(r["batch_images"] for r in steps)
    assert done["tokens_seen"] == latest["extra"]["seen"]["tokens"] == sum(r["batch_tokens"] for r in steps)


def test_subset_check_never_replaces_the_best_checkpoint(workspace, monkeypatch):
    # subset-only check at step 1, full validation at 2 and at 3 (the final step)
    run, records = train(workspace, monkeypatch, "subset", "--set", "training.max_steps=3",
                   "--set", "training.eval_every_steps=1", "--set", "training.full_eval_every_steps=2")
    evals = {r["step"]: r for r in records if r["phase"] == "eval"}
    assert [evals[s]["full_validation"] for s in (1, 2, 3)] == [False, True, True]
    best = load_checkpoint(run / "checkpoints/best.pt")
    assert best["step"] in (2, 3)
    assert load_checkpoint(run / "checkpoints/latest.pt")["step"] == 3
    assert best["extra"]["best"] == min(evals[2]["val"]["loss"], evals[3]["val"]["loss"])


def test_resume_continues_exactly_like_an_uninterrupted_run(workspace, monkeypatch):
    cadence = ["--set", "training.eval_every_steps=3", "--set", "training.full_eval_every_steps=3"]
    straight, _ = train(workspace, monkeypatch, "straight", *cadence)
    train(workspace, monkeypatch, "resumed", *cadence, "--set", "training.max_steps=3")
    resumed, records = train(workspace, monkeypatch, "resumed", *cadence, "--resume")

    a = load_checkpoint(straight / "checkpoints/latest.pt")
    b = load_checkpoint(resumed / "checkpoints/latest.pt")
    assert a["step"] == b["step"] == 6 and a["epoch"] == b["epoch"]
    assert a["extra"]["batches_done_in_epoch"] == b["extra"]["batches_done_in_epoch"]
    assert {k: v for k, v in a["extra"]["seen"].items() if k != "seconds"} == {
        k: v for k, v in b["extra"]["seen"].items() if k != "seconds"}
    for key, value in a["state_dict"].items():
        assert torch.equal(value, b["state_dict"][key]), key
    for name in ("main", "aux"):
        sa, sb = a["optimizers"][name]["state"], b["optimizers"][name]["state"]
        assert all(torch.equal(sa[k]["exp_avg"], sb[k]["exp_avg"]) for k in sa)
    resume = next(r for r in records if r["phase"] == "resume")
    assert resume["step"] == 3 and resume["restored"]["rng_restored"]
    assert [r["step"] for r in records if r["phase"] == "train"] == [1, 2, 3, 4, 5, 6]
