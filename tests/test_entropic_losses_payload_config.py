import pytest
import torch

from vit_compression_lab.entropic import payload
from vit_compression_lab.entropic.bottleneck import EntropicStudent
from vit_compression_lab.entropic.config import SplitSpec, StudentConfig, VisionDims
from vit_compression_lab.entropic.losses import kd_token_loss, stage1_loss, stage2_loss

DIMS = VisionDims(depth=4, hidden_size=16, out_hidden_size=24, patch_size=2)


def test_split_validation():
    SplitSpec("block_2", "block_3").validate(DIMS)
    with pytest.raises(ValueError):
        SplitSpec("block_3", "block_2").validate(DIMS)
    with pytest.raises(ValueError):
        SplitSpec("merger", "block_4").validate(DIMS)
    with pytest.raises(ValueError):
        SplitSpec("block_9", "merger").validate(DIMS)
    assert SplitSpec("merger", "merger").option == "C"
    assert SplitSpec("block_2", "block_2").option == "B"
    assert SplitSpec("pixels", "block_1").option == "A"


def test_config_rejects_unknown_keys_and_fills_defaults():
    with pytest.raises(ValueError):
        StudentConfig.from_dict({"name": "x", "bogus": 1}, DIMS)
    config = StudentConfig.from_dict({"name": "x"}, DIMS)
    assert config["rate"]["beta"] == 0.08 and config["encoder"]["type"] == "token_mlp"


def test_stage1_loss_runs_and_is_differentiable():
    config = StudentConfig.from_dict({"name": "x", "encoder": {"latent_channels": 4, "hidden": 8},
                                      "decoder": {"hidden": 8}}, DIMS)
    student = EntropicStudent(config, DIMS)
    grid = torch.tensor([[1, 4, 6]])
    target = torch.randn(6, 24)
    out = student(target + 0.1 * torch.randn_like(target), grid)
    out["deep"] = {}
    loss, log = stage1_loss(out, {"merger": target}, {"token": [6], "patch": [24]}, 6, config)
    loss.backward()
    assert torch.isfinite(loss) and "bits_per_visual_token" in log


def test_kd_loss_zero_for_identical_logits():
    logits = torch.randn(5, 11)
    mask = torch.ones(5, dtype=torch.bool)
    assert kd_token_loss(logits, logits, mask).abs().item() < 1e-6
    total, log = stage2_loss(logits, logits, torch.zeros(5, dtype=torch.long), mask, alpha=0.5)
    assert "ce" in log and torch.isfinite(total)


def test_payload_round_trip_and_accounting():
    header = payload.make_header(SplitSpec("merger", "merger"), torch.tensor([[1, 4, 6]]), [6], 4)
    strings = [b"\x01\x02\x03", b""]
    blob = payload.pack(header, strings)
    header2, strings2 = payload.unpack(blob)
    assert header2 == header and strings2 == strings
    acct = payload.accounting(blob, strings)
    assert acct["string_bytes"] == 3 and acct["total_bytes"] == len(blob)
    with pytest.raises(ValueError):
        payload.unpack(blob + b"x")
