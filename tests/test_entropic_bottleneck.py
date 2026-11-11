import torch

from vit_compression_lab.entropic.bottleneck import EntropicStudent, patch_coordinates
from vit_compression_lab.entropic.config import StudentConfig, VisionDims

DIMS = VisionDims(depth=4, hidden_size=16, out_hidden_size=24, patch_size=2, temporal_patch_size=2)
GRID = torch.tensor([[1, 4, 6], [1, 2, 2]])  # 24 + 4 patches, 6 + 1 tokens


def make(split, encoder, extra=None):
    data = {"name": "t", "split": split, "encoder": encoder,
            "decoder": {"type": "mirror", "hidden": 8, "gdn": True}}
    data.update(extra or {})
    return EntropicStudent(StudentConfig.from_dict(data, DIMS), DIMS)


def test_option_c_token_mlp_shapes():
    student = make({"encoder_input": "merger", "decoder_target": "merger"},
                   {"type": "token_mlp", "hidden": 8, "latent_channels": 4})
    u = torch.randn(7, 24)
    out = student(u, GRID)
    assert out["z"].shape == (7, 4) and out["recon"].shape == (7, 24)
    assert out["latent_counts"] == [6, 1]
    assert out["likelihood"].shape == (7, 4)


def test_option_b_window_mlp_shapes():
    student = make({"encoder_input": "block_2", "decoder_target": "block_2"},
                   {"type": "window_mlp", "hidden": 8, "latent_channels": 4})
    u = torch.randn(28, 16)
    out = student(u, GRID)
    assert out["z"].shape == (7, 4) and out["recon"].shape == (28, 16)


def test_option_a_pixels_with_stem_and_body():
    student = make({"encoder_input": "pixels", "decoder_target": "block_1"},
                   {"type": "window_mlp", "hidden": 8, "latent_channels": 4},
                   {"stem": {"type": "linear", "dim": 12}, "body": {"type": "attention", "depth": 1, "heads": 2}})
    u = torch.randn(28, DIMS.pixel_dim)
    out = student(u, GRID)
    assert out["recon"].shape == (28, 16)
    counts = student.parameter_counts()
    assert counts["stem"] > 0 and counts["body"] > 0


def test_grid_conv_round_trip_shapes_with_odd_grid():
    student = make({"encoder_input": "merger", "decoder_target": "merger"},
                   {"type": "grid_conv", "hidden": 8, "latent_channels": 4, "num_down": 1, "kernel": 3})
    grid = torch.tensor([[1, 6, 10]])  # 3 x 5 merged tokens: odd width needs padding
    u = torch.randn(15, 24)
    out = student(u, grid)
    assert out["z"].shape == (2 * 3, 4)  # ceil(3/2) * ceil(5/2)
    assert out["recon"].shape == (15, 24)


def test_compress_decompress_matches_forward_in_round_mode():
    torch.manual_seed(0)
    student = make({"encoder_input": "merger", "decoder_target": "merger"},
                   {"type": "token_mlp", "hidden": 8, "latent_channels": 4}).eval()
    u = torch.randn(7, 24)
    strings, counts, ideal = student.compress(u, GRID)
    assert len(strings) == 2 and counts == [6, 1] and all(b >= 0 for b in ideal)
    recon = student.decompress(strings, GRID)
    expected = student(u, GRID, quant_mode="round")["recon"]
    assert torch.allclose(recon, expected, atol=1e-5)


def test_freeze_client_side_leaves_decoder_trainable():
    student = make({"encoder_input": "merger", "decoder_target": "merger"},
                   {"type": "token_mlp", "hidden": 8, "latent_channels": 4})
    student.freeze_client_side()
    assert all(not p.requires_grad for p in student.encoder.parameters())
    assert all(not p.requires_grad for p in student.entropy.parameters())
    assert all(p.requires_grad for p in student.decoder.parameters())


def test_patch_coordinates_follow_merge_block_order():
    coords = patch_coordinates(torch.tensor([[1, 4, 4]]))
    assert coords.shape == (16, 2)
    # Packed patches 0..3 are the top-left window: rows {0,1}, cols {0,1} (normalised by 3).
    assert set(coords[:4, 0].tolist()) == {0.0, 1 / 3} and set(coords[:4, 1].tolist()) == {0.0, 1 / 3}
