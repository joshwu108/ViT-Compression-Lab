import torch

from vit_compression_lab.entropic import grid as G


def test_windows_round_trip():
    x = torch.randn(8, 3)
    w = G.patches_to_windows(x)
    assert w.shape == (2, 12)
    assert torch.equal(G.windows_to_patches(w, 3), x)


def test_patch_grid_round_trip_and_adjacency():
    grid_thw = torch.tensor([[1, 4, 6]])  # 24 patches, 6 merged tokens
    x = torch.arange(24.0)[:, None]
    g = G.tokens_to_grid(x, 1, 4, 6, "patch")
    assert g.shape == (1, 1, 4, 6)
    # The first four packed patches form the top-left 2x2 window in raster space.
    assert sorted(g[0, 0, :2, :2].reshape(-1).tolist()) == [0.0, 1.0, 2.0, 3.0]
    assert torch.equal(G.grid_to_tokens(g, "patch"), x)


def test_token_grid_round_trip_multiple_images():
    grid_thw = torch.tensor([[1, 4, 6], [2, 2, 4]])
    counts = G.token_counts(grid_thw)
    assert counts == [6, 4]
    x = torch.randn(sum(counts), 5)
    grids = G.packed_to_grids(x, grid_thw, "token")
    assert grids[0].shape == (1, 5, 2, 3) and grids[1].shape == (2, 5, 1, 2)
    assert torch.equal(G.grids_to_packed(grids, "token"), x)
