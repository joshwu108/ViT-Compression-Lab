"""Reshaping between Qwen's packed token sequences and per-frame grids.

The image processor emits patches in merge-block order
``(t, h/m, w/m, m, m)`` with ``m = spatial_merge_size = 2``: every run of four
consecutive patches is one 2x2 window and becomes one merged LLM token. The
merged tokens of one image are therefore in raster order ``(t, h/m, w/m)``.

All functions take ``grid_thw`` as an integer tensor ``[num_images, 3]``.
"""

import torch

MERGE = 2


def patch_counts(grid_thw):
    return (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()


def token_counts(grid_thw, merge=MERGE):
    return [n // (merge * merge) for n in patch_counts(grid_thw)]


def counts_for_level(grid_thw, level, merge=MERGE):
    if level == "patch":
        return patch_counts(grid_thw)
    if level == "token":
        return token_counts(grid_thw, merge)
    raise ValueError(f"Unknown level {level!r}")


def split_packed(x, counts):
    if sum(counts) != x.shape[0]:
        raise ValueError(f"Packed length {x.shape[0]} != sum of counts {sum(counts)}")
    return list(torch.split(x, counts, dim=0))


def patches_to_windows(x, merge=MERGE):
    """[N, d] -> [N / merge^2, merge^2 * d] (consecutive patches share a window)."""
    unit = merge * merge
    if x.shape[0] % unit:
        raise ValueError(f"{x.shape[0]} patches is not a multiple of {unit}")
    return x.reshape(x.shape[0] // unit, unit * x.shape[1])


def windows_to_patches(x, dim, merge=MERGE):
    unit = merge * merge
    if x.shape[1] != unit * dim:
        raise ValueError(f"Window width {x.shape[1]} != {unit} * {dim}")
    return x.reshape(x.shape[0] * unit, dim)


def frame_shapes(grid_thw, level, merge=MERGE):
    """(t, H, W) per image at the given level."""
    shapes = []
    for t, h, w in grid_thw.tolist():
        if level == "patch":
            shapes.append((t, h, w))
        else:
            shapes.append((t, h // merge, w // merge))
    return shapes


def tokens_to_grid(x, t, h, w, level, merge=MERGE):
    """One image's packed tokens [t*H*W, C] -> [t, C, H, W] in raster order."""
    c = x.shape[1]
    if level == "token":
        return x.reshape(t, h, w, c).permute(0, 3, 1, 2)
    # patch level: undo the merge-block order (t, h/m, w/m, m, m)
    x = x.reshape(t, h // merge, w // merge, merge, merge, c)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(t, h, w, c)
    return x.permute(0, 3, 1, 2)


def grid_to_tokens(g, level, merge=MERGE):
    """Inverse of tokens_to_grid: [t, C, H, W] -> [t*H*W, C] in Qwen order."""
    t, c, h, w = g.shape
    x = g.permute(0, 2, 3, 1)  # [t, H, W, C]
    if level == "token":
        return x.reshape(t * h * w, c)
    x = x.reshape(t, h // merge, merge, w // merge, merge, c)
    x = x.permute(0, 1, 3, 2, 4, 5)
    return x.reshape(t * h * w, c)


def packed_to_grids(x, grid_thw, level, merge=MERGE):
    counts = counts_for_level(grid_thw, level, merge)
    parts = split_packed(x, counts)
    return [
        tokens_to_grid(part, t, h, w, level, merge)
        for part, (t, h, w) in zip(parts, frame_shapes(grid_thw, level, merge))
    ]


def grids_to_packed(grids, level, merge=MERGE):
    return torch.cat([grid_to_tokens(g, level, merge) for g in grids], dim=0)
