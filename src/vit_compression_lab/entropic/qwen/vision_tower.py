"""Load only the Qwen3.5 vision tower and split it at any block.

The 27 B checkpoint keeps all 333 ``model.visual.*`` tensors in one shard
(``model-00001-of-00018.safetensors``), so the teacher ViT (~0.44 B params,
~0.9 GB in BF16) can be loaded without the language model. Everything is read
from the local Hugging Face cache (``local_files_only=True``); nothing is
downloaded.

``TeacherSplit`` re-implements ``Qwen3_5VisionModel.forward`` as explicit
stages so a client can stop after block ``k`` and a server can resume from a
reconstructed activation at block ``m``:

    depth -1          pixel_values              [N, 1536]
    depth 0           stem = patch_embed + pos  [N, 1152]
    depth k (1..27)   output of block k         [N, 1152]
    depth 28          merger output             [N/4, 5120]

All position information (bilinear position-table taps, 2-D RoPE, cu_seqlens)
is a function of ``grid_thw`` only and is recomputed by ``geometry()`` on
either side. ``verify()`` must be run once per split depth before results are
trusted; it compares the staged path with the unmodified forward pass.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
from transformers.utils.hub import cached_file
from transformers.vision_utils import (
    get_vision_attention_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

from ..config import MERGER, PIXELS, VisionDims, point_depth

VISUAL_PREFIX = "model.visual."


def snapshot_dir(checkpoint, revision):
    """Directory of the cached snapshot; raises if it is not cached."""
    config_path = cached_file(checkpoint, "config.json", revision=revision, local_files_only=True)
    return Path(config_path).parent


def load_config(checkpoint, revision):
    return AutoConfig.from_pretrained(checkpoint, revision=revision, local_files_only=True)


def vision_state_dict(checkpoint, revision):
    snap = snapshot_dir(checkpoint, revision)
    weight_map = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    by_shard = {}
    for key, shard in weight_map.items():
        if key.startswith(VISUAL_PREFIX):
            by_shard.setdefault(shard, []).append(key)
    if not by_shard:
        raise RuntimeError(f"No {VISUAL_PREFIX}* tensors in the checkpoint index")
    state = {}
    for shard, keys in by_shard.items():
        with safe_open(str(snap / shard), framework="pt", device="cpu") as f:
            for key in keys:
                state[key[len(VISUAL_PREFIX) :]] = f.get_tensor(key)
    return state


def load_vision_tower(checkpoint, revision, device="cpu", dtype=torch.bfloat16,
                      attn_implementation="sdpa"):
    """Frozen ``Qwen3_5VisionModel`` with the checkpoint's weights"""

    # extract the Qwen3_5 config set attention implementation
    config = load_config(checkpoint, revision)
    vision_config = config.vision_config
    vision_config._attn_implementation = attn_implementation

    # construct vision network from extracted settings
    model = Qwen3_5VisionModel(vision_config)
    missing, unexpected = model.load_state_dict(vision_state_dict(checkpoint, revision), strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected vision tensors: {unexpected[:5]}")
    # Non-persistent rotary buffers are legitimately absent from the shard.
    real_missing = [k for k in missing if not k.endswith("inv_freq")]
    if real_missing:
        raise RuntimeError(f"Missing vision tensors: {real_missing[:5]}")

    # set model device, prceision, and evaluation mode
    model = model.to(device=device, dtype=dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@dataclass
class Geometry:
    interp_indices: torch.Tensor
    interp_weights: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int | None
    position_embeddings: tuple


class TeacherSplit:
    def __init__(self, visual):
        self.visual = visual
        self.depth = len(visual.blocks)
        self.merger_depth = self.depth + 1
        self.dims = VisionDims.from_vision_config(visual.config)

    @property
    def device(self):
        return self.visual.patch_embed.proj.weight.device

    @property
    def dtype(self):
        return self.visual.patch_embed.proj.weight.dtype

    def depth_of(self, point):
        return point_depth(point, self.depth)

    @torch.no_grad()
    def geometry(self, grid_thw):
        visual = self.visual
        grid_thw = grid_thw.to(self.device)

        # function from transformer library
        interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side=visual.num_grid_per_side,
            mode=visual.interpolation_mode,
            align_corners=visual.interpolation_align_corners,
            spatial_merge_size=visual.config.spatial_merge_size,
        )
        # get the positional information
        position_ids = get_vision_position_ids(grid_thw, visual.spatial_merge_size)
        # get the cumulative and maximum lengths for attention sequences
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, visual.config)
        anchor = torch.empty(0, device=self.device)
        # include cosine and sine tensors for spatial position within attention
        position_embeddings = visual.rotary_pos_emb(anchor, position_ids)
        return Geometry(interp_indices, interp_weights, cu_seqlens, max_seqlen, position_embeddings)

    def stem(self, pixel_values, geo):
        # attain the patch embeddings + positional information
        hidden = self.visual.patch_embed(pixel_values.to(self.device))
        # use the object's interp_indices table to determine the positions
        pos = (self.visual.pos_embed(geo.interp_indices) * geo.interp_weights[:, :, None]).sum(1)
        return hidden + pos.to(hidden.dtype)

    def run_blocks(self, hidden, start, stop, geo):
        """Apply blocks[start:stop]"""
        for blk in self.visual.blocks[start:stop]:
            hidden = blk(hidden, cu_seqlens=geo.cu_seqlens, max_seqlen=geo.max_seqlen,
                         position_embeddings=geo.position_embeddings)
        return hidden

    def prefix(self, pixel_values, grid_thw, depth, geo=None):
        """Request input from start to depth(input itself, merger output, block output)"""
        if depth == -1:
            return pixel_values
        geo = geo or self.geometry(grid_thw)
        hidden = self.stem(pixel_values, geo)
        if depth == self.merger_depth:
            hidden = self.run_blocks(hidden, 0, self.depth, geo)
            return self.visual.merger(hidden)
        return self.run_blocks(hidden, 0, depth, geo)

    def suffix(self, hidden, grid_thw, from_depth, geo=None):
        """Resume after ``from_depth`` and return merged features [N/4, 5120]."""
        if from_depth == self.merger_depth:
            return hidden
        if from_depth < 0:
            raise ValueError("suffix() cannot start from pixels; use prefix()")
        geo = geo or self.geometry(grid_thw)
        hidden = self.run_blocks(hidden.to(self.dtype), from_depth, self.depth, geo)
        return self.visual.merger(hidden)

    @torch.no_grad()
    def collect(self, pixel_values, grid_thw, points):
        """{point: activation} for several split points in one pass."""
        depths = sorted({self.depth_of(p) for p in points})
        geo = self.geometry(grid_thw)
        out = {}
        current_depth = -1
        hidden = pixel_values
        for depth in depths:
            if depth == -1:
                out[depth] = pixel_values
                continue
            if current_depth == -1:
                hidden = self.stem(pixel_values, geo)
                current_depth = 0
            if depth == self.merger_depth:
                hidden = self.run_blocks(hidden, current_depth, self.depth, geo)
                out[depth] = self.visual.merger(hidden)
                current_depth = self.depth
            else:
                hidden = self.run_blocks(hidden, current_depth, depth, geo)
                current_depth = depth
                out[depth] = hidden
        names = {}
        for p in points:
            names[p] = out[self.depth_of(p)]
        return names

    @torch.no_grad()
    def full(self, pixel_values, grid_thw):
        """The unmodified forward pass (what inject_eval.py's encode_image uses)."""
        return self.visual(pixel_values.to(self.device), grid_thw=grid_thw.to(self.device)).pooler_output

    @torch.no_grad()
    def verify(self, pixel_values, grid_thw, depth):
        """Staged path vs the unmodified forward. Returns diagnostics; run before trusting a split."""
        staged = self.suffix(self.prefix(pixel_values, grid_thw, depth), grid_thw, depth)
        reference = self.full(pixel_values, grid_thw)
        diff = (staged.float() - reference.float()).abs()
        return {"depth": depth, "max_abs_diff": diff.max().item(), "mean_abs_diff": diff.mean().item(),
                "bit_exact": torch.equal(staged, reference), "shape": list(reference.shape)}


def point_names(depth):
    return [PIXELS] + [f"block_{k}" for k in range(depth + 1)] + [MERGER]
