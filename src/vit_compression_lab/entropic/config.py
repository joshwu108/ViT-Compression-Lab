"""Configuration for the Entropic Student pipeline (plain JSON, no framework).

A ``SplitSpec`` names where the student encoder takes its input and which
teacher activation the decoder reconstructs:

    encoder_input  : "pixels" | "block_0" .. "block_27" | "merger"
    decoder_target : "block_0" .. "block_27" | "merger"

``block_0`` is the stem output (patch embedding + resampled position table),
``block_k`` the output of the k-th ViT block (1-based, as in the model config),
and ``merger`` the ``[tokens, 5120]`` features injected into the LLM. The
target depth must not be shallower than the input depth. The three options of
the plan are:

    A  encoder_input=pixels|block_0, decoder_target=block_m
    B  encoder_input=block_k,        decoder_target=block_k (or block_m, m>k)
    C  encoder_input=merger,         decoder_target=merger
"""

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

PIXELS = "pixels"
MERGER = "merger"


def point_depth(point, depth):
    """Integer depth of a split point: pixels=-1, block_k=k, merger=depth+1."""
    if point == PIXELS:
        return -1
    if point == MERGER:
        return depth + 1
    if point.startswith("block_"):
        k = int(point[len("block_") :])
        if not 0 <= k <= depth:
            raise ValueError(f"{point}: block index must be in 0..{depth}")
        return k
    raise ValueError(f"Unknown split point {point!r}")


def point_level(point):
    """'patch' for per-patch activations, 'token' for merged LLM tokens."""
    return "token" if point == MERGER else "patch"


@dataclass(frozen=True)
class VisionDims:
    """The few vision-config numbers the student needs. Defaults are the
    verified Qwen3.8-27B values (vision_config in the cached config.json)."""

    depth: int = 27
    hidden_size: int = 1152
    out_hidden_size: int = 5120
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2

    @property
    def pixel_dim(self):
        return self.in_channels * self.temporal_patch_size * self.patch_size**2

    @property
    def merge_unit(self):
        return self.spatial_merge_size**2

    def dim_at(self, point):
        if point == PIXELS:
            return self.pixel_dim
        if point == MERGER:
            return self.out_hidden_size
        point_depth(point, self.depth)
        return self.hidden_size

    @classmethod
    def from_vision_config(cls, vision_config):
        return cls(
            depth=vision_config.depth,
            hidden_size=vision_config.hidden_size,
            out_hidden_size=vision_config.out_hidden_size,
            in_channels=vision_config.in_channels,
            patch_size=vision_config.patch_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            spatial_merge_size=vision_config.spatial_merge_size,
        )


@dataclass(frozen=True)
class SplitSpec:
    encoder_input: str
    decoder_target: str

    def validate(self, dims):
        d_in = point_depth(self.encoder_input, dims.depth)
        d_tgt = point_depth(self.decoder_target, dims.depth)
        if self.decoder_target == PIXELS:
            raise ValueError("decoder_target cannot be 'pixels'")
        if d_tgt < d_in:
            raise ValueError(
                f"decoder_target {self.decoder_target} is shallower than "
                f"encoder_input {self.encoder_input}"
            )
        if self.encoder_input == MERGER and self.decoder_target != MERGER:
            raise ValueError("A merger-level input can only target the merger")
        return self

    def input_depth(self, dims):
        return point_depth(self.encoder_input, dims.depth)

    def target_depth(self, dims):
        return point_depth(self.decoder_target, dims.depth)

    @property
    def input_level(self):
        return point_level(self.encoder_input)

    @property
    def target_level(self):
        return point_level(self.decoder_target)

    @property
    def option(self):
        if self.encoder_input == MERGER:
            return "C"
        if self.encoder_input in (PIXELS, "block_0"):
            return "A"
        return "B"

    def as_dict(self):
        return {"encoder_input": self.encoder_input, "decoder_target": self.decoder_target}


DEFAULTS = {
    "name": None,
    "split": {"encoder_input": "merger", "decoder_target": "merger"},
    "stem": {"type": "none"},  # only for encoder_input == "pixels": {"type": "linear", "dim": 384}
    "body": {"type": "none"},  # optional student body: {"type": "attention", "depth": 2, "heads": 6}
    "encoder": {"type": "token_mlp", "hidden": 1024, "latent_channels": 256, "gdn": True},
    "decoder": {"type": "mirror", "hidden": 1024, "gdn": True},
    "entropy": {"type": "factorized", "filters": [3, 3, 3], "init_scale": 10.0, "tail_mass": 1e-9},
    "quantization": {"train": "noise", "eval": "round"},  # train: noise | ste
    "distortion": {"type": "mse", "reduction": "sum_per_image", "cosine_weight": 0.0},
    "deep_supervision": [],  # e.g. [{"point": "merger", "weight": 1.0, "type": "standardized_mse"}]
    "rate": {"beta": 0.08, "reduction": "sum_per_image"},
    "training": {
        "epochs": 10, "lr": 1e-3, "aux_lr": 1e-3, "weight_decay": 0.0,
        "milestones": [5, 8], "gamma": 0.1, "grad_clip": 1.0,
        "max_patches_per_batch": 16384, "seed": 0, "log_every": 20,
        "eval_every_epoch": True, "num_workers": 4,
    },
    "data": {
        "train_manifests": [], "val_manifests": [], "train_limit": None, "val_limit": 64,
        "feature_cache": None, "channel_stats": None,
    },
    "teacher": {
        "checkpoint": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "dtype": "bfloat16", "attn_implementation": "sdpa",
    },
    "stage2": {
        "alpha": 0.0, "tau": 1.0, "lr": 1e-4, "epochs": 1, "train_server_suffix": False,
        "answer_source": "teacher_greedy", "max_new_tokens": 32, "grad_checkpointing": True,
    },
}


def _merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


@dataclass
class StudentConfig:
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data, dims=None):
        unknown = set(data) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        merged = _merge(DEFAULTS, data)
        config = cls(merged)
        config.split.validate(dims or VisionDims())
        if config.raw["name"] is None:
            raise ValueError("Config needs a 'name' (used for the run directory)")
        return config

    @classmethod
    def load(cls, path, dims=None):
        return cls.from_dict(json.loads(Path(path).read_text()), dims)

    @property
    def name(self):
        return self.raw["name"]

    @property
    def split(self):
        return SplitSpec(**self.raw["split"])

    def __getitem__(self, key):
        return self.raw[key]

    def get(self, key, default=None):
        return self.raw.get(key, default)

    def to_json(self):
        return json.dumps(self.raw, indent=2, sort_keys=True) + "\n"
