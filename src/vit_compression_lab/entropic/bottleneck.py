"""Student encoders, decoders and the EntropicStudent wrapper.

Encoders map the client-side activation ``u`` ``[N_in, d_in]`` to a latent
``z`` ``[L, C_z]``; decoders map the dequantised latent back to the decoder
target ``[N_out, d_out]``. All modules take ``grid_thw`` so that ragged packed
sequences can be reshaped per image. Three families are provided:

    token_mlp   per-token MLP (latent per input token)
    window_mlp  the 4 patches of a 2x2 merge window are concatenated first, so
                the latent grid equals the LLM token grid (patch-level input only)
    grid_conv   the paper's conv + GDN stack on the per-frame token grid, with
                ``num_down`` stride-2 stages (latent grid downsampled 2^num_down)

The decoder type "mirror" (default) picks the family of the encoder.
"""

import math

import torch
from torch import nn

from . import grid as G
from .config import MERGER, PIXELS, StudentConfig, VisionDims
from .entropy_model import FactorizedEntropyModel
from .gdn import GDN

LEVEL_SCALE = {("patch", "patch"): 1.0, ("token", "token"): 1.0,
               ("patch", "token"): 0.5, ("token", "patch"): 2.0}


def _mlp(d_in, hidden, d_out, gdn, inverse, pre_norm):
    layers = []
    if pre_norm:
        layers.append(nn.LayerNorm(d_in))
    layers.append(nn.Linear(d_in, hidden))
    layers.append(GDN(hidden, inverse=inverse) if gdn else nn.GELU())
    layers.append(nn.Linear(hidden, d_out))
    return nn.Sequential(*layers)


# ------------------------------------------------------------------ MLP family

class TokenMLPEncoder(nn.Module):
    family = "token_mlp"

    def __init__(self, d_in, hidden, latent, level, gdn=True, pre_norm=True):
        super().__init__()
        self.latent_level = level
        self.net = _mlp(d_in, hidden, latent, gdn, inverse=False, pre_norm=pre_norm)

    def forward(self, u, grid_thw):
        return self.net(u)

    def latent_counts(self, grid_thw):
        return G.counts_for_level(grid_thw, self.latent_level)


class TokenMLPDecoder(nn.Module):
    def __init__(self, latent, hidden, d_out, latent_level, out_level, gdn=True):
        super().__init__()
        if latent_level != out_level:
            raise ValueError("token_mlp decoder needs latent and target at the same level")
        self.net = _mlp(latent, hidden, d_out, gdn, inverse=True, pre_norm=False)

    def forward(self, z_hat, grid_thw):
        return self.net(z_hat)


class WindowMLPEncoder(nn.Module):
    """Patch-level input; one latent per 2x2 merge window (= one LLM token)."""

    family = "window_mlp"

    def __init__(self, d_in, hidden, latent, level, gdn=True, pre_norm=True, merge=G.MERGE):
        super().__init__()
        if level != "patch":
            raise ValueError("window_mlp encoder needs patch-level input")
        self.merge = merge
        self.latent_level = "token"
        self.net = _mlp(merge * merge * d_in, hidden, latent, gdn, inverse=False, pre_norm=pre_norm)

    def forward(self, u, grid_thw):
        return self.net(G.patches_to_windows(u, self.merge))

    def latent_counts(self, grid_thw):
        return G.token_counts(grid_thw, self.merge)


class WindowMLPDecoder(nn.Module):
    """Token-level latent -> patch-level target (un-windowed) or token-level target."""

    def __init__(self, latent, hidden, d_out, latent_level, out_level, gdn=True, merge=G.MERGE):
        super().__init__()
        if latent_level != "token":
            raise ValueError("window_mlp decoder expects a token-level latent")
        self.merge = merge
        self.out_level = out_level
        self.d_out = d_out
        width = d_out if out_level == "token" else merge * merge * d_out
        self.net = _mlp(latent, hidden, width, gdn, inverse=True, pre_norm=False)

    def forward(self, z_hat, grid_thw):
        y = self.net(z_hat)
        if self.out_level == "token":
            return y
        return G.windows_to_patches(y, self.d_out, self.merge)


# ------------------------------------------------------------- conv family

class _ChannelLastGDN(nn.Module):
    def __init__(self, channels, inverse):
        super().__init__()
        self.gdn = GDN(channels, inverse=inverse)

    def forward(self, x):  # [t, C, H, W]
        return self.gdn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def _pad_to_multiple(g, multiple):
    t, c, h, w = g.shape
    ph = (-h) % multiple
    pw = (-w) % multiple
    if ph or pw:
        g = nn.functional.pad(g, (0, pw, 0, ph))
    return g


class GridConvEncoder(nn.Module):
    """Conv5x5 (+GDN) stack with ``num_down`` stride-2 stages, per frame."""

    family = "grid_conv"

    def __init__(self, d_in, hidden, latent, level, num_down=1, kernel=5, gdn=True, num_layers=None):
        super().__init__()
        self.level = level
        self.latent_level = level  # same token layout family; counts come from latent_counts()
        self.num_down = int(num_down)
        num_layers = max(self.num_down, 1) if num_layers is None else int(num_layers)
        layers = []
        c_in = d_in
        for i in range(num_layers):
            stride = 2 if i < self.num_down else 1
            c_out = latent if i == num_layers - 1 else hidden
            layers.append(nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=kernel // 2))
            if i < num_layers - 1:
                layers.append(_ChannelLastGDN(c_out, inverse=False) if gdn else nn.GELU())
            c_in = c_out
        self.net = nn.Sequential(*layers)

    def latent_shape(self, t, h, w):
        s = 2**self.num_down
        return t, math.ceil(h / s), math.ceil(w / s)

    def latent_counts(self, grid_thw):
        return [t * hh * ww for t, hh, ww in
                (self.latent_shape(*shape) for shape in G.frame_shapes(grid_thw, self.level))]

    def forward(self, u, grid_thw):
        latents = []
        for g in G.packed_to_grids(u, grid_thw, self.level):
            g = _pad_to_multiple(g, 2**self.num_down)
            y = self.net(g)  # [t, latent, Hl, Wl]
            latents.append(y.permute(0, 2, 3, 1).reshape(-1, y.shape[1]))
        return torch.cat(latents, dim=0)


class GridConvDecoder(nn.Module):
    def __init__(self, latent, hidden, d_out, latent_level, out_level, num_down=1, kernel=5,
                 gdn=True, encoder=None):
        super().__init__()
        if encoder is None:
            raise ValueError("grid_conv decoder needs its encoder for the latent geometry")
        self.encoder = [encoder]  # list: not registered as a submodule
        self.in_level = encoder.level
        self.out_level = out_level
        scale = 2**num_down * LEVEL_SCALE[(self.in_level, out_level)]
        if scale < 1 or scale != int(scale) or (int(scale) & (int(scale) - 1)):
            raise ValueError("grid_conv: output grid must be latent grid times a power of two")
        self.num_up = int(math.log2(scale))
        layers = []
        c_in = latent
        steps = max(self.num_up, 1)
        for i in range(steps):
            c_out = d_out if i == steps - 1 else hidden
            if i < self.num_up:
                layers.append(nn.ConvTranspose2d(c_in, c_out, kernel, stride=2,
                                                 padding=kernel // 2, output_padding=1))
            else:
                layers.append(nn.Conv2d(c_in, c_out, kernel, padding=kernel // 2))
            if i < steps - 1:
                layers.append(_ChannelLastGDN(c_out, inverse=True) if gdn else nn.GELU())
            c_in = c_out
        self.net = nn.Sequential(*layers)

    def forward(self, z_hat, grid_thw):
        encoder = self.encoder[0]
        in_shapes = G.frame_shapes(grid_thw, self.in_level)
        out_shapes = G.frame_shapes(grid_thw, self.out_level)
        counts = encoder.latent_counts(grid_thw)
        outputs = []
        for part, in_shape, (t, h_out, w_out) in zip(G.split_packed(z_hat, counts), in_shapes, out_shapes):
            t_l, h_l, w_l = encoder.latent_shape(*in_shape)
            g = part.reshape(t_l, h_l, w_l, -1).permute(0, 3, 1, 2)
            y = self.net(g)[:, :, :h_out, :w_out]
            outputs.append(G.grid_to_tokens(y, self.out_level))
        return torch.cat(outputs, dim=0)


# ------------------------------------------------------ optional student parts

def patch_coordinates(grid_thw, merge=G.MERGE):
    """Normalised (row, col) in [0, 1] of every patch, in Qwen's packed order."""
    coords = []
    for t, h, w in grid_thw.tolist():
        rows = torch.arange(h).view(h // merge, merge, 1, 1).expand(h // merge, merge, w // merge, merge)
        cols = torch.arange(w).view(1, 1, w // merge, merge).expand(h // merge, merge, w // merge, merge)
        # packed order is (hb, wb, i, j); rows/cols above are (hb, i, wb, j)
        rows = rows.permute(0, 2, 1, 3).reshape(-1).float() / max(h - 1, 1)
        cols = cols.permute(0, 2, 1, 3).reshape(-1).float() / max(w - 1, 1)
        coords.append(torch.stack([rows, cols], dim=-1).repeat(t, 1))
    return torch.cat(coords, dim=0)


def sinusoidal_2d(coords, dim):
    half = dim // 2
    freqs = torch.exp(torch.arange(0, half, 2, dtype=torch.float32) * (-math.log(10000.0) / half))
    out = []
    for axis in range(2):
        angle = coords[:, axis : axis + 1] * 1000.0 * freqs[None, :]
        out.extend([torch.sin(angle), torch.cos(angle)])
    emb = torch.cat(out, dim=-1)
    if emb.shape[1] < dim:
        emb = nn.functional.pad(emb, (0, dim - emb.shape[1]))
    return emb[:, :dim]


class PixelStem(nn.Module):
    """Student replacement for the teacher stem: linear patch embed + 2-D sinusoidal position."""

    def __init__(self, pixel_dim, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(pixel_dim, dim)

    def forward(self, pixels, grid_thw):
        x = self.proj(pixels)
        pos = sinusoidal_2d(patch_coordinates(grid_thw), self.dim).to(x.device, x.dtype)
        return x + pos


class AttentionBody(nn.Module):
    """A few pre-LN self-attention blocks run per image on the packed sequence."""

    def __init__(self, dim, depth=2, heads=6, mlp_ratio=4.0, level="patch"):
        super().__init__()
        self.level = level
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(dim),
                "attn": nn.MultiheadAttention(dim, heads, batch_first=True),
                "norm2": nn.LayerNorm(dim),
                "mlp": nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                     nn.Linear(int(dim * mlp_ratio), dim)),
            }))

    def forward(self, x, grid_thw):
        outputs = []
        for part in G.split_packed(x, G.counts_for_level(grid_thw, self.level)):
            h = part.unsqueeze(0)
            for blk in self.blocks:
                n = blk["norm1"](h)
                h = h + blk["attn"](n, n, n, need_weights=False)[0]
                h = h + blk["mlp"](blk["norm2"](h))
            outputs.append(h.squeeze(0))
        return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------- wrapper

ENCODERS = {"token_mlp": TokenMLPEncoder, "window_mlp": WindowMLPEncoder, "grid_conv": GridConvEncoder}
DECODERS = {"token_mlp": TokenMLPDecoder, "window_mlp": WindowMLPDecoder, "grid_conv": GridConvDecoder}


def build_encoder(cfg, d_in, level):
    kind = cfg["type"]
    if kind not in ENCODERS:
        raise ValueError(f"Unknown encoder type {kind!r}; choose from {sorted(ENCODERS)}")
    kwargs = {k: v for k, v in cfg.items() if k not in ("type", "latent_channels")}
    return ENCODERS[kind](d_in, latent=cfg["latent_channels"], level=level, **kwargs)


def build_decoder(cfg, encoder, latent, d_out, out_level):
    kind = encoder.family if cfg.get("type", "mirror") == "mirror" else cfg["type"]
    if kind not in DECODERS:
        raise ValueError(f"Unknown decoder type {kind!r}; choose from {sorted(DECODERS)}")
    kwargs = {k: v for k, v in cfg.items() if k != "type"}
    if kind == "grid_conv":
        kwargs.setdefault("num_down", getattr(encoder, "num_down", 1))
        kwargs["encoder"] = encoder
    return DECODERS[kind](latent, d_out=d_out, latent_level=encoder.latent_level,
                          out_level=out_level, **kwargs)


class EntropicStudent(nn.Module):
    """stem? -> body? -> encoder -> entropy model -> decoder."""

    def __init__(self, config: StudentConfig, dims: VisionDims | None = None):
        super().__init__()
        dims = dims or VisionDims()
        self.config = config
        self.dims = dims
        split = config.split.validate(dims)
        self.split = split
        d_in = dims.dim_at(split.encoder_input)
        d_out = dims.dim_at(split.decoder_target)
        in_level, out_level = split.input_level, split.target_level

        stem_cfg = config["stem"]
        if split.encoder_input == PIXELS:
            if stem_cfg["type"] != "linear":
                raise ValueError("encoder_input 'pixels' needs stem.type == 'linear'")
            self.stem = PixelStem(d_in, stem_cfg["dim"])
            d_in = stem_cfg["dim"]
        else:
            if stem_cfg["type"] != "none":
                raise ValueError("A stem is only used with encoder_input 'pixels'")
            self.stem = None

        body_cfg = config["body"]
        if body_cfg["type"] == "attention":
            self.body = AttentionBody(d_in, body_cfg.get("depth", 2), body_cfg.get("heads", 6), level=in_level)
        elif body_cfg["type"] == "none":
            self.body = None
        else:
            raise ValueError(f"Unknown body type {body_cfg['type']!r}")

        self.encoder = build_encoder(config["encoder"], d_in, in_level)
        latent = config["encoder"]["latent_channels"]
        ent = config["entropy"]
        if ent["type"] != "factorized":
            raise ValueError("Only the factorized entropy model is implemented")
        self.entropy = FactorizedEntropyModel(latent, filters=ent["filters"],
                                              init_scale=ent["init_scale"], tail_mass=ent["tail_mass"])
        self.decoder = build_decoder(config["decoder"], self.encoder, latent, d_out, out_level)
        self.latent_channels = latent
        self.d_in, self.d_out = d_in, d_out

    # --------------------------------------------------------- pieces

    def encode(self, u, grid_thw):
        if self.stem is not None:
            u = self.stem(u, grid_thw)
        if self.body is not None:
            u = self.body(u, grid_thw)
        return self.encoder(u, grid_thw)

    def decode(self, z_hat, grid_thw):
        return self.decoder(z_hat, grid_thw)

    def latent_counts(self, grid_thw):
        return self.encoder.latent_counts(grid_thw)

    def forward(self, u, grid_thw, quant_mode=None):
        """Training/eval forward. quant_mode: noise | ste | round (default by mode)."""
        if quant_mode is None:
            quant_mode = self.config["quantization"]["train" if self.training else "eval"]
        z = self.encode(u, grid_thw)
        z_hat, likelihood = self.entropy(z, mode=quant_mode)
        recon = self.decode(z_hat, grid_thw)
        return {"z": z, "z_hat": z_hat, "likelihood": likelihood, "recon": recon,
                "latent_counts": self.latent_counts(grid_thw)}

    # --------------------------------------------------------- coding

    @torch.no_grad()
    def compress(self, u, grid_thw):
        """Returns (strings per image, latent counts, ideal bits per image)."""
        self.entropy.update()
        z = self.encode(u, grid_thw)
        counts = self.latent_counts(grid_thw)
        strings, ideal = [], []
        for part in G.split_packed(z, counts):
            strings.append(self.entropy.compress(part.float()))
            ideal.append(self.entropy.ideal_bits(part.float()))
        return strings, counts, ideal

    @torch.no_grad()
    def decompress(self, strings, grid_thw, device=None):
        counts = self.latent_counts(grid_thw)
        parts = [self.entropy.decompress(s, n, device=device) for s, n in zip(strings, counts)]
        z_hat = torch.cat(parts, dim=0)
        return self.decode(z_hat, grid_thw)

    # --------------------------------------------------------- parameters

    def aux_parameters(self):
        return [self.entropy.quantiles]

    def main_parameters(self):
        aux = {id(p) for p in self.aux_parameters()}
        return [p for p in self.parameters() if id(p) not in aux]

    def freeze_client_side(self):
        """Stage 2: encoder side and entropy model fixed so the bitrate cannot change."""
        for module in (self.stem, self.body, self.encoder, self.entropy):
            if module is not None:
                for p in module.parameters():
                    p.requires_grad_(False)

    def parameter_counts(self):
        def count(m):
            return 0 if m is None else sum(p.numel() for p in m.parameters())
        return {"stem": count(self.stem), "body": count(self.body), "encoder": count(self.encoder),
                "entropy": count(self.entropy), "decoder": count(self.decoder),
                "client_side": count(self.stem) + count(self.body) + count(self.encoder)}
