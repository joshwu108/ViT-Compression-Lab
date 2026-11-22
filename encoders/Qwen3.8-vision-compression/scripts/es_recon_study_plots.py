"""Plots for the reconstruction study. Needs only json + matplotlib (no torch), so it can
run under any Python that has matplotlib (the project .venv does not).

    python es_recon_study_plots.py curves OUT.png LABEL=RUN_DIR [LABEL=RUN_DIR ...]
    python es_recon_study_plots.py rate OUT.png DIAGNOSTICS.json [--skip NAME ...]
    python es_recon_study_plots.py channels OUT.png DIAGNOSTICS.json [--skip NAME ...]

curves    rounded evaluation of the fixed training subset (solid) and validation set
          (dashed) against optimizer updates, one colour per run
rate      bits per visual token vs. standardized fraction of variance unexplained (validation)
channels  channel 3994 vs. all other channels, per method (validation)
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "lines.linewidth": 1.8, "legend.frameon": False,
})


def evals(run_dir):
    lines = (Path(run_dir) / "metrics.jsonl").read_text().splitlines()
    return [r for r in map(json.loads, lines) if r["phase"] == "eval"]


def curves(args):
    panels = [
        ("Standardized MSE (1.0 = training-mean predictor)", lambda e: e["standardized_mse"], True),
        ("Raw relative L2 error", lambda e: e["relative_l2"], False),
        ("Coded bits per visual token (rANS)", lambda e: e.get("coded", {}).get("bits_per_visual_token"), False),
        ("Channel 3994 MSE", lambda e: e["ch3994"]["mse"], True),
        ("MSE over the other 5119 channels", lambda e: e["other_channels"]["mse"], True),
        ("Training objective (rounded)", lambda e: e["loss"], True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, (title, get, log) in zip(axes.flat, panels):
        for color, item in zip(COLORS, args.items):
            label, _, run_dir = item.rpartition("=")
            records = evals(run_dir)
            for split, style in (("train", "-"), ("val", "--")):
                points = [(r["step"], get(r[split])) for r in records if split in r and get(r[split]) is not None]
                if points:
                    ax.plot(*zip(*points), style, color=color, label=f"{label} ({split})")
        ax.set_title(title, loc="left")
        if log:
            ax.set_yscale("log")
    for ax in axes[-1]:
        ax.set_xlabel("optimizer updates")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(args.title or "Rounded evaluation: fixed training subset (solid) vs validation (dashed)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(args.output, dpi=140, bbox_inches="tight")


def val_rows(args):
    rows = json.loads(Path(args.items[0]).read_text())["rows"]
    return [r for r in rows if r["split"] == "val" and r["method"] not in args.skip]


def kind(method):
    if method in ("zero", "train_mean") or method.startswith("pca"):
        return 2  # reference, no real bitstream
    return 1 if method.startswith("int") else 0


def rate(args):
    names = ["trained student (rANS coded)", "fixed codec", "reference (no bitstream)"]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    seen = set()
    for r in val_rows(args):
        bits = r["bits_per_visual_token"]
        if bits is None or bits == 0:
            continue  # shown in the channel plot / table instead
        k = kind(r["method"])
        ax.scatter(bits, r["fvu_standardized"], s=46, color=COLORS[k], edgecolor=SURFACE, linewidth=1.2,
                   zorder=3, label=None if k in seen else names[k])
        seen.add(k)
        ax.annotate(r["method"], (bits, r["fvu_standardized"]), xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color=INK)
    ax.axhline(1.0, color=MUTED, linewidth=0.8, linestyle=":")
    ax.text(0.01, 1.0, "training-mean predictor (0 bits)", va="bottom", fontsize=8, color=MUTED,
            transform=ax.get_yaxis_transform())
    ax.set_xlabel("bits per visual token (payload bits / tokens, validation images)")
    ax.set_ylabel("standardized fraction of variance unexplained (lower is better)")
    ax.set_title(args.title or "Bitrate vs reconstruction, validation images", loc="left")
    ax.legend(loc="lower left")
    fig.savefig(args.output, dpi=140, bbox_inches="tight")


def channels(args):
    rows = val_rows(args)
    labels = [r["method"] for r in rows][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.34 * len(rows) + 1.6), sharey=True)
    for ax, title, get in (
            (axes[0], "Channel 3994: fraction of variance unexplained", lambda r: r["ch3994"]["fvu"]),
            (axes[1], "Other 5119 channels: median per-channel FVU", lambda r: r["channels"]["fvu_median"])):
        values = [get(r) for r in rows][::-1]
        low = min(v for v in values if v > 0) / 3  # exact reconstructions (0) are drawn at the axis floor
        ax.set_xscale("log")
        ax.barh(labels, [max(v, low) - low for v in values], left=low, height=0.6, color=COLORS[0])
        for y, v in enumerate(values):
            ax.text(max(v, low), y, f" {v:.3g}", va="center", fontsize=8, color=INK)
        ax.set_xlim(low, max(values) * 6)
        ax.axvline(1.0, color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    fig.suptitle(args.title or "Per-channel error on validation images (1.0 = training-mean predictor)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.output, dpi=140, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=["curves", "rate", "channels"])
    parser.add_argument("output", type=Path)
    parser.add_argument("items", nargs="+")
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument("--title")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    {"curves": curves, "rate": rate, "channels": channels}[args.kind](args)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
