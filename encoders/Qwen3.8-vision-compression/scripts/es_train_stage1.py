"""Stage 1: rate–distortion training of the Entropic Student against the frozen teacher.

Only the vision tower is loaded (never the LLM). With a feature cache that
holds the encoder input and the decoder target, and no deep-supervision terms,
no teacher model is loaded at all (``--cache-only``).

    python encoders/Qwen3.8-vision-compression/scripts/es_train_stage1.py \
        --config configs/qwen3_8/entropic_student/c_merger_tokenmlp_beta0.08.json \
        --runs-dir runs/qwen3.8/entropic-student

Writes runs/<name>/{config.json, run_info.json, metrics.jsonl, checkpoints/}.
Resume with --resume (loads checkpoints/latest.pt). Nothing is downloaded.

Step mode (training.max_steps set, e.g. --set training.max_steps=1000
--set training.eval_every_steps=50) trains to a total number of optimizer
updates. Every eval_every_steps it evaluates a fixed training subset and the
validation set with the eval quantization (rounding) and pooled dataset-level
metrics, then writes latest.pt / best.pt; no per-epoch checkpoints are kept.
With training.full_eval_every_steps set there are two cadences: every
eval_every_steps the training subset and data.val_fast_manifests ("val_fast")
are evaluated and latest.pt is written; every full_eval_every_steps (and at
max_steps) the full validation set ("val") is evaluated as well, and only
those evaluations can replace best.pt (lowest full-validation loss).
--init-checkpoint continues another run's checkpoint in a new run directory.

Resume restores model, entropy tables, both optimizers, scheduler, epoch, step
and the position inside the epoch. Checkpoints written by this version also
carry the torch RNG state; older ones do not, so continuing them gives a valid
but not bit-identical continuation of the original noise stream.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from vit_compression_lab.entropic import payload as payload_lib  # noqa: E402
from vit_compression_lab.entropic.bottleneck import EntropicStudent  # noqa: E402
from vit_compression_lab.entropic.checkpoint import (  # noqa: E402
    RunDir,
    hardware_info,
    load_checkpoint,
    load_student_state,
    save_checkpoint,
    software_info,
)
from vit_compression_lab.entropic.config import MERGER, StudentConfig, VisionDims  # noqa: E402
from vit_compression_lab.entropic.diagnostics import FeatureErrorStats  # noqa: E402
from vit_compression_lab.entropic.grid import counts_for_level, token_counts  # noqa: E402
from vit_compression_lab.entropic.losses import distortion_loss, stage1_loss  # noqa: E402
from vit_compression_lab.entropic.metrics import reconstruction_metrics  # noqa: E402
from vit_compression_lab.entropic.qwen.data import (  # noqa: E402
    FeatureCache,
    ImageManifestDataset,
    PatchBudgetBatchSampler,
    collate_packed,
    estimate_patch_count,
    load_processor,
    read_manifest,
)
from vit_compression_lab.entropic.qwen.vision_tower import TeacherSplit, load_vision_tower  # noqa: E402
from vit_compression_lab.run_checkpoint import sha256_file  # noqa: E402


def needed_points(config):
    split = config.split
    points = {split.encoder_input, split.decoder_target}
    for term in config["deep_supervision"]:
        points.add(term["point"])
    return points


def propagate(split, recon, grid_thw, from_depth, point, geo):
    """Run the frozen teacher from the decoder target to a deeper point, with gradient."""
    to_depth = split.depth_of(point)
    if to_depth < from_depth:
        raise ValueError(f"deep-supervision point {point} is shallower than the decoder target")
    hidden = recon.to(split.dtype)
    if point == MERGER:
        return split.suffix(hidden, grid_thw, from_depth, geo)
    return split.run_blocks(hidden, from_depth, to_depth, geo)


def teacher_activations(split, batch, points, device):
    """returns the intermediate feature tensors"""

    # if required features cached in batch, move to device
    if all(p in batch["features"] for p in points):
        return {p: batch["features"][p].to(device) for p in points}
    if split is None:
        raise RuntimeError("Feature cache lacks required points and no teacher is loaded")

    # run teacher to collect them
    with torch.no_grad():
        return split.collect(batch["pixel_values"].to(device, split.dtype), batch["grid_thw"].to(device), points)


def run_batch(student, split, batch, config, points, channel_stats, device, train):
    """
    student - trainiable EntropicStudent in bottleneck.py
    split - access to frozen teacher's vision layers
    batch - data for the group of images with their grids and cached features
    config - settings for splits, student, quantization, and loss
    points - names of teacher feature locations needed for configuration
    channel_stats - channel standard deviations used to normalize reconstruction errors
    device - where computation runs
    train - whether for training or evaluation
    """

    # move image layout information to the GPU and fetch teacher activations
    grid = batch["grid_thw"].to(device)
    acts = teacher_activations(split, batch, points, device)
    sp = config.split # config settings for student encoder

    # look up the teacher features at student input location
    u = acts[sp.encoder_input].float() #[total tokens, 5120]
    n_visual_tokens = sum(token_counts(grid))
    geo = split.geometry(grid) if (split is not None and config["deep_supervision"]) else None

    # run student with input, grid
    # quantization comes from config["quantization"]: "train" mode here, "eval" under student.eval()
    out = student(u, grid)
    out["deep"] = {}
    for term in config["deep_supervision"]:
        out["deep"][term["point"]] = propagate(split, out["recon"], grid, sp.target_depth(split.dims),
                                               term["point"], geo)
    # count features at patch and token resolutions
    counts_by_level = {lvl: counts_for_level(grid, lvl) for lvl in ("patch", "token")}
    targets = {p: acts[p].float() for p in points}

    # calculate the training objective
    loss, log = stage1_loss(out, targets, counts_by_level, n_visual_tokens, config, channel_stats)
    return loss, log, out, acts, grid


@torch.no_grad()
def evaluate(student, split, loader, config, points, channel_stats, device, coded_examples):
    student.eval()
    logs, coded = [], []

    # run model for each batch of images
    for batch in loader:
        loss, log, out, acts, grid = run_batch(student, split, batch, config, points, channel_stats, device, False)
        entry = {k: float(v) for k, v in log.items()}
        sp = config.split
        if sp.decoder_target != MERGER and split is not None and MERGER in acts:
            # The LLM-facing error even when the decoder target is inside the ViT.
            h_hat = propagate(split, out["recon"], grid, sp.target_depth(split.dims), MERGER, split.geometry(grid))
            entry.update({f"merger/{k}": v for k, v in reconstruction_metrics(acts[MERGER], h_hat).items()})
        elif sp.decoder_target == MERGER:
            entry.update({f"merger/{k}": v for k, v in reconstruction_metrics(acts[MERGER], out["recon"]).items()})
        logs.append(entry)
        if len(coded) < coded_examples:
            # Actual entropy coding on the first few images: bytes, overhead, exact round trip.
            student.entropy.update(force=True)
            u = acts[sp.encoder_input].float()
            strings, counts, ideal = student.compress(u, grid)
            header = payload_lib.make_header(sp, grid.cpu(), counts, student.latent_channels)
            blob = payload_lib.pack(header, strings)
            z_round = student.entropy.quantize(student.encode(u, grid), "round")
            z_dec = torch.cat([student.entropy.decompress(s, n, device=device) for s, n in zip(strings, counts)])
            coded.append({"coded_bytes": len(blob), "ideal_bits": sum(ideal),
                          "overhead_ratio": 8 * len(blob) / max(sum(ideal), 1e-9),
                          "bits_per_visual_token": 8 * len(blob) / sum(token_counts(grid)),
                          "roundtrip_exact": torch.equal(z_round.to(z_dec.dtype), z_dec)})
    summary = {k: sum(e[k] for e in logs if k in e) / max(1, sum(1 for e in logs if k in e))
               for k in {k for e in logs for k in e}}
    if coded:
        summary["coded"] = {k: (sum(c[k] for c in coded) / len(coded) if k != "roundtrip_exact"
                                else all(c[k] for c in coded)) for k in coded[0]}
    student.train()
    return summary


@torch.no_grad()
def evaluate_dataset(student, split, loader, config, points, channel_stats, mean, device, coded_examples,
                     channel=None, started=None):
    """Step-mode evaluation. ``loader`` yields ONE image per batch.
    Measures how good the student performs on a whole set of images in setp mode

        student - EntropicStudent being trained
        split - where the frozne teacher's vision tower is
        loader - DataLoader that yields one image per batch in fixed order
        config - StudentConfig with settings
        points - names of where the teacher feature locations are needed (encoder input, decoder target, etc)
        channel_stats - dictionary from point to per channel std
        mean - per-channel training mean at decoder target
        device - GPU or CPU
        coded_examples - how many images to entropy code
        channel - channel index whose error is reported separately
    """
    started = started or time.time()
    student.eval() # eval quantization rounding
    student.entropy.update(force=True) # rebuild integer CDF tables the entropy encoder uses

    sp = config.split
    codable = config["quantization"]["eval"] != "none"
    stats = None # error accumulator
    logs, coded, bits, tokens = [], [], 0.0, 0

    # iterate over images
    for batch in loader:
        # run student on this image
        _, log, out, acts, grid = run_batch(student, split, batch, config, points, channel_stats, device, False)
        target = acts[sp.decoder_target].float()

        if stats is None:
            # create error accumulator object, defined in daignostics.py
            c = target.shape[1]
            std = channel_stats[sp.decoder_target] if channel_stats else torch.ones(c, device=device)
            stats = FeatureErrorStats(mean if mean is not None else torch.zeros(c, device=device), std)

        # add errors to running sums
        stats.add(target, out["recon"])
        logs.append({k: float(v) for k, v in log.items()})
        n = sum(token_counts(grid))
        bits += float(log["bits"])
        tokens += n
        if codable and len(coded) < coded_examples:
            # Real entropy coding: bytes, exact latent round trip, and decoded features == forward().
            u = acts[sp.encoder_input].float()
            strings, counts, _ = student.compress(u, grid)
            blob = payload_lib.pack(payload_lib.make_header(sp, grid.cpu(), counts, student.latent_channels), strings)
            z_dec = torch.cat([student.entropy.decompress(s, k, device=device) for s, k in zip(strings, counts)])
            recon_dec = student.decompress(strings, grid, device=device)
            coded.append({"bytes": len(blob), "tokens": n,
                          "latent_exact": torch.equal(out["z_hat"].to(z_dec.dtype), z_dec),
                          "recon_max_abs_diff": (recon_dec - out["recon"]).abs().max().item()})
    summary = {
        "aggregation": "loss/distortion/rate_nats: mean over images; all other metrics pooled over tokens",
        "images": len(logs),
        "loss": sum(e["loss"] for e in logs) / len(logs),
        "distortion": sum(e["distortion"] for e in logs) / len(logs),
        "rate_nats": sum(e["rate"] for e in logs) / len(logs),
        "estimated_bits_per_visual_token": bits / tokens if codable else None,
        **stats.result(channel),
    }
    if coded:
        summary["coded"] = {
            "images": len(coded),
            "bits_per_visual_token": 8 * sum(c["bytes"] for c in coded) / sum(c["tokens"] for c in coded),
            "roundtrip_exact": all(c["latent_exact"] for c in coded),
            "recon_max_abs_diff": max(c["recon_max_abs_diff"] for c in coded),
        }
    summary["seconds"] = time.time() - started
    student.train() # back to training behavior
    return summary


def apply_overrides(raw, assignments):
    """--set a.b=value: replace an existing config entry; the value is parsed as JSON."""
    for item in assignments:
        key, _, value = item.partition("=")
        *parents, leaf = key.split(".")
        node = raw
        for part in parents:
            node = node[part]
        if leaf not in node:
            raise KeyError(f"--set {key}: no such config entry")
        node[leaf] = json.loads(value)


def grad_norm(parameters):
    grads = [p.grad for p in parameters if p.grad is not None]
    return torch.sqrt(sum(g.pow(2).sum() for g in grads)) if grads else torch.tensor(0.0)


def rng_state(device):
    state = {"torch": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def main():

    # argument parsing
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/qwen3.8/entropic-student"))
    parser.add_argument("--device", default="cuda:1", help="GPU 0 is shared on this host")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Override data.train_limit (smoke tests)")
    parser.add_argument("--epochs", type=int, help="Override training.epochs")
    parser.add_argument("--cache-only", action="store_true",
                        help="Never load the teacher; all needed points must be cached")
    parser.add_argument("--coded-examples", type=int, default=4,
                        help="Images actually entropy coded at each evaluation")
    parser.add_argument("--run-name", help="Run directory name (default: the config's name)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a config entry, e.g. --set rate.beta=0 (repeatable; JSON value)")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Start a NEW run from another run's checkpoint: model, entropy tables, "
                             "optimizers, scheduler, epoch and step are all restored")
    parser.add_argument("--diagnostic-channel", type=int, default=3994,
                        help="Channel whose error is reported separately in step-mode evaluation")
    args = parser.parse_args()

    device = torch.device(args.device)
    config = StudentConfig.load(args.config)
    apply_overrides(config.raw, args.set)
    if args.run_name:
        config.raw["name"] = args.run_name
    config = StudentConfig.from_dict(config.raw)  # re-validate after overrides
    training_config = config["training"]
    if args.epochs is not None:
        training_config["epochs"] = args.epochs
    if args.limit is not None:
        config.raw["data"]["train_limit"] = args.limit
    max_steps, eval_every = training_config["max_steps"], training_config["eval_every_steps"]
    step_mode = max_steps is not None
    if step_mode and not eval_every:
        parser.error("training.max_steps needs training.eval_every_steps")
    if args.resume and args.init_checkpoint:
        parser.error("--init-checkpoint starts a new run; use --resume alone to continue this one")
    torch.manual_seed(training_config["seed"])

    full_eval_every = training_config["full_eval_every_steps"]
    if full_eval_every and (not step_mode or full_eval_every % eval_every):
        parser.error("training.full_eval_every_steps needs step mode and a multiple of eval_every_steps")

    teacher = config["teacher"] # extract the teacher mdoel
    # --cache-only reads features and grids from the cache: no image preprocessing, no processor
    processor = None if args.cache_only else load_processor(teacher["checkpoint"], teacher["revision"])
    points = needed_points(config) # teacher feature locations needed
    split = None
    dims = VisionDims()
    if not args.cache_only:
        visual = load_vision_tower(teacher["checkpoint"], teacher["revision"], device=device,
                                   dtype=getattr(torch, teacher["dtype"]),
                                   attn_implementation=teacher["attn_implementation"])
        split = TeacherSplit(visual)
        dims = split.dims
    elif config["deep_supervision"]:
        parser.error("--cache-only cannot propagate deep-supervision terms through the teacher")

    data = config["data"]
    if not data["train_manifests"]:
        parser.error("config.data.train_manifests is empty: no training images are in this repository")

    # set up cache for tensors to be cached from teacher feature extraction
    cache = FeatureCache(data["feature_cache"]) if data["feature_cache"] else None
    if args.cache_only and cache is None:
        parser.error("--cache-only needs data.feature_cache")
    cache_points = sorted(points) if (cache is not None and args.cache_only) else []

    # read train + validation images from manifests
    train_rows = read_manifest(data["train_manifests"], limit=data["train_limit"])
    val_rows = read_manifest(data["val_manifests"], limit=data["val_limit"]) if data["val_manifests"] else []
    val_fast_rows = read_manifest(data["val_fast_manifests"]) if data["val_fast_manifests"] else []
    if not {r["id"] for r in val_fast_rows} <= {r["id"] for r in val_rows}:
        parser.error("data.val_fast_manifests must be a subset of the validation rows")
    if cache is not None and not args.cache_only:
        # Use cached activations where present (teacher is still available for the rest).
        cache_points = sorted(p for p in points if all(cache.has(r["id"]) for r in train_rows + val_rows))

    def patch_count(row):
        """Patches of one example: the cached grid in cache-only mode, else an estimate from the image size."""
        return cache.patch_count(row["id"]) if args.cache_only else estimate_patch_count(row, processor)

    def make_dataset(rows):
        # cache-only: features + grid_thw only (no pixel tensors are read or needed)
        return ImageManifestDataset(rows, processor, cache, cache_points, load_pixels=not args.cache_only)

    def make_loader(rows, shuffle):
        """Patch-budget batches. An image above the whole budget is a batch of its own, never dropped."""
        counts = [patch_count(r) for r in rows] # defines the patches needed per image

        # determines which indices belong in each batch
        sampler = PatchBudgetBatchSampler(counts, training_config["max_patches_per_batch"], shuffle=shuffle,
                                          seed=training_config["seed"], drop_oversize=False)
        if sampler.oversize:
            print(f"{sampler.oversize} examples exceed max_patches_per_batch; each is a single-example batch")

        # use pytorch module for DataLoader. Its own generator keeps loader bookkeeping from
        # drawing on the global RNG, so evaluations and resumes do not shift the training noise.
        return DataLoader(make_dataset(rows), batch_sampler=sampler, collate_fn=collate_packed,
                          num_workers=training_config["num_workers"], generator=torch.Generator()), sampler

    # create the loader
    train_loader, train_sampler = make_loader(train_rows, shuffle=True)
    val_loader = make_loader(val_rows, shuffle=False)[0] if (val_rows and not step_mode) else None

    def make_eval_loader(rows):
        # step mode: one image per batch, fixed order, so dataset-level metrics need no batch weighting
        return DataLoader(make_dataset(rows), batch_size=1, shuffle=False, collate_fn=collate_packed,
                          generator=torch.Generator())

    channel_stats, channel_mean = None, None
    if data["channel_stats"]:
        # load previously saved feature statistics (computed on the training manifest only)
        stats = torch.load(data["channel_stats"], weights_only=False)
        channel_stats = {p: v["std"].to(device) for p, v in stats.items()}
        channel_mean = {p: v["mean"].to(device) for p, v in stats.items()}
    target_mean = (channel_mean or {}).get(config.split.decoder_target)

    # create the student model described in config
    student = EntropicStudent(config, dims).to(device)
    needs_stats = student.output_mode == "normalized" or config["decoder"]["bias_init"] == "train_mean"
    if needs_stats and target_mean is None:
        parser.error("decoder.output 'normalized' / bias_init 'train_mean' need data.channel_stats")
    if student.output_mode == "normalized":
        student.set_output_stats(target_mean, channel_stats[config.split.decoder_target])
    if config["decoder"]["bias_init"] == "train_mean":
        student.init_output_bias(target_mean)

    # optimizer for encoder, decoder, entropy distribution (objective = reconstruction error + rate penalty)
    main_opt = torch.optim.Adam(student.main_parameters(), lr=training_config["lr"], weight_decay=training_config["weight_decay"])
    # updates the qunatiles for student.entropy
    aux_opt = torch.optim.Adam(student.aux_parameters(), lr=training_config["aux_lr"])
    # schedules certain milestones where learning rate is reduced
    scheduler = torch.optim.lr_scheduler.MultiStepLR(main_opt, milestones=training_config["milestones"], gamma=training_config["gamma"])

    run = RunDir(args.runs_dir, config.name, resume=args.resume)
    start_epoch, step, best, skip_batches = 0, 0, math.inf, 0
    seen = {"images": 0, "tokens": 0, "seconds": 0.0}  # cumulative over the whole run, across resumes
    resuming = args.resume and run.checkpoint_path("latest").exists()
    source = run.checkpoint_path("latest") if resuming else args.init_checkpoint
    restored = None
    if source is not None:
        # restore a full training state into the student the optimizers were built on
        ckpt = load_checkpoint(source, map_location=device)
        if "optimizers" not in ckpt:
            parser.error(f"{source} has no optimizer state (use a latest.pt)")
        load_student_state(student, ckpt["state_dict"])
        main_opt.load_state_dict(ckpt["optimizers"]["main"])
        aux_opt.load_state_dict(ckpt["optimizers"]["aux"])
        scheduler.load_state_dict(ckpt["optimizers"]["scheduler"])
        extra = ckpt.get("extra", {})
        step, best = ckpt["step"], extra.get("best", math.inf)
        seen.update(extra.get("seen", {}))
        if "batches_done_in_epoch" in extra:
            # saved inside epoch ckpt["epoch"]: redo that epoch's (deterministic) order, skipping what ran
            start_epoch, skip_batches = ckpt["epoch"], extra["batches_done_in_epoch"]
        else:
            start_epoch = ckpt["epoch"] + 1  # epoch-boundary checkpoint
        if "rng" in extra:
            torch.set_rng_state(extra["rng"]["torch"].cpu())
            if "cuda" in extra["rng"] and device.type == "cuda":
                torch.cuda.set_rng_state(extra["rng"]["cuda"].cpu(), device)
        adam_steps = {name: int(next(iter(opt.state_dict()["state"].values()))["step"])
                      for name, opt in (("main", main_opt), ("aux", aux_opt))}
        restored = {"path": str(source), "sha256": sha256_file(source), "epoch": ckpt["epoch"], "step": step,
                    "skip_batches": skip_batches, "rng_restored": "rng" in extra, "best": best,
                    "adam_steps": adam_steps, "scheduler_last_epoch": scheduler.last_epoch,
                    "cdf_table_shape": list(student.entropy._quantized_cdf.shape)}
        print(f"restored {source}: epoch {start_epoch}, step {step}, skipping {skip_batches} batches, "
              f"rng restored: {'rng' in extra}")
    if resuming:
        run.log({"phase": "resume", "step": step, "restored": restored, "training": training_config})
    else:
        run.write_config(config)
        run.write_run_info({
            "command": sys.argv, "overrides": args.set, "seed": training_config["seed"],
            "init_checkpoint": restored,
            "stage": 1, "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "teacher": teacher, "teacher_loaded": split is not None, "points": sorted(points),
            "cache_points": cache_points, "split": config.split.as_dict(), "option": config.split.option,
            "parameters": student.parameter_counts(), "train_examples": len(train_rows),
            "val_examples": len(val_rows), "software": software_info(), "hardware": hardware_info(device),
            "val_fast_examples": len(val_fast_rows), "train_eval_ids": [r["id"] for r in
                                                                         train_rows[: data["train_eval_limit"]]],
            "val_fast_ids": [r["id"] for r in val_fast_rows],
            "manifests": {str(p): sha256_file(p) for p in
                          data["train_manifests"] + data["val_manifests"] + data["val_fast_manifests"]},
            "channel_stats_sha256": sha256_file(data["channel_stats"]) if data["channel_stats"] else None,
            "batching": {"train_batches_per_epoch": len(train_sampler), "oversize_single_example_batches":
                         train_sampler.oversize, "dropped": train_sampler.dropped,
                         "train_patches": sum(train_sampler.patch_counts),
                         "max_patches_one_image": max(train_sampler.patch_counts)},
        })

    train_eval_loader = make_eval_loader(train_rows[: data["train_eval_limit"]]) if step_mode else None
    val_eval_loader = make_eval_loader(val_rows) if (step_mode and val_rows) else None
    val_fast_loader = make_eval_loader(val_fast_rows) if (step_mode and val_fast_rows) else None

    def save_latest(epoch, batches_done=None):
        extra = {"best": best, "rng": rng_state(device), "seen": dict(seen)}
        if batches_done is not None:
            extra["batches_done_in_epoch"] = batches_done
        optimizers = {"main": main_opt, "aux": aux_opt, "scheduler": scheduler}
        save_checkpoint(run.checkpoint_path("latest"), student, epoch, step, optimizers, extra=extra)

    def step_evaluation(epoch, batches_done, save=True, full=True):
        """Rounded evaluation, then checkpoints.

        Always: the fixed training subset and the fixed validation subset ("val_fast").
        ``full``: the whole validation set ("val") too. Only a full evaluation can replace best.pt,
        so the criterion is always the same images; latest.pt is written either way.
        """
        nonlocal best
        record = {"phase": "eval", "epoch": epoch, "step": step, "quantization": config["quantization"]["eval"],
                  "full_validation": full, "images_seen": seen["images"], "tokens_seen": seen["tokens"]}
        loaders = [("train", train_eval_loader), ("val_fast", val_fast_loader)]
        if full:
            loaders.append(("val", val_eval_loader))
        for name, loader in loaders:
            if loader is not None:
                record[name] = evaluate_dataset(student, split, loader, config, points, channel_stats, target_mean,
                                                device, args.coded_examples, args.diagnostic_channel)
        run.log(record)
        names = [name for name, _ in loaders if name in record]
        if any(not record[k]["coded"]["roundtrip_exact"] for k in names if "coded" in record[k]):
            run.log({"phase": "abort", "step": step, "reason": "entropy round trip failed"})
            raise SystemExit("entropy round trip failed; run stopped (latest.pt is the last good state)")
        print(f"eval step {step}: " + " | ".join(
            f"{k} loss {record[k]['loss']:.0f} std-mse {record[k]['standardized_mse']:.4f} "
            f"relL2 {record[k]['relative_l2']:.4f}" for k in names), flush=True)
        if not save:
            return
        if full:
            score = record.get("val", record["train"])["loss"]
            if score < best:
                best = score
                save_checkpoint(run.checkpoint_path("best"), student, epoch, step, extra={"best": best})
        save_latest(epoch, batches_done)
        milestone = training_config["milestone_every_steps"]
        if milestone and step % milestone == 0:
            save_checkpoint(run.checkpoint_path(f"step-{step:06d}"), student, epoch, step, extra={"best": best})

    student.train() #enable training behavior
    if step_mode and not resuming:
        step_evaluation(start_epoch, skip_batches, save=False)  # starting point (initialisation or source run)
    epoch = start_epoch
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    window = {"data": 0.0, "compute": 0.0, "steps": 0}  # seconds since the last logged step
    while (step < max_steps) if step_mode else (epoch < training_config["epochs"]):
        train_sampler.set_epoch(epoch)
        train_sampler.start_batch = skip_batches  # resume inside an epoch without loading what already ran
        epoch_start = time.time() #record start time
        stop = False
        tick = time.time()
        for batch_index, batch in enumerate(train_loader, start=skip_batches):
            loaded = time.time()
            # run student and compute main objective
            loss, log, _, _, grid = run_batch(student, split, batch, config, points, channel_stats, device, True)
            aux = student.entropy.aux_loss()
            main_opt.zero_grad(set_to_none=True) #clear previous gradients
            loss.backward() # compute gradients of loss with respect tio the main parameters
            log_now = (step + 1) % training_config["log_every"] == 0
            if log_now:  # per-module gradient norms, before clipping
                grads = {f"grad_norm/{name}": grad_norm(module.parameters()).item() for name, module in
                         (("encoder", student.encoder), ("decoder", student.decoder), ("entropy", student.entropy))}
            if training_config["grad_clip"]:
                total_grad = torch.nn.utils.clip_grad_norm_(student.main_parameters(), training_config["grad_clip"])
            else:
                total_grad = grad_norm(student.main_parameters())
            if not (torch.isfinite(loss) and torch.isfinite(total_grad)):
                run.log({"phase": "abort", "epoch": epoch, "step": step, "reason": "non-finite loss or gradient",
                         "loss": loss.item(), "grad_norm": total_grad.item()})
                raise SystemExit(f"non-finite loss or gradient at step {step}; no update applied, run stopped")
            main_opt.step()

            # auxiliary parameters update
            aux_opt.zero_grad(set_to_none=True)
            aux.backward()
            if log_now:
                grads["grad_norm/aux_quantiles"] = grad_norm(student.aux_parameters()).item()
            aux_opt.step()

            step += 1
            seen["images"] += len(batch["ids"])
            seen["tokens"] += sum(token_counts(grid))
            if device.type == "cuda":
                torch.cuda.synchronize(device)  # so the timings below include the GPU work
            now = time.time()
            window["data"] += loaded - tick
            window["compute"] += now - loaded
            window["steps"] += 1
            seen["seconds"] += now - tick
            if log_now:
                run.log({"phase": "train", "epoch": epoch, "step": step, "lr": scheduler.get_last_lr()[0],
                         "aux_loss": aux.item(), "grad_norm": total_grad.item(), **grads,
                         **{k: float(v) for k, v in log.items()},
                         "batch_images": len(batch["ids"]), "batch_tokens": sum(token_counts(grid)),
                         "images_seen": seen["images"], "tokens_seen": seen["tokens"],
                         "train_seconds": seen["seconds"],
                         "data_seconds_per_step": window["data"] / window["steps"],
                         "compute_seconds_per_step": window["compute"] / window["steps"],
                         "gpu_peak_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
                print(f"epoch {epoch} step {step} loss {log['loss']:.4f} mse {log['mse']:.5f} "
                      f"bits/tok {log['bits_per_visual_token']:.1f}", flush=True)
                window = {"data": 0.0, "compute": 0.0, "steps": 0}
            if step_mode:
                if step % eval_every == 0 or step >= max_steps:
                    step_evaluation(epoch, batch_index + 1,
                                    full=not full_eval_every or step % full_eval_every == 0 or step >= max_steps)
                if step >= max_steps:
                    stop = True
                    break
            tick = time.time()  # evaluation and checkpoint time is not training time
        skip_batches = 0
        if stop:
            # scheduler.step() for this epoch happens on resume, after the skipped batches
            run.log({"phase": "done", "step": step, "epoch": epoch, "images_seen": seen["images"],
                     "tokens_seen": seen["tokens"], "train_seconds": seen["seconds"],
                     "train_examples": len(train_rows), "image_passes": seen["images"] / len(train_rows),
                     "best_full_validation_loss": best})
            break
        scheduler.step()
        if step_mode:
            epoch += 1
            continue
        record = {"phase": "epoch", "epoch": epoch, "step": step, "seconds": time.time() - epoch_start}

        # run evaluation
        if val_loader is not None and training_config["eval_every_epoch"]:
            val = evaluate(student, split, val_loader, config, points, channel_stats, device, args.coded_examples)
            record["val"] = val
            score = val.get("loss", math.inf)
            if score < best:
                best = score
                save_checkpoint(run.checkpoint_path("best"), student, epoch, step, extra={"best": best, "val": val})
        run.log(record)
        student.entropy.update(force=True)
        save_latest(epoch)
        save_checkpoint(run.checkpoint_path(f"epoch-{epoch:03d}"), student, epoch, step, extra={"best": best})
        print(f"epoch {epoch} done: {record.get('val', {})}")
        epoch += 1


if __name__ == "__main__":
    main()
