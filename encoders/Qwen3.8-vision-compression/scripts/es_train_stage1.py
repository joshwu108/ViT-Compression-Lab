"""Stage 1: rate–distortion training of the Entropic Student against the frozen teacher.

Only the vision tower is loaded (never the LLM). With a feature cache that
holds the encoder input and the decoder target, and no deep-supervision terms,
no teacher model is loaded at all (``--cache-only``).

    python encoders/Qwen3.8-vision-compression/scripts/es_train_stage1.py \
        --config configs/qwen3_8/entropic_student/c_merger_tokenmlp_beta0.08.json \
        --runs-dir runs/qwen3.8/entropic-student

Writes runs/<name>/{config.json, run_info.json, metrics.jsonl, checkpoints/}.
Resume with --resume (loads checkpoints/latest.pt). Nothing is downloaded.
"""

import argparse
import hashlib
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
    save_checkpoint,
    software_info,
)
from vit_compression_lab.entropic.config import MERGER, StudentConfig, VisionDims  # noqa: E402
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
    out = student(u, grid, quant_mode=None if train else "round")
    out["deep"] = {}
    for term in config["deep_supervision"]:
        out["deep"][term["point"]] = propagate(split, out["recon"], grid, sp.target_depth(split.dims),
                                               term["point"], geo)
    # count features at patch and token resolutions
    counts_by_level = {lvl: counts_for_level(grid, lvl) for lvl in ("patch", "token")}
    targets = {p: acts[p].float() for p in points}

    # clalculate the training objective
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
    args = parser.parse_args()

    device = torch.device(args.device)
    config = StudentConfig.load(args.config)
    training_config = config["training"]
    if args.epochs is not None:
        training_config["epochs"] = args.epochs
    if args.limit is not None:
        config.raw["data"]["train_limit"] = args.limit
    torch.manual_seed(training_config["seed"])

    teacher = config["teacher"] # extract the teacher mdoel
    processor = load_processor(teacher["checkpoint"], teacher["revision"]) # processor associated with that checkpoint
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
    cache_points = sorted(points) if (cache is not None and args.cache_only) else []

    # read train + validation images from manifests
    train_rows = read_manifest(data["train_manifests"], limit=data["train_limit"])
    val_rows = read_manifest(data["val_manifests"], limit=data["val_limit"]) if data["val_manifests"] else []
    if cache is not None and not args.cache_only:
        # Use cached activations where present (teacher is still available for the rest).
        cache_points = sorted(p for p in points if all(cache.has(r["id"]) for r in train_rows + val_rows))

    def make_loader(rows, shuffle):
        """helper function to """
        dataset = ImageManifestDataset(rows, processor, cache, cache_points) # defines how to extract image data
        counts = [estimate_patch_count(r, processor) for r in rows] # defines the patches needed per image

        # determines which indices belong in each batch
        sampler = PatchBudgetBatchSampler(counts, training_config["max_patches_per_batch"], shuffle=shuffle, seed=tr["seed"])
        if sampler.dropped:
            print(f"{sampler.dropped} examples exceed max_patches_per_batch and are skipped")

        # use pytorch module for DataLoader
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_packed,
                          num_workers=training_config["num_workers"]), sampler

    # create the loader
    train_loader, train_sampler = make_loader(train_rows, shuffle=True)
    val_loader = make_loader(val_rows, shuffle=False)[0] if val_rows else None

    channel_stats = None
    if data["channel_stats"]:
        # load previously saved feature statistics
        stats = torch.load(data["channel_stats"], weights_only=False)
        channel_stats = {p: v["std"].to(device) for p, v in stats.items()}

    # create the student model described in config
    student = EntropicStudent(config, dims).to(device)

    # optimizer for encoder, decoder, entropy distribution (objective = reconstruction error + rate penalty)
    main_opt = torch.optim.Adam(student.main_parameters(), lr=training_config["lr"], weight_decay=training_config["weight_decay"])
    # updates the qunatiles for student.entropy
    aux_opt = torch.optim.Adam(student.aux_parameters(), lr=training_config["aux_lr"])
    # schedules certain milestones where learning rate is reduced
    scheduler = torch.optim.lr_scheduler.MultiStepLR(main_opt, milestones=training_config["milestones"], gamma=training_config["gamma"])

    run = RunDir(args.runs_dir, config.name, resume=args.resume)
    start_epoch, step, best = 0, 0, math.inf
    if args.resume and run.checkpoint_path("latest").exists():
        # restore existing run
        ckpt = load_checkpoint(run.checkpoint_path("latest"), map_location=device)
        student.load_state_dict(ckpt["state_dict"])
        main_opt.load_state_dict(ckpt["optimizers"]["main"])
        aux_opt.load_state_dict(ckpt["optimizers"]["aux"])
        scheduler.load_state_dict(ckpt["optimizers"]["scheduler"])
        start_epoch, step = ckpt["epoch"] + 1, ckpt["step"]
        best = ckpt.get("extra", {}).get("best", math.inf)
        print(f"resumed at epoch {start_epoch}, step {step}")
    else:
        run.write_config(config)
        run.write_run_info({
            "stage": 1, "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "teacher": teacher, "teacher_loaded": split is not None, "points": sorted(points),
            "cache_points": cache_points, "split": config.split.as_dict(), "option": config.split.option,
            "parameters": student.parameter_counts(), "train_examples": len(train_rows),
            "val_examples": len(val_rows), "software": software_info(), "hardware": hardware_info(device),
        })

    student.train() #enable training behavior
    for epoch in range(start_epoch, training_config["epochs"]):
        train_sampler.set_epoch(epoch)
        epoch_start = time.time() #record start time
        for batch in train_loader:
            # run student and compute main objective
            loss, log, _, _, _ = run_batch(student, split, batch, config, points, channel_stats, device, True)
            aux = student.entropy.aux_loss()
            main_opt.zero_grad(set_to_none=True) #clear previous gradients
            loss.backward() # compute gradients of loss with respect tio the main parameters
            if training_config["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(student.main_parameters(), training_config["grad_clip"])
            main_opt.step()

            # auxiliary parameters update
            aux_opt.zero_grad(set_to_none=True)
            aux.backward()
            aux_opt.step()

            step += 1
            if step % training_config["log_every"] == 0:
                run.log({"phase": "train", "epoch": epoch, "step": step, "lr": scheduler.get_last_lr()[0],
                         "aux_loss": aux.item(), **{k: float(v) for k, v in log.items()}})
                print(f"epoch {epoch} step {step} loss {log['loss']:.4f} mse {log['mse']:.5f} "
                      f"bits/tok {log['bits_per_visual_token']:.1f}")
        scheduler.step()
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
        optimizers = {"main": main_opt, "aux": aux_opt, "scheduler": scheduler}
        save_checkpoint(run.checkpoint_path("latest"), student, epoch, step, optimizers, extra={"best": best})
        save_checkpoint(run.checkpoint_path(f"epoch-{epoch:03d}"), student, epoch, step, extra={"best": best})
        print(f"epoch {epoch} done: {record.get('val', {})}")


if __name__ == "__main__":
    main()
