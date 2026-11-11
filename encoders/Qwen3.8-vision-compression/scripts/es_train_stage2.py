"""Stage 2: task-level distillation through the frozen Qwen LLM.

Adaptation of the paper's Stage 2 (CE + KD on classifier logits) to language
modelling: the encoder side and the entropy model are frozen (so the bitrate
is fixed and rounding replaces noise), the decoder (optionally also a private
copy of the server-side ViT suffix) is trained to minimise

    L_2 = alpha * CE(student logits, answer tokens)
        + (1 - alpha) * tau^2 * KL(teacher next-token dist. || student next-token dist.)

over the answer positions, teacher-forced on the teacher's own greedy answer
(default) or the manifest reference. Teacher logits use the original merged
features; student logits use the reconstructed ones; the LLM is frozen and
gradients flow only into the injected visual embeddings.

    python encoders/Qwen3.8-vision-compression/scripts/es_train_stage2.py \
        --student-checkpoint runs/qwen3.8/entropic-student/<name>/checkpoints/best.pt \
        --manifests data/train/vqa_manifest.jsonl --runs-dir runs/qwen3.8/entropic-student

Hardware: the 27B model (~54 GB BF16) plus activations for a backward pass
through 64 layers; requires an otherwise idle 80 GB GPU. Nothing is downloaded.
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import GenerationConfig, Qwen3_5ForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from vit_compression_lab.entropic.checkpoint import (  # noqa: E402
    RunDir,
    hardware_info,
    load_checkpoint,
    save_checkpoint,
    software_info,
    student_from_checkpoint,
)
from vit_compression_lab.entropic.config import MERGER  # noqa: E402
from vit_compression_lab.entropic.losses import stage2_loss  # noqa: E402
from vit_compression_lab.entropic.qwen.data import load_processor, read_manifest  # noqa: E402
from vit_compression_lab.entropic.qwen.injection import (  # noqa: E402
    injected_kwargs,
    load_inject_eval,
    teacher_forced_inputs,
)
from vit_compression_lab.entropic.qwen.vision_tower import TeacherSplit  # noqa: E402


class ServerSuffix:
    """Runs the ViT from the decoder target to the merger, frozen or as a trainable private copy."""

    def __init__(self, split, from_depth, trainable):
        self.split = split
        self.from_depth = from_depth
        self.trainable = trainable
        self.blocks = None
        self.merger = None
        if trainable and from_depth != split.merger_depth:
            self.blocks = copy.deepcopy(split.visual.blocks[from_depth:]).float()
            self.merger = copy.deepcopy(split.visual.merger).float()
            for p in list(self.blocks.parameters()) + list(self.merger.parameters()):
                p.requires_grad_(True)

    def parameters(self):
        if self.blocks is None:
            return []
        return list(self.blocks.parameters()) + list(self.merger.parameters())

    def __call__(self, recon, grid, geo):
        if self.blocks is None:
            return self.split.suffix(recon, grid, self.from_depth, geo)
        hidden = recon.float()
        for blk in self.blocks:
            hidden = blk(hidden, cu_seqlens=geo.cu_seqlens, max_seqlen=geo.max_seqlen,
                         position_embeddings=geo.position_embeddings)
        return self.merger(hidden)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True, help="Rows need a question")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/qwen3.8/entropic-student"))
    parser.add_argument("--name", help="Run name (default: <stage1 name>-stage2)")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    ie = load_inject_eval()

    ckpt = load_checkpoint(args.student_checkpoint, map_location=device)
    student, config = student_from_checkpoint(ckpt)
    student = student.to(device)
    s2 = config["stage2"]
    epochs = args.epochs or s2["epochs"]
    student.entropy.update(force=True)
    student.freeze_client_side()
    student.train()

    teacher_cfg = config["teacher"]
    processor = load_processor(teacher_cfg["checkpoint"], teacher_cfg["revision"])
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        teacher_cfg["checkpoint"], revision=teacher_cfg["revision"], local_files_only=True,
        dtype=torch.bfloat16, device_map={"": device})
    for p in model.parameters():
        p.requires_grad_(False)
    if s2["grad_checkpointing"]:
        # Checkpointing is active only in train mode; this model has no dropout (attention_dropout 0.0).
        model.gradient_checkpointing_enable()
        model.train()
    else:
        model.eval()
    defaults = model.generation_config
    generation_config = GenerationConfig(do_sample=False, max_new_tokens=s2["max_new_tokens"],
                                         eos_token_id=defaults.eos_token_id,
                                         pad_token_id=defaults.pad_token_id, bos_token_id=defaults.bos_token_id)
    eos_ids = set(defaults.eos_token_id if isinstance(defaults.eos_token_id, list) else [defaults.eos_token_id])

    split = TeacherSplit(model.model.visual)
    sp = config.split
    in_depth, tgt_depth = sp.input_depth(split.dims), sp.target_depth(split.dims)
    suffix = ServerSuffix(split, tgt_depth, s2["train_server_suffix"])

    params = [p for p in student.decoder.parameters() if p.requires_grad] + suffix.parameters()
    optimizer = torch.optim.AdamW(params, lr=s2["lr"], weight_decay=0.0)
    rows = read_manifest(args.manifests, limit=args.limit, require_question=True)
    name = args.name or f"{config.name}-stage2"
    run = RunDir(args.runs_dir, name, resume=args.resume)
    run.write_config(config)
    run.write_run_info({"stage": 2, "student_checkpoint": str(args.student_checkpoint), "stage2": s2,
                        "trainable_parameters": sum(p.numel() for p in params), "examples": len(rows),
                        "teacher": teacher_cfg, "software": software_info(), "hardware": hardware_info(device)})

    teacher_cache = {}  # id -> (answer_ids, teacher logits at answer positions, on CPU)
    step = 0
    for epoch in range(epochs):
        for row in rows:
            with Image.open(row["image_path"]) as source:
                image = source.convert("RGB")
            inputs = ie.build_inputs(processor, image, row["question"])
            inputs = {k: v.to(device) for k, v in inputs.items()}
            grid = inputs["image_grid_thw"]
            with torch.no_grad():
                h = ie.encode_image(model, inputs)
                if row["id"] in teacher_cache:
                    answer_ids, teacher_logits = teacher_cache[row["id"]]
                    teacher_logits = teacher_logits.to(device)
                else:
                    if s2["answer_source"] == "teacher_greedy":
                        # Generation with the KV cache needs eval mode (checkpointing disables the cache).
                        model.eval()
                        new_ids = ie.generate(model, injected_kwargs(inputs, h), generation_config)
                        if s2["grad_checkpointing"]:
                            model.train()
                        answer_ids = [t for t in new_ids.tolist() if t not in eos_ids]
                    else:
                        answer_ids = processor.tokenizer(row["expected_answer"], add_special_tokens=False)["input_ids"]
                    if not answer_ids:
                        continue
                    extended, mask, labels = teacher_forced_inputs(processor, inputs, answer_ids)
                    model.model.rope_deltas = None
                    teacher_logits = model(**injected_kwargs(extended, h), use_cache=False).logits[mask].float()
                    teacher_cache[row["id"]] = (answer_ids, teacher_logits.to(torch.float16).cpu())
                extended, mask, labels = teacher_forced_inputs(processor, inputs, answer_ids)
                geo = split.geometry(grid)
                u = split.prefix(inputs["pixel_values"].to(split.dtype), grid, in_depth, geo)

            out = student(u.float(), grid, quant_mode="round")
            h_hat = suffix(out["recon"], grid, geo).to(h.dtype)
            model.model.rope_deltas = None
            student_logits = model(**injected_kwargs(extended, h_hat), use_cache=False).logits[mask]
            loss, log = stage2_loss(student_logits, teacher_logits, labels[mask], torch.ones_like(labels[mask], dtype=torch.bool),
                                    alpha=s2["alpha"], tau=s2["tau"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            step += 1
            if step % 10 == 0:
                run.log({"phase": "train", "epoch": epoch, "step": step, "id": row["id"],
                         **{k: float(v) for k, v in log.items()}})
                print(f"epoch {epoch} step {step} kd {log['kd']:.4f}")
        extra = {"stage": 2, "epoch": epoch}
        if suffix.blocks is not None:
            extra["server_suffix_state"] = {"blocks": suffix.blocks.state_dict(), "merger": suffix.merger.state_dict(),
                                            "from_depth": tgt_depth}
        save_checkpoint(run.checkpoint_path("latest"), student, epoch, step, {"decoder": optimizer}, extra=extra)
        save_checkpoint(run.checkpoint_path(f"epoch-{epoch:03d}"), student, epoch, step, extra=extra)
        run.log({"phase": "epoch", "epoch": epoch, "step": step, "time": time.time()})


if __name__ == "__main__":
    main()
