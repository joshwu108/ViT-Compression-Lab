"""Run directories, metric logging, and student checkpoints.

    runs/<name>/config.json          the resolved StudentConfig
    runs/<name>/run_info.json        software, hardware, teacher revision, parameter counts
    runs/<name>/metrics.jsonl        one JSON line per logged step or evaluation
    runs/<name>/checkpoints/latest.pt, best.pt, epoch-XXX.pt

A checkpoint holds the config, the student state dict (including the entropy
model's integer CDF tables once update() has run), optimiser states, and the
epoch/step counters. Loading rebuilds the student from the stored config.
"""

import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..run_checkpoint import append_record, write_json
from .bottleneck import EntropicStudent
from .config import StudentConfig, VisionDims


def now():
    return datetime.now(timezone.utc).isoformat()


class RunDir:
    def __init__(self, root, name, resume=False):
        self.path = Path(root) / name
        if self.path.exists() and not resume and any(self.path.iterdir()):
            raise FileExistsError(f"{self.path} exists; pass --resume or choose a new name")
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "checkpoints").mkdir(exist_ok=True)

    def write_config(self, config):
        (self.path / "config.json").write_text(config.to_json())

    def write_run_info(self, info):
        write_json(self.path / "run_info.json", info)

    def log(self, record):
        append_record(self.path / "metrics.jsonl", {"time": now(), **record})

    def checkpoint_path(self, tag):
        return self.path / "checkpoints" / f"{tag}.pt"


def software_info():
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
    }


def hardware_info(device):
    if device is None or torch.device(device).type != "cuda":
        return {"device": str(device), "cpu": platform.processor() or platform.machine()}
    props = torch.cuda.get_device_properties(torch.device(device))
    return {"device": str(device), "gpu": props.name, "gpu_memory_bytes": props.total_memory,
            "cpu": platform.processor() or platform.machine()}


def save_checkpoint(path, student, epoch, step, optimizers=None, extra=None):
    payload = {
        "format": "entropic-student-v1",
        "config": student.config.raw,
        "dims": student.dims.__dict__,
        "state_dict": student.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "saved_at": now(),
    }
    if optimizers:
        payload["optimizers"] = {k: v.state_dict() for k, v in optimizers.items()}
    if extra:
        payload["extra"] = extra
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(path, map_location="cpu"):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format") != "entropic-student-v1":
        raise ValueError(f"{path} is not an Entropic Student checkpoint")
    return payload


def student_from_checkpoint(payload, strict=True):
    dims = VisionDims(**payload["dims"])
    config = StudentConfig.from_dict(payload["config"], dims)
    student = EntropicStudent(config, dims)
    state = payload["state_dict"]
    # Integer CDF tables are sized at update() time; resize buffers before loading.
    for name in ("_quantized_cdf", "_offset", "_cdf_length"):
        key = f"entropy.{name}"
        if key in state:
            getattr(student.entropy, name).resize_(state[key].shape)
    student.load_state_dict(state, strict=strict)
    return student, config
