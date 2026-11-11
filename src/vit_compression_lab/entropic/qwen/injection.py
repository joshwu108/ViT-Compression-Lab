"""Helpers for feeding reconstructed features to the frozen Qwen LLM.

``inject_eval.py`` already proves the injection path (identity checks, hooks
that assert the encoder did not run and that the decoder saw exactly the
injected rows). Rather than copying that code, this module loads the script as
a module so its functions (``build_inputs``, ``injected_inputs``,
``prefill_logits``, ``generate``, ``check_identity``, ``timed_generation``,
``load_manifests``) are reused unchanged.
"""

import importlib.util
import sys
from pathlib import Path

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling

REPO_ROOT = Path(__file__).resolve().parents[4]
INJECT_EVAL = REPO_ROOT / "encoders" / "Qwen3.8-vision-compression" / "scripts" / "inject_eval.py"


def load_inject_eval():
    if "inject_eval" in sys.modules:
        return sys.modules["inject_eval"]
    spec = importlib.util.spec_from_file_location("inject_eval", INJECT_EVAL)
    module = importlib.util.module_from_spec(spec)
    sys.modules["inject_eval"] = module
    spec.loader.exec_module(module)
    return module


def encoder_outputs(features, modality="image"):
    """``mm_encoder_outputs`` holding one image's or one video's merged features."""
    return {modality: BaseModelOutputWithPooling(pooler_output=(features,))}


def injected_kwargs(inputs, features, modality="image"):
    """Same text inputs and grid metadata, precomputed features instead of pixels.

    Mirrors inject_eval.injected_inputs() and adds the video case. A fresh dict
    is built on every call because generate() writes into mm_encoder_outputs.
    """
    kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "mm_token_type_ids": inputs["mm_token_type_ids"],
        "mm_encoder_outputs": encoder_outputs(features, modality),
    }
    if modality == "image":
        kwargs["image_grid_thw"] = inputs["image_grid_thw"]
    elif modality == "video":
        kwargs["video_grid_thw"] = inputs["video_grid_thw"]
    else:
        raise ValueError(f"Unknown modality {modality!r}")
    return kwargs


def teacher_forced_inputs(processor, inputs, answer_ids):
    """Append answer tokens to a prompt for one teacher-forced forward pass.

    Returns (kwargs_without_features, mask, labels): ``mask[0, t]`` is true at
    the logit positions that predict an answer token, ``labels`` holds those
    tokens (shifted so labels[0, t] is the token predicted at position t).
    """
    input_ids = inputs["input_ids"]
    device = input_ids.device
    answer_ids = torch.as_tensor(answer_ids, dtype=input_ids.dtype, device=device).reshape(1, -1)
    prompt_len = input_ids.shape[1]
    full_ids = torch.cat([input_ids, answer_ids], dim=1)
    total = full_ids.shape[1]
    extended = {
        "input_ids": full_ids,
        "attention_mask": torch.ones_like(full_ids),
        "mm_token_type_ids": torch.cat(
            [inputs["mm_token_type_ids"], torch.zeros_like(answer_ids)], dim=1),
    }
    for key in ("image_grid_thw", "video_grid_thw"):
        if key in inputs:
            extended[key] = inputs[key]
    mask = torch.zeros(1, total, dtype=torch.bool, device=device)
    mask[0, prompt_len - 1 : total - 1] = True
    labels = torch.zeros(1, total, dtype=input_ids.dtype, device=device)
    labels[0, prompt_len - 1 : total - 1] = answer_ids[0]
    return extended, mask, labels
