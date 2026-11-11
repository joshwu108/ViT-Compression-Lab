"""Rate–distortion, feature-distillation and task-distillation losses.

Stage 1 (Matsubara et al. Eq. 2, per image, averaged over the batch):

    L_1 = 1/2 * ||h - h_hat||^2  (+ deep-supervision terms)  - beta * log p(z_tilde)

``distortion.type``:
    mse               plain squared error (the paper)
    standardized_mse  squared error divided by per-channel teacher std
                      (deviation: needed because merger channel 3994 is ~61x RMS)
``distortion.reduction``: sum_per_image (paper) | mean_per_element
``rate.reduction``:       sum_per_image (paper, nats) | per_visual_token
"""

import math

import torch
import torch.nn.functional as F


def segment_ids(counts, device):
    return torch.repeat_interleave(
        torch.arange(len(counts), device=device), torch.tensor(counts, device=device)
    )


def per_image_sums(per_token, counts):
    """per_token: [N] -> [B] sums using the packed layout."""
    ids = segment_ids(counts, per_token.device)
    return torch.zeros(len(counts), device=per_token.device, dtype=per_token.dtype).index_add_(0, ids, per_token)


def distortion_loss(recon, target, counts, cfg, channel_std=None):
    recon = recon.float()
    target = target.float()
    error = recon - target
    kind = cfg.get("type", "mse")
    if kind == "standardized_mse":
        if channel_std is None:
            raise ValueError("standardized_mse needs per-channel teacher std (data.channel_stats)")
        error = error / channel_std.to(error.device, error.dtype).clamp_min(1e-6)
    elif kind != "mse":
        raise ValueError(f"Unknown distortion type {kind!r}")
    squared = error.pow(2)
    per_token = squared.sum(-1)
    reduction = cfg.get("reduction", "sum_per_image")
    if reduction == "sum_per_image":
        loss = 0.5 * per_image_sums(per_token, counts).mean()
    elif reduction == "mean_per_element":
        loss = 0.5 * squared.mean()
    else:
        raise ValueError(f"Unknown distortion reduction {reduction!r}")
    cosine = F.cosine_similarity(recon, target, dim=-1)
    cosine_weight = float(cfg.get("cosine_weight", 0.0))
    if cosine_weight:
        loss = loss + cosine_weight * (1 - cosine).mean()
    with torch.no_grad():
        plain_mse = (recon - target).pow(2).mean()
    return {"loss": loss, "mse": plain_mse, "cosine": cosine.mean().detach(),
            "cosine_min": cosine.min().detach()}


def rate_loss(likelihood, latent_counts, n_visual_tokens, cfg):
    nats_per_symbol = -torch.log(likelihood.float())
    per_token = nats_per_symbol.sum(-1)
    nats_per_image = per_image_sums(per_token, latent_counts)
    total_bits = nats_per_image.sum() / math.log(2.0)
    reduction = cfg.get("reduction", "sum_per_image")
    if reduction == "sum_per_image":
        loss = nats_per_image.mean()
    elif reduction == "per_visual_token":
        loss = nats_per_image.sum() / float(n_visual_tokens)
    else:
        raise ValueError(f"Unknown rate reduction {reduction!r}")
    return {"loss": loss, "bits": total_bits.detach(),
            "bits_per_visual_token": (total_bits / float(n_visual_tokens)).detach(),
            "bits_per_image": nats_per_image.detach() / math.log(2.0)}


def stage1_loss(out, targets, counts_by_level, n_visual_tokens, config, channel_stats=None):
    """``targets``: {point: tensor}; ``out``: EntropicStudent forward dict plus
    ``out['deep'] = {point: tensor}`` for reconstructions propagated through the
    frozen suffix. ``channel_stats``: {point: std tensor}."""
    split = config.split
    tgt_cfg = config["distortion"]
    std = (channel_stats or {}).get(split.decoder_target)
    d = distortion_loss(out["recon"], targets[split.decoder_target],
                        counts_by_level[split.target_level], tgt_cfg, std)
    total = d["loss"]
    log = {"distortion": d["loss"].detach(), "mse": d["mse"], "cosine": d["cosine"],
           "cosine_min": d["cosine_min"]}
    for term in config["deep_supervision"]:
        point = term["point"]
        level = "token" if point == "merger" else "patch"
        term_cfg = {**tgt_cfg, **{k: v for k, v in term.items() if k not in ("point", "weight")}}
        dd = distortion_loss(out["deep"][point], targets[point], counts_by_level[level],
                             term_cfg, (channel_stats or {}).get(point))
        total = total + float(term["weight"]) * dd["loss"]
        log[f"deep/{point}/mse"] = dd["mse"]
        log[f"deep/{point}/cosine"] = dd["cosine"]
    r = rate_loss(out["likelihood"], out["latent_counts"], n_visual_tokens, config["rate"])
    beta = float(config["rate"]["beta"])
    total = total + beta * r["loss"]
    log.update({"rate": r["loss"].detach(), "bits": r["bits"],
                "bits_per_visual_token": r["bits_per_visual_token"], "loss": total.detach()})
    return total, log


def kd_token_loss(student_logits, teacher_logits, mask, tau=1.0):
    """tau^2 * KL(teacher || student) averaged over positions where mask is true."""
    s = F.log_softmax(student_logits.float() / tau, dim=-1)
    t = F.log_softmax(teacher_logits.float() / tau, dim=-1)
    kl = (t.exp() * (t - s)).sum(-1)
    mask = mask.to(kl.dtype)
    return tau**2 * (kl * mask).sum() / mask.sum().clamp_min(1.0)


def stage2_loss(student_logits, teacher_logits, labels, mask, alpha=0.0, tau=1.0):
    """Adaptation of Eq. 7: alpha * CE(labels) + (1 - alpha) * tau^2 * KL.

    ``labels`` are the next tokens at the masked positions (teacher-forced).
    With alpha = 0 (default) only the distributional term is used, which needs
    no ground-truth answers.
    """
    kd = kd_token_loss(student_logits, teacher_logits, mask, tau)
    log = {"kd": kd.detach()}
    total = (1 - alpha) * kd
    if alpha > 0:
        ce = F.cross_entropy(student_logits.float()[mask], labels[mask])
        total = total + alpha * ce
        log["ce"] = ce.detach()
    log["loss"] = total.detach()
    return total, log
