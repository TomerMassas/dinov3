from __future__ import annotations

from typing import Dict, Literal, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from train_pictime.eval.embed import load_paths


def get_head(model, which: Literal["teacher", "student"]):
    mdl = model.teacher if which == "teacher" else model.student
    dino_head = mdl["dino_head"]
    return dino_head



@torch.no_grad()
def prototype_utilization(
    model,
    E: torch.Tensor,
    which: Literal["teacher", "student"] = "teacher",
    batch_size: int = 64,
    device: str = "cuda",
    teacher_temp: float = 0.07,   # only used for optional confidence metric
) -> Dict[str, float]:
    """
    Computes prototype usage histogram + entropy/perplexity stats.
    Uses argmax over prototypes as "assignment".
    """
    head = get_head(model, which=which)
    ht = head.training
    head.eval()

    try:
        # We don't know K until first forward
        counts = None
        K = None
        total = 0
        maxprob_sum = 0.0 # confidence of assignments (mean max softmax prob)

        for i in range(0, E.shape[0], batch_size):
            out = E[i:i + batch_size].to(device)  # [B, D]
            logits = head(out)  # [B, K]
            if K is None:
                K = int(logits.shape[-1])
                counts = torch.zeros((K,), dtype=torch.long, device="cpu")

            # assignments
            proto = torch.argmax(logits, dim=-1).detach().cpu()  # [B]
            counts += torch.bincount(proto, minlength=K)
            total += proto.numel()

            # optional: how peaky are predictions
            probs = torch.softmax(logits.float() / teacher_temp, dim=-1)
            maxprob_sum += float(probs.max(dim=-1).values.mean().item()) * proto.numel()

        assert counts is not None and K is not None and total > 0

        p = counts.float() / float(total)
        p_nz = p[p > 0]

        entropy = float((-p_nz * torch.log(p_nz)).sum().item())
        perplexity = float(torch.exp(torch.tensor(entropy)).item())
        used = int((counts > 0).sum().item())

        top1 = float(p.max().item())
        top10 = float(torch.topk(p, k=min(10, K)).values.sum().item())

        # KL(p || uniform)
        kl = float((p_nz * torch.log(p_nz * K)).sum().item())

        return {
            "K": float(K),
            "total_samples": float(total),
            "used_count": float(used),
            "used_frac": float(used / K),
            "entropy": entropy,
            "perplexity": perplexity,
            "top1_share": top1,
            "top10_share": top10,
            "kl_to_uniform": kl,
            "mean_maxprob": float(maxprob_sum / total),
        }
    finally:
        head.train(ht)