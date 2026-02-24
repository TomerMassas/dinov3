from __future__ import annotations

from pathlib import Path
from typing import Literal, List

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def load_paths(txt_path: str) -> List[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def get_backbone(model_arch, which: Literal["teacher", "student"]):
    mdl = model_arch.teacher if which == "teacher" else model_arch.student
    backbone = mdl["backbone"]
    return backbone


@torch.no_grad()
def extract_embeddings(
    model,
    paths: List[str,],
    which: Literal["teacher", "student"] = "teacher",
    batch_size: int = 64,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Returns: E [N, D] float32 on CPU, L2-normalized.
    """
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    backbone = get_backbone(model, which=which)
    was_training = backbone.training
    backbone.eval()

    try:
        outs = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            imgs = torch.stack([tfm(Image.open(p).convert("RGB")) for p in batch_paths]).to(device, non_blocking=True)

            out = backbone(imgs)
            emb = F.normalize(out.float(), dim=-1)  # [B, D]
            outs.append(emb.cpu())
        return torch.cat(outs, dim=0)
    finally:
        backbone.train(was_training)
