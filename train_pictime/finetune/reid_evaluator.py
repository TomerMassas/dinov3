from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from train_pictime.finetune.reid_dataset import (
    ReIDCropDataset, ReIDSample, load_project, build_global_identity_map, get_val_transform,
)
from train_pictime.wandb_logger import log_wandb


# ---------------------------------------------------------------------------
# CMC / mAP computation
# ---------------------------------------------------------------------------

def compute_cmc_map(
    query_embs: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_embs: torch.Tensor,
    gallery_labels: torch.Tensor,
) -> dict[str, float]:
    """Compute CMC (Rank-1/5/10) and mAP from query/gallery embeddings.

    Args:
        query_embs:    [Q, D] L2-normalized
        query_labels:  [Q] integer identity IDs
        gallery_embs:  [G, D] L2-normalized
        gallery_labels:[G] integer identity IDs

    Returns:
        {"rank1": float, "rank5": float, "rank10": float, "mAP": float}
    """
    # Cosine similarity (embeddings are L2-normed)
    sim = query_embs @ gallery_embs.T  # [Q, G]
    indices = sim.argsort(dim=1, descending=True)  # [Q, G] sorted by similarity

    gallery_labels_expanded = gallery_labels.unsqueeze(0).expand_as(sim)  # [Q, G]
    sorted_labels = gallery_labels_expanded.gather(1, indices)  # [Q, G]
    matches = (sorted_labels == query_labels.unsqueeze(1))  # [Q, G] bool

    Q = query_embs.shape[0]
    cmc = torch.zeros(gallery_embs.shape[0])
    all_ap = []

    for i in range(Q):
        match_positions = matches[i].nonzero(as_tuple=False).squeeze(1)
        if match_positions.numel() == 0:
            continue

        # CMC: first correct match position
        first_match = match_positions[0].item()
        cmc[first_match:] += 1

        # AP: average precision for this query
        n_correct = match_positions.numel()
        precisions = torch.arange(1, n_correct + 1, dtype=torch.float32) / (match_positions.float() + 1)
        all_ap.append(precisions.mean().item())

    cmc = cmc / Q  # cumulative -> percentage

    return {
        "rank1": cmc[0].item() if len(cmc) > 0 else 0.0,
        "rank5": cmc[min(4, len(cmc) - 1)].item() if len(cmc) > 4 else cmc[-1].item(),
        "rank10": cmc[min(9, len(cmc) - 1)].item() if len(cmc) > 9 else cmc[-1].item(),
        "mAP": sum(all_ap) / len(all_ap) if all_ap else 0.0,
    }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ReIDEvaluator:
    """In-process ReID evaluator. Call maybe_eval() from the training loop."""

    def __init__(
        self,
        val_project_dirs: list[Path],
        eval_every: int,
        seed: int,
        device: str = "cuda",
        batch_size: int = 64,
        min_k: int = 2,  # identities need >= 2 samples (1 query + 1 gallery)
    ):
        self.eval_every = eval_every
        self.device = device
        self.batch_size = batch_size

        # Load val data
        all_samples: list[ReIDSample] = []
        for d in val_project_dirs:
            all_samples.extend(load_project(d))

        if not all_samples:
            raise ValueError("No validation samples found")

        id_map, labels = build_global_identity_map(all_samples)
        self.dataset = ReIDCropDataset(all_samples, labels, transform=get_val_transform(), min_k=min_k)

        # Split query / gallery: 1 query per identity, rest gallery
        rng = random.Random(seed)
        identity_to_indices = self.dataset.identity_to_indices

        self.query_indices = []
        self.gallery_indices = []
        for gid, idx_list in identity_to_indices.items():
            if len(idx_list) < min_k:
                continue
            chosen = rng.choice(idx_list)
            self.query_indices.append(chosen)
            self.gallery_indices.extend([i for i in idx_list if i != chosen])

        print(f"ReID Eval: {len(self.query_indices)} queries, {len(self.gallery_indices)} gallery, "
              f"{len(identity_to_indices)} identities")

    @torch.no_grad()
    def _extract(self, backbone, proj_head, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract embeddings for given dataset indices."""
        backbone.eval()
        proj_head.eval()

        embs, labs = [], []
        for i in range(0, len(indices), self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
            imgs = torch.stack([self.dataset[j][0] for j in batch_idx]).to(self.device)
            labels = torch.tensor([self.dataset[j][1] for j in batch_idx])

            out = backbone(imgs)
            out = out["x_norm_clstoken"] if isinstance(out, dict) else out
            out = proj_head(out)
            out = F.normalize(out.float(), dim=-1)

            embs.append(out.cpu())
            labs.append(labels)

        return torch.cat(embs), torch.cat(labs)

    @torch.no_grad()
    def maybe_eval(
        self,
        backbone,
        proj_head,
        iteration: int,
        wandb_run: Any = None,
    ) -> dict[str, float] | None:
        """Run evaluation if due this iteration. Returns metrics or None."""
        if self.eval_every <= 0 or iteration <= 0 or iteration % self.eval_every != 0:
            return None

        was_training_bb = backbone.training
        was_training_ph = proj_head.training

        query_embs, query_labels = self._extract(backbone, proj_head, self.query_indices)
        gallery_embs, gallery_labels = self._extract(backbone, proj_head, self.gallery_indices)

        metrics = compute_cmc_map(query_embs, query_labels, gallery_embs, gallery_labels)

        # Restore training mode
        backbone.train(was_training_bb)
        proj_head.train(was_training_ph)

        # Log
        log_wandb(wandb_run, {
            "eval/rank1": metrics["rank1"],
            "eval/rank5": metrics["rank5"],
            "eval/rank10": metrics["rank10"],
            "eval/mAP": metrics["mAP"],
        }, step=iteration)

        print(f"[Eval @ iter {iteration}] Rank-1={metrics['rank1']:.4f} "
              f"Rank-5={metrics['rank5']:.4f} mAP={metrics['mAP']:.4f}")

        return metrics
