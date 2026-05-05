from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import silhouette_score

from train_pictime.finetune.reid_dataset import (
    ReIDCropDataset, build_global_identity_map, get_val_transform,
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
        image_paths,
        bboxes,
        bbox_indices,
        project_ids,
        cluster_ids,
        seed: int,
        device: str = "cuda",
        batch_size: int = 64,
        min_k: int = 2,  # identities need >= 2 samples (1 query + 1 gallery)
        silhouette_max_samples: int = 8000,
    ):
        self.device = device
        self.batch_size = batch_size
        self.seed = seed
        self.silhouette_max_samples = silhouette_max_samples

        if len(image_paths) == 0:
            raise ValueError("No validation samples found")

        labels = build_global_identity_map(project_ids, cluster_ids)
        self.dataset = ReIDCropDataset(
            image_paths, bboxes, bbox_indices, project_ids, cluster_ids, labels,
            transform=get_val_transform(), min_k=min_k,
            centroid_distances_filename=None,
        )

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

        subset = Subset(self.dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False,
        )

        embs, labs = [], []
        for imgs, labels in loader:
            imgs = imgs.to(self.device, non_blocking=True)

            out = backbone(imgs)
            out = out["x_norm_clstoken"] if isinstance(out, dict) else out
            out = proj_head(out.float())
            out = F.normalize(out, dim=-1)

            embs.append(out.cpu())
            labs.append(labels)

        return torch.cat(embs), torch.cat(labs)

    def _compute_silhouette(
        self, embs: torch.Tensor, labels: torch.Tensor,
    ) -> float | None:
        """Silhouette score on a stratified subsample (k per identity)."""
        cap = self.silhouette_max_samples
        if cap <= 0:
            return None

        embs_np = embs.numpy()
        labels_np = labels.numpy()

        # Group indices by identity, discard ids with < 4 samples
        id_to_idx: dict[int, list[int]] = defaultdict(list)
        for i, lab in enumerate(labels_np):
            id_to_idx[int(lab)].append(i)
        id_to_idx = {gid: idx for gid, idx in id_to_idx.items() if len(idx) >= 4}

        if len(id_to_idx) < 2:
            return None

        n_ids = len(id_to_idx)
        k = max(4, cap // n_ids)

        rng = random.Random(self.seed)
        per_id: list[tuple[int, list[int]]] = []
        for gid in sorted(id_to_idx):
            idx = id_to_idx[gid]
            take = min(k, len(idx))
            per_id.append((gid, rng.sample(idx, take) if take < len(idx) else list(idx)))

        # If over cap, shuffle identities and keep whole groups until budget fills
        chosen: list[int] = []
        total = sum(len(v) for _, v in per_id)
        if total <= cap:
            for _, v in per_id:
                chosen.extend(v)
        else:
            rng.shuffle(per_id)
            for _, v in per_id:
                if len(chosen) + len(v) > cap:
                    break
                chosen.extend(v)

        if len(set(labels_np[chosen])) < 2:
            return None

        return float(silhouette_score(embs_np[chosen], labels_np[chosen], metric="cosine"))

    @torch.no_grad()
    def maybe_eval(
        self,
        backbone,
        proj_head,
        iteration: int,
        wandb_run: Any = None,
    ) -> dict[str, float] | None:
        """Run evaluation if due this iteration. Returns metrics or None."""
        if iteration <= 0:
            return None

        was_training_bb = backbone.training
        was_training_ph = proj_head.training

        all_indices = self.query_indices + self.gallery_indices
        all_embs, all_labels = self._extract(backbone, proj_head, all_indices)

        n_query = len(self.query_indices)
        query_embs, query_labels = all_embs[:n_query], all_labels[:n_query]
        gallery_embs, gallery_labels = all_embs[n_query:], all_labels[n_query:]

        metrics = compute_cmc_map(query_embs, query_labels, gallery_embs, gallery_labels)
        sil = self._compute_silhouette(all_embs, all_labels)
        if sil is not None:
            metrics["silhouette"] = sil

        # Restore training mode
        backbone.train(was_training_bb)
        proj_head.train(was_training_ph)

        # Log
        log_dict = {
            "eval/rank1": metrics["rank1"],
            "eval/rank5": metrics["rank5"],
            "eval/rank10": metrics["rank10"],
            "eval/mAP": metrics["mAP"],
        }
        if "silhouette" in metrics:
            log_dict["eval/silhouette"] = metrics["silhouette"]
        log_wandb(wandb_run, log_dict, step=iteration)

        sil_str = f" Silhouette={metrics['silhouette']:.4f}" if "silhouette" in metrics else ""
        print(f"[Eval @ iter {iteration}] Rank-1={metrics['rank1']:.4f} "
              f"Rank-5={metrics['rank5']:.4f} mAP={metrics['mAP']:.4f}{sil_str}")

        return metrics
