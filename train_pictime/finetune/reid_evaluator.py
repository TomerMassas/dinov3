from __future__ import annotations

import math
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

def _tier_suffix(tier: float) -> str:
    return f"top{int(round(tier * 100))}"


class ReIDEvaluator:
    """In-process ReID evaluator. Call maybe_eval() from the training loop.

    Tiered eval (when `centroid_distances_filename` is given): each tier T in
    `eval_tiers` restricts every identity's pool to the top-T fraction of crops
    closest to the cluster centroid (V11-derived), then runs Q/G split + metrics
    on the restricted pool. cluster_id=-1 identities are dropped (no centroid).
    All tiers share the same identity universe so comparisons are clean.
    """

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
        centroid_distances_filename: str | None = None,
        eval_tiers: list[float] | None = None,
    ):
        self.device = device
        self.batch_size = batch_size
        self.seed = seed
        self.silhouette_max_samples = silhouette_max_samples
        self.min_k = min_k

        if len(image_paths) == 0:
            raise ValueError("No validation samples found")

        # Tiered eval requires centroid distances. Drop cluster_id=-1 (no centroid)
        # so all tiers share the same identity universe.
        if centroid_distances_filename is not None:
            keep_mask = cluster_ids != -1
            n_dropped = int((~keep_mask).sum())
            if n_dropped > 0:
                print(f"ReID Eval: dropping {n_dropped} cluster_id=-1 samples (no centroid)")
            image_paths = image_paths[keep_mask]
            bboxes = bboxes[keep_mask]
            bbox_indices = bbox_indices[keep_mask]
            project_ids = project_ids[keep_mask]
            cluster_ids = cluster_ids[keep_mask]
            self.eval_tiers = sorted(eval_tiers) if eval_tiers else [1.0]
        else:
            # Backwards-compat: single full-pool eval, no tier filtering.
            self.eval_tiers = [1.0]

        labels = build_global_identity_map(project_ids, cluster_ids)
        self.dataset = ReIDCropDataset(
            image_paths, bboxes, bbox_indices, project_ids, cluster_ids, labels,
            transform=get_val_transform(), min_k=min_k,
            centroid_distances_filename=centroid_distances_filename,
        )

        # Per-tier (query_indices, gallery_indices). Same RNG seed across tiers
        # so the random query pick is consistent given the same pool.
        rng = random.Random(seed)
        self.tier_to_qg: dict[float, tuple[list[int], list[int]]] = {}

        if centroid_distances_filename is not None:
            sorted_indices_map = self.dataset.identity_to_sorted_indices
            assert sorted_indices_map is not None, \
                "Tiered eval requires identity_to_sorted_indices on the dataset"
            for tier in self.eval_tiers:
                q_indices, g_indices = [], []
                for gid, sorted_idx_list in sorted_indices_map.items():
                    keep = max(min_k, math.ceil(tier * len(sorted_idx_list)))
                    pool = sorted_idx_list[:keep]
                    if len(pool) < min_k:
                        continue
                    chosen = rng.choice(pool)
                    q_indices.append(chosen)
                    g_indices.extend([i for i in pool if i != chosen])
                self.tier_to_qg[tier] = (q_indices, g_indices)
                print(f"ReID Eval [tier={_tier_suffix(tier)}]: "
                      f"{len(q_indices)} queries, {len(g_indices)} gallery, "
                      f"{len(q_indices)} identities")
        else:
            q_indices, g_indices = [], []
            for gid, idx_list in self.dataset.identity_to_indices.items():
                if len(idx_list) < min_k:
                    continue
                chosen = rng.choice(idx_list)
                q_indices.append(chosen)
                g_indices.extend([i for i in idx_list if i != chosen])
            self.tier_to_qg[1.0] = (q_indices, g_indices)
            print(f"ReID Eval: {len(q_indices)} queries, {len(g_indices)} gallery, "
                  f"{len(q_indices)} identities")

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
        """Run evaluation if due this iteration. Returns flat dict or None.

        Returns per-tier keys like `mAP_top50`, `silhouette_top100`, plus
        a top-level `silhouette` mirror of the tier-100 value (for the
        BestCheckpointTracker which reads `metrics["silhouette"]`).
        """
        if iteration <= 0:
            return None

        was_training_bb = backbone.training
        was_training_ph = proj_head.training

        # Extract once over the union of all tier indices (saves repeated forward passes).
        union: set[int] = set()
        for q, g in self.tier_to_qg.values():
            union.update(q)
            union.update(g)
        all_indices = sorted(union)
        all_embs, all_labels = self._extract(backbone, proj_head, all_indices)
        idx_to_row: dict[int, int] = {idx: i for i, idx in enumerate(all_indices)}

        log_dict: dict[str, float] = {}
        result: dict[str, float] = {}
        print_lines: list[str] = []

        for tier in self.eval_tiers:
            q_idxs, g_idxs = self.tier_to_qg[tier]
            if not q_idxs or not g_idxs:
                continue
            q_rows = torch.tensor([idx_to_row[i] for i in q_idxs], dtype=torch.long)
            g_rows = torch.tensor([idx_to_row[i] for i in g_idxs], dtype=torch.long)

            q_embs = all_embs[q_rows]
            q_labels = all_labels[q_rows]
            g_embs = all_embs[g_rows]
            g_labels = all_labels[g_rows]

            metrics = compute_cmc_map(q_embs, q_labels, g_embs, g_labels)

            tier_embs = torch.cat([q_embs, g_embs])
            tier_labels = torch.cat([q_labels, g_labels])
            sil = self._compute_silhouette(tier_embs, tier_labels)

            suffix = _tier_suffix(tier)
            log_dict[f"eval/rank1_{suffix}"] = metrics["rank1"]
            log_dict[f"eval/rank5_{suffix}"] = metrics["rank5"]
            log_dict[f"eval/rank10_{suffix}"] = metrics["rank10"]
            log_dict[f"eval/mAP_{suffix}"] = metrics["mAP"]
            log_dict[f"eval/n_identities_{suffix}"] = float(len(q_idxs))
            if sil is not None:
                log_dict[f"eval/silhouette_{suffix}"] = sil

            result[f"rank1_{suffix}"] = metrics["rank1"]
            result[f"rank5_{suffix}"] = metrics["rank5"]
            result[f"rank10_{suffix}"] = metrics["rank10"]
            result[f"mAP_{suffix}"] = metrics["mAP"]
            result[f"n_identities_{suffix}"] = float(len(q_idxs))
            if sil is not None:
                result[f"silhouette_{suffix}"] = sil

            sil_str = f" Sil={sil:.4f}" if sil is not None else ""
            print_lines.append(
                f"  [{suffix}] mAP={metrics['mAP']:.4f} R1={metrics['rank1']:.4f} "
                f"R5={metrics['rank5']:.4f} R10={metrics['rank10']:.4f}{sil_str} "
                f"(n_id={len(q_idxs)})"
            )

        # Restore training mode
        backbone.train(was_training_bb)
        proj_head.train(was_training_ph)

        log_wandb(wandb_run, log_dict, step=iteration)

        # Tracker compat: surface tier-100 silhouette as top-level `silhouette`.
        if "silhouette_top100" in result:
            result["silhouette"] = result["silhouette_top100"]

        print(f"[Eval @ iter {iteration}]")
        for line in print_lines:
            print(line)

        return result if result else None
