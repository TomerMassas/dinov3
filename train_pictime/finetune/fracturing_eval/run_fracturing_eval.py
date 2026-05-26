"""Fracturing eval — measure how often the finetuned model splits a single
true identity into multiple predicted sub-clusters.

Per project (from HELD_OUT_PROJECTS_FILE):
  1. Load clusters_fixed.json (ground truth) + detections.json
  2. Drop GT cluster_id == -1; drop GT clusters with < MIN_GT_CLUSTER_SIZE crops
  3. Embed surviving crops with the finetuned backbone + projection head
  4. HDBSCAN cluster them (same params as production realworld_eval)
  5. For each surviving GT cluster, compute:
       - fracturing_count = # distinct non-noise predicted IDs (0 if all noise)
       - sub_cluster_sizes (descending), sub_cluster_pcts (% of non-noise crops)

Output: summary.json with one entry per (project, gt_cluster). Plots
generated separately by plot.py.

Usage:
    python3 -m train_pictime.finetune.fracturing_eval.run_fracturing_eval
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from dinov3.configs import setup_job
from train_pictime.finetune.finetune_reid import (
    CFG_PATH, REPO_ROOT, build_projection_head, load_backbone,
)
from train_pictime.finetune.realworld_eval.cluster_test_set import (
    cluster_embeddings_hdbscan,
    crop_bbox,
    embed_project,
)
from train_pictime.finetune.reeval_tiered import find_best_silhouette_ckpt
from train_pictime.finetune.reid_dataset import get_val_transform
from train_pictime.finetune.fracturing_eval.config import (
    DATASET_ROOT,
    FINETUNE_VERSION_DIR,
    HELD_OUT_PROJECTS_FILE,
    MIN_GT_CLUSTER_SIZE,
    OUTPUT_BASE,
)
from train_pictime.finetune.fracturing_eval.plot import (
    plot_fracturing_histogram,
    plot_subcluster_percentages,
)


# ---------------------------------------------------------------------------
# Held-out project list
# ---------------------------------------------------------------------------

def load_held_out_pids(path: str) -> list[str]:
    """Accepts bare list ['pid', ...] or {'project_ids': ['pid', ...]}.
    Mirrors the EXCLUDE_FILE convention in realworld_eval."""
    with open(path, "r") as f:
        payload = json.load(f)
    pids = payload if isinstance(payload, list) else payload.get("project_ids", [])
    if not pids:
        raise RuntimeError(f"No project_ids in {path}")
    return list(pids)


# ---------------------------------------------------------------------------
# Per-project dataset (active crops only)
# ---------------------------------------------------------------------------

class ActiveCropDataset(Dataset):
    """Flat dataset of crops for ONE project, restricted to GT-labeled entries
    that survived the -1 + MIN_GT_CLUSTER_SIZE filters.

    Items: (entry_idx, crop_tensor, is_valid) — same interface as
    realworld_eval.ProjectCropDataset, so embed_project consumes it unchanged.
    """

    def __init__(self,
                 project_dir: str,
                 active_entries: list[tuple[str, int, list, int]],
                 transform,
                ):
        self.project_dir = project_dir
        self.entries = active_entries  # (fname, bbox_idx, bbox, gt_cluster_id)
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        fname, bbox_idx, bbox, _gid = self.entries[idx]
        img_path = os.path.join(self.project_dir, "images", fname)
        try:
            img = Image.open(img_path).convert("RGB")
            crop = crop_bbox(img, bbox)
            if crop.size[0] < 4 or crop.size[1] < 4:
                return idx, torch.zeros(3, 224, 224), False
            return idx, self.transform(crop), True
        except Exception:
            return idx, torch.zeros(3, 224, 224), False


# ---------------------------------------------------------------------------
# Per-project pipeline
# ---------------------------------------------------------------------------

def build_active_entries(clusters_fixed: dict,
                         detections: dict,
                         min_gt_size: int,
                        ) -> tuple[list[tuple[str, int, list, int]], dict[int, int]]:
    """Build the (fname, bbox_idx, bbox, gt_cluster_id) list for surviving crops.

    Drops:
      - GT entries with cluster_id == -1
      - GT clusters with < min_gt_size members
      - Entries where (fname, bbox_idx) lacks a matching detection bbox

    Returns:
        active_entries: list ready to feed ActiveCropDataset
        surviving_gt_sizes: dict[gt_cluster_id, size_after_detection_pairing]
    """
    gt_to_crops: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for fname, entries in clusters_fixed.items():
        for e in entries:
            gid = int(e["cluster_id"])
            if gid == -1:
                continue
            gt_to_crops[gid].append((fname, int(e["bbox_index"])))

    surviving = {gid: crops for gid, crops in gt_to_crops.items() if len(crops) >= min_gt_size}

    active: list[tuple[str, int, list, int]] = []
    surviving_gt_sizes: dict[int, int] = {}
    for gid, crops in surviving.items():
        kept = 0
        for fname, bbox_idx in crops:
            dets = detections.get(fname)
            if dets is None or bbox_idx >= len(dets):
                continue
            bbox = dets[bbox_idx]["bbox"]
            active.append((fname, bbox_idx, bbox, gid))
            kept += 1
        if kept > 0:
            surviving_gt_sizes[gid] = kept

    # Defensive: drop entries belonging to GT clusters that lost everyone
    active = [e for e in active if e[3] in surviving_gt_sizes]
    return active, surviving_gt_sizes


def compute_fracturing_entries(pid: str,
                               surviving_gt_sizes: dict[int, int],
                               active_entries: list[tuple[str, int, list, int]],
                               valid_indices: list[int],
                               pred_labels: np.ndarray,
                              ) -> list[dict]:
    """Per surviving GT cluster, compute fracturing metrics from the predicted
    cluster labels (HDBSCAN output)."""
    gt_to_preds: dict[int, list[int]] = defaultdict(list)
    for emb_idx, active_idx in enumerate(valid_indices):
        gid = active_entries[active_idx][3]
        gt_to_preds[gid].append(int(pred_labels[emb_idx]))

    results = []
    for gid, gt_size in surviving_gt_sizes.items():
        all_preds = gt_to_preds.get(gid, [])
        n_valid_embedded = len(all_preds)

        # Drop predicted -1 (model abstained / noise)
        non_noise = [p for p in all_preds if p != -1]
        n_noise_dropped = n_valid_embedded - len(non_noise)

        if not non_noise:
            results.append({"project_id": pid,
                            "gt_cluster_id": gid,
                            "gt_size": gt_size,
                            "n_valid_embedded": n_valid_embedded,
                            "n_noise_dropped": n_noise_dropped,
                            "fracturing_count": 0,
                            "sub_cluster_sizes": [],
                            "sub_cluster_pcts": [],
                           })
            continue

        counts = Counter(non_noise)
        sizes_sorted = sorted(counts.values(), reverse=True)
        total = sum(sizes_sorted)
        pcts_sorted = [round(s * 100.0 / total, 4) for s in sizes_sorted]

        results.append({"project_id": pid,
                        "gt_cluster_id": gid,
                        "gt_size": gt_size,
                        "n_valid_embedded": n_valid_embedded,
                        "n_noise_dropped": n_noise_dropped,
                        "fracturing_count": len(counts),
                        "sub_cluster_sizes": sizes_sorted,
                        "sub_cluster_pcts": pcts_sorted,
                       })
    return results


def process_project(pid: str,
                    dataset_root: str,
                    backbone,
                    proj_head,
                    transform,
                    device: str,
                    min_gt_size: int,
                   ) -> tuple[str, list[dict]]:
    """Returns (status, entries). Status is 'done' or a skip reason."""
    project_dir = os.path.join(dataset_root, pid)
    clusters_fixed_path = os.path.join(project_dir, "clusters_fixed.json")
    detections_path = os.path.join(project_dir, "detections.json")

    if not os.path.exists(clusters_fixed_path):
        return "skip_no_clusters_fixed", []
    if not os.path.exists(detections_path):
        return "skip_no_detections", []

    with open(clusters_fixed_path, "r") as f:
        clusters_fixed = json.load(f)
    with open(detections_path, "r") as f:
        detections = json.load(f)

    active_entries, surviving_gt_sizes = build_active_entries(clusters_fixed,
                                                              detections,
                                                              min_gt_size,
                                                             )
    if not surviving_gt_sizes:
        return "skip_no_valid_gt_clusters", []
    if len(active_entries) < 2:
        return "skip_too_few_crops", []

    dataset = ActiveCropDataset(project_dir, active_entries, transform)
    embeddings, valid_indices = embed_project(backbone, proj_head, dataset, device)
    if len(embeddings) < 2:
        return "skip_no_valid_embeddings", []

    pred_labels = cluster_embeddings_hdbscan(embeddings)
    entries = compute_fracturing_entries(pid,
                                         surviving_gt_sizes,
                                         active_entries,
                                         valid_indices,
                                         pred_labels,
                                        )
    return "done", entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = OmegaConf.load(CFG_PATH)
    setup_job(output_dir=None, seed=cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Find best finetune ckpt ---
    ckpt_dir = Path(FINETUNE_VERSION_DIR) / "ckpt"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"FINETUNE_VERSION_DIR has no ckpt/: {ckpt_dir}")
    ckpt_path, it, train_sil = find_best_silhouette_ckpt(ckpt_dir)
    version_name = Path(FINETUNE_VERSION_DIR).name
    output_dir = Path(OUTPUT_BASE) / f"{version_name}_iter{it}_sil{train_sil:.4f}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nFinetune ckpt:  {ckpt_path}")
    print(f"  iteration   = {it}")
    print(f"  train sil   = {train_sil:.4f}")
    print(f"Output dir:     {output_dir}")

    # --- Held-out projects ---
    pids = load_held_out_pids(HELD_OUT_PROJECTS_FILE)
    print(f"\nHeld-out projects: {len(pids)} from {HELD_OUT_PROJECTS_FILE}")

    # --- Load model ---
    print("\nLoading backbone...")
    pretrain_cfg_path = str(REPO_ROOT / cfg.pretrain_config)
    backbone, embed_dim = load_backbone(pretrain_cfg_path,
                                        cfg.pretrained_weights,
                                        cfg.backbone_which,
                                       )
    backbone.cuda()
    print(f"Backbone loaded, embed_dim={embed_dim}")

    print("\nLoading finetune state...")
    state = torch.load(ckpt_path, map_location="cuda")
    backbone.load_state_dict(state["backbone_state_dict"], strict=True)
    proj_head = build_projection_head(embed_dim,
                                      cfg.proj_hidden_dim,
                                      cfg.proj_output_dim,
                                     ).cuda()
    proj_head.load_state_dict(state["proj_head_state_dict"], strict=True)
    backbone.eval()
    proj_head.eval()
    del state
    torch.cuda.empty_cache()
    print("Finetune state loaded.")

    # --- Per-project loop ---
    transform = get_val_transform()
    all_entries: list[dict] = []
    status_counts: dict[str, int] = defaultdict(int)


    for pid in tqdm(pids, desc="Projects"):
        try:
            status, entries = process_project(pid,
                                              DATASET_ROOT,
                                              backbone,
                                              proj_head,
                                              transform,
                                              device,
                                              MIN_GT_CLUSTER_SIZE,
                                             )
        except Exception as e:
            tqdm.write(f"[{pid}] ERROR: {e!r}")
            status_counts["error"] += 1
            continue
        status_counts[status] += 1
        all_entries.extend(entries)

    # --- Write summary ---
    summary = {
        "model_version": version_name,
        "ckpt_path": str(ckpt_path),
        "iteration": int(it),
        "train_silhouette": float(train_sil),
        "held_out_projects_file": HELD_OUT_PROJECTS_FILE,
        "dataset_root": DATASET_ROOT,
        "min_gt_cluster_size": MIN_GT_CLUSTER_SIZE,
        "status_counts": dict(status_counts),
        "n_gt_clusters": len(all_entries),
        "entries": all_entries,
    }
    summary_path = output_dir / "summary.json"
    tmp = str(summary_path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, summary_path)
    print(f"\nWrote summary: {summary_path}")

    # --- Aggregate stats ---
    if all_entries:
        fracturing_counts = [e["fracturing_count"] for e in all_entries]
        n_perfect = sum(1 for f in fracturing_counts if f == 1)
        n_fractured = sum(1 for f in fracturing_counts if f > 1)
        n_all_noise = sum(1 for f in fracturing_counts if f == 0)
        mean_frac = float(np.mean(fracturing_counts))
        print(f"\n===== Fracturing Summary =====")
        print(f"GT clusters analyzed:  {len(all_entries)}")
        print(f"Perfectly grouped:     {n_perfect}  ({100.0*n_perfect/len(all_entries):.1f}%)")
        print(f"Fractured (>1):        {n_fractured}  ({100.0*n_fractured/len(all_entries):.1f}%)")
        print(f"Model-noise-only (0):  {n_all_noise}")
        print(f"Mean fracturing:       {mean_frac:.2f}")
        print(f"Max fracturing:        {max(fracturing_counts)}")
    print(f"\nStatus counts: {dict(status_counts)}")

    # --- Plots ---
    if all_entries:
        title_suffix = f" — {version_name} iter{it} (sil={train_sil:.3f})"
        plot_fracturing_histogram(all_entries,
                                  output_dir / "plot_1_histogram.png",
                                  title_suffix,
                                 )
        plot_subcluster_percentages(all_entries,
                                    output_dir / "plot_2_per_rank.png",
                                    title_suffix,
                                   )
    print(f"\nDone. To re-plot without re-running: "
          f"python3 -m train_pictime.finetune.fracturing_eval.plot")


if __name__ == "__main__":
    main()
