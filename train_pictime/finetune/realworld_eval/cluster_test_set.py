"""Real-world ReID evaluation — cluster 100 unseen projects with the
current-best finetune ckpt, write HDBSCAN clusters + crop JPGs ready for
the standalone HTML viewer.

Pipeline:
1. Sample 100 projects from Portraits[26] that are NOT in
   single_cluster_projects_v2.json (the train+eval pool). Filter to
   projects with >= MIN_BBOXES detections so the clusters are meaningful.
   Test set is FROZEN on first run: subsequent ckpt evals load the same
   test_projects.json and only re-embed/re-cluster.
2. Load V11 backbone once + Trial-2-winner finetune state (backbone +
   proj_head from `ckpt_iter*_sil*.pt`).
3. Per project: read detections.json, crop each bbox, forward through
   backbone + proj_head + F.normalize → [N, 128] L2-normed embeddings.
   HDBSCAN cluster (same params as cluster_embeddings.py).
4. Save per-ckpt clusters JSON (cluster_embeddings.py schema) under
   <OUTPUT_BASE>/V<n>_iter<N>_sil<S>/clusters/. Save cropped JPGs (for
   the HTML viewer) under the GLOBAL <OUTPUT_BASE>/crops/ — they're
   model-independent so they're shared across all ckpt evals.

Usage:
    python3 -m train_pictime.finetune.realworld_eval.cluster_test_set
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    from hdbscan import HDBSCAN

from dinov3.configs import setup_job
from train_pictime.finetune.finetune_reid import (
    CFG_PATH, REPO_ROOT, build_projection_head, load_backbone,
)
from train_pictime.finetune.reeval_tiered import find_best_silhouette_ckpt
from train_pictime.finetune.reid_dataset import get_val_transform
from train_pictime.finetune.realworld_eval.config import (
    BATCH_SIZE, DATASET_ROOT, EXCLUDE_FILE, FINETUNE_VERSION_DIR,
    HDBSCAN_METRIC, HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES,
    MIN_BBOXES, N_SAMPLE, NUM_WORKERS, OUTPUT_BASE, SEED,
    VIEWER_CROP_JPEG_QUALITY, VIEWER_CROP_MAX_EDGE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def crop_bbox(img_pil: Image.Image, bbox_normalized) -> Image.Image:
    """Crop a normalized [0,1] bbox from a PIL image. Mirrors extract_embeddings.crop_bbox."""
    w, h = img_pil.size
    x1, y1, x2, y2 = bbox_normalized
    px1 = max(0, min(int(x1 * w), w))
    py1 = max(0, min(int(y1 * h), h))
    px2 = max(0, min(int(x2 * w), w))
    py2 = max(0, min(int(y2 * h), h))
    return img_pil.crop((px1, py1, px2, py2))


def safe_stem(filename: str) -> str:
    """Strip extension and replace any path-unsafe char with '_' for crop JPG naming."""
    stem = os.path.splitext(filename)[0]
    return re.sub(r"[^A-Za-z0-9._-]", "_", stem)


def count_bboxes(detections: dict) -> int:
    return sum(len(dets) for dets in detections.values())


def list_candidate_projects(dataset_root: Path,
                            exclude_pids: set[str],
                            min_bboxes: int,
                           ) -> list[str]:
    """All project_ids in dataset_root that are not excluded, have detections.json, and >= min_bboxes total."""
    candidates: list[str] = []
    for entry in tqdm(sorted(os.scandir(dataset_root), key=lambda e: e.name), desc="Scanning projects"):
        if not entry.is_dir():
            continue
        pid = entry.name
        if pid in exclude_pids:
            continue
        det_path = os.path.join(entry.path, "detections.json")
        if not os.path.exists(det_path):
            continue
        try:
            with open(det_path, "r") as f:
                detections = json.load(f)
        except Exception:
            continue
        if count_bboxes(detections) < min_bboxes:
            continue
        candidates.append(pid)
    return candidates


def sample_test_projects(dataset_root: Path,
                         exclude_pids: set[str],
                         n_sample: int,
                         min_bboxes: int,
                         seed: int,
                        ) -> tuple[list[str], int]:
    """Seed-deterministic random sample of n_sample project_ids. Returns (sampled, pool_size)."""
    candidates = list_candidate_projects(dataset_root, exclude_pids, min_bboxes)
    if len(candidates) < n_sample:
        raise RuntimeError(f"Only {len(candidates)} candidate projects (need {n_sample}) "
                           f"after filtering by min_bboxes={min_bboxes}. Relax the threshold.")
    rng = random.Random(seed)
    sampled = sorted(rng.sample(candidates, n_sample))
    return sampled, len(candidates)


def load_or_create_test_set(test_set_path: Path,
                            dataset_root: Path,
                            exclude_pids: set[str],
                            n_sample: int,
                            min_bboxes: int,
                            seed: int,
                           ) -> list[str]:
    """Global, ckpt-independent test set: write once at OUTPUT_BASE, reuse forever."""
    if test_set_path.exists():
        with open(test_set_path, "r") as f:
            data = json.load(f)
        print(f"Loaded existing test set ({len(data['project_ids'])} projects) from {test_set_path}")
        return data["project_ids"]

    print(f"Sampling fresh test set (n={n_sample}, min_bboxes={min_bboxes}, seed={seed})...")
    sampled, pool_size = sample_test_projects(dataset_root, exclude_pids, n_sample, min_bboxes, seed)

    test_set_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(test_set_path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump({
            "project_ids": sampled,
            "n_sample": n_sample,
            "seed": seed,
            "min_bboxes": min_bboxes,
            "candidate_pool_size": pool_size,
            "exclude_file": str(EXCLUDE_FILE),
            "dataset_root": str(dataset_root),
        }, f, indent=2)
    os.replace(tmp, test_set_path)
    print(f"Saved test set ({len(sampled)} projects, pool was {pool_size}) → {test_set_path}")
    return sampled


# ---------------------------------------------------------------------------
# Per-project inference + clustering
# ---------------------------------------------------------------------------

class ProjectCropDataset(Dataset):
    """Flat dataset of crops for ONE project. Items: (entry_idx, crop_tensor, is_valid)."""

    def __init__(self, project_dir: str, detections: dict, transform):
        self.project_dir = project_dir
        self.transform = transform
        self.entries: list[tuple[str, int, list]] = []  # (filename, bbox_idx, bbox)
        for fname, dets in detections.items():
            img_path = os.path.join(project_dir, "images", fname)
            if not os.path.exists(img_path):
                continue
            for bbox_idx, det in enumerate(dets):
                self.entries.append((fname, bbox_idx, det["bbox"]))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        fname, bbox_idx, bbox = self.entries[idx]
        img_path = os.path.join(self.project_dir, "images", fname)
        try:
            img_pil = Image.open(img_path).convert("RGB")
            crop_pil = crop_bbox(img_pil, bbox)
            if crop_pil.size[0] < 4 or crop_pil.size[1] < 4:
                return idx, torch.zeros(3, 224, 224), False
            return idx, self.transform(crop_pil), True
        except Exception:
            return idx, torch.zeros(3, 224, 224), False


@torch.no_grad()
def embed_project(backbone, proj_head, dataset: ProjectCropDataset, device: str) -> tuple[np.ndarray, list[int]]:
    """Returns (embeddings [N_valid, 128], valid_entry_indices)."""
    loader = DataLoader(dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=NUM_WORKERS,
                        pin_memory=True,
                       )
    all_embs: list[np.ndarray] = []
    valid_entry_indices: list[int] = []
    for batch_indices, batch_tensors, batch_valid in loader:
        valid_mask = batch_valid.bool()
        if not valid_mask.any():
            continue
        valid_tensors = batch_tensors[valid_mask].to(device, non_blocking=True)

        out = backbone(valid_tensors)
        out = out["x_norm_clstoken"] if isinstance(out, dict) else out
        out = proj_head(out.float())
        out = F.normalize(out, dim=-1)

        all_embs.append(out.cpu().numpy())
        valid_entry_indices.extend(int(i) for i in batch_indices[valid_mask].tolist())

    if not all_embs:
        return np.empty((0, 128), dtype=np.float32), []
    return np.concatenate(all_embs, axis=0), valid_entry_indices


def cluster_embeddings_hdbscan(embeddings: np.ndarray) -> np.ndarray:
    """Same params as train_pictime/cluster_embeddings.py."""
    n = len(embeddings)
    if n < 2:
        return np.array([-1] * n, dtype=int)
    clusterer = HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                        min_samples=HDBSCAN_MIN_SAMPLES,
                        metric=HDBSCAN_METRIC,
                       )
    return clusterer.fit_predict(embeddings)


def build_clusters_dict(dataset: ProjectCropDataset,
                        valid_entry_indices: list[int],
                        labels: np.ndarray,
                       ) -> dict:
    """Mirror train_pictime/cluster_embeddings.py schema: {filename: [{bbox_index, cluster_id}, ...]}."""
    result: dict[str, list[dict]] = {}
    for emb_idx, entry_idx in enumerate(valid_entry_indices):
        fname, bbox_idx, _ = dataset.entries[entry_idx]
        result.setdefault(fname, []).append({
            "bbox_index": int(bbox_idx),
            "cluster_id": int(labels[emb_idx]),
        })
    return result


def save_viewer_crops(project_dir: str,
                      out_dir: Path,
                      dataset: ProjectCropDataset,
                      entry_indices: list[int],
                     ) -> None:
    """Save resized crop JPGs the HTML viewer references (only for the given indices)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for entry_idx in entry_indices:
        fname, bbox_idx, bbox = dataset.entries[entry_idx]
        img_path = os.path.join(project_dir, "images", fname)
        try:
            img_pil = Image.open(img_path).convert("RGB")
            crop_pil = crop_bbox(img_pil, bbox)
            crop_pil.thumbnail((VIEWER_CROP_MAX_EDGE, VIEWER_CROP_MAX_EDGE))
            out_path = out_dir / f"{safe_stem(fname)}__bb{bbox_idx}.jpg"
            crop_pil.save(out_path, format="JPEG", quality=VIEWER_CROP_JPEG_QUALITY)
        except Exception as e:
            tqdm.write(f"  crop save failed for {fname}#{bbox_idx}: {e!r}")


def missing_crop_indices(out_dir: Path,
                         dataset: ProjectCropDataset,
                         valid_entry_indices: list[int],
                        ) -> list[int]:
    """Return the subset of valid_entry_indices whose JPG is not yet on disk.

    Per-file check (not just dir-non-empty) so a previously-interrupted
    project gets completed rather than left half-cropped.
    """
    if not out_dir.exists():
        return list(valid_entry_indices)
    missing: list[int] = []
    for entry_idx in valid_entry_indices:
        fname, bbox_idx, _ = dataset.entries[entry_idx]
        out_path = out_dir / f"{safe_stem(fname)}__bb{bbox_idx}.jpg"
        if not out_path.exists():
            missing.append(entry_idx)
    return missing


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
    print(f"\nFinetune ckpt:  {ckpt_path}")
    print(f"  iteration   = {it}")
    print(f"  train sil   = {train_sil:.4f}")
    print(f"Output dir:     {output_dir}")

    # --- Build or load global, ckpt-independent test set ---
    with open(EXCLUDE_FILE, "r") as f:
        exclude_payload = json.load(f)
    exclude_pids = set(exclude_payload if isinstance(exclude_payload, list)
                       else exclude_payload.get("project_ids", []))
    print(f"\nExcluded pool:  {len(exclude_pids)} project_ids in {EXCLUDE_FILE}")

    test_set_path = Path(OUTPUT_BASE) / "test_projects.json"
    test_pids = load_or_create_test_set(test_set_path=test_set_path,
                                        dataset_root=Path(DATASET_ROOT),
                                        exclude_pids=exclude_pids,
                                        n_sample=N_SAMPLE,
                                        min_bboxes=MIN_BBOXES,
                                        seed=SEED,
                                       )

    # --- Load model ---
    print("\nLoading V11 backbone...")
    pretrain_cfg_path = str(REPO_ROOT / cfg.pretrain_config)
    backbone, embed_dim = load_backbone(pretrain_cfg_path, cfg.pretrained_weights, cfg.backbone_which)
    backbone.cuda()
    print(f"V11 backbone loaded, embed_dim={embed_dim}")

    print("\nLoading finetune state...")
    state = torch.load(ckpt_path, map_location="cuda")
    backbone.load_state_dict(state["backbone_state_dict"], strict=True)
    proj_head = build_projection_head(embed_dim, cfg.proj_hidden_dim, cfg.proj_output_dim).cuda()
    proj_head.load_state_dict(state["proj_head_state_dict"], strict=True)
    backbone.eval()
    proj_head.eval()
    del state
    torch.cuda.empty_cache()
    print("Finetune state loaded.")

    # --- Per-project: embed → cluster → save ---
    # clusters/ is per-ckpt (model-dependent); crops/ is global (model-independent
    # bbox crops from the source images, shared across all ckpt evals).
    transform = get_val_transform()

    clusters_root = output_dir / "clusters"
    crops_root = Path(OUTPUT_BASE) / "crops"
    clusters_root.mkdir(parents=True, exist_ok=True)
    crops_root.mkdir(parents=True, exist_ok=True)

    n_done, n_skipped, n_errors = 0, 0, 0
    total_crops, total_clusters, total_noise = 0, 0, 0

    for pid in tqdm(test_pids, desc="Projects"):
        clusters_path = clusters_root / f"{pid}.json"
        if clusters_path.exists():
            n_skipped += 1
            continue

        project_dir = os.path.join(DATASET_ROOT, pid)
        det_path = os.path.join(project_dir, "detections.json")
        try:
            with open(det_path, "r") as f:
                detections = json.load(f)
            dataset = ProjectCropDataset(project_dir, detections, transform)
            if len(dataset) == 0:
                tqdm.write(f"[{pid}] empty after filtering — skipped")
                n_skipped += 1
                continue

            embeddings, valid_entry_indices = embed_project(backbone, proj_head, dataset, device)
            if len(embeddings) == 0:
                tqdm.write(f"[{pid}] no valid crops — skipped")
                n_skipped += 1
                continue

            labels = cluster_embeddings_hdbscan(embeddings)
            clusters_dict = build_clusters_dict(dataset, valid_entry_indices, labels)

            tmp = str(clusters_path) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(clusters_dict, f)
            os.replace(tmp, clusters_path)

            # Save viewer crops to GLOBAL crops dir (shared across ckpt evals).
            # Only save the missing ones — completes any partial dirs from a prior
            # interrupted run, and a no-op when all crops are already there.
            pid_crop_dir = crops_root / pid
            missing = missing_crop_indices(pid_crop_dir, dataset, valid_entry_indices)
            if missing:
                save_viewer_crops(project_dir, pid_crop_dir, dataset, missing)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int(np.sum(labels == -1))
            total_crops += len(labels)
            total_clusters += n_clusters
            total_noise += n_noise
            n_done += 1
        except Exception as e:
            tqdm.write(f"[{pid}] ERROR: {e!r}")
            n_errors += 1

    print(f"\n===== Summary =====")
    print(f"Done:     {n_done}")
    print(f"Skipped:  {n_skipped}")
    print(f"Errors:   {n_errors}")
    if n_done > 0:
        print(f"Mean clusters / project: {total_clusters / n_done:.2f}")
        print(f"Mean crops / project:    {total_crops / n_done:.1f}")
        print(f"Noise fraction:          {total_noise / max(total_crops, 1):.3f}")
    print(f"\nOutputs under: {output_dir}")
    print(f"Next: python3 -m train_pictime.finetune.realworld_eval.build_html_viewer")


if __name__ == "__main__":
    main()
