"""Generate per-project labeling files for a dataset (e.g. Wedding[1]) using the
SAME finetune model + tuned HDBSCAN as the real-world eval HTML view, so what the
labeling team reviews is exactly what you validated in the viewer.

For every project dir under realworld_eval.config.DATASET_ROOT, writes INTO the
project dir (tag = the finetune version dir name lowercased, e.g. "v51"):

    embeddings_<tag>.npz       filenames, bbox_indices, embeddings [N,128] L2-normed
    clusters_<tag>.json        {filename: [{bbox_index, cluster_id}, ...]}
    crop_distances_<tag>.json  {cluster_id: [{filename, bbox_index, distance}, ...]}  (sorted asc)

Reuses realworld_eval's model load + embed_project + cluster_embeddings_hdbscan
(single source of truth via realworld_eval/config.py) and replicates
build_centroids' distance math on the in-memory embeddings.

Per project: skipped if clusters_fixed.json exists (reviewer truth) or all three
outputs already exist (unless FORCE). Atomic writes. Crops are NOT saved — the
review UI crops on-the-fly from images/ + detections.json.

Set the model in realworld_eval/config.py (FINETUNE_VERSION_DIR / FINETUNE_CKPT_PATH)
and the HDBSCAN params BEFORE running — they decide the tag and the clustering.

    python3 -m train_pictime.finetune.realworld_eval.prep_labeling_files
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from dinov3.configs import setup_job
from train_pictime.finetune.finetune_reid import (
    CFG_PATH, REPO_ROOT, build_projection_head, load_backbone,
)
from train_pictime.finetune.reeval_tiered import find_best_silhouette_ckpt
from train_pictime.finetune.reid_dataset import get_val_transform
from train_pictime.finetune.realworld_eval.config import (
    DATASET_ROOT, FINETUNE_CKPT_PATH, FINETUNE_VERSION_DIR,
)
from train_pictime.finetune.realworld_eval.cluster_test_set import (
    ProjectCropDataset, build_clusters_dict, cluster_embeddings_hdbscan, embed_project,
)

CLUSTERS_FIXED_FILENAME = "clusters_fixed.json"
FORCE = False   # re-generate even if outputs exist (never overrides clusters_fixed.json)


# ---------------------------------------------------------------------------
# Model + distances
# ---------------------------------------------------------------------------

def load_model(cfg, device):
    """Load backbone (arch from reid_config) + finetune ckpt (backbone + proj_head),
    exactly as cluster_test_set does. Honors FINETUNE_CKPT_PATH, else best ckpt."""
    if FINETUNE_CKPT_PATH:
        ckpt_path = Path(FINETUNE_CKPT_PATH)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"FINETUNE_CKPT_PATH not found: {ckpt_path}")
    else:
        ckpt_path, _it, _sil = find_best_silhouette_ckpt(Path(FINETUNE_VERSION_DIR) / "ckpt")
    print(f"Finetune ckpt: {ckpt_path}")

    backbone, embed_dim = load_backbone(str(REPO_ROOT / cfg.pretrain_config),
                                        cfg.pretrained_weights, cfg.backbone_which)
    backbone.to(device)
    state = torch.load(ckpt_path, map_location=device)
    backbone.load_state_dict(state["backbone_state_dict"], strict=True)
    proj_head = build_projection_head(embed_dim, cfg.proj_hidden_dim, cfg.proj_output_dim).to(device)
    proj_head.load_state_dict(state["proj_head_state_dict"], strict=True)
    backbone.eval()
    proj_head.eval()
    del state
    torch.cuda.empty_cache()
    return backbone, proj_head


def compute_crop_distances(embeddings, filenames, bbox_indices, labels) -> dict:
    """Per-cluster cosine distance to centroid, sorted ascending; excludes -1.
    Mirrors build_centroids.build_centroids_for_project (embeddings are L2-normed)."""
    result: dict[str, list] = {}
    for cid in sorted(set(int(c) for c in labels)):
        if cid == -1:
            continue
        idxs = [k for k in range(len(labels)) if int(labels[k]) == cid]
        embs = embeddings[idxs]
        centroid = embs.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        dists = 1.0 - embs @ centroid
        result[str(cid)] = sorted(
            ({"filename": filenames[idxs[j]], "bbox_index": int(bbox_indices[idxs[j]]),
              "distance": float(dists[j])} for j in range(len(idxs))),
            key=lambda e: e["distance"],
        )
    return result


def _save_json(path: Path, obj) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _save_npz(path: Path, filenames, bbox_indices, embeddings) -> None:
    tmp = str(path) + ".tmp.npz"
    np.savez(tmp,
             filenames=np.array(filenames, dtype=object),
             bbox_indices=np.array(bbox_indices, dtype=np.int32),
             embeddings=embeddings.astype(np.float32))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = OmegaConf.load(CFG_PATH)
    setup_job(output_dir=None, seed=cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone, proj_head = load_model(cfg, device)
    transform = get_val_transform()

    tag = Path(FINETUNE_VERSION_DIR).name.lower()             # e.g. "v51"
    emb_name = f"embeddings_{tag}.npz"
    cl_name = f"clusters_{tag}.json"
    dist_name = f"crop_distances_{tag}.json"
    print(f"\nDataset:  {DATASET_ROOT}")
    print(f"Writing per project:  {emb_name} | {cl_name} | {dist_name}\n")

    project_dirs = sorted(e.path for e in os.scandir(DATASET_ROOT) if e.is_dir())

    processed = skipped = skipped_fixed = errors = 0
    total_crops = total_clusters = total_noise = 0

    for project_dir in tqdm(project_dirs, desc="Projects"):
        pdir = Path(project_dir)
        if (pdir / CLUSTERS_FIXED_FILENAME).exists():
            skipped_fixed += 1
            continue
        emb_p, cl_p, dist_p = pdir / emb_name, pdir / cl_name, pdir / dist_name
        if not FORCE and emb_p.exists() and cl_p.exists() and dist_p.exists():
            skipped += 1
            continue

        det_path = pdir / "detections.json"
        if not det_path.exists():
            skipped += 1
            continue

        try:
            with open(det_path) as f:
                detections = json.load(f)
            dataset = ProjectCropDataset(str(pdir), detections, transform)
            if len(dataset) == 0:
                skipped += 1
                continue

            embeddings, valid_idx = embed_project(backbone, proj_head, dataset, device)
            if len(embeddings) == 0:
                skipped += 1
                continue

            labels = cluster_embeddings_hdbscan(embeddings)
            filenames = [dataset.entries[i][0] for i in valid_idx]
            bbox_idxs = [dataset.entries[i][1] for i in valid_idx]

            _save_npz(emb_p, filenames, bbox_idxs, embeddings)
            _save_json(cl_p, build_clusters_dict(dataset, valid_idx, labels))
            _save_json(dist_p, compute_crop_distances(embeddings, filenames, bbox_idxs, labels))

            n_clusters = len(set(int(l) for l in labels) - {-1})
            n_noise = int(np.sum(labels == -1))
            total_crops += len(labels)
            total_clusters += n_clusters
            total_noise += n_noise
            processed += 1
        except Exception as e:
            tqdm.write(f"[{pdir.name}] ERROR: {e!r}")
            errors += 1

    print(f"\n===== Summary =====")
    print(f"Processed: {processed}")
    print(f"Skipped (already done): {skipped}")
    print(f"Skipped (clusters_fixed present): {skipped_fixed}")
    print(f"Errors: {errors}")
    if processed > 0:
        print(f"Mean clusters / project: {total_clusters / processed:.2f}")
        print(f"Mean crops / project:    {total_crops / processed:.1f}")
        print(f"Noise fraction:          {total_noise / max(total_crops, 1):.3f}")


if __name__ == "__main__":
    main()