"""Offline index builder for ReID finetuning.

Scans all project directories, reads detections.json and clusters_fixed.json
(or clusters.json), and writes a single .npz index file.

Run once before training:
    python3 -m train_pictime.finetune.build_index
"""

import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

MODEL_SOURCE = "v17_ckpt19750"  # "foundation_b16" | "v11_ckpt13k" | "v17_ckpt19750"

# Maps each backbone source to its input/output filenames — NEVER use ternaries here.
CLUSTERS_FILENAME_BY_MODEL = {
    "foundation_b16": "clusters.json",
    "v11_ckpt13k":    "clusters_v2.json",
    "v17_ckpt19750":  "clusters_v3.json",
}
# Project filter: v1/v2 trained on the single-cluster ALLOWLIST; v3 trains on the
# WHOLE dataset except the reviewer-labeled held-out projects (DENYLIST — these are
# the fracturing-eval test set and must never enter train/val).
FILTER_MODE_BY_MODEL = {
    "foundation_b16": "allowlist",
    "v11_ckpt13k":    "allowlist",
    "v17_ckpt19750":  "denylist",
}
FILTER_FILENAME_BY_MODEL = {
    "foundation_b16": "single_cluster_projects.json",
    "v11_ckpt13k":    "single_cluster_projects_v2.json",
    "v17_ckpt19750":  "fracturing_eval/approved_projects.json",   # held-out reviewer set (denylist); also the fracturing-eval input
}
INDEX_FILENAME_BY_MODEL = {
    "foundation_b16": "reid_index.npz",
    "v11_ckpt13k":    "reid_index_v2.npz",
    "v17_ckpt19750":  "reid_index_v3.npz",
}
CLUSTERS_FILENAME = CLUSTERS_FILENAME_BY_MODEL[MODEL_SOURCE]
FILTER_MODE = FILTER_MODE_BY_MODEL[MODEL_SOURCE]
FILTER_FILENAME = FILTER_FILENAME_BY_MODEL[MODEL_SOURCE]
INDEX_FILENAME = INDEX_FILENAME_BY_MODEL[MODEL_SOURCE]

# Reviewer-corrected clusters always win over per-version clusters.
CLUSTERS_FIXED_FILENAME = "clusters_fixed.json"

DATA_BASE_PATH = Path("/data/AI/Tomer/person_reid/dataset_utils/dataset_finetune/Portraits[26]")
FILTER_PATH = Path("/data/AI/Tomer/dinov3/train_pictime/finetune") / FILTER_FILENAME
OUTPUT_PATH = DATA_BASE_PATH / INDEX_FILENAME


def load_filter_pids(path: Path) -> set[str]:
    """Accepts a bare list ["pid", ...] or {"project_ids": ["pid", ...]}."""
    with open(path) as f:
        payload = json.load(f)
    pids = payload if isinstance(payload, list) else payload.get("project_ids", [])
    if not pids:
        raise RuntimeError(f"No project ids in filter file {path}")
    return set(pids)


def build_index(data_base: Path) -> list[dict]:
    filter_pids = load_filter_pids(FILTER_PATH)
    if FILTER_MODE == "allowlist":
        project_dirs = sorted([d for d in data_base.iterdir() if d.is_dir() and d.name in filter_pids])
    elif FILTER_MODE == "denylist":
        project_dirs = sorted([d for d in data_base.iterdir() if d.is_dir() and d.name not in filter_pids])
    else:
        raise ValueError(f"Unknown FILTER_MODE: {FILTER_MODE}")
    print(f"Filter: {FILTER_MODE} ({len(filter_pids)} ids in {FILTER_FILENAME})")
    print(f"Found {len(project_dirs)} projects")

    samples = []
    skipped = 0

    for project_dir in tqdm(project_dirs, desc="Building index", file=sys.stdout):
        det_path = project_dir / "detections.json"
        if not det_path.exists():
            skipped += 1
            continue

        cluster_path = project_dir / CLUSTERS_FIXED_FILENAME
        if cluster_path.exists() and FILTER_MODE == "denylist":
            # Reviewer-labeled project NOT in the denylist — likely a hole in the
            # held-out list. Including it would leak test-set projects into train.
            print(f"[LEAK WARNING] {project_dir.name} has {CLUSTERS_FIXED_FILENAME} "
                  f"but is not in {FILTER_FILENAME} — check the held-out list!")
        if not cluster_path.exists():
            cluster_path = project_dir / CLUSTERS_FILENAME
        if not cluster_path.exists():
            skipped += 1
            continue

        with open(det_path) as f:
            detections = json.load(f)
        with open(cluster_path) as f:
            clusters = json.load(f)

        project_id = project_dir.name

        for fname, cluster_entries in clusters.items():
            if fname not in detections:
                continue
            det_list = detections[fname]

            for entry in cluster_entries:
                bbox_idx = entry["bbox_index"]
                cluster_id = entry["cluster_id"]

                if bbox_idx >= len(det_list):
                    continue

                bbox = det_list[bbox_idx]["bbox"]
                image_path = str(project_dir / "images" / fname)
                samples.append({
                    "image_path": image_path,
                    "bbox": bbox,
                    "bbox_index": bbox_idx,
                    "project_id": project_id,
                    "cluster_id": cluster_id,
                })

    print(f"Total samples: {len(samples)}, Skipped projects: {skipped}")
    return samples


def main():
    samples = build_index(DATA_BASE_PATH)

    # Save as columnar numpy arrays for instant loading
    image_paths = np.array([s["image_path"] for s in samples], dtype=object)
    bboxes = np.array([s["bbox"] for s in samples], dtype=np.float32)  # [N, 4]
    bbox_indices = np.array([s["bbox_index"] for s in samples], dtype=np.int32)
    project_ids = np.array([s["project_id"] for s in samples], dtype=object)
    cluster_ids = np.array([s["cluster_id"] for s in samples], dtype=np.int32)

    print(f"Writing index to {OUTPUT_PATH}...")
    np.savez(OUTPUT_PATH,
             image_paths=image_paths,
             bboxes=bboxes,
             bbox_indices=bbox_indices,
             project_ids=project_ids,
             cluster_ids=cluster_ids,
            )
    print("Done.")


if __name__ == "__main__":
    main()
