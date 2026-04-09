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

DATA_BASE_PATH = Path("/data/AI/Tomer/person_reid/dataset_utils/dataset_finetune/Portraits[26]")
OUTPUT_PATH = DATA_BASE_PATH / "reid_index.npz"


def build_index(data_base: Path) -> list[dict]:
    project_dirs = sorted([d for d in data_base.iterdir() if d.is_dir()])
    print(f"Found {len(project_dirs)} projects")

    samples = []
    skipped = 0

    for project_dir in tqdm(project_dirs, desc="Building index", file=sys.stdout):
        det_path = project_dir / "detections.json"
        if not det_path.exists():
            skipped += 1
            continue

        cluster_path = project_dir / "clusters_fixed.json"
        if not cluster_path.exists():
            cluster_path = project_dir / "clusters.json"
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
                image_path = str(project_dir / fname)
                samples.append({
                    "image_path": image_path,
                    "bbox": bbox,
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
    project_ids = np.array([s["project_id"] for s in samples], dtype=object)
    cluster_ids = np.array([s["cluster_id"] for s in samples], dtype=np.int32)

    print(f"Writing index to {OUTPUT_PATH}...")
    np.savez(
        OUTPUT_PATH,
        image_paths=image_paths,
        bboxes=bboxes,
        project_ids=project_ids,
        cluster_ids=cluster_ids,
    )
    print("Done.")


if __name__ == "__main__":
    main()
