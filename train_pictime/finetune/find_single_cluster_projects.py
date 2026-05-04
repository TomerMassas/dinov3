"""Find projects where clustering produced exactly 1 identity (single cluster).

These are "clean" projects — high confidence that all detections belong to
the same person. Useful for early finetune experiments before the full
curated dataset is ready.

Outputs a JSON list of project folder names to:
    <DATA_BASE_PATH>/single_cluster_projects.json

Run:
    python3 -m train_pictime.finetune.find_single_cluster_projects
"""

import json
import sys
from pathlib import Path

from tqdm import tqdm

MODEL_SOURCE = "v11_ckpt13k"  # "foundation_b16" | "v11_ckpt13k"

# Maps each backbone source to the (non-fixed) clusters filename and output filename.
CLUSTERS_FILENAME_BY_MODEL = {
    "foundation_b16": "clusters.json",
    "v11_ckpt13k":    "clusters_v2.json",
}
OUTPUT_FILENAME_BY_MODEL = {
    "foundation_b16": "single_cluster_projects.json",
    "v11_ckpt13k":    "single_cluster_projects_v2.json",
}
CLUSTERS_FILENAME = CLUSTERS_FILENAME_BY_MODEL[MODEL_SOURCE]
OUTPUT_FILENAME = OUTPUT_FILENAME_BY_MODEL[MODEL_SOURCE]
CLUSTERS_FIXED_FILENAME = "clusters_fixed.json"  # always wins over per-version clusters

DATA_BASE_PATH = Path("/data/AI/Tomer/person_reid/dataset_utils/dataset_finetune/Portraits[26]")
OUTPUT_PATH = f"/data/AI/Tomer/dinov3/train_pictime/finetune/{OUTPUT_FILENAME}"


def main():
    project_dirs = sorted([d for d in DATA_BASE_PATH.iterdir() if d.is_dir()])
    print(f"Found {len(project_dirs)} projects")

    single_cluster = []
    multi_cluster = 0
    no_cluster_file = 0

    for project_dir in tqdm(project_dirs, desc="Scanning", file=sys.stdout):
        cluster_path = project_dir / CLUSTERS_FIXED_FILENAME
        is_fixed = cluster_path.exists()
        if not is_fixed:
            cluster_path = project_dir / CLUSTERS_FILENAME
        if not cluster_path.exists():
            no_cluster_file += 1
            continue

        # Reviewer-verified: trust unconditionally, no cluster-count check needed.
        if is_fixed:
            single_cluster.append(project_dir.name)
            continue

        # HDBSCAN-derived: keep the empirical "exactly 2 non-noise clusters" filter.
        with open(cluster_path) as f:
            clusters = json.load(f)

        all_ids = set()
        for entries in clusters.values():
            for entry in entries:
                cid = entry["cluster_id"]
                if cid != -1:
                    all_ids.add(cid)

        if len(all_ids) == 2:
            single_cluster.append(project_dir.name)
        else:
            multi_cluster += 1

    print(f"\nSingle-cluster (clean): {len(single_cluster)}")
    print(f"Multi-cluster:          {multi_cluster}")
    print(f"No cluster file:        {no_cluster_file}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(single_cluster, f)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()