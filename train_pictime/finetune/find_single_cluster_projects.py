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

DATA_BASE_PATH = Path("/data/AI/Tomer/person_reid/dataset_utils/dataset_finetune/Portraits[26]")
OUTPUT_PATH =  "/data/AI/Tomer/dinov3/train_pictime/finetune/single_cluster_projects.json"


def main():
    project_dirs = sorted([d for d in DATA_BASE_PATH.iterdir() if d.is_dir()])
    print(f"Found {len(project_dirs)} projects")

    single_cluster = []
    multi_cluster = 0
    no_cluster_file = 0

    for project_dir in tqdm(project_dirs, desc="Scanning", file=sys.stdout):
        cluster_path = project_dir / "clusters_fixed.json"
        if not cluster_path.exists():
            cluster_path = project_dir / "clusters.json"
        if not cluster_path.exists():
            no_cluster_file += 1
            continue

        with open(cluster_path) as f:
            clusters = json.load(f)

        # Collect all unique cluster IDs, ignoring noise (-1)
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