"""Scan the finetune dataset for reviewer-labeled projects.

A project is "labeled" once the review team has produced a clusters_fixed.json
for it. This writes the current list of all such project_ids to a JSON file,
which build_store.py consumes to decide which projects to (re)incorporate.

Labels are append-only, so this list only grows across cycles.

Run:
    python3 -m train_pictime.finetune.scan_labeled_projects
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from tqdm import tqdm


DATASET_ROOT = Path("/data/AI/Tomer/person_reid/dataset_utils/dataset_finetune/Portraits[26]")
CLUSTERS_FIXED_FILENAME = "clusters_fixed.json"
OUTPUT_PATH = Path(__file__).parent / "labeled_projects.json"


def main():
    labeled = []
    for entry in tqdm(sorted(os.scandir(DATASET_ROOT), key=lambda e: e.name), desc="Scanning"):
        if not entry.is_dir():
            continue
        if os.path.exists(os.path.join(entry.path, CLUSTERS_FIXED_FILENAME)):
            labeled.append(entry.name)

    tmp = str(OUTPUT_PATH) + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"project_ids": labeled, "count": len(labeled)}, f, indent=2)
    os.replace(tmp, OUTPUT_PATH)
    print(f"Found {len(labeled)} labeled projects (have {CLUSTERS_FIXED_FILENAME}) -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()