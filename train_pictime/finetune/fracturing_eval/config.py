"""Fracturing eval — single source of truth for config.

Mirrors the realworld_eval/config.py pattern. Imports `FINETUNE_VERSION_DIR`
+ HDBSCAN params from realworld_eval so V<n> + clustering knob changes
propagate to both evals automatically (no chance of drift).
"""

from train_pictime.finetune.realworld_eval.config import (
    BATCH_SIZE,
    FINETUNE_VERSION_DIR,
    HDBSCAN_METRIC,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    NUM_WORKERS,
)


# ---------------------------------------------------------------------------
# Held-out projects + dataset root
# ---------------------------------------------------------------------------

# Path to a JSON file listing project IDs that are EXCLUDED from finetune
# train data and used as the held-out test set for fracturing eval.
# Schema: either a bare list ["pid1", "pid2", ...] or {"project_ids": [...]}.
HELD_OUT_PROJECTS_FILE = "/data/AI/Tomer/dinov3/train_pictime/finetune/fracturing_eval/approved_projects.json"

# Project dirs containing `clusters_fixed.json` + `detections.json` live here.
DATASET_ROOT = "/data/AI/Tomer/person_reid/dataset_utils/dataset_finetune/Portraits[26]"


# ---------------------------------------------------------------------------
# Eval knobs
# ---------------------------------------------------------------------------

# Skip GT clusters with fewer than this many crops. Tiny GT clusters
# fracture mechanically (any noise = high fracturing %) and add noise to
# the histogram tail.
MIN_GT_CLUSTER_SIZE = 5


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

OUTPUT_BASE = "/data/AI/Tomer/dinov3/train_pictime/finetune/fracturing_eval/results"
