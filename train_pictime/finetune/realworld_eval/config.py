"""Real-world ReID evaluation — single source of truth for config.

Both cluster_test_set.py and build_html_viewer.py read constants from here.
Edit here once and both scripts stay in sync (no chance of e.g. one script
pointing at V31 while the other still points at V28).
"""

# ---------------------------------------------------------------------------
# Ckpt + paths
# ---------------------------------------------------------------------------

# Trial 2 winner: n_blocks=6, lr_backbone=1e-4 on V11/ckpt/13000. From the
# V<n> dir, `find_best_silhouette_ckpt` picks the highest-silhouette of
# the saved best-3 ckpts.
FINETUNE_VERSION_DIR = "/data/AI/Tomer/dinov3/train_pictime/finetune_experiments/V31"

DATASET_ROOT = "/data/AI/Tomer/person_reid/dataset_utils/dataset_finetune/Portraits[26]"

# Filter / exclude file (train+eval pool) — same path used by build_index.FILTER_PATH.
EXCLUDE_FILE = "/data/AI/Tomer/dinov3/train_pictime/finetune/single_cluster_projects_v2.json"

OUTPUT_BASE = "/data/AI/Tomer/realworld_eval"


# ---------------------------------------------------------------------------
# Sampling (test set built once, frozen at OUTPUT_BASE/test_projects.json)
# ---------------------------------------------------------------------------

N_SAMPLE = 100
SEED = 42
MIN_BBOXES = 50


# ---------------------------------------------------------------------------
# HDBSCAN — identical to train_pictime/cluster_embeddings.py
# ---------------------------------------------------------------------------

HDBSCAN_MIN_CLUSTER_SIZE = 3
HDBSCAN_MIN_SAMPLES = None
HDBSCAN_METRIC = "euclidean"


# ---------------------------------------------------------------------------
# Inference + viewer crop saving
# ---------------------------------------------------------------------------

BATCH_SIZE = 64
NUM_WORKERS = 4
VIEWER_CROP_MAX_EDGE = 300       # px; max edge of the saved viewer JPG (preserves aspect)
VIEWER_CROP_JPEG_QUALITY = 85
