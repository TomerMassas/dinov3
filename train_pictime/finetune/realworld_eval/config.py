"""Real-world ReID evaluation — single source of truth for config.

Both cluster_test_set.py and build_html_viewer.py read constants from here.
Edit here once and both scripts stay in sync (no chance of e.g. one script
pointing at V31 while the other still points at V28).
"""

# ---------------------------------------------------------------------------
# Ckpt + paths
# ---------------------------------------------------------------------------

# New model: V51 finetune (ViT-S/16). From the V<n> dir, `find_best_silhouette_ckpt`
# picks the highest-silhouette of the saved ckpt_iter*_sil*.pt checkpoints.
FINETUNE_VERSION_DIR = "/data/AI/Tomer/dinov3/train_pictime/finetune_experiments/V51"

# Optionally pin an EXACT ckpt — e.g. a `last_iter*` file, which find_best_silhouette_ckpt
# won't match (it only globs ckpt_iter*_sil*.pt). None -> auto-pick the best ckpt.
FINETUNE_CKPT_PATH = "/data/AI/Tomer/dinov3/train_pictime/finetune_experiments/V51/ckpt/last_iter26274_sil0.4679.pt"

# Flip TEST_SET_NAME to switch test sets. Each test set lives under its own
# subdir of OUTPUT_BASE so they coexist without collision.
DATASET_PARENT = "/data/AI/Tomer/person_reid/dataset_utils/dataset_finetune"
TEST_SET_NAME = "Wedding[1]"
DATASET_ROOT = f"{DATASET_PARENT}/{TEST_SET_NAME}"

# Filter / exclude file (train+eval pool) — same path used by build_index.FILTER_PATH.
# Set to None when the test set has no overlap with the train pool (e.g. Wedding[1]).
EXCLUDE_FILE = None #"/data/AI/Tomer/dinov3/train_pictime/finetune/single_cluster_projects_v2.json"

OUTPUT_BASE = "/data/AI/Tomer/dinov3/train_pictime/finetune/realworld_eval/results"


# ---------------------------------------------------------------------------
# Sampling (test set built once, frozen at OUTPUT_BASE/<TEST_SET_NAME>/test_projects.json)
# ---------------------------------------------------------------------------

N_SAMPLE = 100
SEED = 42
MIN_BBOXES = 0


# ---------------------------------------------------------------------------
# HDBSCAN — tuned new-model params (matches model_comparison/config.py NEW_CLUSTER).
# cluster_selection_epsilon>0 + allow_single_cluster=True requires the standalone
# `hdbscan` package (sklearn's HDBSCAN crashes on that combo).
# ---------------------------------------------------------------------------

HDBSCAN_MIN_CLUSTER_SIZE = 3
HDBSCAN_MIN_SAMPLES = None                  # None -> uses min_cluster_size
HDBSCAN_CLUSTER_SELECTION_EPSILON = 0.1     # merges over-split sub-clusters (anti-fracturing)
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"    # "eom" (fewer/larger) | "leaf" (more/finer)
HDBSCAN_ALLOW_SINGLE_CLUSTER = False# True         # single-identity gallery -> one cluster, not all-noise
HDBSCAN_METRIC = "euclidean"


# ---------------------------------------------------------------------------
# Inference + viewer crop saving
# ---------------------------------------------------------------------------

BATCH_SIZE = 64
NUM_WORKERS = 4
VIEWER_CROP_MAX_EDGE = 300       # px; max edge of the saved viewer JPG (preserves aspect)
VIEWER_CROP_JPEG_QUALITY = 85
