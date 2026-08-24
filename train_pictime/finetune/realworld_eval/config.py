"""Real-world ReID evaluation — single source of truth for config.

Both cluster_test_set.py and build_html_viewer.py read constants from here.
Edit here once and both scripts stay in sync (no chance of e.g. one script
pointing at V31 while the other still points at V28).
"""

# ---------------------------------------------------------------------------
# Ckpt + paths
# ---------------------------------------------------------------------------

# New model: V52 finetune (ViT-S/16). From the V<n> dir, `find_best_silhouette_ckpt`
# picks the highest-silhouette of the saved ckpt_iter*_sil*.pt checkpoints.
# NOTE: this dir's NAME is also the output tag (prep_labeling_files -> clusters_v52.json).
FINETUNE_VERSION_DIR = "/data/AI/Tomer/dinov3/train_pictime/finetune_experiments/V52"

# Optionally pin an EXACT ckpt — e.g. a `last_iter*` file, which find_best_silhouette_ckpt
# won't match (it only globs ckpt_iter*_sil*.pt). None -> auto-pick the best ckpt.
FINETUNE_CKPT_PATH = "/data/AI/Tomer/dinov3/train_pictime/finetune_experiments/V52/ckpt/ckpt_iter8991_sil0.4692.pt"

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
# HDBSCAN — V52-optimized params, handed over as the production spec:
#     body_distance_function       cosine
#     body_clustering_distance     0.08
#     body_clustering_size         3
#     body_clustering_min_samples  2
#     body_clustering_method       hdbscan4_1   (prod method id, not an HDBSCAN arg)
#
# METRIC stays euclidean deliberately — this is NOT a deviation from that spec.
# embed_project L2-normalizes, and on unit vectors d_euclid = sqrt(2 * d_cos), a strictly
# monotone map: core distances, mutual reachability and the MST all keep their ordering, so
# the cluster hierarchy is identical. min_cluster_size, min_samples, selection_method and
# allow_single_cluster are scale-invariant; only cluster_selection_epsilon is an absolute
# distance and must be converted:  cosine 0.08 -> euclidean sqrt(2 * 0.08) = 0.40.
# Cosine is in neither BallTree.valid_metrics nor KDTree.valid_metrics, so asking hdbscan
# for it either errors or forces a dense O(n^2) matrix per project; euclidean keeps the
# tree path for identical output.
#
# Scale check: the previous 0.1 euclidean was cosine 0.005, so this is 16x looser — expect
# markedly fewer, larger clusters. allow_single_cluster stays False (the 06-24 wedding fix;
# True fits one loose root cluster and dumps everyone else to noise on multi-identity
# galleries). epsilon>0 + allow_single_cluster=True would need the standalone `hdbscan`
# package, since sklearn's HDBSCAN crashes on that combination.
# ---------------------------------------------------------------------------

HDBSCAN_MIN_CLUSTER_SIZE = 3
HDBSCAN_MIN_SAMPLES = 2
HDBSCAN_CLUSTER_SELECTION_EPSILON = 0.40    # == cosine 0.08 on L2-normalized embeddings
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"
HDBSCAN_ALLOW_SINGLE_CLUSTER = False
HDBSCAN_METRIC = "euclidean"                # see the conversion note above


# ---------------------------------------------------------------------------
# Inference + viewer crop saving
# ---------------------------------------------------------------------------

BATCH_SIZE = 64
NUM_WORKERS = 4
VIEWER_CROP_MAX_EDGE = 300       # px; max edge of the saved viewer JPG (preserves aspect)
VIEWER_CROP_JPEG_QUALITY = 85
