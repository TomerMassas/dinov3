"""Single source of truth for the OLD (ResNet) vs NEW (DINOv3 ViT-S/16) body-
embedding comparison. Fully isolated — nothing here is imported by the rest of
the training pipeline.

Comparison is on the ~150 NEWLY-labeled projects (reviewer clusters_fixed.json =
ground truth) that were added after the 826 snapshot. See prepare_eval_set.py.
"""
from pathlib import Path

# --- Repos / data (VM paths) ---
DINOV3_REPO = Path("/data/AI/Tomer/dinov3")
PERSON_REID_REPO = Path("/data/AI/Tomer/person_reid")   # old ResNet lives in its reid_src/
DATASET_ROOT = Path("/data/AI/Tomer/person_reid/dataset_utils/dataset_finetune/Portraits[26]")

CLUSTERS_FIXED_FILENAME = "clusters_fixed.json"          # reviewer ground truth
DETECTIONS_FILENAME = "detections.json"
IMAGES_SUBDIR = "images"

# --- Eval-set selection ---
# Pick the TOP_N approved projects with the highest num_crops/num_clusters
# (mean cluster size) — the galleries with the biggest per-identity clusters,
# i.e. the richest galleries for a ReID/clustering comparison.
HERE = Path(__file__).parent
COMPLETION_LOG = HERE / "completion_log.json"           # approved-project metadata (num_crops, num_clusters)
TOP_N = None #200

# Names this test set — drives both the results subdir and the project-ids file,
# so different test sizes/compositions sit side by side under results/.
TEST_SET_NAME = "all_approved"
OUTPUT_DIR = HERE / "results" / TEST_SET_NAME           # all artifacts for this test set
NEW_PROJECTS_FILE = OUTPUT_DIR / f"proj_ids_{TEST_SET_NAME}.json"   # locked test set (TOP_N)

# --- GT filtering (match fracturing_eval) ---
MIN_GT_CLUSTER_SIZE = 5    # drop GT clusters smaller than this; GT cluster_id == -1 always dropped

# --- OLD model: ResNet50 CTL (person-reID reid_src), 2048-d, NOT normalized ---
OLD_WEIGHTS = PERSON_REID_REPO / "models" / "reid_model.pt"   # confirm exact filename after unzip
OLD_EMB_DIM = 2048
OLD_RESIZE_HW = (256, 128)                                    # production INPUT.SIZE_TEST (H, W)
# Faithful production body clustering (IdentityClustering CONFIG): agglomerative.
OLD_CLUSTER = dict(
    method="agglomerative",
    distance_threshold=0.85,     # body_clustering_distance
    linkage="average",
    dist_func="euclidean",       # normalize then euclidean (body_distance_function)
    merge_centroid_cosine=0.2,   # post-merge: cluster centroids closer than this (cosine) are merged
)

# --- NEW model: DINOv3 ViT-S/16 finetune (deployed V44), 128-d, L2-normalized ---
NEW_CKPT = PERSON_REID_REPO / "models" / "ckpt_iter15000_sil0.4556.pt"   # backbone+proj_head state dicts
PICTIME_CFG = DINOV3_REPO / "train_pictime/pictime_vitl_im1k_lin834.yaml"
# New-model provenance, rendered as a table in comparison.md (one row per training
# stage). Update per checkpoint so each report self-documents which model it scored.
# Columns are derived from the first row's keys, so keep keys consistent across rows.
NEW_MODEL_INFO = {
    "backbone (SSL pretrain)": {
        "method":     "DINOv3 self-supervised (ViT-S/16)",
        "data":       "Pictime pretrain images, NO face-blur (V18 line)",
        "labels":     "none (self-supervised)",
        "checkpoint": "V18, ckpt 19750",
    },
    "finetune (metric learning)": {
        "method":     "SupCon, PK sampling (P16×K4), progressive unfreeze",
        "data":       "full finetune set − reviewer-held-out",
        "labels":     "HDBSCAN clusters_v3 (pseudo, NOT reviewer truth)",
        "checkpoint": "V44 ckpt_iter15000",
    },
}
NEW_EMB_DIM = 128
PROJ_HIDDEN_DIM = 384
PROJ_OUTPUT_DIM = 128
BACKBONE_WHICH = "teacher"
# Tuned clustering for the new model (HDBSCAN). Knobs that matter for ReID
# galleries / fracturing — see cluster_hdbscan:
#   min_samples=1 + allow_single_cluster=True mirror production-HDBSCAN and handle
#   single-identity galleries; cluster_selection_epsilon>0 merges over-split
#   sub-clusters (raise it if fracturing is high); 'eom' keeps clusters fewer/larger.
NEW_CLUSTER = dict(
    method="hdbscan",
    min_cluster_size=10,
    min_samples=None,# None--> min_samples=min_cluster_size # how aggressively things get called noise, low min_samples tries to keep every crop but risks chaining one person's crop into a different person's cluster
    cluster_selection_epsilon=0.1, # 1 mergre all into 1 cluster, 0 pure hdbscan (0,1) playground for decreasing fracturing effect
    cluster_selection_method="eom", # eom, leaf
    allow_single_cluster=True,
    metric="euclidean",
)

# --- Inference ---
BATCH_SIZE = 64
NUM_WORKERS = 4

# --- Metrics ---
SILHOUETTE_MAX_SAMPLES = 10000
SEED = 42

# --- Output (caches live alongside results in the per-test-set dir) ---
OLD_CACHE = OUTPUT_DIR / "old_embeddings.npz"
NEW_CACHE = OUTPUT_DIR / "new_embeddings.npz"