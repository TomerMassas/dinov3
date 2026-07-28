"""Single source of truth for the body-part fragment classifier.

POSITIVE class = "this crop is ONLY body parts" (a hand, a leg, a torso sliver) —
the crops we want to DISCARD before ReID clustering.
NEGATIVE class = the no-face crops the reviewer pruned in the labeling app: real,
usable person crops that simply have no visible face.

Face-bearing crops never enter training OR inference: the face detector is gate 1
and this classifier is gate 2, so the training distribution matches deployment.

Pipeline (run in this order):
    embed.py    labeled crops -> cached 384-d CLS + geometry   [GPU, must run on the VM]
    train.py    cached features -> LogisticRegression + thresholds   [CPU, iterate locally]
    predict.py  full gallery -> p_fragment per crop            [GPU, VM]

Self-contained apart from reusing finetune_reid.load_backbone, extract_embeddings.crop_bbox
and reid_dataset's ImageNet constants / val transform.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo / data (VM paths)
# ---------------------------------------------------------------------------

DINOV3_REPO = Path("/data/AI/Tomer/dinov3")
DATASET_ROOT = Path("/data/AI/Tomer/person_reid/dataset_utils/dataset_finetune/Wedding[1]")

DETECTIONS_FILENAME = "detections.json"
DETECTIONS_BASELINE_FILENAME = "bodyfilter_baseline.json"   # post-face-filter candidate pool
LABELS_FILENAME = "bodyfilter_result.json"    # written by the labeling app, per gallery
IMAGES_SUBDIR = "images"

# ---------------------------------------------------------------------------
# Galleries
# ---------------------------------------------------------------------------

# None -> auto-discover every <proj_id> under DATASET_ROOT that has LABELS_FILENAME.
# Only galleries the reviewer actually Saved/Done have that file, so a newly
# labeled gallery is picked up with no config change.
# Labeled as of 2026-07-27:  17601187 (367 pos) | 18226778 (170 pos) | 21833423 (8 pos)
GALLERIES = None

# Gallery for the within-gallery selection CV (run 1). This run picks transform ×
# feature-set × C; it is optimistic in absolute terms (same venue, same people)
# but valid for RANKING variants, since every variant shares the same optimism.
# None -> the labeled gallery with the most positives.
CV_GALLERY = None

# Held-out gallery highlighted in the cross-gallery report (run 2). It still
# appears as one leave-one-gallery-out fold either way; this only marks it.
# None -> no gallery singled out.
TEST_GALLERY = None    # e.g. "21833423"

# A leave-one-gallery-out fold whose held-out gallery has fewer positives than
# this is reported but flagged as statistically thin — recall measured on a
# handful of positives has a CI wide enough to be meaningless. Never silently
# dropped; the report lists which folds were flagged.
MIN_POSITIVES_PER_FOLD = 20

# ---------------------------------------------------------------------------
# Backbone: V18 SSL pretrain (PRE-finetune), ViT-S/16, 384-d CLS
# ---------------------------------------------------------------------------
# Deliberately the pretrain backbone, NOT a finetune ckpt: SupCon is trained to
# map a hand crop and a full-body crop of the same person to the same place, i.e.
# it destroys exactly the crop-completeness signal this classifier needs.

PRETRAIN_CFG = DINOV3_REPO / "train_pictime/pictime_vitl_im1k_lin834.yaml"
PRETRAIN_CKPT = "/data/AI/Tomer/dinov3/train_pictime/experiments_V2/V18/ckpt/19750"
BACKBONE_WHICH = "teacher"
BACKBONE_TAG = "v18"
CLS_DIM = 384

# ---------------------------------------------------------------------------
# Crop transform
# ---------------------------------------------------------------------------
# reid_dataset.get_val_transform is Resize(256) -> CenterCrop(224): Resize with an
# int scales the SHORT side, so a 100x300 full-body crop becomes 256x768 and the
# centre crop keeps roughly the torso only — the ViT then sees a torso for BOTH
# "full body" and "torso only" inputs, and the aspect ratio is gone. That is the
# exact distinction this classifier exists to make, so it gets its own transform.
#
#   "letterbox"  pad to square with the ImageNet mean, then resize
#                (keeps all content AND makes the shape visible to the ViT)
#   "warp"       resize straight to CROP_SIZE x CROP_SIZE
#                (keeps all content, loses shape — geometry features restore it)
#   "reid_val"   the finetune val transform (baseline / negative control)

TRANSFORMS = ("letterbox", "warp", "reid_val")
TRANSFORM = "letterbox"        # fallback; train.py picks the winner and predict.py reads it
RUN_ALL_TRANSFORMS = True      # embed.py caches all three in ONE image-decode pass
CROP_SIZE = 224

# Re-embed even when a valid cache already exists. Normally leave False: the cache
# is fingerprinted, so a genuinely stale one is detected and rejected on load —
# this flag is only for forcing a rebuild by hand.
EMBED_FORCE = False

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
#   "cls"       384-d CLS only
#   "geom"      geometry / context scalars only (cheap, backbone-free baseline)
#   "cls+geom"  both — expected best, since geometry carries what the transform hides

FEATURE_SETS = ("cls+geom", "cls", "geom")
FEATURE_SET = "cls+geom"       # the one predict.py uses; train.py picks the winner

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
# n is small (hundreds of positives) against 384+12 features, so a default
# LogisticRegression(C=1.0) separates the training set perfectly and generalises
# badly. C is swept low and chosen by CV, never by train score.

SEED = 42
CV_FOLDS = 5
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
CLASS_WEIGHT = "balanced"
MAX_ITER = 2000
KNN_K = 5                      # sanity baseline; if it wins, collect data rather than add capacity

# ---------------------------------------------------------------------------
# Thresholds — two of them, tuned for OPPOSITE objectives. Do not collapse.
# ---------------------------------------------------------------------------
# LABELING: gates which crops the labeling app shows next round. A fragment that
#   falls below it is never shown, never labeled, and silently becomes a false
#   negative in the dataset — a permanent error. Target very high recall and
#   accept a smaller labeling speed-up; the speed-up compounds in later rounds.
# DEPLOY: gates which crops are dropped before ReID clustering. An error here
#   costs one crop, and HDBSCAN already absorbs stray noise.

LABELING_TARGET_RECALL = 0.99
DEPLOY_TARGET_PRECISION = 0.95

# ---------------------------------------------------------------------------
# predict.py — the labeling display filter
# ---------------------------------------------------------------------------

# None -> use model_path(), the file every train run overwrites. Set to an explicit
# path only to score with an archived model.
MODEL_PATH = None

# The app shows: every crop above the labeling threshold, PLUS this many crops
# sampled at random from the SUPPRESSED pool. The random quota is what keeps a
# live supply of ordinary negatives coming in from every new venue — without it,
# `deleted_keys` degenerates to boundary negatives only as the model improves,
# negative diversity freezes at the first few galleries, and the training prior
# drifts away from deployment (which silently moves the thresholds).
# Sized in whole UI batches so it lands cleanly in the reviewer's workflow.
UI_BATCH_SIZE = 100
RANDOM_BATCHES = 3
RANDOM_QUOTA = UI_BATCH_SIZE * RANDOM_BATCHES

# Sampling from the suppressed pool (not the whole baseline) means the two parts
# compose cleanly: above-threshold crops are 100% reviewed (a census) and the
# suppressed pool is sampled, so
#     estimated missed fragments = (fragments found in the sample) / sampling_fraction
# That estimate is the only direct read on the classifier's false-negative rate.

# Score only crops in the baseline pool. Face-bearing crops are out of
# distribution — the model never saw one in training — so scoring them would
# produce meaningless numbers. Galleries with no baseline file are skipped and
# counted rather than scored against raw detections.
REQUIRE_BASELINE = True

# Re-score galleries that already have a scores file, overwriting it. True after a
# retrain so the whole dataset reflects the new model; False to score only projects
# that have never been scored.
# NOTE the overwrite discards the previous round's file. Full provenance lives inside
# each file so the current one always self-documents, but per-round history is lost —
# archive them app-side if you ever want to audit which model shaped which round.
PREDICT_FORCE = True

# Never re-display a crop the reviewer has already judged. kept_keys | deleted_keys
# from bodyfilter_result.json are subtracted from BOTH show_keys and audit_keys, so a
# labeled gallery can be re-scored with a newer model without showing the same crop
# twice. They are still scored and still appear in `scores` — which is what lets you
# measure real precision/recall per gallery straight from the scores file.
EXCLUDE_REVIEWED = True

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "results" / BACKBONE_TAG

# Model-tied artifacts keyed by backbone tag (never built with a ternary) so it
# stays obvious which backbone produced which cache / model as versions accumulate.
#
# EMBED_CACHE and SCORES live INSIDE each project dir. The `classifier_` prefix keeps
# them clearly apart from the embeddings_<v51>.npz ReID files already in those dirs.
# The embedding cache is written ONCE per gallery and read by both train.py and
# predict.py forever after — the backbone is frozen, so those vectors never change.
EMBED_CACHE = {"v18": "classifier_embeddings_v18_{transform}.npz"}
SCORES_FILENAME = {"v18": "classifier_scores_v18.json"}

# One fixed filename, overwritten by every train run — so predict.py always picks up
# the newest model without the path changing. Nothing is lost by dropping the winning
# transform / feature set from the name: both are stored inside the bundle, printed by
# predict.py at startup, and recorded in report.md and every scores file.
MODEL_FILE = {"v18": "model_v18.pkl"}

REPORT_FILE = OUTPUT_DIR / "report.md"
CURVES_FILE = OUTPUT_DIR / "curves.png"


def embed_cache_name(transform: str) -> str:
    """Filename only — the cache lives in the project dir, not OUTPUT_DIR."""
    return EMBED_CACHE[BACKBONE_TAG].format(transform=transform)


def model_path() -> Path:
    return OUTPUT_DIR / MODEL_FILE[BACKBONE_TAG]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

BATCH_SIZE = 64
NUM_WORKERS = 4
