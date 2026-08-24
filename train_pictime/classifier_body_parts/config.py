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

# The labeling app's approval registry - one entry per completed gallery, carrying the
# admin approval flag. Every entry with `approved: true` trains the model; that is the
# only condition. Lives in the UI repo, NOT under DATASET_ROOT, because it spans
# datasets. Read by train.approved_galleries.
COMPLETION_LOG = Path("/data/AI/Tomer/UI_dataset_view/bodyfilter_completion_log.json")

# ---------------------------------------------------------------------------
# Galleries
# ---------------------------------------------------------------------------
# Which galleries train the model is NOT configured here. It is read from
# COMPLETION_LOG: every entry with `approved: true`, and nothing else. Approving a
# gallery in the labeling app is the whole mechanism -- see train.approved_galleries.




# ---------------------------------------------------------------------------
# Backbone — ViT-S/16, always the 384-d CLS (never the 128-d projection output)
# ---------------------------------------------------------------------------
# "finetune" (default): the DEPLOYED ReID backbone, so production runs ONE forward
#   pass that feeds both the body embedding and this classifier. The projection head
#   is never built — we take the pre-head CLS. The head is where SupCon's
#   nuisance-invariance is enforced, so the backbone still carries crop-completeness
#   signal; and mode-C only unfreezes the last few blocks, so most of it is still the
#   V18 pretrain weights.
# "pretrain": the V18 SSL backbone, fully untouched by SupCon. Cleanest features in
#   principle, but costs production a second forward pass per crop.
#
# Both coexist — BACKBONE_TAG namespaces the caches and the model — so this is a
# measurable comparison, not a guess. See README.
BACKBONE_SOURCE = "finetune"
BACKBONE_TAG = "ft_v52"
BACKBONE_WHICH = "teacher"
CLS_DIM = 384

# "pretrain" source
PRETRAIN_CFG = DINOV3_REPO / "train_pictime/pictime_vitl_im1k_lin834.yaml"
PRETRAIN_CKPT = "/data/AI/Tomer/dinov3/train_pictime/experiments_V2/V18/ckpt/19750"

# "finetune" source — V44, the ckpt production actually loads
# (mirrors model_comparison/config.py NEW_CKPT). The base arch/weights come from
# reid_config.yaml rather than being hardcoded, so they cannot drift from whatever
# the finetune run actually started from.
REID_CONFIG = DINOV3_REPO / "train_pictime/finetune/reid_config.yaml"
FINETUNE_CKPT = "/data/AI/Tomer/person_reid/models/ckpt_V52_iter8991_sil0.4692.pt"


def backbone_ckpt() -> str:
    """The checkpoint the active BACKBONE_SOURCE loads. Part of the feature signature,
    so changing it invalidates every cache AND forces the classifier to be retrained."""
    return FINETUNE_CKPT if BACKBONE_SOURCE == "finetune" else str(PRETRAIN_CKPT)

# ---------------------------------------------------------------------------
# Crop transform — PINNED to reid_val, deliberately
# ---------------------------------------------------------------------------
# The point of running this classifier on the DEPLOYED ReID backbone is that production
# serves the body embedding and the fragment score from ONE forward pass. That only
# holds if both see the SAME PIXELS. The body embedding uses
# reid_dataset.get_val_transform, so this classifier must use it too — any other
# transform here silently reintroduces the second forward pass that switching to the
# finetune backbone was meant to remove.
#
# reid_val's known weakness — the original reason letterbox and warp existed — is that
# Resize(256) scales the SHORT side, so a 100x300 full-body crop becomes 256x768 and the
# centre crop keeps roughly the torso: the ViT then sees a torso for BOTH "full body"
# and "torso only", and the aspect ratio is gone. The GEOMETRY features (log_aspect,
# sqrt_rel_area, log_crop_px) are what put that information back, which is why pinning
# costs so little. Measured on the ft_v52 run-1 selection (cls+geom, best C):
#
#   letterbox  PR-AUC 0.9738    retired
#   warp       PR-AUC 0.9718    retired
#   reid_val   PR-AUC 0.9719 <- pinned; ~0.2% relative, for halving production passes
#
# Both retired transforms are still implemented in dataset.get_transform so the ft_v44 /
# ft_v52 ablation stays reproducible; they are simply never embedded or selected. Do not
# put them back here without re-deciding the two-forward-pass question.

TRANSFORMS = ("reid_val",)     # NOTE the comma — ("reid_val") is a str, and iterating it
                               # yields 'r','e','i','d',... through get_transform
TRANSFORM = "reid_val"         # the only transform; train.py "selects" it trivially

# Decorative while reid_val is the only transform: get_val_transform hardcodes
# Resize(256) -> CenterCrop(224) and ignores this. It still sits in the feature
# signature, so changing it invalidates every cache while altering no pixels.
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

FEATURE_SET = "cls+geom"       # the one predict.py uses; train.py picks the winner

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
# n is small (hundreds of positives) against 384+12 features, so a default
# LogisticRegression(C=1.0) separates the training set perfectly and generalises
# badly. C is swept low and chosen by CV, never by train score.

SEED = 42
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
CLASS_WEIGHT = "balanced"
MAX_ITER = 2000

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

# Both thresholds are calibrated on the POOLED leave-one-gallery-out predictions —
# every gallery's held-out scores concatenated into one set, then walked once. That
# pools the DATA rather than averaging per-gallery thresholds: the threshold -> recall
# map is non-linear, so averaging thresholds lands short of the target, and lands short
# specifically on the galleries where the model is weakest. Pooling also weights each
# gallery by its size automatically — and by the right size per metric, since positives
# drive recall while negatives drive FPR.
#

# ---------------------------------------------------------------------------
# Training admission
# ---------------------------------------------------------------------------
# Approved in COMPLETION_LOG. That is the whole rule, and there is no knob for it.
# Approving a gallery in the labeling app is what puts it into training.

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

# Artifact names carry BACKBONE_TAG so it stays obvious which backbone produced which
# cache / model as versions accumulate — but as templates rather than tag-keyed dicts,
# which would KeyError the moment a new tag is set. (Deliberate deviation from the
# {model_id: filename} convention; the point of that rule — provenance in the filename
# — is preserved.)
#
# The embedding cache lives INSIDE each project dir. The `classifier_` prefix keeps it
# clearly apart from the embeddings_<v51>.npz ReID files already in those dirs. It is
# written ONCE per gallery and read by both train.py and predict.py forever after.
EMBED_CACHE = "classifier_embeddings_{tag}_{transform}.npz"

# One fixed filename, overwritten by every train run — so predict.py always picks up
# the newest model without the path changing. Nothing is lost by dropping the winning
# transform / feature set from the name: both are stored inside the bundle, printed by
# predict.py at startup, and recorded in report.md and every scores file.
MODEL_FILE = "model_{tag}.pkl"

# DELIBERATELY NOT tagged: the labeling UI reads this path, and it must stay stable
# across backbone swaps. Provenance is not lost — the file's `model` block records the
# source, ckpt, transform, feature set and galleries that produced it.
SCORES_FILENAME = "classifier_scores.json"

REPORT_FILE = OUTPUT_DIR / "report.md"
CURVES_FILE = OUTPUT_DIR / "curves.png"


def embed_cache_name(transform: str) -> str:
    """Filename only — the cache lives in the project dir, not OUTPUT_DIR."""
    return EMBED_CACHE.format(tag=BACKBONE_TAG, transform=transform)


def model_path() -> Path:
    return OUTPUT_DIR / MODEL_FILE.format(tag=BACKBONE_TAG)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

BATCH_SIZE = 64
NUM_WORKERS = 4
