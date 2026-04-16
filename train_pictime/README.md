# train_pictime

Person re-identification (ReID) pipeline built on top of Meta's DINOv3 codebase. This directory contains everything Pictime-specific: self-supervised pretraining on ~6.5M images, finetune dataset preparation (detection, embedding, clustering), and supervised ReID finetuning.

The pretrained backbone produced here feeds into the production ReID pipeline (separate repo).

---

## Directory Structure

```
train_pictime/
├── train_dino_grad_accum.py   # Main pretraining script (with gradient accumulation)
├── train_dino.py              # Original pretraining script (no grad accumulation, legacy)
├── pictime_vitl_im1k_lin834.yaml  # Pretrain config (model, optimizer, augmentations)
├── pictime_dataset.py         # PicTimeImageDataset — reads image paths from .txt file
├── run_name.py                # W&B run name builder
├── wandb_logger.py            # W&B logging helpers (log_wandb, log_paired, log_paired_variant)
├── migrate_old_runs.py        # One-off script to backfill old W&B runs with new key format
├── test_wandb_grouping.py     # Test for log_paired / log_paired_variant
├── extract_embeddings.py      # Extract DINOv3 ViT-B/16 embeddings for person crops
├── cluster_embeddings.py      # HDBSCAN clustering of person embeddings per project
├── eval/
│   ├── evaluator.py           # In-process pretrain evaluator (views, rank, geom, proto)
│   ├── eval_config.yaml       # Eval frequencies, sample sizes, metric params
│   ├── embed.py               # Shared embedding extraction for eval
│   ├── metrics_views.py       # Multi-view consistency metrics
│   ├── metrics_rank.py        # Embedding variance & effective rank
│   ├── metrics_geometry.py    # Geometric metrics (cosine distributions, kNN)
│   └── metrics_prototype.py   # Prototype utilization metrics
├── finetune/
│   ├── finetune_reid.py       # Main finetune training script
│   ├── reid_config.yaml       # Finetune hyperparameters
│   ├── reid_dataset.py        # ReIDCropDataset + PKBatchSampler
│   ├── reid_evaluator.py      # ReID eval (Rank-1/5/10, mAP, silhouette score)
│   ├── supcon_loss.py         # Supervised Contrastive Loss (Khosla et al. 2020)
│   ├── build_index.py         # Offline: scan projects → reid_index.npz
│   └── find_single_cluster_projects.py  # Find "clean" projects (for test runs)
└── utils_pictime/
    └── utils_dataset.py
```

---

## 1. Pretraining

Self-supervised ViT pretraining using DINOv3 (DINO + iBOT + KoLeo losses) on ~6.5M Pictime images.

### Model

- Architecture: **ViT-Small/16** with RoPE positional embeddings
- Pretrained weights: `dinov3_vits16_pretrain_lvd1689m` (LVD-142M foundation)
- Teacher/student setup with EMA (standard DINOv3)

### How it works

1. `PicTimeImageDataset` (`pictime_dataset.py`) reads image paths from a .txt file
2. DINOv3 augmentation pipeline: global crops (224px) + 8 local crops (96px), color jitter, horizontal flips
3. Gradient accumulation to simulate large batches on a single GPU
4. Cosine LR schedule with warmup, cosine weight decay, teacher EMA momentum schedule

### Config

All in `pictime_vitl_im1k_lin834.yaml`:

| Parameter | Value |
|---|---|
| GPU batch size | 16 |
| Effective batch size | 1024 (via grad accumulation) |
| Optimizer | AdamW (lr=0.001, 10 epoch warmup) |
| Epochs | 100 |
| OFFICIAL_EPOCH_LENGTH | 30,000 iters |
| Checkpointing | Every 5,000 iters, keep last 3 |
| Eval | Every 12,500 iters |
| W&B logging | Every 10 iters |

### Running

```bash
# Fresh start (auto-creates next V{n} directory)
python3 -m train_pictime.train_dino_grad_accum

# Resume from last checkpoint
# Set resume_training = True in main() of train_dino_grad_accum.py
python3 -m train_pictime.train_dino_grad_accum
```

Output goes to `/data/AI/Tomer/dinov3/train_pictime/experiments/V{n}/`.

### Resume support

- Loads model + optimizer from latest DCP checkpoint
- Restores exact iteration; LR/WD/momentum/teacher_temp schedules pick up at correct step
- Data loader sampler advances to skip already-seen samples
- W&B run resumed via persisted `wandb_run_id.txt`

### Pretrain evaluation

The evaluator (`eval/evaluator.py`) runs during training at configured intervals and computes:

- **Views pack** (every 1K iters): multi-view consistency between two crops of same image
- **Rank pack** (every 2K iters): embedding variance, effective rank (raw + centered)
- **Proto pack** (every 2K iters): prototype utilization in DINO/iBOT heads
- **Geom pack** (every 10K iters): cosine distance distributions, kNN metrics

All metrics logged for both teacher and student on the same W&B chart (via `log_paired` / `log_paired_variant` which use `wandb.plot.line_series`).

### W&B

- Project: `person-reid-dinov3`
- Run name format: `person_reid_vits16_effbs1024_lr0.001`
- Train keys: `train/total_loss`, `train/ibot_loss`, `train/dino_global`, `train/koleo`, `train/lr`, `train/wd`, `train/mom`
- Eval keys: grouped under `eval/` prefix with teacher/student paired charts

### NaN loss guard

If loss diverges to NaN/Inf, the optimizer step is skipped and gradients are zeroed. Training continues from the next iteration. The eval's `eigvalsh` call also has a try/except to avoid crashes on ill-conditioned covariance matrices.

---

## 2. Finetune Dataset Preparation

The finetune dataset comes from ~50K project folders under `Portraits[26]/`. Each project is a photo session with ~100 images of the same person(s). The goal is to label which person crops belong to the same identity.

### Pipeline

```
detect persons (YOLO11) → extract embeddings (ViT-B/16) → cluster (HDBSCAN) → human review
```

### 2.1 Person Detection

Done externally using YOLO11 (code in `body_detection_src` from separate repo, invoked via `my_prompt`). Produces `detections.json` per project:

```json
{
  "photo1.jpg": [
    {"bbox": [0.1, 0.2, 0.5, 0.8]},
    {"bbox": [0.6, 0.1, 0.9, 0.9]}
  ]
}
```

Bboxes are normalized [0, 1] coordinates: `[x1, y1, x2, y2]`.

### 2.2 Embedding Extraction

`extract_embeddings.py` — runs DINOv3 **ViT-B/16 foundation model** (not the pretrained ViT-S/16) over all person crops.

```bash
python3 -m train_pictime.extract_embeddings
```

- Reads `detections.json` per project, crops each bbox, runs through ViT-B/16
- Outputs `embeddings.npz` per project: `filenames`, `bbox_indices`, `embeddings` [M, 768] L2-normalized
- Has resume/skip logic: skips projects where `embeddings.npz` already matches bbox count
- `DEBUG = True` enables `debug_visualize()` for bbox verification
- Dataset path: `/data/AI/Tomer/person_reid/dataset_utils/dataset_finetune/Portraits[26]/`

### 2.3 Clustering

`cluster_embeddings.py` — HDBSCAN clustering per project to group crops by identity.

```bash
python3 -m train_pictime.cluster_embeddings
```

- Algorithm: HDBSCAN (euclidean on L2-normed embeddings = cosine distance)
- `min_cluster_size=3`, `min_samples=None`
- Outputs `clusters.json` per project:

```json
{
  "photo1.jpg": [
    {"bbox_index": 0, "cluster_id": 0},
    {"bbox_index": 1, "cluster_id": 1}
  ]
}
```

- `cluster_id = -1` means noise/outlier (valid after human review)
- Atomic saves, skip logic (`FORCE = False` to skip already-clustered)
- Test path: `/data/AI/Tomer/UI_dataset_view/data`

### 2.4 Human Review

Reviewers correct cluster assignments using a separate UI repo that displays project images with clusters in separate columns. Corrected labels are saved as `clusters_fixed.json` (preferred over `clusters.json` at load time).

---

## 3. Finetune Training

Supervised contrastive learning on the labeled person crops. Trains a projection head on top of the frozen pretrained teacher backbone.

### How it works

1. **`build_index.py`** scans all projects, reads `clusters_fixed.json` (or `clusters.json`), and writes a single `reid_index.npz` — run this once before training
2. **`finetune_reid.py`** loads the index, splits by project into train/val, and trains

### Running

```bash
# Step 1: Build index (run once, or re-run after data changes)
python3 -m train_pictime.finetune.build_index

# Step 2: Train
python3 -m train_pictime.finetune.finetune_reid
```

The index currently filters to projects listed in `single_cluster_projects.json` (clean projects with exactly 1-2 identity clusters, for early experiments).

### Architecture

- **Backbone**: Teacher network from pretrain checkpoint (DCP format), ViT-S/16, embed_dim=384
- **Projection head**: MLP `384 → 384 → 128` with BatchNorm + ReLU, L2-normalized output
- **Loss**: Supervised Contrastive Loss (SupCon, Khosla et al. 2020)

### Sampling: PK Batch Sampler

Each batch: **P=16** projects, **1 identity per project**, **K=4** crops per identity = batch of 64. All negatives are cross-project (guaranteed different people). Identities with < K samples are never selected.

### Freeze Modes

| Mode | Description |
|---|---|
| **A** (default) | Backbone fully frozen. Only the projection head trains. |
| **C** | Head trains first. At iteration `unfreeze_after`, the last `unfreeze_n_blocks` transformer blocks + final layer norm are unfrozen with a lower LR (`lr_backbone`). |

### Config

All in `finetune/reid_config.yaml`:

| Parameter | Value |
|---|---|
| Pretrained checkpoint | `experiments/V9/ckpt/15000` |
| Backbone | Teacher, ViT-S/16 |
| P x K | 16 x 4 = 64 per batch |
| Epochs | 10 |
| LR (head) | 1e-3, cosine decay to 1e-6 |
| LR (backbone, mode C) | 1e-5 |
| Warmup | 500 iters |
| Temperature | 0.07 |
| Val ratio | 5% of projects |

### Evaluation

`reid_evaluator.py` runs during training every `ckpt_every` iterations:

- **Query/gallery protocol**: 1 random query per identity, rest as gallery
- **Rank-1/5/10**: cumulative match characteristic
- **mAP**: mean average precision
- **Silhouette score**: cosine-based, stratified subsample capped at 8K samples (configurable)

### Checkpointing

- Best 3 checkpoints by **silhouette score** (higher = better)
- Filename format: `ckpt_iter{N}_sil{score}.pt`
- Output: `/data/AI/Tomer/dinov3/train_pictime/finetune_experiments/V{n}/ckpt/`

### Filtering to clean projects

`find_single_cluster_projects.py` scans all projects and saves a JSON list of "clean" folder names (projects with exactly 2 non-noise clusters). This list is loaded by `build_index.py` to filter the index for early test runs.

```bash
# Generate the filter list
python3 -m train_pictime.finetune.find_single_cluster_projects

# Rebuild index with filter applied
python3 -m train_pictime.finetune.build_index
```

---

## Data Locations (VM)

| What | Path |
|---|---|
| Pretrain images list | `/data/AI/Tomer/dinov3/train_pictime/train_paths.txt` |
| Pretrain val images list | `/data/AI/Tomer/dinov3/train_pictime/val_paths_100K.txt` |
| Pretrain experiments | `/data/AI/Tomer/dinov3/train_pictime/experiments/V{n}/` |
| Pretrained weights (foundation) | `/data/AI/Tomer/dinov3/dinov3/weights/` |
| Finetune dataset | `/data/AI/Tomer/person_reid/dataset_utils/dataset_finetune/Portraits[26]/` |
| Finetune index | `Portraits[26]/reid_index.npz` |
| Finetune experiments | `/data/AI/Tomer/dinov3/train_pictime/finetune_experiments/V{n}/` |
| Single-cluster filter | `/data/AI/Tomer/dinov3/train_pictime/finetune/single_cluster_projects.json` |
| UI test data | `/data/AI/Tomer/UI_dataset_view/data` |

---

## Per-Project File Layout

Each project folder under `Portraits[26]/` can contain:

```
ProjectName/
├── images/              # Original photos
├── detections.json      # YOLO11 person bboxes (normalized)
├── embeddings.npz       # ViT-B/16 embeddings per crop
├── clusters.json        # HDBSCAN cluster assignments
└── clusters_fixed.json  # Human-reviewed cluster assignments (preferred)
```