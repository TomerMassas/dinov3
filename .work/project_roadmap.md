---
name: Roadmap
description: Active todos, priorities, and project goals for DINOv3 pretraining
type: project
originSessionId: aa1f4eaa-d19c-4dc3-b9d3-e14c6f92fa58
---
# Roadmap

## Goals
- [ ] Pretrain ViT on 6.5M Pictime images using DINOv3, continuing from the LVD-142M foundation
- [ ] Produce embeddings suitable for downstream person ReID
- [ ] Prepare finetune dataset — cluster 500K images by photo session for ReID labels

## Active Todos (post 2026-04-23 foundation-load fix)

- [ ] **W&B missing train-loss charts — finish diagnosis** (in progress as of 2026-04-23). Diagnostic print is live in `wandb_logger.py:56-57` ("[W&B LOG @ iter N] keys=…" every 100 iters). Waiting for Tomer's first stdout lines past ~200 iters. Then apply case-specific fix per plan file and remove the diagnostic print.
- [ ] **Launch V10 fresh** — no `--resume`. Should log `Loading FOUNDATION weights from …` and `Foundation load done — missing: 0, unexpected: 0`. First `train/dino_global` should be ~4-6 (foundation) not ~11 (random). If missing/unexpected ≠ 0, arch drift — reconcile before training.
- [ ] **Consider lower peak LR for V10** — `cfg.optim.lr=0.001` was tuned for from-scratch training. Continued pretrain from a strong foundation typically wants 5e-4 or 1e-4 peak. Non-blocking, but worth deciding before the cosine schedule commits.
- [ ] **Re-baseline finetune at V10/ckpt/~5-10K** — rerun Mode-A and Mode-C SupCon against the V10 backbone. V9-backbone results (mAP 0.37 green, 0.415 blue) are no longer the baseline; with a real foundation-continued backbone, both absolute numbers and the mode-A-vs-C gap may shift.
- [ ] **Re-visit "base backbone" finetune flag** (the original ask from this session) — becomes a clean 3-way comparison once V10 has enough iters: LVD-142M-direct (via hub) vs V10 (foundation + continued-pretrain on Pictime) vs V9 (random-init + SSL on Pictime). Plan file `currently-the-arcaface-seems-kind-lovelace.md` already has the hub-loader design from the earlier iteration; worth revisiting.
- [ ] **Re-examine ArcFace viability on the V10 backbone** — Trials 5/6 were against V9 random-init; the head-distortion pattern may or may not reproduce on a stronger backbone. Reset the ArcFace trials table after V10.
- [ ] **Optional faithfulness tweaks** (can be folded into V10 pre-launch if desired):
  - `norm_layer: layernormbf16` in pictime yaml — matches foundation's LayerNorm eps (1e-5 vs current 1e-6). Same state-dict keys, just different forward.
  - `pos_embed_rope_rescale_coords: 2` — enables the coord-rescale training aug the foundation was trained with.
- [ ] Run clustering on full dataset (50K projects) and validate results
- [ ] Iterate on clustering quality (face masking, parameter tuning if needed)

## Superseded / Obsolete
- [~] **Trial 6 — ArcFace refinement** (m=17.2° + warmup + classifier group split) — the plumbing is kept, but the eval comparison was against a random-init backbone. Re-decide whether to rerun after V10 stabilizes.
- [~] **Trial 6b — reduce classifier LR** — same caveat; contingent on Trial 6 rerun.
- [~] **Trial 7 — Mode-C ArcFace on blue's recipe** — same caveat.
- [~] **Trial 3 — Bigger PK batch (P=32/24, K=4)** — still a valid SupCon idea, but lower priority than re-baselining on V10.
- [~] **Trial 4 — Label quality audit** — still valid, diagnostic, orthogonal to the pretrain fix.
- [~] **PK vs random sampling diagnostic** — still valid, orthogonal.

## Completed
- [x] **2026-04-23 — Foundation loader fix** (see progress.md 2026-04-23 entry)
  - Created `train_pictime/foundation_loader.py` with `load_foundation_into_backbone()`
  - Wired into both `train_dino_grad_accum.py` (resume-guarded) and `train_dino.py` (pretrained-guarded)
  - Fixed arch drift: `pictime_vitl_im1k_lin834.yaml` flipped to `n_storage_tokens: 4`, `mask_k_bias: true` to match LVD-142M release
  - Removed dead `parse_args` + `make_setup_args` from both scripts; inline `SimpleNamespace` in `main()`
- [x] **Trial 2 — Stronger Mode C** (blue, `unfreeze4_lr1e-4`): unfreeze 4 blocks @ iter 2000, lr_backbone=1e-4. **Previous leader** (on V9 backbone): mAP 0.415, Rank-1 0.680, silhouette +0.015. Note: all finetune numbers are against the now-invalidated V9 random-init-pretrained backbone.
- [x] **Trial 5 — ArcFace (first attempt)** (orange, m=28.6° frozen): ran ~4k iters, failed — train loss decreased but eval decayed. Diagnosed as head distortion under aggressive margin. Note: against V9 random-init backbone.
- [x] Plumbed ArcFace refinement config: `arcface_margin_warmup_iters`, `arcface_classifier_lr`, `arcface_classifier_wd`; split classifier into own AdamW param group; mutable `criterion.margin` warmup (committed `0bec9ad`)
- [x] First finetune run (mode A, frozen backbone) — baseline established (against V9 random-init backbone)
- [x] Rerun mode C finetune with the unwrap-FSDP2 + autocast(bf16) fix — trained stably past iter 2000, beats mode A by ~2pp mAP (small win, plateaus early)
- [x] Analyzed mode A vs mode C final metrics; identified plateau + negative silhouette as main issues
- [x] Created `experiment.md` log with hypotheses, ranked trials, decisions checklist
- [x] Implemented generic experiment-naming plumbing: `experiment_tag` + `experiment_group` config fields → run name suffix + W&B group
- [x] **Trial 1 — SupCon τ sweep** — superseded by Trial 2's clear Mode-C win (mode-A saturates fast, so τ sweep on mode-A is lower ceiling than tuning unfreezing)
- [x] Separate finetune W&B project + improved run naming
- [x] Filter build_index.py to single-cluster projects for test runs
- [x] Optimize eval extraction (DataLoader + merged query/gallery pass)
- [x] Document full pipeline in train_pictime/README.md
- [x] Test finetune pipeline end-to-end (debugged path, dtype, scheduler issues)
- [x] Resume pretraining (resumed from ~15K iters — now known to have been random-init the whole time)
- [x] Implement HDBSCAN clustering script (`cluster_embeddings.py`)
- [x] Extract DINOv3 embeddings for finetune dataset (ViT-B/16 foundation model)
- [x] Revamp W&B logging — consistent train keys, teacher/student grouped on same graphs
- [x] Update run name format to show effective batch size
- [x] Switch to target_batch_size=256 (training in progress — but on random-init backbone until V10)
- [x] Download 500K finetune images + run YOLO11 person detection
