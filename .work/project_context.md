---
name: Project Context
description: DINOv3 repo — ViT pretraining for person ReID on 6.5M Pictime images, separate from production ReID repo
type: project
originSessionId: aa1f4eaa-d19c-4dc3-b9d3-e14c6f92fa58
---
- Architecture: DINOv3 (self-supervised ViT pretraining)
- Model: ViT-Small, patch size 16, RoPE positional embeddings
- Pretrained weights: dinov3_vits16_pretrain_lvd1689m (ViT-S/16 from LVD-142M)
- Task: Person re-identification (ReID)
- Pretraining data: ~6.5M images from Pictime
- This repo: Pretraining only — the production ReID pipeline is a separate repository
- Part of a larger pipeline: pretrain here → fine-tune/deploy in production ReID repo

## Training Setup
- Main training script: `train_pictime/train_dino_grad_accum.py` (current)
- Sister script: `train_pictime/train_dino.py` (no grad accumulation; legacy but kept patched)
- Config: `train_pictime/pictime_vitl_im1k_lin834.yaml`
- GPU batch size: 16, effective batch size: 256 (was ~1024 target in grad_accum script)
- Optimizer: AdamW, cosine LR schedule, lr=0.001, 10 epoch warmup
- OFFICIAL_EPOCH_LENGTH: 30,000 iters, 100 epochs
- Checkpointing: every 5000 iters, keep last 3
- Logging: W&B every 10 iters
- Eval: every 12,500 iters
- Arch fields (post 2026-04-23 fix, now matched to LVD-142M release): `n_storage_tokens=4`, `mask_k_bias=true`, `layerscale=1e-5`, patch=16, arch=vit_small
- Foundation loader: `train_pictime/foundation_loader.py::load_foundation_into_backbone()` — loads the LVD-142M `.pth` into the FSDP2-wrapped `student["backbone"]` and re-syncs the EMA teacher. Called from both train scripts after `model.init_weights()`.
- Scripts: no argparse anymore — paths are inlined as `SimpleNamespace(...)` at the top of `main()`; flip `resume=False/True` in `train_dino_grad_accum.py` to switch between fresh/resume.
- Status (as of 2026-04-23): V1-V9 DCPs are random-init from-scratch DINOv3 on Pictime (due to the pre-fix bug). V10 onwards: proper continued-pretrain from LVD-142M. Launch V10 fresh (no `--resume`) to get the first correct run.

## Finetune Dataset Preparation
- Location on VM: `/data/AI/Tomer/person_reid/dataset_utils/dataset_finetune/Portraits[26]/`
- 500K images, ~50K project folders total; current training uses a **single-cluster-projects subset → ~21K identities/classes** (filtered via `single_cluster_projects.json`)
- Each project: ~100 images of the same person
- Each project: `images/` folder + `detections.json` (YOLO11 person bboxes, normalized [0,1] coords)
- Detection script: `train_pictime/my_prompt` (original code, uses `body_detection_src` from separate repo)
- Embedding script: `train_pictime/extract_embeddings.py` (ViT-B/16 foundation model, outputs `embeddings.npz` per project)
- Goal: extract DINOv3 embeddings per crop → cluster by photo session (same clothes)
- Clustering script: `train_pictime/cluster_embeddings.py` (HDBSCAN, outputs `clusters.json` per project)
- Pipeline: detect persons → extract embeddings → cluster → reviewers correct → use as finetune labels
- Reviewer-corrected labels: `clusters_fixed.json` per project (preferred over `clusters.json`)
- cluster_id=-1 is valid (reviewers verified)
- UI viewer: separate repo, displays project images with clusters in separate columns (test data at `/data/AI/Tomer/UI_dataset_view/data`)

## Finetune Training (train_pictime/finetune/)
- Loss: SupCon (default) or ArcFace (angular-margin classification, pytorch-metric-learning). Selected via `loss:` config.
- **ArcFace knobs** (added 2026-04-20 after Trial 5 head-distortion failure):
  - `arcface_m_deg` — margin in degrees (library stores it in radians after `init_margin()`)
  - `arcface_margin_warmup_iters` — ramp margin 0→target over first N iters (runtime mutation of `criterion.margin` in radians; library doesn't cache cos(m), so this is safe)
  - `arcface_classifier_lr` / `arcface_classifier_wd` — classifier `W` goes into its own AdamW param group (wd=0 recommended since prototypes are cosine-normalized)
  - `maybe_unfreeze()` preserves the 3-group split (head / classifier / backbone) for future Mode-C ArcFace runs
- **Original SupCon details:**
- Loss: SupCon (Supervised Contrastive Learning, Khosla et al. 2020)
- Backbone: Teacher from pretrain DCP checkpoint (V9/ckpt/15000), ViT-S/16 embed_dim=384
- Mixed precision: fp32 params/grads/optimizer state + `torch.autocast(bf16)` around backbone forward. Backbone is loaded unwrapped (no FSDP2) — DCP is read via a throwaway FSDP2 SSLMetaArch, full state dict extracted via `get_model_state_dict(full_state_dict=True)`, then loaded into a fresh `build_model_from_cfg` backbone. Reason: pretrain's FSDP2 `MixedPrecisionPolicy(param_dtype=bf16)` + fp32 grad_dtype crashes when backbone blocks unfreeze (bf16 grads can't be assigned to fp32 grad slots). Standard mixed-precision recipe avoids this and avoids bf16-grad update swamping on small `lr_backbone`.
- Head: MLP projection (384 → 384 → 128), L2-normalized output
- Sampling: PK sampler — P=16 projects × K=4 crops, cross-project negatives only
- Freeze modes: A (frozen backbone) or C (progressive unfreezing of last N blocks)
- Eval: Query/gallery ReID protocol — Rank-1/5/10, mAP, silhouette score (cosine, stratified subsample capped at 8K)
- Checkpointing: best 3 by silhouette score; `ckpt_every` controls both eval and checkpoint frequency
- Index: offline `build_index.py` generates `.npz` for fast loading; currently filtered to single-cluster projects via `single_cluster_projects.json`
- W&B: separate project `person-reid-finetune` (not shared with pretrain)
- Run name format: `finetune_reid_vits16_bs64[_frozen][_<experiment_tag>]`
- Experiment tagging (loss-agnostic, plumbed through in 2026-04-19 session):
  - `experiment_tag` in `reid_config.yaml`: free-form string appended as suffix (e.g. `tau0.1`, `arcface_m0.5`)
  - `experiment_group` in `reid_config.yaml`: W&B group for overlaying trial runs (e.g. `trial1_tau_sweep`)
  - Both default `null` → no change to unannotated runs. `init_wandb()` in `wandb_logger.py` accepts `group` kwarg
- Experiment log: `train_pictime/finetune/experiment.md` — runs table, hypotheses, ranked trials
