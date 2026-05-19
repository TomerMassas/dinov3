---
name: Progress Log
description: Session-by-session log of work done together on DINOv3 pretraining
type: project
originSessionId: aa1f4eaa-d19c-4dc3-b9d3-e14c6f92fa58
---
# Progress Log

## 2026-03-12
- Set up persistent memory system for cross-session context
- Documented project context: DINOv3 ViT pretraining for person ReID on 6.5M Pictime images
- Saved working guidelines (no unsolicited changes, explain before acting, stay concise)
- Identified training setup: ViT-S/16 with grad accumulation (effective batch 128), training started
- Revamped W&B logging:
  - Fixed train metric keys in `train_dino_grad_accum.py` to match `train_dino.py` (explicit `train/ibot_loss`, `train/dino_global`, `train/koleo` instead of raw `**avg_metrics`)
  - Added `log_paired` and `log_paired_variant` helpers in `wandb_logger.py` to group teacher/student on the same graph
  - Updated all eval logging calls in `evaluator.py` to use new helpers — cuts eval graph count ~in half
## 2026-03-15
- Fixed W&B paired logging — `log_paired` / `log_paired_variant` (using `wandb.plot.line_series` + `_paired_history` accumulation) weren't working in practice
- Tested and verified fix with `test_wandb_grouping.py` — `log_paired` / `log_paired_variant` now work correctly
- Evaluator uses `log_paired` / `log_paired_variant` (line_series charts grouping teacher/student)
- Created `migrate_old_runs.py` script to backfill 2 old runs (`pictime_vits16_bs16_lc8_lr0.0005` x2) with new format:
  - Train: renames flat keys (`ibot_loss`→`train/ibot_loss`, `dino_global_crops_loss`→`train/dino_global`, `koleo_loss`→`train/koleo`)
  - Eval: converts flat `eval/<group>/teacher/<metric>` + `eval/<group>/student/<metric>` keys into `line_series` charts matching new `log_paired` format
  - Run names NOT changed (stays `pictime_...`)

## 2026-03-16
- Training now running at effective batch_size=256
- Started finetune dataset preparation pipeline
- Created `train_pictime/extract_embeddings.py` — extracts DINOv3 ViT-B/16 foundation model embeddings for person crops
  - Reads `detections.json` (normalized [0,1] bboxes), crops each person, runs through model
  - Uses `dinov3_vitb16()` from hub (not bare `vit_base()` — needs layerscale, storage_tokens, mask_k_bias params to match checkpoint)
  - Outputs `embeddings.npz` per project (filenames, bbox_indices, embeddings [M, 768] L2-normed)
  - Includes `debug_visualize()` for bbox verification (DEBUG flag)
  - Skip logic for resumability
  - Dataset path on VM: `/data/AI/Tomer/person_reid/dataset_utils/dataset_finetune/Portraits[26]/`

## 2026-03-18
- Finished extracting DINOv3 ViT-B/16 foundation model embeddings for the full finetune dataset (500K images)
- Added checkpoint resume support to `train_dino_grad_accum.py`:
  - `--resume` boolean flag: resumes from latest checkpoint in latest version dir
  - Loads model weights + optimizer state via `load_checkpoint` (DCP format)
  - Restores exact iteration, schedules (LR/WD/momentum/teacher_temp) pick up at correct step
  - Data loader sampler advances to skip already-seen samples
  - W&B run resumed via persisted `wandb_run_id.txt` in output dir
  - `init_wandb` in `wandb_logger.py` updated with optional `resume_id` param
- Training was paused at ~15K iterations (GPU was needed for embedding extraction), ready to resume

## 2026-03-19
- Training resumed from ~15K iterations (done by Tomer on 2026-03-18)
- Planned and implemented HDBSCAN clustering for finetune dataset
  - Created `train_pictime/cluster_embeddings.py` — clusters person embeddings per project
  - Algorithm: HDBSCAN (euclidean on L2-normed = cosine), min_cluster_size=3
  - Output: `clusters.json` per project (grouped by filename, each entry has bbox_index + cluster_id)
  - Noise/outlier points labeled as cluster_id=-1
  - Skip logic, atomic saves, tqdm progress, summary stats
  - Hardcoded constants (no argparse) — Tomer's preference
  - Test path: `/data/AI/Tomer/UI_dataset_view/data`
- Tomer has a separate UI repo for visualizing project data, uses cluster JSON to display each cluster in a separate column

## 2026-03-23
- Training crashed at iter 32999 — loss diverged to NaN, then eval's `eigvalsh` failed on ill-conditioned covariance matrix
- Added NaN loss guard in `train_dino_grad_accum.py`: skips `optimizer.step()` when loss is NaN/Inf, zeros grads, continues training
- Added try/except around `torch.linalg.eigvalsh` in `metrics_rank.py` to return zeros gracefully instead of crashing
- Training resumed from last good checkpoint and is running again

## 2026-04-09
- Fixed W&B logging bug: metrics stopped appearing after ~14K iters
  - Root cause: `run.log(step=step)` conflicted with `define_metric("*", step_metric="train/iter")`
  - Fix: removed `step=step` from all `run.log()` calls in `wandb_logger.py`, added `"train/iter": step` to each payload instead
- Built ReID finetune training pipeline (all files in `train_pictime/finetune/`):
  - `supcon_loss.py` — Supervised Contrastive Loss (Khosla et al. 2020)
  - `reid_dataset.py` — ReIDCropDataset + PKBatchSampler (P projects × K crops, cross-project negatives only)
  - `reid_evaluator.py` — Query/gallery ReID eval with Rank-1/5/10 + mAP
  - `reid_config.yaml` — All hyperparameters, mode A (frozen backbone) / mode C (progressive unfreezing)
  - `finetune_reid.py` — Main training script, loads teacher backbone from DCP checkpoint
  - `build_index.py` — Offline script to scan 53K projects and save as .npz index for fast loading
- Design decisions: SupCon loss, teacher backbone, frozen backbone first (mode A), PK sampling with cross-project negatives, best-3-checkpoints by mAP
- Status: code written, first run hit FileNotFoundError (fixed), not yet fully tested by Tomer

## 2026-04-12
- Added silhouette score to ReID evaluation (`reid_evaluator.py`):
  - `_compute_silhouette()` — stratified subsample, cosine metric, configurable cap (`silhouette_max_samples: 8000`)
  - Logged to W&B as `eval/silhouette`, printed alongside Rank-1/5/mAP
- Switched checkpoint tracking metric from mAP to silhouette score (`finetune_reid.py`)
  - Filenames: `ckpt_iter*_sil*.pt`
- Fixed `build_index.py` — image paths were missing `/images/` subdirectory
- Fixed bf16→float32 dtype mismatch in both training loop and evaluator (`embs.float()` before proj_head)
- Unified eval/checkpoint frequency: removed redundant `eval_every` config, `ckpt_every` now controls both
- Created `find_single_cluster_projects.py` — scans all projects, saves list of folders with exactly 2 clusters to JSON (Tomer changed threshold from 1→2 for his use case)
  - Purpose: identify "clean" projects for test training before full curated dataset is ready
- Tomer commented out the tmp_trip debug subset and committed all changes

## 2026-04-16
- Added single-cluster project filtering to `build_index.py` — loads `single_cluster_projects.json` as allowlist
- Separated finetune W&B into its own project (`person-reid-finetune`), added `wandb_project` param to `init_wandb`
- Updated finetune run name to include backbone arch + frozen tag: `finetune_reid_vits16_bs64_frozen`
- Created detailed `train_pictime/README.md` documenting full pretrain + finetune pipeline
- Optimized `reid_evaluator.py`:
  - Replaced sequential `_extract()` with DataLoader (4 workers + pin_memory)
  - Merged query+gallery extraction into single DataLoader pass
- Ran efficiency audit of finetune pipeline; identified pre-cropping as future optimization

## 2026-04-18
- Reviewed mode A finetune results: silhouette=-0.02, Rank-1/5/10~0.7, mAP=0.36 (first run, 9370 total iters)
- Fixed bug in `finetune_reid.py:maybe_unfreeze()` — LR schedule was overwriting backbone param group with head LR (missing `lr_multiplier`)
- Configured mode C (progressive unfreezing): `unfreeze_after=2000`, `num_epochs=20`, `unfreeze_n_blocks=2`
- Added `train/lr_backbone` W&B logging after unfreezing to verify LR ratio is correct
- Mode C crashed at iter 2000 (the unfreeze step) with FSDP2 grad_dtype mismatch:
  `assign bf16 grad to tensor with grad_dtype Float`
  - Root cause: `load_backbone()` returned an FSDP2-wrapped backbone with `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)` carried over from pretrain. While frozen, no grads flowed → dormant. After unfreeze, bf16 backward grads couldn't be assigned to the fp32 grad slots.
- Fix (option 2 — standard mixed-precision recipe):
  - `load_backbone()` now loads DCP via the FSDP2 SSLMetaArch only to read weights, extracts full unsharded state dict via `get_model_state_dict(..., full_state_dict=True)`, strips `_orig_mod.` / `_checkpoint_wrapped_module.` prefixes, and loads into a fresh *unwrapped* backbone from `build_model_from_cfg(cfg, only_teacher=True)` → fp32 params on CUDA
  - Loud-fail guards: `load_state_dict(strict=True)` + explicit fp32 dtype assertion
  - Training loop wraps backbone forward in `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` — bf16 matmul speed, fp32 params/grads/optimizer state (avoids update swamping on small `lr_backbone`)
- Discussion on why option 1 (let grads be bf16) was rejected: bf16's ~7-bit mantissa loses small updates on small LR — real quality risk for finetune, not theoretical

## 2026-03-12 (continued)
- Updated run name format: `person_reid_vits16_effbs128_lr0.001` (shows effective batch size, dropped `lc`)
  - `run_name.py`: new `effective_bs` param, prefix default changed to `person_reid`
  - `train_dino_grad_accum.py`: moved `target_batch_size=128` up, passes it to `make_run_name`
  - Removed dead `make_run_name` / `_arch_to_tag` duplicate from `wandb_logger.py`

## 2026-04-20
- Reviewed W&B screenshots comparing 3 runs: green SupCon mode-A baseline (mAP 0.370), blue SupCon mode-C with `unfreeze4_lr1e-4` (mAP 0.415, sil +0.015 — **current leader**), orange ArcFace m=28.6° frozen (mAP 0.25 and **decaying** while train/loss dropped — textbook head-distortion failure)
- Corrected memory: training uses single-cluster-projects subset → ~21K identities (not 50K)
- Verified pytorch-metric-learning ArcFaceLoss behavior: `init_margin()` stores margin in radians as a plain float, used directly each forward, nothing precomputed → runtime mutation for warmup is safe (must write radians)
- Implemented Trial 6 refinement (committed in `0bec9ad`): reduce ArcFace margin 28.6°→17.2° (≈0.3 rad), add 1k-iter margin warmup (0→target), split classifier into own AdamW param group with `weight_decay=0` and tunable `arcface_classifier_lr` (default = head lr). W&B now logs `train/arcface_margin_deg` + `train/lr_classifier`. Plumbed `maybe_unfreeze()` to preserve 3-group split (head/classifier/backbone) for future Mode-C ArcFace trials.
- Updated `experiment.md` with blue + orange run rows, H7 hypothesis (margin engages too aggressively on frozen head), full Trial 6 spec + fallback plan (if eval still decays, reduce classifier LR; if that fails, try SubCenter-ArcFace)

## 2026-04-19
- Reviewed 5 W&B screenshots comparing mode A (frozen, ~9.4k iters) vs mode C (unfreeze 2 blocks @ 2000, ~18k iters) finetune runs
  - Mode C beats A by ~2pp mAP (0.390 vs 0.370), ~1.3pp Rank-1, ~2pp Rank-5. Silhouette C=-0.005, A=-0.014 (both negative)
  - No visible kick at iter 2000 unfreeze step → backbone LR too conservative, 2 blocks is too few
  - Both plateau quickly (mAP flat after ~6k iters) — capacity/LR-starved, not data-starved
  - Negative silhouette across the board is the main warning sign — embedding space not forming tight same-person clusters
- Created `train_pictime/finetune/experiment.md` — living experiment log with runs table (incl. backbone ckpt column), 6 hypotheses, 5 ranked trials, decisions checklist
- Added **H6** + **Trial 5 (ArcFace)**: angular-margin classification loss may better suit noisy per-project labels — class prototypes average out intra-class outliers. Variants noted: SubCenter-ArcFace (noise), AdaFace (variable crop quality), CosFace
- Gave Tomer a thorough explanation of SupCon temperature τ: sharpness knob on softmax, `1/τ` gradient scaling + softmax sharpening compound → low τ focuses on hardest pairs (dangerous under label noise), high τ is democratic/noise-tolerant
- Implemented generic experiment-tagging plumbing for finetune runs (loss-agnostic, works for any trial):
  - Added `experiment_tag` (run name suffix) + `experiment_group` (W&B group) fields to `reid_config.yaml`
  - Extended `wandb_logger.py:init_wandb()` to accept `group` kwarg → forwards to `wandb.init(group=...)`
  - Updated `finetune_reid.py:237-245` to append `tag_suffix` to run name and pass group through
  - Both fields default to `null` → zero regression for unannotated runs
- Tomer configured first Trial 1 run: τ=0.1, mode A (frozen, unfreeze_after=0), 10 epochs (~9k iters matches plateau from baseline)
  - Reasoning: τ=0.1 chosen first because low τ amplifies gradients on hardest pairs; if labels noisy, hardest pairs are often mislabeled ones. Start noise-tolerant and move lower only if it works
  - Flagged config bugs: `experiment_tag: "tau0.05"` vs actual `temperature: 0.1` mismatch; `"trail_temperature"` typo (should be "trial")

## 2026-04-23 — MAJOR FINDING: pretrain was actually from scratch

Initial ask: add a finetune config flag to swap the DCP backbone for the vanilla LVD-142M release (ArcFace trials were stalling; hypothesis was the continued-pretrain backbone was the bottleneck).

While tracing how `train_dino_grad_accum.py` bootstraps the base backbone, discovered a load-path bug that invalidates the whole continued-pretrain chain.

### The bug
- `train_dino_grad_accum.py:249` and `train_dino.py:241` set `cfg.student.pretrained_weights = args.pretrained` (pointing at `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`).
- Nothing in the SSL training path ever reads `cfg.student.pretrained_weights`. The only consumer is `dinov3/eval/setup.py` (eval pipeline).
- `SSLMetaArch.init_weights()` only loads a `.pth` if `cfg.student.resume_from_teacher_chkpt` is truthy. That field has been `''` in all 3 git commits of `pictime_vitl_im1k_lin834.yaml` (Feb 17, Mar 8, Mar 26 2026).
- Net: fresh pretrain runs went through `DinoVisionTransformer.init_weights()` which is pure random init (trunc_normal_, zeros_, reset_parameters). LVD-142M never got loaded.
- **Every `Vk` DCP**, including V9/ckpt/15000, traces back to random init — they're "DINOv3 from scratch on 6.5M Pictime images", not "LVD-142M further pretrained".
- This very likely explains the finetune plateau and why the ArcFace trials all sit in the same eval range.

### The fix
Three changes, all in `train_pictime/`:

1. **`train_pictime/foundation_loader.py`** (new) — `load_foundation_into_backbone(model, pth_path)`:
   - `torch.load(pth_path, map_location="cpu")` → flat state dict
   - Defensive: unwrap `"teacher"/"model"/"state_dict"` wrapper keys if present, strip `"module."`/`"backbone."` prefixes
   - Shard with `distribute_tensor` over `dinov3.distributed.get_process_subgroup()` world mesh; `rope_embed.periods` and `qkv.bias_mask` kept unsharded (mirrors `init_fsdp_model_from_checkpoint` at `dinov3/checkpointer/checkpointer.py:268-302`)
   - `model.student["backbone"].load_state_dict(sharded, strict=True)` — targets the backbone submodule only, so head random-init (from `init_weights()`) is preserved
   - Re-sync EMA: `model.model_ema.load_state_dict(model.student.state_dict())` (`self.model_ema IS self.teacher` per `ssl_meta_arch.py:131`)

2. **`train_dino_grad_accum.py`** — import + call after `model.init_weights()`, guarded `if not args.resume and args.pretrained:` (resume overwrites via `load_checkpoint` anyway).

3. **`train_dino.py`** (sister script, kept faithfully patched per Tomer's preference to avoid legacy-script footguns) — same import + call, guarded `if args.pretrained:` (no resume flag in this script).

### Arch audit — pictime yaml vs LVD-142M hub spec
First V10 attempt failed strict-load with unexpected `storage_tokens` and `blocks.i.attn.qkv.bias_mask` keys. Hub's `dinov3_vits16` (`dinov3/hub/backbones.py:201-237`) hardcodes `n_storage_tokens=4, mask_k_bias=True`. Pictime yaml had inherited `0/false` from Meta's `vitl_im1k_lin834` IM1K-linear-probe ablation config.

- **Flipped** `train_pictime/pictime_vitl_im1k_lin834.yaml:73-74`: `n_storage_tokens: 4`, `mask_k_bias: true`.
- Full arch audit confirmed all other state-dict-affecting fields match after the flip. Remaining benign differences:
  - `norm_layer: layernorm` vs hub's `layernormbf16` — both resolve to `nn.LayerNorm` via `norm_layer_dict` (`vision_transformer.py:27-31`), differ only in `eps` (1e-6 vs 1e-5). Same state-dict keys.
  - `pos_embed_rope_rescale_coords: null` vs hub's `2` — runtime-only training aug (applied inside forward when `self.training=True`); no buffer impact.
  - `pos_embed_rope_dtype: bf16` — not passed through `build_model` (`dinov3/models/__init__.py:37-58` omits it); effectively inert.

### Cleanup
- Removed dead `parse_args` + unused `make_setup_args` from both scripts; replaced with inline `SimpleNamespace(...)` in `main()`. Dropped `argparse` and `os` imports.

### Known-invalid / stale state
- **V1-V9 DCPs** = random-init DINOv3 on 6.5M Pictime, ~33K total iters across the resume chain. Useful as a "from-scratch on our data" baseline; not a "continued pretrain" backbone.
- **ArcFace Trials 5/6 eval results** (orange/green/blue in the comparison) are against the V9 random-init backbone. They say something about ArcFace-vs-SupCon with a weak backbone but are not load-bearing for ArcFace's actual viability — need re-running once V10 exists.
- **Trial 6/6b/7 plans** in the roadmap are obsoleted by the need to re-baseline; see updated roadmap.

### Next action for Tomer
Launch V10 fresh (no `--resume`). Expected log lines:
- `Loading FOUNDATION weights from /data/AI/Tomer/dinov3/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`
- `Foundation load done — missing: 0, unexpected: 0`
- Initial `train/dino_global` should drop from ~11 (random-init baseline on the W&B history) to ~4-6 (foundation-init).

### 2026-04-23 (continued) — W&B missing train-loss charts investigation (in progress)
- Tomer reported that recent runs show only `train/total_loss` chart — `train/ibot_loss`, `train/dino_global`, `train/koleo` are missing. Same symptom class as the 2026-04-09 regression (that fix — remove `step=step` from `run.log()`, put `train/iter` in payload — is still intact in `wandb_logger.py:53-57`).
- Full code audit of the pipeline (`log_wandb` → `run.log` → define_metric → `compute_losses` loss_dict keys → accumulation loop) found no obvious break. All three keys (`dino_global_crops_loss`, `koleo_loss`, `ibot_loss`) are unconditionally set in `ssl_meta_arch.py:632, 637, 648`; `train_dino_grad_accum.py:337-344` copies all of them into `avg_metrics`; `train_dino_grad_accum.py:377-386` logs them under `train/*` names.
- Ranked hypotheses: (A) `define_metric("*", step_metric="train/iter")` wildcard fails to match slashed keys in the installed wandb SDK version; (B) W&B workspace layout / chart-panel collapse (not a code bug); (C) silent exception in accumulation that zeroes the three keys.
- **Step 1 applied**: added a throttled diagnostic print in `log_wandb` (`wandb_logger.py:56-57`) — prints `[W&B LOG @ iter N] keys=[…]` every 100 iters. Committed alongside the foundation-loader work.
- Tomer is running with the diagnostic now; will report which of the three cases fires once he's past ~200 iters. Next step is case-specific fix (explicit per-key `define_metric` for Case A; tracedown for Case B; UI-layout reset for Case C). Diagnostic print gets removed once the fix lands.

## 2026-05-04 — V11 finetune trial 1 win, curriculum sampler design, .work/ migration

### V11 trial 1 — decisive win on the continued-pretrain backbone
- Trial: SupCon Mode-C `unfreeze4_lr1e-4` on V11/ckpt/13000. exp_tag `unfreeze4_lr1e-4_v11ckpt13k`, exp_group `trial_unfreeze_size_and_lr_bb`
- vs V9 "blue" baseline: **mAP 0.42 → 0.62 (+0.20)**, Rank-1 0.68 → 0.83, **Silhouette +0.015 → +0.18** (first decisively positive — embedding space finally healthy)
- Confirms 2026-04-23 foundation-loader fix paid off. V1–V9 random-init was the bottleneck, not loss/recipe.
- Curve pattern: mAP/Rank-1 plateau ~5–6k iters; Rank-5/10 + silhouette still climbing at 9k → 15–20 epochs may extract more.
- V11 = first proper continued-pretrain (V10 was a smoke run). ~10 days train. ckpt/13000 is the new finetune source.

### Curriculum sampler — design locked, not yet coded
- Concept: per-identity centroid-based curriculum. Stage 1 = top-fraction `p` of crops closest to centroid; ramp `p(t): 0.5 → 1.0` over first 30% iters.
- Critical control: include `p=0.5 fixed` ablation. Without it can't separate curriculum-learning from data-cleaning.
- Centroid embeddings: **V11 pretrain (NOT finetune backbone)** — finetune was told "all crops in project = one identity" via SupCon, would mis-merge multi-person un-fixed projects on re-clustering.
- Per-cluster L2-normed mean; cosine distance.

### v2 data-prep pipeline status
Modify-in-place pattern, `MODEL_SOURCE → filename` dict mapping in each script. v1 artifacts untouched.
- ✅ `extract_embeddings.py` — V11 via `load_backbone` from `finetune_reid.py` (cross-script import). Out: `embeddings_v2.npz`. **Running on VM.**
- ✅ `cluster_embeddings.py` — same HDBSCAN params; skips projects with `clusters_fixed.json` (reviewer truth wins). Out: `clusters_v2.json`. **About to run.**
- ✅ `find_single_cluster_projects.py` — resolver hierarchy (fixed > v2 > skip). Reviewer-fixed accepted unconditionally; HDBSCAN-derived keep `len(non-noise)==2`. Out: `single_cluster_projects_v2.json`. **Code ready.**
- ⏳ `build_centroids.py` (new) — planned. Per-project `crop_distances_v2.json` keyed by cluster_id with sorted (filename, bbox_index, distance).
- ⏳ Sampler integration in `reid_dataset.py` + new config fields — planned.

### Memory infrastructure — `.work/` migration
- Auto-memory moved from `~/.claude-work/projects/.../memory/` into `<repo>/.work/` (git-tracked). Auto-memory now a pointer stub.
- Renames: `progress.md` → `project_progress.md`, `roadmap.md` → `project_roadmap.md`.
- New slash commands at `~/.claude-work/commands/` (global): `/session-start`, `/session-end`, `/handoff`, `/park`, `/pushback`, `/migrate-to-work-dir`. The last automates this migration in other work repos.
- Working-guidelines memory updated with two new rules: 4-option approval menu; dict-based filename mapping for model-tied artifacts.

### Deferred
- Latent bug in `extract_embeddings.py:should_skip_project`: compares saved-count to total-bbox-count, but invalid crops (small/missing/PIL-fail) reduce saved count → projects with any invalid crop get re-processed every run. Fix: `os.path.exists()` only. Apply after step 1 completes.
- JetBrains plugin only establishes IDE integration via its button; terminal-launched `claude-work` loses popup diffs. No fix; live with terminal diffs.

## 2026-05-05 — Curriculum sampler shipped end-to-end; first trial launched

### Pipeline complete: v2 data prep → curriculum-aware sampler
Stage-by-stage build matches the design locked yesterday.

- **`train_pictime/build_centroids.py` (new)** — per-project per-cluster centroid
  + sorted cosine distances. Output: `crop_distances_v2.json`. Loud-fail on
  missing `(filename, bbox_index)` entries. Skips `cluster_id == -1` (noise).
  `MODEL_SOURCE` dict pattern, atomic save, `FORCE` flag.
  - Run on full dataset: 53,650 projects processed, 152 errors (cascade from
    the n<3 cluster_embeddings crash). Tomer chose not to fix the n<3 guard
    in `cluster_embeddings.py:25` — those projects are <3 detections, useless
    for ReID anyway, and would be filtered by `min_k=4`.

- **`train_pictime/finetune/build_index.py`** — refactored to v2 mode.
  Added `MODEL_SOURCE` + 3 `*_BY_MODEL` dicts (`CLUSTERS`/`SINGLE_CLUSTER`/
  `INDEX`). New `bbox_indices` field in the npz. Output:
  `reid_index_v2.npz`. Verified: 8317 projects, 306,888 samples.

- **`train_pictime/finetune/reid_dataset.py`** —
  - `load_index()` returns 5-tuple incl. `bbox_indices`.
  - `ReIDCropDataset.__init__` takes `bbox_indices`, `cluster_ids`,
    `centroid_distances_filename`. When the filename is given, builds
    `identity_to_sorted_indices: dict[global_id, list[dataset_idx]]` —
    per-identity dataset indices ordered by ascending centroid distance.
    Loud-fail on missing identities or unmatchable entries.
  - `PKBatchSampler` adds `curriculum_p_start/p_end/end_frac` (defaults
    1.0/1.0/0.3 = no-op). Linear ramp `p(t)` over `end_frac * num_batches`
    iters, then flat. Pool slice `pool = sorted_idx[:max(K, ceil(p*len))]`
    — clamp guarantees ≥K crops (graded effect on small identities, accepted
    as a known limitation per pushback).

- **`train_pictime/finetune/reid_evaluator.py`** — added `bbox_indices` arg,
  forwards `centroid_distances_filename=None` (eval is curriculum-free; -1
  identities kept in val pool).

- **`train_pictime/finetune/reid_config.yaml`** — `reid_index_filename`
  (defaults to v2) + `curriculum:` block at the bottom (`enabled` defaults
  false; vanilla regression preserved).

- **`train_pictime/finetune/finetune_reid.py`** — config-driven index path,
  5-tuple destructure, train-only `cluster_id == -1` filter when curriculum
  is on, dataset/sampler arg pass-through, optional `train/curriculum_p`
  W&B log every 10 iters (formula duplicated from sampler with a header
  comment flagging the duplication).

### Pushback session — head-vs-backbone curriculum framing retracted
I flagged that the unfreeze step at iter 2000 lands inside the curriculum
ramp window (ends iter 7000 at p=0.5) and could "anchor the backbone to
easy data". Tomer pushed back: curriculum philosophy doesn't distinguish
between head and backbone training. I retracted — that framing was
overcooked. The legitimate (but mild) concern is in the *opposite*
direction: backbone has strong pretrained knowledge from V11, and
curriculum-easy-first could narrow that, but `lr_backbone=1e-4` (10×
smaller than head) limits per-step degradation. Not a blocker.

### Trial 1 launched
Config:
- `experiment_tag: curriculum_p0.1to0.5_frac0.7`
- `experiment_group: trial_curriculum_v1`
- `p_start=0.1, p_end=0.5, end_frac=0.7`
- Backbone: V11/ckpt/13000 (same as Trial 1 V11 baseline)
- Mode C unfreeze: 4 blocks @ iter 2000 (same as Trial 1)

Index pool: 8317 single-cluster v2 projects; ~16K identities expected
post -1 filter.

### Tradeoffs accepted (won't be fixed)
- **No "vanilla on v2" baseline** — Trial 1 V11 (mAP 0.62) was on v1 index;
  curriculum is on v2 index. Comparison entangles curriculum + data-source
  shift. Tomer: "we will have many more trials from now on" — accepted.
- **Curriculum clamp neuters small identities** — `max(K, ceil(p*len))`
  means identities with <K/p_start crops feel no curriculum. With K=4 and
  p_start=0.1, threshold = 40 crops. Many v2 identities are smaller →
  graded effect. Accepted as documented.
- **n<3 cluster crash** in `cluster_embeddings.py` left unfixed (152
  projects, useless for ReID anyway).

### Side task — UI repo prompt
Drafted a self-contained prompt for the reviewer UI repo (separate Claude
session) to add a `clusters_fixed.json → clusters_v2.json → clusters.json`
resolver hierarchy. Goal: reviewers get cleaner V11 starting clusters on
unreviewed projects.

### Open
Trial 1 running on the VM. Awaiting first-iter sanity (curriculum_p chart,
filtered-sample-count log) and eval-vs-baseline comparison.

### Trial 1 result — curriculum lost decisively at iter 7500
At iter 7500 (right after curriculum ramp completes at iter 6300):

| Metric | V11 baseline (brown) | Curriculum p0.1→0.5 (pink) | Δ |
|---|---|---|---|
| eval/mAP | 0.631 | 0.555 | **−0.076** |
| eval/Rank-1 | 0.821 | 0.743 | **−0.078** |
| eval/Rank-5 | 0.912 | 0.859 | −0.053 |
| eval/Rank-10 | 0.944 | 0.900 | −0.044 |
| eval/silhouette | +0.196 | +0.126 | −0.07 |

Both runs still climbing at 7500 — neither plateaued — but curriculum's gap stays
roughly constant, not closing. No sharp inflection at the curriculum-end iter.

### The train/loss signature flips the diagnosis
**`train/loss` is consistently LOWER on pink (~1.0–1.3) than brown (~1.3–1.5)**,
yet eval is worse. Textbook curriculum-overfit-to-easy-mode signature:
- Lower train loss because pink sees mechanically-easier crops (pool restriction).
- Worse eval because the representation memorizes the easy mode instead of
  learning the variation that matters at query time.
- Even past iter 7000 (when p locks at 0.5) the gap doesn't close — early-iter
  narrow distribution has already shaped the weights.

This is stronger than yesterday's pushback discussion predicted. If v2 data
shift alone were the cause, pink train loss would be *higher* (harder data →
harder fit). We're seeing the opposite. **Curriculum itself is the likely
culprit, not the v2 data shift.**

### `train/curriculum_p` ✓
Clean linear ramp 0.1 → 0.5 over iters 0–6300, flat at 0.5 thereafter.
Schedule formula is correct (ramp_end = 0.7 × 9000 ≈ 6300).

### Vanilla-on-v2 launched
To confirm attribution (curriculum vs data shift) — same config as the curriculum
run with `curriculum.enabled: false`, exp_tag `vanilla_v11ckpt13k_v2index`,
group `trial_curriculum_v1` (overlays on same plot).
- If vanilla-on-v2 ≈ pink (0.555) → curriculum is neutral, loss is data.
- If vanilla-on-v2 ≈ brown (0.62) → curriculum is actively hurting.
- Working hypothesis based on train-loss signature: vanilla-on-v2 is closer to brown.

### New rule (working guidelines)
"When committing, stage all modified + untracked files (`git add -A`).
Don't pick subsets and don't ask which files to include." Saved to feedback memory.

## 2026-05-06 — Tiered eval shipped; noisy-eval hypothesis partially confirmed

### Curriculum trial 1 attribution — vanilla-on-v2 lands between brown/pink
Vanilla-on-v2 (blue) finished overnight: mAP 0.59 at 10K iters. Final 3-way on
v2 val:
- brown (v1 baseline): 0.625 — but ON v1 VAL POOL (not directly comparable)
- blue  (vanilla v2): 0.59
- pink  (curriculum p0.1→0.5): 0.555

Noticed while tracing the eval path: brown's val pool came from the v1 index,
blue/pink's from v2. The 0.625 vs 0.59 gap is partly val-pool drift, not just
train-data shift. Tomer confirmed they'd already considered this — moving on.

### Diagnosis: noisy eval penalizes cleaner-trained models
Tomer raised this directly: HDBSCAN labels in the eval pool have false positives.
A curriculum-trained model that doesn't memorize the noise gets penalized at
eval time. Standard ReID literature finding.

His initial proposal — "eval at p_end" (eval level matches train level) — got
pushback for three reasons:
1. mAP becomes non-comparable across runs (different eval pools per run)
2. Smaller cleaner gallery mechanically inflates scores → can't separate "model
   got better" from "test got easier"
3. Production-mismatch: at deploy, the model sees full crop distribution

Counter-proposal that landed: fixed-tier eval (top-50 / 75 / 100), same tiers
applied to all runs. Plus -1 dropped from val (no centroid → can't tier).

### Code shipped
- `train_pictime/finetune/reid_evaluator.py` — rewritten. New args
  `centroid_distances_filename`, `eval_tiers`. Per-tier Q/G built upfront,
  single extract pass over union of tier indices, per-tier mAP/Rank/silhouette
  logged as `eval/mAP_top50` etc. `silhouette_top100` mirrored to `silhouette`
  for `BestCheckpointTracker` compat. Backwards-compat path: no centroid file
  → single full-pool eval.
- `train_pictime/finetune/reid_config.yaml` — added `eval_tiers: [0.5, 0.75, 1.0]`
  + `eval_centroid_distances_filename: "crop_distances_v2.json"`.
- `train_pictime/finetune/finetune_reid.py` — passes new args through.
- `train_pictime/finetune/reeval_tiered.py` (new) — load V11 once, swap saved
  finetune state per run, run tiered eval, print markdown row. Hardcoded RUNS
  dict with V<n> dirs; auto-picks best-silhouette ckpt per run.

### Reeval result — partial confirmation of noisy-eval penalty
Blue (V14) and red (V15) re-evaluated on v2 val with tier filtering. Pink (V13)
had no `ckpt_iter*_sil*.pt` files — root cause not investigated, deferred.

| Tier | Blue mAP | Red mAP | Δ red−blue | Blue R1 | Red R1 | Δ |
|------|----------|---------|------------|---------|--------|---|
| top50  | 0.8090 | 0.8092 | **+0.0002** | 0.9214 | 0.9363 | **+0.0149** |
| top75  | 0.7614 | 0.7498 | -0.0116 | 0.9065 | 0.9065 | 0 |
| top100 | 0.6914 | 0.6827 | -0.0087 | 0.8957 | 0.8848 | -0.0109 |

Silhouette: top50 — red 0.366 vs blue 0.364 (red slightly higher); top100 —
red 0.246 vs blue 0.252 (blue higher). Gap reverses across tiers.

**Translation**: soft curriculum (red, p=0.5→1.0) is at parity-to-marginally-better
than vanilla on clean labels (top-50), slight loss on noisy labels (top-100).
The 0.575 vs 0.59 W&B gap was real but partly an artifact of eval-noise penalty.
Curriculum is not the disaster the W&B charts suggested.

Also notable: dropping -1 from val + cleaning to top-50 closes a **12-point
mAP gap** (0.69 → 0.81 for blue). The eval bias was huge.

### Follow-up run launched
After seeing the tier results, Tomer changed the config with three combined
knobs:
- `unfreeze_after`: 2000 → 1000 (earlier unfreeze)
- `unfreeze_n_blocks`: 4 → 6 (50% of ViT-S)
- `curriculum.p_end`: 1.0 → 0.8 (perma-filter bottom 20%, never fully release)

Flagged the entanglement (3 knobs, can't isolate cause if it wins/loses) and
the stale `experiment_tag: "vanilla_v11ckpt13k_v2index"`. Tomer renamed the
tag and launched.

### Deferred / open
- Pink V13 missing ckpts — root cause unknown
- Whether to switch `BestCheckpointTracker` from full-tier to top-50
  silhouette (cleaner signal). Wait until more tier data lands.
- New run is running on VM; results will tell us if more capacity + perma-
  filtered curriculum beats vanilla on tier-50.

## 2026-05-07 — Cluster-id=-1 unification, code-style sweep, Trial 1 launched

### `cluster_id == -1` filter unified across train + eval
- Resolved the 10k vs 8k iter mystery between vanilla and curriculum runs:
  vanilla kept -1 in train (~+20% more identities → more iters/epoch),
  curriculum dropped them → fewer iters. Now -1 filtered globally right
  after `load_index()` in both `finetune_reid.py` and `reeval_tiered.py`.
- Removed redundant conditional filters in `finetune_reid.py` (curriculum-
  only block) and `reid_evaluator.py` (tiered-only block).
- Doc cleanup: `reid_config.yaml`, `README.md`.
- Eval is functionally unchanged on the current tiered config (tiered eval
  was already dropping -1). Train side is the visible change; vanilla and
  curriculum runs from now on share the same train pool.

### Code-style rule saved + repo-wide sweep
- Saved `.work/feedback_function_signatures.md` (multi-line `def` AND
  function calls: first arg on `(` line, args column-aligned, closing `)`
  aligned with opening `(`). Project line length is 120 (from
  `pyproject.toml`).
- **24 def-signature edits** across 10 files (TOO_LONG wraps,
  VERTICAL→horizontal rewraps, OVER_WRAPPED collapses, MIXED rewraps).
- **30 function-call edits** across 12 files (12 TOO_LONG wraps, 12
  VERTICAL→horizontal rewraps, 6 MIXED — 3 collapsed, 3 strict-wrapped).
- Skipped: Pattern A (list-as-single-arg, e.g. `transforms.Compose([...])`),
  Pattern B (dict/list-of-dicts as arg, e.g. `torch.optim.AdamW([{...}])`,
  `torch.save({...}, path)`), Pattern C (nested-lambda calls, e.g.
  `model._apply(lambda t: torch.full_like(...))` × 4 files), and 12
  unscanned files. Survey-first / edits-second worked well.

### Trial brainstorm — full list ranked, methodology locked
Discussed all 7 ideas from `my_prompt`. Methodological constraints locked:
- **One knob at a time** (V15 entangled run was the lesson).
- **Curriculum off** unless the trial is specifically about curriculum.
- **Tier-50 is primary metric**.
- **V11/ckpt/13000 fixed** as backbone (until #4 ViT-B exploration).

Recommended order: (1) head-vs-no-head eval [no training cost],
(2) n_blocks sweep, (3) τ sweep, (4) K sweep, (5) P sweep + whole dataset,
(6) ArcFace solo on V11, (7) ViT-B cheap path.

### Trial 1 — n_blocks sweep launched
- Sweep `n_blocks ∈ {0, 2, 4, 6, 8, 12}` with everything else held fixed
  (V11/ckpt/13000 teacher, SupCon τ=0.07, head lr=1e-3, lr_backbone=1e-4,
  unfreeze_after=2000 except n_blocks=0 → Mode A, P=16, K=4, 10 epochs,
  curriculum off). W&B group `trial_n_blocks`.
- **Refactor**: split `finetune_reid.py:main()` into `run_finetune(cfg)`
  + thin `main()`. Direct `python3 finetune_reid.py` still works (function
  renamed to avoid shadowing the local W&B `run` variable).
- **New trial folder** `train_pictime/finetune/trials/` with `__init__.py`
  + `trial_01_nblocks.py` — single-process sequential sweep, deep-copies
  base cfg, per-iter overrides, try/except per iter, `cuda.empty_cache()`
  between. Tomer started the script.
- Follow-up `trial_01b_lr_backbone.py` to be created after this finishes
  (needs `n_blocks*`).

### experiment.md
First wrote a comprehensive version with run-history table + lessons +
backlog → Tomer pushed back ("too much info"). Stripped to Trial 1 setup +
lr_backbone follow-up + Results placeholder. Lesson: keep this file scoped
to active trials, not a historical archive (progress.md is the archive).

### Other
- Stale 2026-05-05 handoff resolved on resumption (was already addressed
  by 2026-05-06 work).

### Open
- Trial 1 (6 runs × ~1.5h ≈ 9h) running on VM. Awaiting tier-50 mAP curve
  across n_blocks values to identify `n_blocks*`.
- Decide `BestCheckpointTracker` metric (full-tier silhouette vs tier-50
  silhouette) once more tier data lands.
- Pink V13 missing ckpts root cause (deferred from 2026-05-06).

## 2026-05-11 — Trial 2 verdict + real-world eval pipeline shipped

### Trial 2 (lr_backbone × n_blocks sweep) — verdict: lr=1e-4 wins, gap is thin

Reviewed n_blocks=6 slice of the W&B group `trial_lrbb_x_nblocks` (Tomer
filtered out n_blocks=2 and 12 for chart clarity; said both were
"slightly lower"). Final values @ ~8k iters:

| LR     | mAP_top50 | mAP_top100 | R1_top50 | R1_top100 | sil_top50 | sil_top100 |
|--------|-----------|------------|----------|-----------|-----------|------------|
| 1e-4   | **0.840** | **0.718**  | ~0.930   | 0.882     | **0.408** | **0.285**  |
| 5e-5   | 0.835     | 0.713      | 0.927    | ~0.885    | 0.402     | 0.280      |
| 5e-4   | 0.827     | 0.702      | ~0.918   | 0.872     | 0.400     | 0.277      |
| 1e-5   | 0.827     | 0.697      | ~0.920   | 0.885     | 0.392     | 0.262      |

- 1e-4 leads consistently on mAP and silhouette (cleanest signal:
  sil_top100 = 0.285 vs 0.262 for 1e-5).
- 5e-5 vs 1e-4 gap is ~0.005 mAP_top50 / ~0.006 silhouette — within plausible
  single-seed variance. Win is real but thin.
- 5e-4 trained stably (no divergence at the top end of the LR range).
- Pre-trial assumption holds: `lr_backbone=1e-4` is the keeper.
- Trial 2 result still needs writing to `experiment.md` Trial 2 Results
  section + the n_blocks=2/12 slices need eyeballing to fully close out
  the original "is 2 enough or does LR unlock 6" question. Deferred.

### Real-world eval pipeline shipped — `train_pictime/finetune/realworld_eval/`

Goal Tomer set: "cluster a test set with the current best finetune weights,
see the clusters we'd be providing in production, see where it excels /
struggles." Built end-to-end.

New files (4):
- `__init__.py`
- `config.py` — **single source of truth** for `FINETUNE_VERSION_DIR`
  (V31, Trial 2 winner), `OUTPUT_BASE`, sampling/HDBSCAN/viewer knobs.
  Both scripts import from here (Tomer pushback during build: don't
  duplicate the V<n> constant across scripts).
- `cluster_test_set.py` — samples 100 projects NOT in
  `single_cluster_projects_v2.json` (filter: ≥50 person bboxes per
  project), loads V11 + V31 best-silhouette ckpt, per-project embed →
  HDBSCAN cluster → save clusters JSON + cropped JPGs. Reuses
  `load_backbone` + `build_projection_head` from `finetune_reid.py` and
  `find_best_silhouette_ckpt` from `reeval_tiered.py`.
- `build_html_viewer.py` — pure stdlib, generates `ui/index.html`
  (100-card grid) + `ui/projects/<pid>.html` (cluster rows with
  prev/next nav). Relative-paths only, no server needed for content;
  Tomer added a comment with the working SSH tunnel command for serving.

### Output layout (test set frozen, model-independent crops shared)

```
/data/AI/Tomer/realworld_eval/
├── test_projects.json           ← GLOBAL — sampled once, reused across ckpts
├── crops/<pid>/<file>__bbN.jpg  ← GLOBAL — model-independent bbox crops
└── V31_iter8520_sil0.2858/
    ├── clusters/<pid>.json      ← per-ckpt cluster labels
    └── ui/                      ← per-ckpt HTML viewer
```

Future ckpt evals reuse the same 100 projects + crops; only `clusters/`
and `ui/` get regenerated.

### Build-time decisions Tomer pushed back on (logged for context)
- Test set + crops moved from per-ckpt subdir to global `OUTPUT_BASE`
  level after Tomer flagged "test set should be picked once, reuse".
- Initial guess for Trial 2 winner V<n> was V28; Tomer corrected to V31.
- Crops dir skip logic switched from "dir non-empty → skip" to per-file
  existence check, so an interrupted run gets completed rather than
  left half-cropped.
- `find_output_dir` sort switched from lexicographic to mtime so
  "newest" actually means newest filesystem-time.

### First real-world run — V31 ckpt
- Ckpt: `V31/ckpt/ckpt_iter8520_sil0.2858.pt`
- 100/100 projects done, 0 skipped, 0 errors
- Mean clusters / project: **10.17**
- Mean crops / project: **139.7**
- Noise fraction: **13.9%**
- Serving via `python3 -m http.server 8080` on VM + SSH tunnel
  (`ssh -L 8080:127.0.0.1:8080 azureuser@10.0.32.13`) → open
  `http://localhost:8080/V31_iter8520_sil0.2858/ui/index.html` locally.

### Open
- Eyeball pass on the 100 projects — Tomer mid-review. Patterns to watch:
  within-person splits (same person, different outfit), between-person
  merges, projects with very high noise %.
- Trial 2 final `experiment.md` write-up + n_blocks=2/12 readings still
  pending.
- "Multiple matching output dirs" branch in `build_html_viewer.py` is
  untested (only fires if same V<n> is evaluated twice with different
  best ckpts — won't happen until V31 has a newer best-3).

## 2026-05-12 — Realworld eval output path fix + multi-test-set support

### OUTPUT_BASE relocation
- Tomer flagged that `cluster_test_set.py` wrote to `/data/AI/Tomer/realworld_eval/`
  (loose dir at /data/AI/Tomer/) instead of nested under the repo. Tomer's
  chosen target: `/data/AI/Tomer/dinov3/train_pictime/finetune/realworld_eval/results/`
  (data colocated with code, but under a `results/` subdir to keep code and
  outputs separate).
- One-line fix in `config.py`. Bash migration command given:
  `mv /data/AI/Tomer/realworld_eval/* .../realworld_eval/results/ && rmdir ...`

### Broken-images diagnosis (no code change)
- After moving, Tomer saw broken-image icons in the HTML viewer. Diagnosed
  through several rounds — wrong `-d` path on the http.server (`-d /dinov3/...`
  was missing the `/data/AI/Tomer/` prefix; absolute path interpretation, not
  relative). Plus likely stale nohup'd server still listening on 8080.
- HTML uses relative `../../../crops/...` paths, so the move itself didn't
  break anything — but the server needs to root at `OUTPUT_BASE` (or any
  ancestor) for the relative paths to resolve.
- Side reference for next session: `pgrep -af "http.server 8080"` (dry-run)
  vs `pkill [-9] -f "http.server 8080"` for cleanup. `kill -9` = SIGKILL.

### Multi-test-set support — `TEST_SET_NAME` knob
Tomer has a new test set `Wedding[1]` (~106 projects, harder galleries with
many people per scene). Wanted to run real-world eval on it alongside the
existing Portraits[26] eval without collision.

Design: single `TEST_SET_NAME` constant in `config.py`. All outputs nest
under `OUTPUT_BASE/<TEST_SET_NAME>/`. The HTML viewer's relative paths
(`../../../crops/...`) still resolve correctly since `crops/` and `V<n>_.../`
stay siblings — they just live one level deeper.

Config knobs for Wedding (locked with Tomer): `N_SAMPLE=100`, `MIN_BBOXES=0`
(don't drop any of the 106), `EXCLUDE_FILE=None` (no train-pool overlap).
HDBSCAN params unchanged (apples-to-apples with Portraits first pass).

### Files changed
- `train_pictime/finetune/realworld_eval/config.py` — added `DATASET_PARENT`,
  `TEST_SET_NAME`, derived `DATASET_ROOT`. `EXCLUDE_FILE → None`,
  `MIN_BBOXES → 0`. `OUTPUT_BASE` corrected to `.../realworld_eval/results`.
- `train_pictime/finetune/realworld_eval/cluster_test_set.py` — import
  `TEST_SET_NAME`, prefix all three output paths (`test_set_path`,
  `crops_root`, `output_dir`) under it. Added `EXCLUDE_FILE is None`
  branch. Docstrings updated ("global" → "per-test-set").
- `train_pictime/finetune/realworld_eval/build_html_viewer.py` — same
  pattern: import `TEST_SET_NAME`, compute `test_set_base` local,
  forward to `find_output_dir`, `test_projects_path`, `crops_root`.
  HTML generation untouched (relative paths still correct).

### Output layout (target)
```
results/
├── Portraits[26]/        ← existing data migrated here
│   ├── test_projects.json, crops/, V31_iter8520_sil0.2858/
└── Wedding[1]/           ← cluster_test_set.py running into this now
    ├── test_projects.json, crops/, V31_iter8520_sil0.2858/
```

### Open
- `cluster_test_set.py` running on VM for Wedding[1] — finishes overnight.
  Tomorrow: build_html_viewer + serve + eyeball pass.
- Portraits HTML viewer reportedly broken after migration — need to rerun
  `build_html_viewer.py` with `TEST_SET_NAME="Portraits[26]"`. Per the
  relative-paths analysis the existing HTML *should* still work, but Tomer
  flagged it as broken — will verify tomorrow.
- Carry-forward from 2026-05-11: Trial 2 `experiment.md` write-up still
  pending; n_blocks=2 and 12 slices not yet read.

## 2026-05-18 — /setup-claude-in-repo command + Trial 3 (face-blur) shipped & launched

### `/setup-claude-in-repo` — new global slash command

Sibling to `/migrate-to-work-dir`. Bootstraps `.work/` in a freshly cloned
repo so /session-start, /session-end, /handoff, /park, /pushback work
identically across repos. Lives at `~/.claude-work/commands/setup-claude-in-repo.md`.

7-step flow:
1. Abort if `<repo>/.work/` exists.
2. Scaffold 8 files (templates baked inline — no dependency on dinov3):
   `MEMORY.md`, `user_profile.md` (Focus line blank), `feedback_working_guidelines.md`,
   `feedback_function_signatures.md`, `project_progress.md` (header + first entry),
   `project_roadmap.md` (empty sections), `project_context.md` (frontmatter only),
   `my_prompt` (empty).
3. Append `.gitignore` lines (`parked.md`, `handoff.md`).
4. Report scaffold inline.
5. Analyze the repo carefully (READMEs, manifests, top-level structure, recent
   commits, entry points — ~10-15 reads max).
6. Draft `project_context.md` body + `Focus:` line in `user_profile.md`, show
   inline, wait for go, then write.
7. Final tree + suggested commit command.

Deliberately NOT copied: `feedback_fix_sister_scripts.md` — project-specific
(script-families pattern doesn't apply universally).

### Trial 3 — face-blur finetune

Hypothesis: model is over-reliant on face features. Blurring faces at train
time forces body-feature learning (clothing, build, posture). Eval is never
blurred → any train→eval gain is body-feature generalization to the unblurred
eval distribution.

Setup (inherits current production config, single knob flipped):
- Backbone: V11/ckpt/13000
- Mode C: `unfreeze_after=1000, unfreeze_n_blocks=6, lr_backbone=1e-4`
- SupCon τ=0.07, head lr=1e-3, P=16, K=4, 10 epochs, curriculum off
- **New:** `face_blur.enabled=true`

### Technique decisions

- **Gaussian blur** (Tomer's pick over mean fill / noise / soft-edge).
- **No padding** — blur stays exactly within the yolov11l-face bbox.
- **Adaptive sigma**: PIL `GaussianBlur(radius = sigma_factor * min(face_w_px, face_h_px))`
  computed in original-image pixels, so blur is invariant to image resolution
  and survives the downstream `Resize(256) → CenterCrop(224)`.
- Default `sigma_factor=0.3` → radius = 30% of smaller face dim.

### Code shipped

- **`train_pictime/finetune/reid_dataset.py`** — new module-level helpers
  `apply_face_blur()` + `_build_face_lookup()`. `ReIDCropDataset.__init__`
  gained 3 kwargs (`face_blur_enabled=False, face_blur_sigma_factor=0.3,
  face_blur_faces_filename="faces.json"`). Lookup is built only when
  enabled (no 5-10s startup cost otherwise) and logs a one-line summary.
  `__getitem__` calls `apply_face_blur` between the PIL crop and the
  transform. Edge cases: missing `faces.json` → skip; face outside crop
  → clip; <2px clipped face → skip.

- **`train_pictime/finetune/reid_config.yaml`** — new `face_blur:` block
  (parallel to `curriculum:`), defaults to disabled. For this run Tomer
  set `experiment_tag: "face_blur_sf0.3_v11ckpt13k"`,
  `experiment_group: "trial_face_blur"`, `face_blur.enabled: true`.

- **`train_pictime/finetune/finetune_reid.py`** — reads `face_blur` via
  `cfg.get(...)` mirroring the curriculum pattern; forwards to dataset.
  Rewrapped the `ReIDCropDataset(...)` constructor call to column-aligned
  style while touching it.

- **`train_pictime/finetune/debug_face_blur.py`** (new) — simple debug
  viz: builds dataset with face_blur on, picks 16 indices with faces from
  the first 200 samples, denormalizes each crop and shows via `plt.show()`.
  (Original version saved to disk; Tomer tweaked to interactive plt
  before launching the trial.)

- **`train_pictime/finetune/reid_evaluator.py`** — untouched. Evaluator
  builds its own dataset (`reid_evaluator.py:126`) without face_blur
  kwargs → eval auto-defaults to no-blur. Confirmed.

### Plan-review pattern that worked

Three rounds of plan refinement before code: (1) initial proposal with
4 technique options + padding knob + 3-panel debug viz, (2) Tomer
narrowed to blur, no padding, simple viz, (3) explicit completeness
re-review surfaced 10+ integration details (current config is post-V15
not Trial-2-pristine; eval pathway independence; init-time lookup
gating; stale experiment_tag; tiny-face edge case; etc.). Single approve
on the third pass.

### Open

- **Trial 3 running on the VM** — awaiting tier-50 mAP curve vs current
  production baseline (same `unfreeze1000_nblocks6_lrbb1e-4` recipe
  without blur). Tomer will update with results.
- **Follow-up ladder if Trial 3 wins:**
  - 3b: SupCon τ adjustment
  - 3c: padding (catch hairline/jaw)
  - 3d: ALSO eval-masked — diagnostic "body-only" score
- **Carried forward (not actioned this session):**
  - Wedding[1] HTML viewer eyeball pass + Portraits HTML viewer rebuild
    (from 2026-05-12 handoff; Wedding cluster_test_set was running overnight)
  - Trial 2 `experiment.md` write-up — n_blocks=2/12 slices not yet read
  - Pink V13 missing ckpts root cause (deferred from 2026-05-06)
  - `BestCheckpointTracker` metric switch (full-tier vs tier-50 silhouette)

## 2026-05-19 — Face-blur configurability at pretrain stage shipped

### Diagnosis: V11 backbone is too face-dependent for finetune to wash out

Trial 3 (finetune face-blur, shipped 2026-05-18) gave the same eval as the
unblurred baseline. Hypothesis: the V11 continued-pretrain backbone already
encodes face features strongly, and ~10 epochs of finetune (~1.5h,
`lr_backbone=1e-4` on 6 blocks) doesn't have the capacity to reshape that.
Fix: re-shape the backbone at PRETRAIN time by blurring faces in the SSL
training images.

### Code shipped — pretrain face-blur, mirrors the finetune config pattern

- **`train_pictime/pictime_dataset.py`** — added module-level
  `_build_face_lookup_from_paths(image_paths, faces_filename)` that dedupes
  `Path(p).parent.parent` to find unique project dirs and loads each
  `<project_dir>/<faces_filename>`. Returns `dict[(project_id, filename),
  list[(x1,y1,x2,y2)]]` — same key shape as finetune's lookup so the same
  `apply_face_blur` consumes it identically. `PicTimeImageDataset.__init__`
  gained 3 kwargs (`face_blur_enabled`, `face_blur_sigma_factor`,
  `face_blur_faces_filename`); lookup built only when enabled, with
  `[face_blur] Loaded N/M projects ...` log line matching finetune.
  `__getitem__` calls `apply_face_blur(img, (0,0,1,1), faces, img.size,
  sigma_factor)` between `Image.open` and the transform call. Imports
  `apply_face_blur` directly from `train_pictime.finetune.reid_dataset`.
- **`train_pictime/pictime_vitl_im1k_lin834.yaml`** — new top-level
  `face_blur:` block (`enabled: false / sigma_factor: 0.3 /
  faces_filename: "faces.json"`). Zero regression on vanilla runs.
- **`train_pictime/train_dino_grad_accum.py`** — `build_data_loader` reads
  `cfg.get("face_blur", None)` (mirrors finetune pattern at
  `finetune_reid.py:310-313`), threads three kwargs through to
  `PicTimeImageDataset(...)` (column-aligned per feedback rule).
- **`train_pictime/train_dino.py`** — sister-script wiring (per
  `feedback_fix_sister_scripts.md`).
- **`train_pictime/debug_face_blur_pretrain.py`** (NEW) — builds dataset
  with face_blur on (no transform), reverse-indexes face-bearing samples
  from `dataset.face_lookup`, shows 16 in 4x4 plt grid for eyeball
  verification.

### Plan-review iteration

Initial plan proposed extracting `apply_face_blur` to
`train_pictime/utils_pictime/face_blur.py` (shared module) per
`feedback_fix_sister_scripts.md`. Tomer pushed back on first ExitPlanMode:
"just import it from its current location." Plan revised: pretrain imports
directly from `train_pictime.finetune.reid_dataset`. Result is one less
file, slightly unusual cross-module dependency (pretrain → finetune) but
simpler diff. Lesson noted: when the existing helper is already in the
right shape, don't move it just because the rule says
shared-module-preferred — import is fine.

### Hook-point rationale

`PicTimeImageDataset.__getitem__` between `Image.open` and the transform
call is the natural spot because:
- It runs on the FULL pretrain image (each
  `<project>/bbox_images/<name>.jpg`) before `DataAugmentationDINO`
  multi-crops it.
- The blur in original-pixel coords survives downstream RandomResizedCrop +
  Resize + per-crop transforms (just like the finetune blur survives the
  Resize(256)+CenterCrop(224) pipeline).
- `apply_face_blur(img, (0,0,1,1), faces, img.size, sigma_factor)` makes
  the function's offset math a no-op (bx1_px=0, by1_px=0) — faces scale
  directly to image pixel dims.

### Prerequisites (out of scope but blocks V12 launch)

- **faces.json for pretrain image set**: doesn't exist yet. The 6.5M
  pretrain images live as flat paths in `train_paths.txt`; each path has
  the structure `<project_dir>/bbox_images/<name>.jpg`. Tomer kicked off
  yolov11l-face detection on the VM today; expected runtime 3-7 days.
- **Coord-system assumption (must verify once detection finishes)**: face
  bboxes in `faces.json` MUST be normalized [0,1] to the
  `bbox_images/<name>.jpg` crop, NOT the original Pictime full-frame.
  If detection outputs full-frame coords, the blur math will land on the
  wrong region. Open one `faces.json` + matching crop and eyeball before
  enabling the flag.

### Scheduled reminder

Created `claude.ai/code/routines/trig_011rzNLsknD1VPv67XtUdWRC` to fire on
Sun May 24, 2026, noon Jerusalem (UTC 9:00). Prompt covers: check
detection completion, coord-system sanity check, run
`debug_face_blur_pretrain.py`, flip `face_blur.enabled: true`, smoke
test (~500 iters), launch V12 from LVD-142M with blur on, downstream
Trial 3 rerun on V12/ckpt/~13000 vs V11/ckpt/13000.

### Open

- Awaiting face-detection on 6.5M pretrain images (3-7 days). V12 launch
  blocks on this + coord-system verification.
- Carried forward (still): Wedding[1] HTML viewer eyeball + Portraits
  viewer rebuild (from 2026-05-12), Trial 2 `experiment.md` write-up
  (n_blocks=2/12 slices), pink V13 missing ckpts root cause,
  `BestCheckpointTracker` metric switch (full-tier vs tier-50 silhouette).
