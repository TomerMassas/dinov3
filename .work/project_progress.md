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

## 2026-05-26 — V12 launch, face-blur pretrain running, fracturing eval shipped

### V12 face-masked pretrain launched (after NVML reload + crash recover)

After ~5 days of yolov11l-face detection running on the VM, the faces.json
files landed. Pre-launch eyeball confirmed coord-system was right
(crop-local normalized [0,1] bboxes against `bbox_images/<name>.jpg`);
debug viz showed faces blurred cleanly.

First launch crashed in `setup_job` → `dist.barrier()` with
`nvmlInit_v2() failed: Driver/library version mismatch` (NVML lib version
580.159) — classic kernel-module-vs-userspace-libs skew, almost certainly
from an unattended-upgrade landing on the VM at some point. Step-by-step
`rmmod nvidia_uvm/drm/modeset/nvidia` → `modprobe nvidia + nvidia_uvm`
reloaded the new driver in-place (no reboot needed). Module size confirmed
the swap: `nvidia.ko` went from 104,136,704 → 104,165,376 bytes. V12
relaunched, training started from foundation init + face_blur enabled.

### V12 training crash + resume (iter 1988 → resume from ckpt 1750)

Training halted at iter ~1988 (cause not investigated). Flipped
`train_dino_grad_accum.py:223 resume=False → True`, relaunched.
`use_latest_version_dir(args)` picked V12, foundation loader skipped
correctly (ckpt has the weights), W&B run resumed via persisted
`wandb_run_id.txt`. Tomer also dropped `eval_period_iterations: 12500 →
500` in the yaml so first eval signal lands quickly after resume.

### W&B "Crashed" badge with training still alive

After resume, W&B run showed "Crashed" while GPU was clearly busy and
stdout iter counter kept advancing. Diagnosed as wandb-client disconnect
(heartbeat stopped, training thread kept running independently). Not
caused by the fracturing eval running concurrently — GPU contention
causes CUDA OOM, not silent W&B sync drop.

Decision: Option A — let training continue (ckpts save to disk every 250
iters, stdout log captures losses + eval prints). Plan on next ckpt save
(iter 2000): kill + rename `wandb_run_id.txt → .crashed-backup` to force
a fresh W&B run on relaunch (avoids wandb's flaky re-attach-to-crashed-run
behavior).

Throughput: ~77 sec/iter, on par with V11 — face-blur isn't a meaningful
regression. The "duplicate iter 1992" tqdm output Tomer flagged was just
a mid-iter display refresh, not a stuck loop.

Loss at iter ~1992 was ~16.55 vs V11's roughly 10-12 baseline. Could be
face-blur making the SSL task harder (expected) or a resume artifact.
Open until iter 2500's first eval.

### Fracturing eval shipped — `train_pictime/finetune/fracturing_eval/`

New eval pipeline: for each ground-truth identity in a held-out gallery,
measure how many predicted sub-clusters the model splits it into, plus
the size distribution.

Design locked via 5-question intake: held-out IDs = parameter; drop
predicted `-1` crops; drop GT `-1` entries; `MIN_GT_CLUSTER_SIZE = 5`;
lives at `train_pictime/finetune/fracturing_eval/`.

Files shipped:
- `config.py` — imports `FINETUNE_VERSION_DIR` + HDBSCAN params from
  `realworld_eval/config.py` (single source of truth for V<n>); adds
  `HELD_OUT_PROJECTS_FILE`
  (`/data/AI/Tomer/dinov3/train_pictime/finetune/fracturing_eval/approved_projects.json`),
  `DATASET_ROOT` (Portraits[26]), `MIN_GT_CLUSTER_SIZE=5`.
- `run_fracturing_eval.py` — embed → HDBSCAN → per-GT-cluster fracturing
  math → `summary.json`. Reuses `embed_project` + `crop_bbox` +
  `cluster_embeddings_hdbscan` from `realworld_eval/cluster_test_set.py`.
  Embeds ACTIVE crops only (the subset surviving filters), so HDBSCAN
  density isn't polluted by noise/tiny-cluster crops.
- `plot.py` — Plot 1 (fracturing histogram, `0` bucket for all-noise GT
  clusters, "12+" tail cap), Plot 2 (per-rank sub-cluster % violins,
  median lines, drops ranks with `n<5`).
- `__init__.py` — empty.

After Tomer asked, added the plot calls into `run_fracturing_eval.py`'s
`main()` so one command runs the full pipeline. `plot.py` stays
standalone for re-plotting from `summary.json` without re-running
embeddings.

### Plot interpretation gotcha (worth remembering)

Plot 2's `n at rank r` = GT clusters with `fracturing_count >= r`
(cumulative), NOT clusters with fracturing exactly `r`. Means rank-1's
violin is heavily dominated by the perfectly-grouped clusters (always at
100%). Sanity check: `n at rank 1 == sum of Plot 1's bars for x>=1`.

If the perfect-cluster mass at 100% makes rank-1 hard to read in future
runs, we can change the plot to exclude `fracturing_count == 1` from
rank-1's violin — flagged, not changed yet.

### Open / carried forward

- V12 training: relaunch with fresh W&B run after next ckpt (iter 2000)
  saves. First real eval at iter 2500.
- Loss ~16.5 vs V11 baseline 10-12 — interpret after first eval lands.
- May 24 reminder routine `trig_011rzNLsknD1VPv67XtUdWRC` is stale and
  can be deleted at https://claude.ai/code/routines.
- Carried forward (still): Wedding[1] HTML viewer eyeball + Portraits
  viewer rebuild (from 2026-05-12), Trial 2 `experiment.md` write-up,
  pink V13 missing ckpts root cause, `BestCheckpointTracker` metric
  switch.

## 2026-06-23

### Checkpoint save: best + last (`finetune_reid.py`)

Replaced `BestCheckpointTracker` (best-N by silhouette) with
`CheckpointManager` keeping exactly two files at all times: `ckpt_iter{N}_sil{S}.pt`
(the single best — name unchanged so `find_best_silhouette_ckpt` and all
downstream eval keep matching exactly the best) and `last_iter{N}_sil{S}.pt`
(rolling most-recent, distinct prefix ignored by the `ckpt_iter*` glob). Both
written every eval; final eval makes `last` the final state. `ckpt_max_keep`
left in config, unused.

### OLD ResNet vs NEW ViT-S/16 comparison — `train_pictime/model_comparison/` (new, isolated)

Built a full, self-contained comparison framework to show the deployed V44
ViT-S/16 (128-d) beats the old production ResNet (2048-d) on the reviewer-labeled
galleries. Decisions locked via a 4-question intake.

Key finding from the **IdentityClustering** repo (pic-time, read via `gh`):
production body clustering is **Agglomerative** (`distance_threshold=0.85`,
average linkage, euclidean-on-normalized, + cosine<0.2 centroid merge) — NOT
HDBSCAN (HDBSCAN is the *face* path). The repo's `config.py` already has
`body_embedding_size: 128` / `body_emb_size {2:128}` — it's pre-migrated for our
new embeddings. So comparison = production-old (ResNet+Agglom@0.85) vs tuned-new
(ViT+HDBSCAN); not apples-to-apples by design, Tomer approved.

Files: `config.py` (single source of truth), `prepare_eval_set.py`
(test-set selection), `models.py` (old reid_src CTL loader via sys.path +
new ViT loader), `clustering.py` (agglom + hdbscan), `metrics.py`,
`embed.py` (per-model cache, hardcoded `MODEL` constant), `evaluate.py`
(report + plots), `report.py` (re-render from results.json), plus the two
diagnostics below.

Metrics, two families: **embedding quality** (silhouette cosine, mAP — later
trimmed to silhouette only per Tomer); **clustering quality** computed
**ignoring predicted noise (-1) entirely** (drop `-1` crops, score only
assigned) — mean fracturing, % perfectly grouped, **cluster-level**
precision/recall (dominant-identity matching, NOT pairwise — Tomer's call),
completeness, homogeneity, ARI, and `cluster_count_delta` (mean pred-true
clusters per project; sign = over/under-cluster).

Env fixes along the way: `pip install yacs` (only missing dep for old
`reid_src`); old weights from `models_v1.zip` (the SAS curl initially saved a
223-byte error body -> `-fL`); old `reid_model.pt` cache truncated at 5.3 MB
(should be ~390 MB for 2048-d) -> re-embed.

### HDBSCAN param optimizer — `optimize_hdbscan.py`

One-param-at-a-time sweep on the cached NEW embeddings; plots all metrics vs the
swept value (left [0,1] axis; fracturing + `cluster_count_delta` on right axis
with a zero line), marks best by `OBJECTIVE`. Wired all HDBSCAN knobs through
`config.NEW_CLUSTER` + `cluster_hdbscan` (`min_samples`,
`cluster_selection_epsilon`, `cluster_selection_method`, `allow_single_cluster`).
Switched to the standalone `hdbscan` package (sklearn's crashes on
epsilon>0 + allow_single_cluster — and it's what production uses).

**Key insight:** initial epsilon sweep said 0.5 was "dramatically best" — but
that's the merge-everything artifact (recall/completeness/fracturing all reward
collapsing to one cluster; only homogeneity penalizes it). Switched the sweep
OBJECTIVE to **ARI** (chance-corrected, punishes merging *and* splitting, can't
be gamed by collapse) and added homogeneity + cluster-count-delta diagnostics.

### GT cluster-size histogram — `gt_cluster_size_hist.py`

Tomer noticed metrics improve as `min_cluster_size` 3->10 and suspected test-set
bias. Histogram of GT identity sizes confirms it: top-200-by-crops/clusters is
skewed to big galleries, so a high floor noises out the few small identities and
you score only easy big ones. `build_histogram()` is reusable — `evaluate.py`
auto-generates it if missing and embeds it in `comparison.md`.

### Per-test-set results dir restructure

`TEST_SET_NAME` in config now drives `results/<name>/` holding the proj-ids file
(`proj_ids_<name>.json`, was `new_projects.json`), both caches, and all outputs —
so multiple test sets sit side by side. `prepare_eval_set.py` supports
`TOP_N=None` (all approved). Current sets: `200_biggest` (done) and `all_approved`
(814 approved+valid projects, in progress).

### Open / carried forward

- Run the full comparison on `all_approved` (embed new + old, then evaluate) to
  see how much of the `min_cluster_size` gain was the 200_biggest bias.
- HDBSCAN params still being tuned (sweep each knob via `optimize_hdbscan`, lock
  into `config.NEW_CLUSTER`); greedy one-at-a-time, may need a re-sweep pass.
- The `1e-12` epsilon in `build_store._compute_distances` vs `build_centroids`
  parity — still unresolved (from prior session).
- Carried (still): Wedding[1] HTML viewer, Trial 2 `experiment.md`, pink V13
  missing ckpts, store pipeline VM end-to-end test.

## 2026-06-24 — Real-world eval on V51 + wedding labeling prep

### Model-comparison report — late refinements (cont. from 06-23)

- Trimmed `comparison.md` to clustering-quality only: `REPORT_GROUPS=("B",)` drops
  the embedding-quality table + `plot_embedding_quality.png` (silhouette/mAP still
  computed into results.json, just not shown).
- Added a `NEW_MODEL_INFO` provenance table (backbone + finetune rows: method /
  data / labels / checkpoint) rendered at the top of the report, so each run
  self-documents which ckpt produced it — for the upcoming ckpt-vs-ckpt compares.

### V44 train/test contamination check

`all_approved` (814) vs V44's denylist (`approved_projects.json`, 709):
**709 clean (held out), 105 seen during training** — they were approved *after*
V44 trained, so at train time they had no clusters_fixed and V44 trained on them
with `clusters_v3` HDBSCAN pseudo-labels. Verdict (Tomer): acceptable — no label
leakage (trained on HDBSCAN labels, scored on reviewer truth) and 105/~50K
training share -> no memorization. Left `all_approved` as-is.

### Standalone clustering script for the labeling teammate — `cluster_bodies.py`

Self-contained (`numpy` + `hdbscan` only, no repo/config) `cluster_bodies(embeddings)`
with defaults = `NEW_CLUSTER`. Sent to Tomer to forward. (lives in scratchpad)

### Real-world eval re-run on V51 — and the key clustering insight

Pointed `realworld_eval` at V51; added a `FINETUNE_CKPT_PATH` override to pin an
exact ckpt (`find_best_silhouette_ckpt` only globs `ckpt_iter*_sil*.pt`, so it
can't pick V51's chosen `last_iter26274_sil0.4679.pt`). Set realworld_eval HDBSCAN
to the tuned `NEW_CLUSTER` params; flipped the import to prefer the `hdbscan` package.

**All galleries came back 0 clusters / all-noise.** Debug chain:
1. `min_cluster_size=10` is wrong for weddings — weddings have MANY guests with FEW
   photos each (small per-identity clusters), the *opposite* of the Portraits
   "biggest galleries" the params were tuned on. Dropped to 3.
2. Still mostly noise at mcs=3 -> isolated to **`allow_single_cluster=True`**: on a
   multi-identity wedding it fits ~one loose root cluster and dumps everyone outside
   its core to noise -> 0-1 clusters + rest noise. `=False` fixed it (V44 test clean).

**Insight (important): the HDBSCAN tuning is domain-specific and does NOT transfer.**
`allow_single_cluster=True` + `mcs=10` suit single-dominant-identity Portraits
galleries but are catastrophic on weddings; weddings need `mcs=3` +
`allow_single_cluster=False`. Keep params per domain.

Caveat flagged: the good run changed BOTH model (V51->V44) and the flag, so the clean
one-variable confirmation — V51 + `allow_single_cluster=False`, same `last_iter26274`
ckpt, delete the clusters dir first — is the pending step. If V51-last is still worse
than V44, the iter-26274 "last" ckpt may be overfit (vs V51's best, or V44).

Bug fixes: ckpt sil-parse regex `[\d.]+` grabbed the trailing dot (`0.4679.`) ->
`[0-9]+\.[0-9]+`; `http.server` port typo `808` (privileged) -> 8080; truncated
`old_embeddings.npz` (5.3 MB vs ~390 MB, interrupted write — disk wasn't full) -> re-embed.

### Wedding labeling prep — `prep_labeling_files.py` (NEW)

Generates the per-project labeling files for `Wedding[1]` using the *same* finetune
model + tuned HDBSCAN as the HTML view (reuses realworld_eval's embed + cluster +
config — single source of truth). Per project writes `embeddings_<tag>.npz`,
`clusters_<tag>.json`, `crop_distances_<tag>.json` (tag = version-dir name, e.g.
`v51`) into the dataset dir. Skips `clusters_fixed.json` + already-done; no crop
JPGs (UI crops on-the-fly). Replicates `build_centroids` distance math in-memory.
Decisions: `_v51` tag, crops on-the-fly, save embeddings (retrain-ready for `build_store`).

### Open / carried forward

- **Confirm V51 + `allow_single_cluster=False` in the viewer** (clean one-variable
  test vs the good V44 run); decide V51-last vs V51-best vs V44 if V51-last looks overfit.
- **Run `prep_labeling_files.py` on `Wedding[1]`** (set realworld_eval/config to the
  chosen V51 ckpt first) -> hand to labeling team -> their `clusters_fixed.json` feeds
  `build_store` next cycle.
- Domain-specific HDBSCAN: `NEW_CLUSTER` (mcs=10, allow_single=True) is Portraits-tuned;
  weddings use mcs=3 + allow_single=False.
- **Uncommitted:** this session's edits + both progress entries aren't committed yet
  (earlier /session-end commit skipped at Tomer's request).
- Carried (still): `all_approved` full comparison run, `1e-12` epsilon parity,
  Trial 2 `experiment.md`, pink V13 ckpts, store pipeline VM end-to-end test.

## 2026-06-30 — Global config (effort/model) + onboarding new wedding galleries

### Global Claude config — effort & model

- **Discovery:** this install's config dir is `CLAUDE_CONFIG_DIR=~/.claude-work` (why
  auto-memory + settings live under `~/.claude-work/`, not `~/.claude/`). Live global
  settings = `~/.claude-work/settings.json`.
- `effortLevel: "xhigh"` was already set there (max-effort default already global). The
  persisted `effortLevel` enum only accepts low/medium/high/xhigh — **no storable `"max"`**;
  "max" is a session-only `/effort` toggle above xhigh.
- Adding `"model": "opus"` to `~/.claude-work/settings.json` was **rejected** -> global
  default model still unset. Revisit if opus-by-default is wanted.
- Cleanup owed: `~/.claude/settings.json` (the UNUSED file for this install — not read,
  since config dir is `.claude-work`) got `model: opus` + `effortLevel: xhigh` written by
  mistake; original was `model: sonnet`, no effortLevel. Revert it.

### Onboarding new galleries into Wedding[1] (new: 52544230, 52544256)

New projects arrive with only `images/`. Pipeline to make them labeling-ready +
old-vs-new-comparable:
1. **Detection** — lives in the **person-reID** repo (`body_detection_src/detect_body.py`,
   `Yolo11PersonDetector`, `yolo11m.pt`), NOT dinov3 (dinov3 only *reads* `detections.json`).
   No offline folder-batch runner in-repo (only a test `__main__` + the prod
   `process_request.py` queue). Output `{filename:[{"bbox":[x1,y1,x2,y2] norm,"conf":..}]}`
   — matches every dinov3 reader. Tomer ran detection himself.
2. **New-model files** — `prep_labeling_files.py` -> `embeddings_v51.npz` /
   `clusters_v51.json` / `crop_distances_v51.json` per project; walks all of Wedding[1]
   but skip-if-exists, so re-running only does the 2 new ones.
3. **Old-model embeddings** — `embed_old_wedding.py` (NEW).

### `embed_old_wedding.py` (NEW, model_comparison/) + a latent bug found

Embeds the 2 projects with the OLD ResNet over ALL detected crops (2048-d, raw) ->
`embeddings_resnet.npz` per project (reuses `model_comparison.models` old
loader/transform/forward). **Bug caught:** the existing `ProjectCropDataset`/`CropDataset`
use a fixed `(3,224,224)` invalid-crop placeholder — matches the *new* transform but
mismatches the old `Resize(256,128)`, so a batch mixing valid+invalid crops would crash
collate. Never bit the Portraits GT run (clean crops); could bite raw wedding detections.
New script uses a placeholder matching `OLD_RESIZE_HW`.

### Open / carried forward

- **Global model default:** decide opus-by-default (the `~/.claude-work` edit was rejected)
  and revert the stray `~/.claude/settings.json` edits.
- On the 2 new projects: detection -> `prep_labeling_files.py` -> `embed_old_wedding.py`.
- Still pending: confirm V51 + `allow_single_cluster=False` in the viewer; run
  `prep_labeling_files.py` on full Wedding[1] -> labeling team.
- **Uncommitted:** several sessions' edits (model_comparison, realworld_eval,
  prep_labeling_files, embed_old_wedding) + 3 progress entries still not committed.
- Carried (still): `all_approved` full comparison, `1e-12` epsilon parity, Trial 2
  `experiment.md`, pink V13 ckpts, store pipeline VM end-to-end test.

## 2026-07-27/28 — Body-part fragment classifier built (`classifier_body_parts/`)

New self-contained package to filter "meaningless" body crops (a hand, a leg, a
torso sliver) BEFORE ReID clustering, so clusters stop being polluted by crops
carrying no identity signal. Tomer labels; this is the training/inference side.

### Design decisions

- **Rejected raising the YOLO conf threshold** (Tomer's option 1). conf answers
  "is this a person, is the box tight", not "is this crop useful for telling
  people apart" — a crisp background torso scores high, an occluded real guest
  scores medium. And weddings run mcs=3, so a global conf cut removes exactly the
  small/distant detections thin identities are made of. conf became a *feature*.
- **Frozen V18 SSL pretrain backbone, NOT V44/V51 finetune.** SupCon is trained to
  map a hand crop and a full-body crop of the same person to the same place — it
  deliberately destroys crop-completeness, the exact signal needed here. The 128-d
  projection output would be the worst possible feature.
- **Architecture:** frozen ViT-S/16 -> 384-d CLS + 12 geometry scalars ->
  StandardScaler -> LogisticRegression. Only 396 weights train. ~545 positives is
  linear-probe territory; finetuning would overfit.
- **Transform bug found:** `reid_dataset.get_val_transform` is
  `Resize(256)->CenterCrop(224)`; Resize with an int scales the SHORT side, so a
  100x300 full-body crop -> 256x768 -> centre crop keeps roughly the torso. The ViT
  sees a torso for BOTH "full body" and "torso only". Added `letterbox` (pad to
  square with ImageNet mean) and `warp` variants; all three ablated.
- **Two thresholds, opposite objectives.** Labeling (recall >= 0.99) gates what the
  app displays — a miss there silently becomes a permanent wrong label. Deploy
  (precision >= 0.95) gates what's dropped before clustering — a miss costs one crop.
- **Three-run eval:** selection (StratifiedKFold within CV_GALLERY, ranked by
  PR-AUC since the prior is fixed), LOGO (ROC-AUC, prior-invariant, + Wilson CIs,
  thin folds flagged never dropped), ship (all data, never evaluated).

### The negative-pool decision (the important one)

`kept_keys | deleted_keys` is exactly what the reviewer was SHOWN. From round 2 on
the app only displays classifier-flagged crops, so **`baseline - kept` is unsafe** —
it absorbs the suppressed pool, i.e. the classifier's own predictions, and every
fragment it misses becomes a hard negative. Blind spots compound each round.

**Decision: negatives = `deleted_keys` only.** Considered and rejected using
suppressed crops as weak negatives: noise rate would be low but sits precisely on
the hardest examples, and negatives aren't the bottleneck (~5000 neg vs 537 pos).

Tomer's follow-up: as the model improves, `deleted_keys` degenerates to *boundary*
negatives only — negative diversity freezes at the first galleries and the training
prior drifts from deployment (moving the thresholds). Fix: **`RANDOM_QUOTA = 300`**
(3 UI batches of 100) sampled from the SUPPRESSED pool and displayed alongside.
Sampling suppressed rather than whole-baseline composes cleanly — above-threshold is
a census, suppressed is sampled — giving the only direct read on false negatives:
`missed = (fragments kept in audit) / sampling_fraction`.

`EXCLUDE_REVIEWED`: already-judged crops are subtracted from both display lists, but
still scored — so per-gallery precision/recall is computable from the scores file alone.

### Cache refactor (Tomer's idea, mid-session)

Nothing upstream of the LR ever changes — the backbone is frozen, geometry comes from
detections.json. So `embed.py` now caches each gallery's **whole baseline pool** once
into `<project>/classifier_embeddings_v18_<transform>.npz`, and both `train.py` and
`predict.py` run off it. **`predict.py` lost its GPU path entirely** — re-scoring the
dataset after a retrain is now CPU + seconds.

Each cache carries a fingerprint (backbone/ckpt/which, transform, crop_size,
geometry_names, detections signature) verified on load. The detections signature is a
**content hash, not size+mtime** — a smoke test caught that mtime changes on
rsync/restore/machine-move and would needlessly invalidate every cache.

`model_v18.pkl` is one fixed filename overwritten every train run, so predict always
picks up the newest model with no config change.

### Files

`config.py` (single source of truth) · `dataset.py` (labels, baseline, geometry,
transforms, fingerprinted cache) · `embed.py` (GPU, only GPU step) · `train.py` (CPU,
3 runs + thresholds + report) · `predict.py` (CPU, scores -> UI decision file) ·
`README.md` (full reference: 15 sections, metrics primer, diagnostics)

Also wrote a prompt for the UI-repo Claude: display `show_keys | audit_keys`, add a
third `suppressed_keys` bucket, and **never** write non-displayed crops to
`deleted_keys`.

### First real run (3 galleries) — signal is real, threshold is the problem

- **`warp` won, not `letterbox`** (PR-AUC 0.5088 vs 0.4868; letterbox lost even to
  `reid_val`). Prediction was wrong — likely the padding borders being OOD for the
  backbone. The ablation existed to settle this and did.
- PR-AUC 0.509 vs a 0.089 baseline, ROC-AUC 0.90. `C=0.001` wins as expected.
- **`cls+geom` 0.5088 ~= `cls` 0.5060** — geometry adds almost nothing on top of CLS.
  `geom` alone 0.273, still 3x baseline.
- Run 2 cross-gallery ROC-AUC 0.865 / 0.889 -> **generalizes; not venue memorization.**
- **THE PROBLEM:** labeling threshold at 0.99 recall shows **68% of the pool** —
  only 1.5x speedup, FPR 0.65-0.82 across galleries. Deploy threshold at 0.95
  precision catches just **4.7%** of fragments. The PR curve falls off a cliff before
  0.99 recall.
- Correction: `21833423` is 8 pos / **65** neg (73 crops) — a *small* gallery, not a
  low-prior one. All three sit at ~9-11% positive rate; its FPR CI is [0.22, 0.44],
  as useless as its recall.

### Environment

NVIDIA driver mismatch on `developer-gpu4` blocked the first embed run — kernel module
580.159.03 vs userspace 580.173.02 (package upgraded, no reboot). Every GPU script on
the VM was affected. Note `torch.cuda.is_available()` returned **True** despite NVML
being dead, so the preflight passed and it surfaced as a NCCL trace.

### Open / carried forward

- **Threshold selection is the weak link** — comes from CV_GALLERY's out-of-fold probs
  alone, so it never improves as galleries are added. Offered: pool LOGO out-of-fold
  probs instead (more data, cross-gallery, less optimistic). Not done.
- **Add generated-at timestamp + per-run gallery list to `report.md`** — offered, not
  done. Would have avoided today's stale-local-copy confusion.
- 4th gallery labeled; the fresh 4-gallery report on the VM not yet read.
- **VM cleanup:** delete `features_v18_*.npz` (dead, pre-refactor global cache) and
  `model_v18_warp_cls_geom.pkl` (old naming).
- UI-side changes not yet implemented (prompt written, not applied).
- App should record `suppressed_keys` + a per-crop display reason
  (`"scored"`/`"random"`) — cheap now, impossible retroactively; enables importance
  weighting later.
- Is `21833423`'s tiny size real, or a labeling-standard difference? Affects how its
  LOGO fold reads.
- Strengthen the CUDA preflight (exercise `device_count()` + a real allocation) —
  offered, not decided.
- Carried (still): `all_approved` full comparison, `1e-12` epsilon parity, Trial 2
  `experiment.md`, pink V13 ckpts, store pipeline VM end-to-end test.
- Committed mid-session (`1089514`), clearing the long-standing uncommitted backlog
  from the 06-23/24/30 sessions along with this package.

## 2026-07-28 (cont.) — Pooled calibration + backbone switched to the deployed ReID model

### The miscalibration that surfaced

The 6-gallery report showed `26648310` at **77%** positive and `31643124` at **69%**,
versus ~9-15% for the first four. Cause: those two were labeled *through* the classifier
filter, so the reviewed pool was the classifier's own output, not the gallery.

`CV_GALLERY = None` auto-picks the gallery with the most positives -> it picked the 77%
one -> **both thresholds were calibrated at a 77% prior while deployment sees ~10%.**
Run 2 showed the cost: FPR 0.87-0.91 on the real-prior galleries, i.e. ~90% of the pool
still displayed — worse than the 68% of the 3-gallery run. ROC-AUC also fell from
0.865/0.889 to 0.79-0.84, because training had become positive-heavy with only boundary
negatives.

### Pooled cross-gallery calibration (Tomer's proposal, refined)

Tomer's instinct: stop calibrating on one gallery — do leave-one-gallery-out and combine.
His version averaged the per-gallery thresholds weighted by gallery size.

**Refinement: pool the DATA, not the thresholds.** The threshold->recall map is
non-linear, so averaging thresholds lands short of the target — and lands short
specifically on the galleries where the model is weakest, which are the ones most worth
protecting. The smoke test measures it: a size-weighted average of per-gallery thresholds
gave **0.9688** recall where pooling gave **0.9908** against a 0.99 target.

Pooling also delivers his size-weighting for free, and by the *right* size per metric —
positives drive recall, negatives drive FPR — which one per-gallery weight cannot do.

Restructure: `run_logo` split into `logo_out_of_fold` (takes no threshold — it produces
what the threshold is derived from), `pool_out_of_fold`, and `fold_report`. Run 2 now
runs *before* the thresholds, breaking the circular dependency that forced
single-gallery calibration. `CALIBRATION_GALLERIES` is the escape hatch.

### Coverage guard — measured, not a hardcoded list

    coverage = |kept ∪ deleted| / |crops in the embedding cache|

Measurable precisely because the cache holds the whole baseline pool. Warns below
`MIN_COVERAGE_WARN = 0.95`; the report flags such galleries `← PARTIAL`. It would have
caught the 77% galleries automatically, and it doubles as a labeling progress meter.

### Backbone switched to the deployed ReID model (`ft_v44`)

Tomer flagged the production cost: body embeddings come from the finetuned model, so a
classifier on V18 pretrain means **two forward passes per crop** in production.

Now `BACKBONE_SOURCE = "finetune"`, tag `ft_v44`, ckpt =
`/data/AI/Tomer/person_reid/models/ckpt_iter15000_sil0.4556.pt` (what production loads).
Takes the **384-d pre-head CLS**; the projection head is never built. Rationale: the head
is where SupCon's nuisance-invariance is enforced, and mode-C only unfreezes the last N
blocks, so most of the backbone is still V18 weights. Base arch/weights are read from
`reid_config.yaml` so they cannot drift from what the finetune actually started from.

Both sources coexist via `BACKBONE_TAG`, so the `v18` vs `ft_v44` comparison stays
available on identical labels.

### Feature signature — the loud retrain guard

Accepted cost: every finetune release now invalidates the classifier. Tomer asked for it
to fail loudly. The fingerprint is split:

- `feature_signature(transform)` — gallery-independent (source, ckpt, which, transform,
  crop_size, geometry_names) — stored **in the model bundle** at train time
- `cache_fingerprint(gallery, transform)` — that **+** the gallery's `detections.json`
  content hash — stamped on each cache

`predict.py` compares the bundle's signature against the live config *before scoring
anything* and refuses with `RETRAIN THE CLASSIFIER`. This catches the dangerous case:
both backbones emit 384-d vectors, so a stale model would otherwise run happily and
score everything wrong with no error. Tested four ways (changed ckpt, changed source,
missing signature, cache fingerprint moving independently).

### Smaller changes

- `SCORES_FILENAME` -> stable untagged **`classifier_scores.json`** so the UI has one
  fixed path. Deliberate deviation from the `{model_id: filename}` convention —
  provenance lives inside the file's `model` block instead.
- `EMBED_CACHE` / `MODEL_FILE` -> templates keyed off `BACKBONE_TAG`; the old tag-keyed
  dicts would `KeyError` the moment a new tag was set.
- `report.md` gained a **generated-at timestamp**, after an hour lost to reading a stale
  local copy that had never been downloaded from the VM.
- `detections_signature` switched from size+mtime to a **content hash** — a smoke test
  caught that mtime changes on rsync/restore/machine-move and would needlessly
  invalidate every cache.

### Corrections logged

- `21833423` is 8 pos / **65** neg (73 crops) — a *small* gallery, not a low-prior one.
  Earlier claims about thousands of negatives there and a ~40x prior gap were wrong.
- Claimed the UI wasn't wired to the scores file; the 77% positive rate proved it was.
  That was an assumption from silence, not evidence.
- Predicted `letterbox` would win the transform ablation. `warp` won and letterbox lost
  even to `reid_val` — padding borders are likely OOD for the backbone.

### Status at session end

Tomer manually reviewed every suppressed crop across all 6 galleries, so **all six are
now 100% covered** — pooling over all of them is valid and the coverage warning should
stay silent. Nothing has run under `ft_v44` yet. 141 checks across 3 smoke suites, green.

### Open / carried forward

- **Run `embed` -> `train` -> `predict` under `ft_v44`** (embed is a full re-embed: new
  namespace, ~10K crops x 3 transforms). Watch for: coverage 100% on all six; positive
  rates for the two filtered galleries dropping from 0.768/0.687 toward ~0.10; **ROC-AUC
  recovering above 0.86** (the falsifiable check on the skew diagnosis); and the pooled
  "shows X% of the pool" — the first honestly-calibrated speedup number.
- **Point the UI at `classifier_scores.json`** and delete the stale
  `classifier_scores_v18.json` files, or the app silently reads a file nothing updates.
- **The weak speedup is still unsolved** — ~68-90% of pool at 99% recall. That is model
  strength, not calibration; levers are more galleries or accepting ~0.95 recall.
- `v18` vs `ft_v44` comparison available but not run — would quantify what the
  single-forward-pass win costs in accuracy.
- Later: V44 -> V52 ckpt swap. Update `FINETUNE_CKPT` -> embed -> train -> predict;
  `predict` refuses if the retrain is skipped.
- App should record `suppressed_keys` + a per-crop display reason (`"scored"`/`"random"`)
  — still not done, still impossible retroactively.
- Strengthen the CUDA preflight (`torch.cuda.is_available()` returned True with NVML
  dead) — offered, not decided.
- Carried (still): `all_approved` full comparison, `1e-12` epsilon parity, Trial 2
  `experiment.md`, pink V13 ckpts, store pipeline VM end-to-end test.
- **Uncommitted:** this session's `classifier_body_parts/` edits (config, dataset, embed,
  train, predict, README) + this entry.

## 2026-08-18 — V52 cluster JSONs for Wedding[1] + the detections.json conf-vintage split

Untracked-on-disk since the 07-28 entry (done outside sessions, Aug 4–9): the `ft_v44` and
`ft_v52` classifier runs, and `tmp_antonia/` — the production-format body-embed path whose
Aug 9 zip covers 8 Wedding[1] galleries.

### Thread 1 — cluster JSONs for all of Wedding[1] under V52

`prep_labeling_files.py` is the entry point (not `cluster_test_set.py`, which samples 100
projects into a separate results tree). Two lines in `realworld_eval/config.py`:
`FINETUNE_VERSION_DIR` → V52 (it drives the output tag) and `FINETUNE_CKPT_PATH` → the exact
ckpt. Everything else was already right: `DATASET_ROOT` = Wedding[1], and HDBSCAN already on
the wedding-tuned `mcs=3` / `allow_single_cluster=False`.

`reid_config.yaml` needed no edit — verified by loading the ckpt directly rather than assuming:
ViT-S/16, 12 blocks, `storage_tokens (1,4,384)`, layerscale present, head 384→384→128, so it
strict-loads against the existing arch + `proj_hidden_dim`/`proj_output_dim`.

Flagged: `load_backbone` still reads the V18 pretrain DCP purely as an arch template before the
V52 weights overwrite it, so deleting old pretrain ckpts breaks this path.

### Thread 2 — 23-gallery V52 body-embed batch, and the conf bug

All 23 requested ids are in Wedding[1] with no overlap with the Aug 9 eight. All 23 failed
with `KeyError('conf')`.

**Root cause: `detections.json` has two vintages inside the same dataset.** Old galleries
carry `det keys=['bbox']`; newer ones carry `['bbox','conf']` (e.g. 0.918). person-reID's
`utils/request_processing.py:61` does a bare `float(bbox['conf'])`, and it **persists** conf
rather than only filtering on it. `process_galleries.py`'s handler printed `{e!r}` with no
traceback, which cost a diagnostic round-trip — now fixed.

**Decision: `MISSING_CONF = 1.0`, fabricated and counted.** Rejected re-detecting: it would
renumber bbox indices and invalidate every `"<filename>_<bbox_index>"` key already written
against these galleries — the fragment classifier's labels, `clusters_*.json`,
`crop_distances_*.json` — including the run being generated right now in thread 1. Rejected
patching `request_processing.py`: in production a detection with no conf IS a bug and should
crash; normalize at the offline-replay boundary instead and leave prod strict.

`counts["synthetic_conf"]` is filled from the **post-`current_revisions`** mapping so it stays
comparable with `counts["crops"]`; filling raw detections would have counted superseded
revisions that are never embedded. Reported per gallery as a fraction and flagged
`MIXED VINTAGE` when `0 < synthetic < crops` — a part-old gallery would otherwise be
indistinguishable from a wholly-old one while sitting real scores beside fabricated ones.

**Caveat on the justification:** `classifier_body_parts/dataset.py:209` already does
`det.get("conf", 1.0)`, but that precedent does not transfer cleanly — there conf is a feature
column fed to a scaler (a benign constant), here it becomes a persisted score in a handoff
artifact. The value is a deliberate fabrication, not an inherited convention.

**Correction to a logged result:** because `dataset.py` defaults conf to 1.0, the classifier's
`geom`-only ablation (PR-AUC 0.273) mixed real conf on some galleries with a constant on
others — `21833423` is bbox-only *and* is one of the 7 training galleries. The shipped model is
`feature_set=cls`, so nothing deployed is affected, but that number is weaker evidence than it
looked.

### Thread 2 result — 23/23 written, and gate 1 validated for free

23 galleries, 0 failed, **every one uniformly 100% synthetic conf** — no `MIXED VINTAGE`, so
the vintage split is per-gallery, not per-image inside a gallery. 12083 crops: 9309 face-exempt
(77%), 2774 scored, 171 dropped (6.2% of scored). Zip packaged.

**Gate 1 validated independently and unplanned:** `21833423` reports 86 crops with 13
face-exempt → 73 no-face crops, which is exactly the classifier's labeled pool for it
(8 pos + 65 neg). The face gate in `process_galleries.py` reproduces the same pool definition
the classifier was trained against.

**And the same gallery puts a number on the weak-recall problem: it dropped 0 of its 8 known
fragments** at the deploy threshold. Consistent with the logged ~4.7%-of-fragments figure
(8 × 0.047 ≈ 0.4, so zero is expected), but note the direction — `21833423` is one of the 7
*training* galleries, so this is **in-sample** recall, and out-of-sample is no better than 0/8.
Read `classifier_kept.json` as a high-precision, very-low-recall filter, not a cleanup. The
classifier also only scored 23% of crops; the other 77% were face-exempt and kept unscored.

### Thread 3 — V52-optimized HDBSCAN params + a recluster-only path

New params arrived as the production spec (`body_distance_function` cosine,
`body_clustering_distance` 0.08, `body_clustering_size` 3, `body_clustering_min_samples` 2,
`body_clustering_method` hdbscan4_1). Translation vocabulary is documented at
`model_comparison/config.py:43-45`, and 0.08 is literally one of `optimize_hdbscan.py`'s
`cluster_selection_epsilon` sweep values, so four of the five map unambiguously.

**`cosine 0.08` is encoded as `euclidean 0.40`, and that is not a deviation.** `embed_project`
L2-normalizes, and on unit vectors `d_euclid = sqrt(2 * d_cos)` — strictly monotone, so core
distances, mutual reachability and the MST keep their ordering and the hierarchy is identical;
every other knob is scale-invariant. Only epsilon is absolute: `sqrt(2 * 0.08) = 0.40`. Reason
to prefer it: `cosine` is in neither `BallTree.valid_metrics` nor `KDTree.valid_metrics`
(checked), so asking hdbscan for it either errors or forces a dense O(n²) matrix per project.

**Scale note:** the previous `0.1` euclidean was `cosine 0.005`, so this is **16x looser**.
Expect markedly fewer, larger clusters — intended if the target was fracturing, but not a nudge.

**`recluster_labeling_files.py` (NEW).** The caches make re-clustering free: nothing upstream of
HDBSCAN changed, so it loads `embeddings_<tag>.npz`, re-clusters, and rewrites only
`clusters_<tag>.json` + `crop_distances_<tag>.json` — no GPU, no model, no image decode. Same
pattern as the classifier cache refactor that cost `predict.py` its GPU path. Params and
distance math are imported from the siblings, never restated, so they cannot drift.

Two design points worth keeping: a **unit-norm guard** (`max |‖v‖-1| < 1e-4`, else refuse) turns
the epsilon conversion from an assumption into a checked precondition — an unnormalized cache
would otherwise cluster at a silently wrong threshold with no error. And it reads the existing
clusters file **before** overwriting to report a per-project before/after delta, which recovers
the comparison otherwise lost to both param sets sharing the `v52` tag.

Consequence: `prep_labeling_files.FORCE` stays `False`, so the expensive embed path stays
guarded and the silent-no-op footgun is gone. 17 smoke checks green, including
`clusters_dict_from_arrays` being JSON-identical to `build_clusters_dict`.

### Open / carried forward

- **Which space were the V52 clustering params tuned in?** `prep_labeling_files` clusters the
  **128-d post-projection-head** output; production serves the **384-d pre-head CLS**. The
  `body_*` naming is prod vocabulary, which suggests 384-d — and a cosine threshold does not
  transfer between differently-shaped spaces. Raised, unresolved; if they were tuned on prod
  embeddings the values may not mean what they should here.
- **`hdbscan4_1` is unmapped** — reads as a prod method-version id, not an HDBSCAN arg. Left
  `cluster_selection_method="eom"` and `allow_single_cluster=False` (the 06-24 wedding fix).
- **`embeddings_<tag>.npz` carries no fingerprint** — reclustering trusts that the cache came
  from the V52 ckpt, with the tag as the only link. Offered to stamp one; not built.
- **Downstream conf semantics never verified** — the grep over person-reID for `conf` consumers
  was requested twice and not run. 1.0 keeps everything if downstream filters, but reads as
  "certain" if it ranks. The one unresolved assumption behind the chosen value.
- **`DETECTION_VERSION = 1` is stamped for both vintages**, so the proto claims one detector
  for two. Same class of provenance gap as the conf value. Undecided.
- **`detection_conf: "synthetic"` marker in the `classifier_kept.json` `model` block** —
  offered, not added; the count is currently the only record.
- Cross-batch conf is meaningless: the Aug 9 zip has real scores, this batch will be all 1.0.
- `21833423` is in the classifier's training galleries, so its keep-list is in-sample and its
  drop-rate is optimistic — 1 of 23, but the manifest reports it per gallery.
- Thread 1 decisions still open: whether to filter fragment crops (via `classifier_kept.json`)
  before clustering, whether to relax the `clusters_fixed.json` skip, and pointing the labeling
  UI at the `_v52` tag.
- Carried (still): `all_approved` full comparison, `1e-12` epsilon parity, Trial 2
  `experiment.md`, pink V13 ckpts, store pipeline VM end-to-end test.
- **Uncommitted:** `tmp_antonia/` + `classifier_body_parts/` edits + `realworld_eval/config.py`
  + `realworld_eval/recluster_labeling_files.py` + this entry.
