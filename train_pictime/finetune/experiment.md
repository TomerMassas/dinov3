# Finetune Experiments Log

---

## Trial 1 — `n_blocks` sweep (+ `lr_backbone` follow-up)

**Goal**: find the best number of last-N transformer blocks to unfreeze in
Mode C, holding everything else fixed. ViT-S/16 has 12 blocks total.

### Setup (held fixed)

| Knob | Value |
|---|---|
| Backbone | V11/ckpt/13000, teacher |
| Loss | SupCon, τ=0.07 |
| Head | MLP 384→384→128, L2-norm |
| Optimizer | AdamW, head lr=1e-3, lr_backbone=1e-4, weight_decay=1e-4 |
| Schedule | 10 epochs, warmup 500, cosine to 1e-6 |
| PK | P=16, K=4, batch=64 |
| `unfreeze_after` | 2000 (skipped when `n_blocks=0`) |
| Curriculum | disabled |
| Eval | tier-50 / 75 / 100, drop -1 |
| `clip_grad` | 5.0 |

### Sweep

`n_blocks ∈ {0, 2, 4, 6, 8, 12}` — 6 runs.

- Run names: `finetune_reid_vits16_bs64_nblocks{N}_v11ckpt13k`
- W&B group: `trial_n_blocks`
- Primary metric: **tier-50 mAP** (and silhouette).
- Cost: ~1.5h × 6 ≈ 9h.

### Results

All 6 runs completed cleanly past 8k iters; no crashes, no divergence
(incl. n_blocks=12). Final-iter readings (eyeballed from W&B at ~8k):

| n_blocks | mAP_top50 | mAP_top75 | mAP_top100 | R1_top50 | sil_top50 |
|---|---|---|---|---|---|
| 0 (frozen) | ~0.790 | ~0.715 | ~0.640 | ~0.90 | ~0.34 |
| 2          | ~0.825 | ~0.775 | ~0.715 | ~0.92 | ~0.40 |
| 4          | ~0.830 | ~0.785 | ~0.720 | ~0.93 | ~0.40 |
| 6          | ~0.835 | ~0.790 | ~0.720 | ~0.93 | ~0.41 |
| 8          | ~0.830 | ~0.785 | ~0.715 | ~0.92 | ~0.40 |
| 12         | ~0.830 | ~0.785 | ~0.720 | ~0.92 | ~0.40 |

### Findings

- **Frozen vs unfrozen: +4pp mAP_top50** — the dominant signal. Mode-C
  unequivocally beats Mode-A on V11.
- **`n_blocks` ceiling at 2.** All unfrozen runs collapse into a ~0.005
  mAP band; n_blocks=6 marginally on top, but well within run-to-run
  noise. Capacity is not the bottleneck at `lr_backbone=1e-4`.
- **n_blocks=12 trained stably** — no need to pre-emptively lower
  lr_backbone for big-unfreeze regions. Validates exploring the
  high-LR × high-n_blocks corner in Trial 2.
- **n_blocks=4 vs V14 baseline** — current run lands at mAP_top50 ≈ 0.830
  vs V14 0.809. Not a regression; consistent with the -1 filter shift
  and tiered-eval recipe.

### Implication for next trial

The flat n_blocks curve says either (a) 2 blocks is genuinely enough,
or (b) `lr_backbone=1e-4` is too low to differentiate. Trial 2's 2D
grid distinguishes these — if the LR sweep separates the n_blocks
column, capacity matters and we just under-utilized it; if the column
stays flat across LRs, 2 blocks is the answer.

---

## Trial 2 — `lr_backbone` × `n_blocks` sweep

**Goal**: Trial 1 showed a flat n_blocks curve (mAP_top50 ≈ 0.83 across
n_blocks ∈ {2, 4, 6, 8, 12}, +4pp over frozen). Hypothesis: current
`lr_backbone=1e-4` isn't extracting differential value from extra
unfrozen blocks. Sweep LR jointly with n_blocks to test whether higher
LR re-engages capacity (or confirm 2 blocks is enough).

### Setup (held fixed)

| Knob | Value |
|---|---|
| Backbone | V11/ckpt/13000, teacher |
| Loss | SupCon, τ=0.07 |
| Head | MLP 384→384→128, L2-norm |
| Optimizer | AdamW, head lr=1e-3, weight_decay=1e-4 |
| Schedule | 10 epochs, warmup 500, cosine to 1e-6 |
| PK | P=16, K=4, batch=64 |
| `unfreeze_after` | 2000 |
| Curriculum | disabled |
| Eval | tier-50 / 75 / 100, drop -1 |
| `clip_grad` | 5.0 |

### Sweep

2D grid: `n_blocks ∈ {2, 6, 12}` × `lr_backbone ∈ {1e-5, 5e-5, 1e-4, 5e-4}`
— 12 runs.

- Iteration order: outer `n_blocks`, inner `lr_backbone` (all 4 LRs at
  n_blocks=2 finish first → useful early signal).
- Run names: `finetune_reid_vits16_bs64_nblocks{N}_lrbb{lr}_v11ckpt13k`
- W&B group: `trial_lrbb_x_nblocks` (all 12 overlay)
- Primary metric: **tier-50 mAP** (and silhouette).
- Cost: ~1.5h × 12 ≈ 18h.

### Watch-outs

- **High-LR × big-unfreeze instability**: `lr_backbone=5e-4` with
  `n_blocks=12` is the most aggressive cell. Could destabilize on V11
  (backbone moves too fast). Per-run try/except keeps the sweep going
  if it crashes — check terminal for `[Trial 2] ... CRASHED:` lines.
- **Low-LR × big-unfreeze underutilization**: `lr_backbone=1e-5` with
  `n_blocks=12` may behave near-frozen (too small to move 12 blocks
  meaningfully in 10 epochs). Expected; useful as a lower-bound point.

### Results

_(fill in after runs complete)_

