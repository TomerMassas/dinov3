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

### Follow-up — `lr_backbone` sweep at `n_blocks*`

After Trial 1, hold `n_blocks = n_blocks*` (winner) and sweep
`lr_backbone ∈ {1e-5, 5e-5, 1e-4, 5e-4}` — 4 runs.

- Run names: `finetune_reid_vits16_bs64_nblocks{N*}_lrbb{lr}_v11ckpt13k`
- W&B group: `trial_lr_backbone_at_nblocks{N*}`
- Cost: 4 × 1.5h ≈ 6h.

### Results

_(fill in after runs complete)_
