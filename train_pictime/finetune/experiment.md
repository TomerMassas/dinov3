# Finetune Experiments Log

Person ReID finetuning of DINOv3 ViT-S/16 (pretrained on 6.5M Pictime images) with SupCon loss.

---

## Runs so far

| Run | Backbone ckpt | Mode | Unfreeze | Iters | mAP | Rank-1 | Rank-5 | Rank-10 | Silhouette |
|-----|---------------|------|----------|-------|-----|--------|--------|---------|------------|
| `finetune_reid_vits16_bs64_frozen` (green) | V9/ckpt/15000 | A — frozen backbone | — | ~9.4k | 0.370 | 0.637 | 0.762 | 0.815 | -0.014 |
| `finetune_reid_vits16_bs64` (pink) | V9/ckpt/15000 | C — progressive unfreeze | 2 blocks @ iter 2000 | ~18k | 0.390 | 0.650 | 0.782 | 0.822 | -0.005 |

Delta (C − A): +0.020 mAP, +0.013 Rank-1, +0.020 Rank-5, +0.007 Rank-10, +0.009 silhouette.

---

## Observations

1. **Mode C fix works end-to-end** — unwrap-FSDP2 + `torch.autocast(bf16)` recipe trained stably past iter 2000 without the bf16→fp32 grad assignment crash.
2. **C beats A, but by a small margin.** ~2pp mAP improvement from unfreezing 2/12 blocks with small `lr_backbone`.
3. **No visible "kick" at iter 2000** in any eval metric when unfreezing kicks in. The backbone update is too gentle to drive a phase change.
4. **Both runs plateau quickly** — mAP essentially flat after ~6k iters, Rank-1 flat after ~4k. Not data-starved; capacity/LR-starved.
5. **Silhouette is negative in both runs.** The embedding space is not forming tight same-person clusters — same-person pairs are on average further apart than cross-project pairs. Ranking metrics look acceptable only because *relative* ordering within a query is OK; absolute cluster tightness is not there.

---

## Hypotheses for the plateau

- **H1. `lr_backbone` too small.** Whatever `lr_multiplier` we set, the backbone is barely moving — no visible kick at unfreeze, tiny delta vs frozen. The update step is in the noise floor.
- **H2. Too few unfrozen blocks.** 2 out of 12 blocks is a small slice of the representation. Lower layers stay generic-DINOv3, possibly misaligned with the crop-level ReID task.
- **H3. SupCon temperature mis-tuned.** If τ is too high, negatives produce weak gradients; too low and the loss is dominated by the hardest (possibly noisy) negatives. Never swept.
- **H4. PK batch too small.** P=16, K=4 → 64 samples per batch, only 16 negative classes. Noisy SupCon estimate, high variance per step.
- **H5. Label noise.** Negative silhouette suggests many "same-person" pairs in the index actually look very different (occlusion, scale, pose, wardrobe changes within a session) or clusters include wrong-person crops. If labels are noisy, every tuning knob is fighting noise.
- **H6. SupCon may be the wrong loss for this data.** SupCon is pair-based: every positive directly pulls the anchor, so a single noisy "same-class" crop in a PK batch can dominate the gradient. Classification-based angular-margin losses (ArcFace family) represent each class by a learned prototype (the classifier weight vector), which averages out intra-class outliers and enforces inter-class separation via an additive angular margin. Likely better-behaved under label noise and more suitable for an open-set-at-eval retrieval task with many classes.

---

## Next trials (ranked)

### Trial 1 — SupCon τ sweep on frozen backbone (H3)
- **Why first:** cheapest experiment (backbone frozen = fast), isolates one variable, and its outcome changes what we do next. Fixing loss hyperparameters *before* tuning backbone LR is the correct order of operations.
- **Setup:** Mode A (frozen), τ ∈ {0.05, 0.07, 0.10, 0.20}. Same PK sampler, same head, same LR.
- **Budget:** ~6k–9k iters each (we saw A saturates by then).
- **Success criterion:** at least one τ beats the current A baseline (mAP 0.370) by a meaningful margin; pick the best for subsequent runs. Track whether silhouette crosses zero.

### Trial 2 — Stronger Mode C (H1 + H2)
- **Setup:** `unfreeze_n_blocks=4`, raise `lr_multiplier` (start 0.3× → 0.5×), `unfreeze_after=2000`. Use the τ chosen in Trial 1.
- **Budget:** 15k–20k iters.
- **Success criterion:** visible kick at iter 2000, higher mAP than current C run (>0.40), silhouette approaches zero or positive.

### Trial 3 — Bigger PK batch (H4)
- **Setup:** P=32, K=4 (bs=128) or P=24, K=4 (bs=96), GPU memory permitting. Same τ from Trial 1, same mode (start frozen for isolation).
- **Budget:** 6k–9k iters.
- **Success criterion:** smoother training curves (lower variance), better mAP at equal step count.

### Trial 4 — Label quality sanity check (H5)
- **Not a training run** — diagnostic. Pick the 5 worst-performing projects at eval (lowest per-project mAP) and inspect their `clusters.json` / `clusters_fixed.json` visually in the UI viewer. If labels are noisy on "hard" projects, that's a ceiling none of the above will break through.
- **Outcome:** either confirms labels are clean (rules out H5), or motivates a reviewer pass / stricter filtering before more compute.

### Trial 5 — ArcFace (angular-margin classification loss) (H6)
- **Why:** our per-project labels are essentially class IDs over ~50K classes. ArcFace treats each class as a learned prototype (classifier weight vector on the hypersphere) and enforces an additive angular margin `m` between the anchor's embedding and its prototype. Benefits specific to our setup:
  - **Robust to intra-class noise** — a single mislabeled crop in a project only nudges that class prototype; it does not directly dominate gradient flow to an anchor (unlike SupCon's pair-based gradient).
  - **No PK sampling dependency** — ArcFace works with random sampling; it does not need K positives per class per batch. Larger effective class count per step.
  - **Geometric inter-class separation** — the margin directly enforces that each embedding sits at least `m` radians away from other class prototypes. A principled way to fix our negative silhouette.
  - **Widely validated on person ReID and face recognition** — typical defaults: `s=30`, `m=0.5`.
- **Setup:** add an ArcFace head (linear `[D, num_classes]`, no bias, weight-normalized). Loss = softmax cross-entropy on margin-adjusted logits. Start mode A (frozen backbone) to isolate the loss change vs SupCon mode A baseline. Embedding head is still the MLP → 128d L2-normed; classifier operates on that.
- **Variants worth noting:**
  - **SubCenter-ArcFace** (multiple sub-prototypes per class, keep the nearest) — explicitly designed for label noise. Strong candidate if plain ArcFace helps but we suspect noise.
  - **AdaFace** — adaptive margin based on feature quality; good when crop quality varies (ours does: occlusion, scale, blur).
  - **CosFace** — additive cosine margin instead of angular. Simpler, often similar performance.
- **Budget:** 6k–9k iters on frozen backbone.
- **Success criterion:** beats mode A SupCon baseline on mAP (>0.37) *and* pushes silhouette closer to zero or positive. If silhouette turns positive while SupCon never did, that's strong evidence loss choice is the bottleneck.
- **Risk / cost:** classifier weight matrix is `num_classes × D` — with 50K classes × 128d = 6.4M extra params, trivially affordable. Softmax over 50K classes per step is the main speed cost but still cheap on a single GPU.

---

## Decisions to lock in as we go

- [ ] Best τ (from Trial 1)
- [ ] Best `lr_multiplier` + `unfreeze_n_blocks` (from Trial 2)
- [ ] Best effective batch size (from Trial 3)
- [ ] Label quality status (from Trial 4)
- [ ] Best loss family: SupCon vs ArcFace (and variant, if ArcFace wins) (from Trial 5)

---

## Notes

- All runs use the V9/ckpt/15000 teacher backbone DCP checkpoint.
- W&B project: `person-reid-finetune`.
- Eval: Query/gallery protocol, silhouette stratified subsample capped at 8000.