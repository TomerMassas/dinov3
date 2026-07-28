# Body-part fragment classifier

Filter out "meaningless" body crops — a hand, a leg, a torso sliver — **before**
they reach ReID clustering, so clusters aren't polluted by crops that carry no
identity information.

*Last updated: 2026-07-28*

---

## Contents

1. [Status](#status)
2. [The problem, and why not just raise the detector threshold](#1-the-problem)
3. [Architecture](#2-architecture)
4. [Why the *pretrain* backbone and not the finetune](#3-why-the-pretrain-backbone)
5. [The transform problem](#4-the-transform-problem)
6. [Geometry features](#5-geometry-features)
7. [Labels and the round-2 trap](#6-labels-and-the-round-2-trap)
8. [Two thresholds](#7-two-thresholds)
9. [The three-run evaluation](#8-the-three-run-evaluation)
10. [Metrics reference](#9-metrics-reference)
11. [Reading the coefficients](#10-reading-the-coefficients)
12. [How to run](#11-how-to-run)
13. [The iterative labeling loop](#12-the-iterative-labeling-loop)
14. [Escalation ladder](#13-escalation-ladder)
15. [Files](#14-files)
16. [Gotchas](#15-gotchas)

---

## Status

**Code written and smoke-tested on synthetic data. Never yet run on real crops** —
blocked on 2026-07-27 by an NVIDIA driver mismatch on `developer-gpu4` (see
[Gotchas](#15-gotchas)).

Labeled galleries under `Wedding[1]`:

| gallery | positives | negatives | positive rate |
|---|---|---|---|
| `17601187` | 367 | ~3500 | ~9.5% |
| `18226778` | 170 | — | — |
| `21833423` | 8 | — | well under 1% |

`21833423` is far too thin to measure recall on (8 positives → 95% CI roughly
±35pp). It is kept as a leave-one-gallery-out fold but flagged `THIN` in the report.

All four scripts plus `predict.py` are written. `predict.py`'s baseline loader is
tolerant of several JSON shapes and reports which one it matched — confirm on the
first real run that it prints the expected shape.

---

## 1. The problem

Detections from YOLO11 include crops that are useless for ReID: a hand, a leg, a
back-of-head at 15 px, a motion blur. They pollute clustering.

### Why raising the detector confidence threshold does not work

YOLO's `conf` answers *"is this a person, and is the box tight"* — **not** *"is
this crop useful for telling people apart."* Those come apart in both directions:

- A crisp, well-localized background torso scores **high** and is useless.
- A partially occluded full body of a real guest scores **medium** and is valuable.

It is also not a calibrated probability (objectness/BCE output, dataset-dependent).
And weddings run HDBSCAN at `min_cluster_size=3` because identities are thin — a
global conf cut removes exactly the small/distant detections those thin identities
are made of.

**Resolution:** don't threshold on `conf`. Feed it to the classifier as one feature
among twelve.

---

## 2. Architecture

```
crop ──► [FROZEN V18 ViT-S/16] ──► 384-d CLS ──┐
                                                ├─► StandardScaler ─► LogisticRegression ─► p_fragment
detections.json ──► 12 geometry scalars ───────┘
```

**Trained:** only the logistic regression — 396 weights + a bias.
**Frozen:** everything else. The ViT is a fixed feature extractor.

Three reasons this is right, not just cheap:

1. **The features already exist.** `cluster_test_set.py:220-222` computes the 384-d
   CLS one line before the projection head. Dumping it costs zero extra compute.
2. **The data regime demands it.** ~545 hand-labeled positives is linear-probe
   territory. Finetuning a CNN/ViT on that overfits and wants 10× more labels.
3. **Iteration speed.** GPU embed runs once; every model experiment after that is
   CPU-local and instant.

### What logistic regression is doing here

Computes `z = w·x + b` (one weight per feature), squashes through a sigmoid to a
probability in [0,1]. Training maximizes the likelihood of the labels under an L2
penalty.

- **`C`** is the *inverse* regularization strength. Small `C` = heavy penalty =
  smaller weights = simpler model. Swept `0.001 → 1.0`, chosen by CV. With ~545
  positives against 396 features, the sklearn default `C=1.0` separates the training
  set perfectly and generalizes badly — expect a small `C` to win.
- **`class_weight="balanced"`** — without it, the ~8% positive class is ignored and
  the model learns "never a fragment".
- **`StandardScaler` first — mandatory.** Mixing 384 CLS dims (~0.05 scale) with
  `conf ∈ [0,1]` and `log_crop_px ∈ [2,8]`; unscaled, the big-magnitude features
  swamp everything.

**Why LR and not something stronger:** it emits a *calibrated probability* (without
which the two thresholds are impossible), its coefficients are *inspectable*, and at
this `n` more capacity overfits.

---

## 3. Why the pretrain backbone

`/data/AI/Tomer/dinov3/train_pictime/experiments_V2/V18/ckpt/19750`, `which=teacher`.
The **SSL pretrain** backbone, deliberately **not** V44/V51 finetune.

SupCon is trained to map a hand crop and a full-body crop *of the same person* to
the same place. It is explicitly optimized to **destroy crop-completeness** — the
exact signal this classifier needs. The 128-d projection output would be the worst
possible feature choice here.

`finetune_reid.load_backbone()` gives the pretrain backbone directly:
`prep_labeling_files.py:71` calls it and only *then* overwrites the state dict with
the finetune ckpt. We skip that second step.

Note: `load_backbone` is **DCP-only**. The raw LVD-142M `.pth` foundation would need
`foundation_loader.py::load_foundation_into_backbone()` instead — a small extra
branch if that ablation is ever wanted.

---

## 4. The transform problem

`reid_dataset.get_val_transform()` is `Resize(256) → CenterCrop(224)`. `Resize` with
an **int** scales the **short side**.

A 100×300 full-body crop → 256×768 → the center crop keeps roughly **the torso only**.
So the ViT sees a torso for *both* "full body" and "torso only" inputs, and the
aspect ratio is gone. **The standard transform destroys the exact distinction this
classifier exists to make.**

Three variants, all output 224×224, all cached in one image-decode pass:

| name | what it does | why |
|---|---|---|
| `letterbox` | pad to square with ImageNet mean, then resize | Keeps all content **and** keeps shape visible — a leg stays a thin sliver inside a padded square. Mean-coloured padding is ~zero after `Normalize`. **Expected winner.** |
| `warp` | `Resize((224,224))` straight to square | Keeps all content, loses shape visually; geometry features restore it numerically |
| `reid_val` | the existing finetune transform | Negative control. If it wins, the reasoning above is wrong and that's worth knowing |

---

## 5. Geometry features

Source: `detections.json` (bbox, `conf`, and every *other* bbox in the same image)
plus image pixel dimensions. Computed in `dataset.py::geometry_features`. No model,
no extra I/O.

| group | features | signal |
|---|---|---|
| **Shape & size** | `log_aspect`, `sqrt_rel_area`, `log_crop_px` | Hand ≈ small and square (`log_aspect ≈ 0`); leg = tall thin sliver (`>> 0`); full body ≈ 2–3. `log_crop_px` catches "20 px, unusable whatever it is" |
| **Detector** | `conf` | Informative as a feature, useless as a gate |
| **Truncation** | `touch_left/top/right/bottom` | A bbox pinned to a frame edge (within 0.01) usually means the body is cut off by the photo boundary. Four separate flags because *which* edge differs in meaning — bottom = legs cut, top = head cut. A hint, not a rule: a full-body portrait with feet at the bottom also touches |
| **Crowding** | `max_iou_sibling`, `log_n_dets`, `area_rank` | Overlapping boxes = occlusion or merged people. `area_rank` (0 = biggest in frame) separates "the subject" from "the 12th person in the background" |
| **Position** | `center_y` | Fragments skew toward frame edges |

### Two details that matter

**(a) Aspect and size are computed in PIXELS, not normalized units.** Normalized
space is stretched by the frame's own aspect ratio. Example — a 1200×800 photo with
a perfectly square 200×200 px bbox:

- `rel_w = 200/1200 = 0.167`, `rel_h = 200/800 = 0.25`
- naive normalized aspect = `0.25/0.167` = **1.5** → looks tall. Wrong.
- multiply back to pixels: `200/200 = 1.0`, `log(1.0) = 0` → correctly square.

**(b) Geometry is not redundant with the CLS vector.** Every transform outputs
224×224, so a 60×300 leg and a 300×300 torso both reach the ViT as identical-shaped
tensors. The backbone **cannot** recover "this was 5× taller than wide" from pixels.
Geometry hands it over exactly. (Letterbox partly restores it visually — which is
why it should beat warp — but the numeric feature is cleaner.)

---

## 6. Labels and the round-2 trap

Written by the labeling app per gallery:
`Wedding[1]/<proj_id>/bodyfilter_result.json`

```json
{
  "kept_keys":       ["<photoId>.jpg_<idx>", ...],   // label 1 = fragment -> discard
  "deleted_keys":    ["<photoId>.jpg_<idx>", ...],   // label 0 = real usable person crop
  "finished_batches": [0, 1, ...],
  "reviewer": "...", "saved_at": "..."
}
```

Face-bearing crops are filtered out **before** the app displays anything, so
`deleted_keys` are real people **with no visible face** — the genuinely hard
negatives, not the trivial "has a face" ones.

### The trap this format already avoids

> "the rest of the gallery can be used as negative labels"

True in round 1, **false from round 2 on.** Once the app only shows crops the
classifier flagged, "the rest of the gallery" contains **fragments the classifier
missed**. Labeling those as negatives trains the model to reproduce its own blind
spots — a self-reinforcing error loop baked into label generation, unfixable
downstream.

**`kept_keys ∪ deleted_keys` is exactly the set that was SHOWN.** Crops suppressed
by the classifier are in neither list. `dataset.py::load_gallery_labels` therefore
treats anything outside that union as **unlabeled** and drops it — a suppressed crop
structurally cannot reach training.

### Parsing gotcha

Keys are `<filename>_<detection_index>` and **filenames contain underscores**:

```python
"10035892690_rot3.jpg_1".rpartition("_")  → ("10035892690_rot3.jpg", "_", "1")  ✓
"10035892690_rot3.jpg_1".split("_")       → ["10035892690", "rot3.jpg", "1"]    ✗
```

A naive split fails **only on `_rot*` images** — silently dropping a systematic
subset rather than crashing.

### The negative pool — settled 2026-07-28

Once the classifier gates the display, **`baseline − kept_keys` is no longer a valid
negative set.** It contains two populations: crops shown-and-deleted (real reviewed
negatives) and crops the classifier *suppressed* (never shown). The second group is
the model's own predictions fed back as ground truth.

**Decision: negatives = `deleted_keys` only.**

The counter-argument was considered and rejected: at a 0.99-recall threshold the
suppressed pool should be ~99.9% genuinely negative, so the noise *rate* is low. But
you can't know recall is 0.99 on an unseen gallery (cross-gallery transfer is the
unproven part), the contamination sits precisely on the hardest examples, and you
already have ~6700 negatives against 545 positives — **negatives are not the
bottleneck.** Zero upside, structural downside.

Recommended app format — three buckets, preserving the "every baseline crop is
accounted for" invariant:

```
kept_keys        displayed, kept      -> label 1  (fragment)
deleted_keys     displayed, deleted   -> label 0  (real person)
suppressed_keys  never displayed      -> UNLABELED, excluded from training

kept ∪ deleted ∪ suppressed == baseline
```

Also worth stamping `model_tag` + `threshold` into the result file, so if a model
turns out to have been bad you can quarantine the rounds its blind spots shaped.

### Why negatives don't dry up (Tomer's question, 2026-07-28)

As the classifier improves, the displayed pool trends toward pure positives, so
`deleted_keys` shrinks — negatives shift from *bulk* to *boundary only*.

Partly this is good: a round-2 `deleted_key` is a crop the classifier **wrongly
flagged**, i.e. a hard negative on the decision boundary. That's hard-negative
mining — 20 boundary negatives can beat 2000 obvious ones. And old negatives
accumulate; they don't disappear.

But two real problems:
1. **Negative diversity freezes.** Positives broaden across venues while negatives
   stay pinned to the first few galleries — exactly the run-2 failure mode.
2. **Calibration drifts.** LR's intercept is prior-sensitive. If the training prior
   swings from 9% positives toward 50%, `p_fragment` stops meaning what it meant, and
   the thresholds move out from under you.

**Fix: a fixed random quota in the display** — `RANDOM_QUOTA = 300` (3 UI batches of
100), sampled from the *suppressed* pool. One mechanism, four payoffs: unbiased
negatives from every new venue, false-negative detection, an honest prior estimate,
and a prior-faithful slice to recalibrate on.

The quota stays constant while the threshold-selected set shrinks each round, so
labeling naturally shifts from bulk discovery to sampling and validation — the right
trajectory. Random crops are mostly obvious negatives you can bulk-delete fast, so
300 costs far less than 300 boundary cases.

Sampling from the **suppressed** pool rather than the whole baseline avoids duplicates
and composes cleanly: above-threshold is a *census* (100% reviewed), suppressed is
*sampled*, so

```
estimated missed fragments = (fragments kept among audit_keys) / sampling_fraction
```

**Cheap insurance not yet implemented:** have the app record *why* each crop was
displayed (`"scored"` vs `"random"`). Costs nothing now and preserves the option of
importance weighting later. Same logic as recording `reviewed` — you can't decide it
retroactively.

### Integrity checks (in `load_gallery_labels`)

- `kept ∩ deleted` non-empty → **raises** (app bug)
- keys that don't resolve to a `detections.json` entry → counted and reported, never
  silently absorbed
- `finished_batches` is recorded but **not used for labeling** — `kept ∪ deleted` is
  the authoritative record of what was actually decided, whatever the batching did

---

## 7. Two thresholds

Same model, same scores, **two different cut points** for opposite jobs.

| | **Labeling** threshold | **Deploy** threshold |
|---|---|---|
| Gates | which crops the app *shows* you next round | which crops are *dropped* before clustering |
| False positive costs | seconds of review | **a good crop destroyed** |
| False negative costs | **a fragment never shown → permanent wrong label** | one junk crop left in; HDBSCAN mostly absorbs it |
| Optimize for | **recall** (`LABELING_TARGET_RECALL = 0.99`) | **precision** (`DEPLOY_TARGET_PRECISION = 0.95`) |
| Threshold value | **low** | **high** |
| FPR is | the price you pay | the thing you're avoiding |

Round-1 risk posture: set the labeling threshold **deliberately too low**. Recall
can't be trusted yet, and **round 1's job is to get safely to 4 galleries, not to
maximize speedup.** The speedup compounds once the estimate firms up.

---

## 8. The three-run evaluation

**Run 1 — SELECTION.** `StratifiedKFold` *within* one gallery (`CV_GALLERY`, auto =
most positives = `17601187`). Optimistic in absolute terms — one venue, one set of
people, one lighting setup — but **every variant shares that optimism**, so it
validly *ranks* transform × feature-set × `C`. Ranked by **PR-AUC**.

**Run 2 — GENERALIZATION.** Leave-one-gallery-out across all labeled galleries with
the selected config. Reported per gallery with **ROC-AUC** (prior-invariant, so
comparable across galleries whose positive rates differ by 10×+). Recall carries a
**Wilson CI**. Thin folds are **flagged, never dropped**.

**Run 3 — SHIP.** Refit on every labeled gallery. Never evaluated — its only job is
maximum data. This is the `.pkl` that `predict.py` loads.

Thresholds come from **run 1's** out-of-fold probabilities (the statistically solid
estimate) and are then *reported* on each run-2 fold.

If kNN wins run 1, the shipped model is still the **best LR** — kNN has no calibrated
probability to threshold and no coefficients to inspect. The script says so explicitly.

---

## 9. Metrics reference

### The curves

The model scores every crop in [0,1]. A threshold turns scores into decisions.
**The curves are "what happens at every possible threshold, plotted."** sklearn uses
every unique predicted score as a threshold (not a coarse grid), and returns the
thresholds alongside the curve — which is how `threshold_at_recall` /
`threshold_at_precision` pick the operating points.

*(Gotcha: `precision_recall_curve` returns one fewer threshold than points, hence the
`rec[:-1]` / `prec[:-1]` slicing in the code.)*

```
ROC  =  plot (FPR, Recall)          PR  =  plot (Recall, Precision)

 1.0┤      ┌──────────           1.0┤──────────┐
 R  │    ┌─┘                      P │           └──┐
 e  │  ┌─┘                        r │              └───┐
 c  │ ┌┘        ·  ·  ← random    e │                  └──┐  ← the cliff
 a  │┌┘   ·  ·                    c │                     └──
 l  ├┘ ·                            │· · · · · · · · · · · ·  ← random = positive rate
 0.0└──────────────              0.0└────────────────────────
    0.0    FPR    1.0               0.0      Recall       1.0
      want TOP-LEFT                       want TOP-RIGHT
```

### Definitions

| metric | definition | note |
|---|---|---|
| **Recall** | TP / all real fragments | drives the labeling threshold |
| **Precision** | TP / everything flagged | denominator = *my predictions* |
| **FPR** | FP / all real people | denominator = *the truth*. This is the confusion point |
| **ROC-AUC** | area under (FPR, Recall) | P(random fragment scores above random real person). **Prior-invariant** |
| **PR-AUC** | area under (Recall, Precision) | **Prior-dependent** — floor equals the positive rate |
| **Wilson CI** | CI for a proportion | 7/8 recall has a 95% interval of ~[0.53, 0.98] |

Deliberately absent: **accuracy**. At an 8% positive rate, "never a fragment" scores
92% and is worthless.

### Precision vs FPR, worked

`21833423`: 8 fragments, ~1200 real people. Model flags 20 crops — 6 true, 14 false.

- Recall = 6/8 = **0.75**
- Precision = 6/20 = **0.30**
- FPR = 14/1200 = **0.012**

### Why ROC is prior-invariant and PR is not

At some threshold: TP=300, FP=100 against 3500 negatives.
→ Precision = 300/400 = **0.75**, FPR = 100/3500 = **0.029**

Same model on a gallery with 10× more negatives, so FP ≈ 1000:
→ Precision = 300/1300 = **0.23** (collapsed), FPR = 1000/35000 = **0.029** (unchanged)

ROC's axes are both rates *within* a class, so scaling negatives cancels. Precision
mixes classes in its denominator, so it tracks the prior.

**Hence:** run 1 (one gallery, fixed prior) → **PR-AUC**, which is also far more
sensitive to the minority class. ROC-AUC can read 0.95+ while precision is dreadful:
with 3500 negatives a mere 5% FPR is 175 false positives drowning 367 true ones.
Run 2 (across galleries) → **ROC-AUC**, the only comparable one.

### What good looks like

| ROC-AUC | reading | | PR-AUC (9.5% prior) | reading |
|---|---|---|---|---|
| 0.50 | random | | 0.095 | random |
| 0.70 | weak | | 0.30 | ok |
| 0.85 | good **cross-gallery** for round 1 | | 0.50 | good |
| 0.95 | strong | | 0.70+ | strong |
| 0.99+ | suspicious — check leakage | | | |

### The two numbers that actually decide the workflow

The AUCs *rank models*. These decide what happens:

**Precision at recall ≥ 0.99** = your labeling speedup. `train.py` prints it as
*"shows X% of the pool"*.

| precision @ 0.99 recall | crops to review (of 3867) | speedup |
|---|---|---|
| 0.15 | ~2450 (63%) | 1.6× |
| 0.30 | ~1220 (32%) | 3× |
| 0.60 | ~610 (16%) | 6× |

**Recall at precision ≥ 0.95** = deployment value: how much junk you remove while
destroying almost no good crops.

### Diagnostic patterns

| symptom | meaning |
|---|---|
| ROC great, PR poor | Expected — imbalance punishing precision. Not a bug |
| Run 1 ≈ 0.95, Run 2 ≈ 0.70 | **Learned the venue, not body parts.** The key check |
| PR falls immediately from the left | Top-scored crops aren't fragments → suspect labels or features |
| `geom` alone ≈ `cls+geom` | Backbone contributes nothing; bbox shape does all the work. Would simplify deployment enormously |
| Everything near baseline | No signal — check the ablation before adding data |

---

## 10. Reading the coefficients

`report.md` prints named geometry weights. Because `StandardScaler` runs first:

> a coefficient `w` means: **if this feature rises by one standard deviation, the
> log-odds of "fragment" changes by `w`**

- **Sign** — positive pushes toward *fragment*, negative toward *real person*.
- **Magnitude** — comparable across features *because* they're standardized. That's
  a large part of why the scaler is there.
- **Near zero** — the model found little use for it **given the other features**. Not
  the same as "uninformative": with correlated features the model can put all weight
  on one and ~0 on the other.

Two caveats: strong L2 shrinks *all* coefficients, so compare **within one model**,
never across different `C`. And log-odds aren't linear in probability — `+0.5` moves
the needle far more near p=0.5 than near p=0.99.

---

## 11. How to run

```bash
cd /data/AI/Tomer/dinov3

# 1. GPU (VM only) — embed each gallery's baseline pool, all 3 transforms, ONE decode pass.
#    Skip-if-cached: a gallery is embedded once, ever.
python3 -m train_pictime.classifier_body_parts.embed

# 2. CPU (seconds) — join caches with labels, sweep, pick thresholds, ship the model
python3 -m train_pictime.classifier_body_parts.train

# 3. CPU (seconds) — score every project, write the UI's decision file
python3 -m train_pictime.classifier_body_parts.predict
```

### The embedding cache — why steps 2 and 3 need no GPU

Nothing upstream of the logistic regression ever changes. The backbone is frozen, so
for a fixed (crop, transform) the 384-d CLS is deterministic; geometry comes from
`detections.json` + image dims. **Only the LR and its two thresholds change per round.**

So `embed.py` writes, once per gallery, into the project dir:

```
<project>/classifier_embeddings_v18_letterbox.npz     key[], cls[N,384], geom[N,G] + fingerprint
<project>/classifier_embeddings_v18_warp.npz
<project>/classifier_embeddings_v18_reid_val.npz
```

covering that gallery's **whole baseline pool** — not just labeled crops. Then:

- `train.py` loads the caches and joins them against `kept_keys`/`deleted_keys` on crop key
- `predict.py` loads the caches and applies the LR

Both are CPU-only. Every round after the first touches **zero images and zero GPU**, so
re-scoring the whole dataset after a retrain takes seconds. The `classifier_` prefix
keeps these clearly apart from the `embeddings_v51.npz` ReID files in the same dirs.

**Fingerprint validation.** Each cache stores a JSON signature of everything it depends
on — `backbone_tag`, `pretrain_ckpt`, `backbone_which`, `transform`, `crop_size`,
`geometry_names`, and a **content hash** of `detections.json` — verified on every load.
A mismatch raises with both signatures printed; `predict.py` skips such galleries and
tells you to re-run `embed.py`. Without this, a stale cache would produce wrong scores
with no error anywhere.

The detections signature is a **content hash, not size+mtime**: mtime changes on rsync,
backup restore, or a move to another machine without any bbox changing, and that would
needlessly invalidate every cache in the dataset.

`EMBED_FORCE = True` rebuilds caches by hand; normally unnecessary since staleness is
detected automatically.

No config edits needed — galleries auto-discover from `bodyfilter_result.json`.

**One model file, always the same path.** `train.py` writes `model_v18.pkl` and
overwrites it every run, so `predict.py` always picks up the newest model with no
config change. The winning transform and feature set aren't in the filename — they're
inside the bundle, printed by `predict.py` at startup, and recorded in `report.md` and
every scores file. `MODEL_PATH` pins an archived model if you ever need one.

`PREDICT_FORCE = True` re-scores everything after a retrain (leave it on); `False`
scores only projects never scored before.

### What `predict.py` writes, per project

`<project>/classifier_scores_v18.json`:

| field | meaning |
|---|---|
| `model` | full provenance — file, ckpt, transform, feature set, `C`, galleries trained on |
| `labeling_threshold` / `deploy_threshold` | both gates, from the bundle |
| `show_keys` | `p_fragment >= labeling_threshold`, **minus already-reviewed crops** |
| `audit_keys` | 300 crops sampled from the unreviewed **suppressed** pool — display these too |
| `scores` | **every** scored key → `p_fragment`, reviewed ones included |
| `counts` / `audit` | baseline / scored / already_reviewed / above / suppressed (pre- and post-cut) / quota / `sampling_fraction` |

**The app should display `show_keys ∪ audit_keys`** and sort each into kept/deleted as
today — audit crops included, so an audited crop you delete becomes an ordinary
reviewed negative. Everything else stays unlabeled — see §6.

### Never show the same crop twice (`EXCLUDE_REVIEWED = True`)

`kept_keys | deleted_keys` are subtracted from **both** display lists, so a labeled
gallery can be re-scored with a newer model without re-showing anything you've already
judged. A fully-reviewed gallery yields empty display lists and says so.

They are still **scored** and still appear in `scores` — deliberately. That's what
makes real precision/recall per gallery computable from this file alone, no re-run
needed: intersect `scores` with `kept_keys`/`deleted_keys` and you have ground truth
next to predictions.

`counts` reports both sides of the cut: `above_threshold` (before) and
`above_threshold_new` (what's actually shown). The audit `sampling_fraction`
denominator is the **unreviewed** suppressed pool, so the missed-fragment estimator
measures what's still outstanding.

### `PREDICT_FORCE`

`True` re-scores every project and overwrites each JSON — what you want after a
retrain. `False` scores only projects never scored before.

The overwrite discards the previous round's file. Provenance lives inside each file so
the current one always self-documents, but per-round history is lost; archive app-side
if you'll ever want to audit which model shaped which round.

Note the audit sample is seeded per gallery, so re-running with the **same** model
redraws the same crops. A **new** model reshuffles it regardless — the suppressed pool
itself changed — which is correct: a new model has new blind spots to audit.

### Outputs

```
train_pictime/classifier_body_parts/results/v18/
├── features_v18_letterbox.npz              # caches (embed.py)
├── features_v18_warp.npz
├── features_v18_reid_val.npz
├── model_v18.pkl                   # model + both thresholds; OVERWRITTEN every train run
├── report.md                               # read this
└── curves.png
```

Nothing is written into gallery dirs — only `predict.py` will do that.

### Sanity-check `embed` before it touches the GPU

It prints a per-gallery table first. Check **positive counts match** (367/170/8),
**`unresolved` is 0** (anything else = dataset/app drift), and **dropped crops ≈ 0**.
Everything downstream inherits these.

Runtime: embed ~5 min for ~7K crops × 3 transforms. Train is seconds.

---

## 12. The iterative labeling loop

```
label a gallery → train → pre-filter the next gallery at the LABELING threshold
   → review far fewer crops → retrain → repeat
```

### Don't label what the model is confident about

Confident predictions are the ones it already gets right. Adding them reinforces
existing bias and inflates validation while real performance stalls.

**Per round, take ~100–200 crops:**

- **~70% highest uncertainty** (`|p − threshold|` smallest) — these move the boundary
- **~20% diversity** — spread across embedding space, so you don't label 100
  near-duplicate hands from one wedding
- **~10% pure random** — the *only* unbiased slice; without it you lose all knowledge
  of the true class prior and real-world precision

Typically reaches the same accuracy as random labeling with 3–5× fewer labels.

### Two measurements that keep you honest

1. **A frozen random test set.** ~200 crops randomly sampled across galleries,
   reviewed exhaustively, **never pre-filtered, never trained on.** The only place
   true recall at a given labeling threshold can be measured, and the only place with
   the true class prior — which shifts every round as the reviewed pool gets richer
   in fragments.
2. **Suppressed-pool audit.** Each round, review ~50 random crops from *below* the
   labeling threshold. That's the live false-negative rate, and every fragment found
   there is a known blind spot — the highest-value label available.

### The gallery is the unit of data, not the crop

Every crop in a gallery shares venue, lighting, photographer and *the same people*.
With one gallery, effective n for generalization is ~1, and you cannot distinguish
"learned what a leg looks like" from "learned this venue's carpet". **Breadth beats
depth** at this stage — 3–4 galleries labeled shallowly beats one labeled deeply.

---

## 13. Escalation ladder

Gated on positive count, not on impatience:

| positives | option |
|---|---|
| ~545 (now) | **LR linear probe** — correct capacity |
| ~1–2K | **MLP head** (396 → 256 → 1, dropout). "Meaningless" is a *union* of failure types — limb / blur / non-person / back-of-head — which one hyperplane may not separate |
| ~1K+ | **LightGBM on the geometry block** (± reduced CLS). Trees handle geometry interactions natively: "small AND square AND low conf" |
| ~5K+ | **Unfreeze the last 2–4 ViT blocks.** Mode-C machinery already exists in `finetune_reid.py` |
| anytime | **Pose keypoints** (YOLO11-pose). Visible-keypoint presence is nearly a direct readout of which body part this is. Costs a second detector pass |
| ~500/class | **Multi-class** instead of binary — needs the app to record *which kind* of fragment |

Cheap by construction: the feature cache is model-agnostic, so trying LightGBM is
one more entry in `run_selection`'s candidate list, not a rewrite.

**On taxonomy:** even while training binary, it's worth *storing* fine-grained labels
(`full_body` / `upper_body` / `head_only` / `limb_fragment` / `no_person` /
`heavy_occlusion` / `too_small`). Same labeling effort, but it lets the keep/drop
boundary move later without relabeling, and tells you *which* junk type you miss.

---

## 14. Files

| file | role |
|---|---|
| `config.py` | Single source of truth. Paths, backbone, transforms, feature sets, `C_GRID`, both threshold targets, output naming |
| `dataset.py` | Label + baseline loading with integrity checks, geometry features, the three transforms, `GalleryImageDataset`, and the fingerprinted per-gallery embedding cache (read + write) |
| `embed.py` | **GPU — the only GPU step.** Embeds each gallery's baseline pool once → per-project caches. Skip-if-cached |
| `train.py` | **CPU.** Joins caches with labels, three runs, threshold picking, model pickle, `report.md` + `curves.png` |
| `predict.py` | **CPU, seconds.** Reads caches → `classifier_scores_v18.json` with `show_keys` + `audit_keys` + provenance. Run after each train phase |

Conventions followed from `model_comparison/` and `realworld_eval/`: hardcoded VM
paths, no argparse, `python3 -m ...` invocation, atomic writes (`.tmp` + `os.replace`),
and model-tied filenames built from an explicit `{model_id: filename}` dict rather
than a ternary.

### Design details worth not re-deriving

- **One item = one IMAGE**, not one crop. Wedding JPEGs are large and several
  detections share a file, so decoding once per image instead of once per crop per
  transform is minutes vs tens of minutes.
- **Invalid-crop placeholder is sized from the active transform**, not hardcoded
  `(3,224,224)`. This is the collate bug hit on 2026-06-30; raw wedding detections
  would trigger it.
- **`embed.py` raises immediately without CUDA** — `load_backbone` hardcodes
  `.to_empty(device="cuda")`, so a CPU run would die with an opaque FSDP error a
  minute in.

- **The embedding cache is the architecture, not an optimization.** `embed.py` is the
  only script that reads images or touches a GPU. If you find yourself adding image I/O
  to `train.py` or `predict.py`, something has gone wrong.

### Smoke tests run 2026-07-27/28

Three suites, all green. Nothing has run on real crops yet.

**End-to-end** (fake dataset root with per-gallery caches, real `train.main()` +
`predict.main()`): the train/predict join, fingerprint staleness detection, and the
reviewed-crop exclusion. Confirms `train.py` ignores never-labeled galleries and
unlabeled baseline crops while `predict.py` scores them, that `predict.py` skips
galleries without a valid cache, and that thresholds/provenance flow through the bundle
into the scores files.

**`dataset.py`** (fabricated gallery): `_rot*` key parsing, kept/deleted overlap
raising, out-of-range key counting, baseline shape tolerance, sibling-aware geometry,
IoU against the analytic value (0.780822, exact), all three transforms, letterbox
padding confirmed (edge 0.006 vs centre 1.366 post-normalize), and that the detections
signature is content-based — identical content rewritten with a new mtime keeps the
same signature.

**`predict.py`** (unit): all four baseline JSON shapes, `load_bundle` ambiguity
guardrail, `show_keys`/`audit_keys` disjointness, count reconciliation on both sides of
the reviewed cut, per-gallery audit determinism, fully-reviewed and tiny-pool edges.

**Threshold picking hit its targets** on synthetic data: 0.9918 recall against a 0.99
target, 0.9508 precision against 0.95. Thin-fold contrast behaved correctly —
`[0.95, 0.99]` on 367 positives vs `[0.68, 1.00]` on 8.

---

## 15. Gotchas

**GPU driver mismatch (hit 2026-07-27).** `nvmlInit_v2() failed: Driver/library
version mismatch` — kernel module `580.159.03` vs userspace `580.173.02`. A driver
package upgraded without a reboot. Affects **every** GPU script on the VM, not just
this one.

```bash
nvidia-smi                        # confirms
cat /proc/driver/nvidia/version   # kernel module version
dpkg -l | grep -i nvidia-driver   # userspace version — these won't match

# fix, if nothing holds the GPU (sudo fuser -v /dev/nvidia*):
sudo systemctl stop nvidia-persistenced
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia
sudo systemctl start nvidia-persistenced
# otherwise: sudo reboot  (shared box — coordinate)
```

Note `torch.cuda.is_available()` returned **True** despite NVML being dead, so the
preflight check passed and the failure surfaced later as a NCCL stack trace. An
option was offered to strengthen the check (exercise `device_count()` + a real
allocation) — **not yet implemented**, pending a decision.

**Domain-specific HDBSCAN, for context.** `NEW_CLUSTER` (mcs=10,
`allow_single_cluster=True`) is Portraits-tuned. Weddings need **mcs=3 +
`allow_single_cluster=False`** — the Portraits params produce all-noise on weddings.
Relevant because the deploy threshold should ultimately be validated against wedding
clustering quality, not Portraits.

---

## Open questions

1. **Is `21833423`'s tiny positive count real** (different photographer, fewer crowd
   shots) or a labeling-standard difference? If the latter, its LOGO fold measures
   label drift rather than model generalization.
2. **Strengthen the CUDA preflight?** (option offered, not decided)
3. **App: record `suppressed_keys` and a per-crop display reason** (`"scored"` /
   `"random"`) — both are cheap now and impossible retroactively. See §6.
4. **Confirm `predict.py`'s baseline shape detection** on the first real run — it
   prints which JSON shape it matched.
