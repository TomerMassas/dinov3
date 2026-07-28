"""Train the body-part fragment classifier off the cached features and pick both
operating thresholds. CPU-only — iterate here freely, embed.py is the GPU step.

Three runs, each answering a different question:

  1. SELECTION   StratifiedKFold within one gallery (config.CV_GALLERY).
                 Optimistic in absolute terms — same venue, same people, same
                 lighting — but every variant shares that optimism, so it validly
                 RANKS transform x feature-set x C. Ranked by PR-AUC: the prior is
                 fixed within a gallery, and PR-AUC is far more sensitive to the
                 minority class than ROC-AUC.

  2. GENERALIZATION   Leave-one-gallery-out with the selected config. Reported per
                 gallery with ROC-AUC (prior-invariant, so it is comparable across
                 galleries whose positive rates differ ~40x). Recall on galleries
                 with few positives carries a Wilson CI, because a raw 7/8 reads far
                 more precisely than it is. Thin folds are FLAGGED, never dropped.

  3. SHIP        Refit on every labeled gallery and save the bundle predict.py loads.
                 Never evaluated — its only job is maximum data.

Thresholds come from run 1's out-of-fold probabilities (the statistically solid
estimate) and are then reported on each run-2 fold.

    python3 -m train_pictime.classifier_body_parts.train
"""
from __future__ import annotations

import math
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_pictime.classifier_body_parts import config as C
from train_pictime.classifier_body_parts import dataset


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k/n — the honest width on the thin galleries.

    A recall of 7/8 has a 95% interval of roughly [0.53, 0.98]; reporting it as
    0.875 invites setting a labeling threshold off pure noise.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_labeled_features(transform: str) -> dict:
    """Join the per-gallery embedding caches with the labeling app's decisions.

    The caches hold every baseline crop; labels come from kept_keys (1) and
    deleted_keys (0). Only crops that appear in BOTH are returned — a labeled key
    missing from the cache is counted and reported, never silently dropped.
    """
    cls_parts, geom_parts, labels, galleries, keys = [], [], [], [], []
    missing_total = 0

    for gallery_id in dataset.discover_galleries():
        cache = dataset.load_gallery_cache(gallery_id, transform)
        index = {k: i for i, k in enumerate(cache["key"].tolist())}
        kept, deleted, _summary = dataset.load_label_keys(gallery_id)

        rows = [(k, 1) for k in sorted(kept)] + [(k, 0) for k in sorted(deleted)]
        hit = [(index[k], k, y) for k, y in rows if k in index]
        missing = len(rows) - len(hit)
        if missing:
            print(f"  {gallery_id}: {missing} labeled keys absent from the embedding cache "
                  f"(unresolved or unreadable at embed time)")
            missing_total += missing

        idxs = np.array([i for i, _k, _y in hit], dtype=np.int64)
        cls_parts.append(cache["cls"][idxs])
        geom_parts.append(cache["geom"][idxs])
        labels.extend(y for _i, _k, y in hit)
        keys.extend(k for _i, k, _y in hit)
        galleries.extend([gallery_id] * len(hit))

    if not labels:
        raise RuntimeError(f"No labeled crops found. Check that {C.DATASET_ROOT} has galleries "
                           f"with both {C.LABELS_FILENAME} and a valid embedding cache.")
    if missing_total:
        print(f"  ({missing_total} labeled keys skipped in total)")

    return {"cls": np.concatenate(cls_parts, axis=0),
            "geom": np.concatenate(geom_parts, axis=0),
            "label": np.array(labels, dtype=np.int8),
            "gallery_id": np.array(galleries, dtype=object),
            "key": np.array(keys, dtype=object),
            "geometry_names": np.array(dataset.GEOMETRY_NAMES, dtype=object),
           }


def build_X(cache, feature_set: str) -> np.ndarray:
    if feature_set == "cls":
        return np.asarray(cache["cls"])
    if feature_set == "geom":
        return np.asarray(cache["geom"])
    if feature_set == "cls+geom":
        return np.concatenate([np.asarray(cache["cls"]), np.asarray(cache["geom"])], axis=1)
    raise ValueError(f"Unknown feature set {feature_set!r} (expected one of {C.FEATURE_SETS})")


def make_lr(c_value: float) -> Pipeline:
    lr = LogisticRegression(C=c_value,
                            class_weight=C.CLASS_WEIGHT,
                            max_iter=C.MAX_ITER,
                            random_state=C.SEED,
                           )
    return Pipeline([("scale", StandardScaler()), ("lr", lr)])


def make_knn() -> Pipeline:
    knn = KNeighborsClassifier(n_neighbors=C.KNN_K, metric="cosine", weights="distance")
    return Pipeline([("scale", StandardScaler()), ("knn", knn)])


def oof_probs(model_factory, X: np.ndarray, y: np.ndarray, n_splits: int) -> np.ndarray:
    """Out-of-fold P(fragment) for every row, via StratifiedKFold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=C.SEED)
    probs = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in skf.split(X, y):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        probs[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return probs


def threshold_at_recall(y: np.ndarray, probs: np.ndarray, target_recall: float) -> float:
    """Highest threshold that still reaches target_recall — the labeling gate."""
    _prec, rec, thr = precision_recall_curve(y, probs)
    ok = np.flatnonzero(rec[:-1] >= target_recall)
    return float(thr[ok[-1]]) if len(ok) else 0.0


def threshold_at_precision(y: np.ndarray, probs: np.ndarray, target_precision: float) -> float:
    """Lowest threshold from which precision STAYS at target_precision — the deploy gate.

    Not simply the first threshold that clears the target: precision is noisy at
    the low end and a single lucky spike there would hand back a threshold that
    does not hold. Taking the point after the last violation is monotone-safe.
    """
    prec, _rec, thr = precision_recall_curve(y, probs)
    violations = np.flatnonzero(prec[:-1] < target_precision)
    idx = int(violations[-1]) + 1 if len(violations) else 0
    return float(thr[idx]) if idx < len(thr) else 1.0


def rates_at(y: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    """Recall / FPR / precision at one threshold, with counts kept for the CIs."""
    pred = probs >= threshold
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    return {"n_pos": n_pos,
            "n_neg": n_neg,
            "tp": tp,
            "fp": fp,
            "recall": tp / n_pos if n_pos else float("nan"),
            "fpr": fp / n_neg if n_neg else float("nan"),
            "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
           }


# ---------------------------------------------------------------------------
# Run 1 — selection
# ---------------------------------------------------------------------------

def run_selection(caches: dict, cv_mask: np.ndarray, y_cv: np.ndarray) -> list[dict]:
    """Sweep transform x feature_set x model within one gallery. Returns all rows."""
    results: list[dict] = []
    n_splits = min(C.CV_FOLDS, int(y_cv.sum()), int((y_cv == 0).sum()))
    if n_splits < 2:
        raise RuntimeError(f"CV gallery has too few of one class for {C.CV_FOLDS}-fold CV "
                           f"({int(y_cv.sum())} pos / {int((y_cv == 0).sum())} neg)")
    if n_splits < C.CV_FOLDS:
        print(f"NOTE: reduced to {n_splits}-fold CV (limited by the minority class)")

    for transform, cache in caches.items():
        for feature_set in C.FEATURE_SETS:
            X = build_X(cache, feature_set)[cv_mask]
            candidates = [(f"lr(C={c:g})", (lambda c=c: make_lr(c))) for c in C.C_GRID]
            candidates.append((f"knn(k={C.KNN_K})", make_knn))
            for name, factory in candidates:
                probs = oof_probs(factory, X, y_cv, n_splits)
                results.append({"transform": transform,
                                "feature_set": feature_set,
                                "model": name,
                                "pr_auc": float(average_precision_score(y_cv, probs)),
                                "roc_auc": float(roc_auc_score(y_cv, probs)),
                                "probs": probs,
                               })
                print(f"  {transform:>9} | {feature_set:>8} | {name:>12} | "
                      f"PR-AUC {results[-1]['pr_auc']:.4f} | ROC-AUC {results[-1]['roc_auc']:.4f}")
    return results


# ---------------------------------------------------------------------------
# Run 2 — leave-one-gallery-out
# ---------------------------------------------------------------------------

def run_logo(X: np.ndarray,
             y: np.ndarray,
             galleries: np.ndarray,
             c_value: float,
             labeling_threshold: float,
            ) -> list[dict]:
    """Train on all galleries but one, score the held-out one. Thin folds flagged."""
    folds: list[dict] = []
    for gallery in sorted(set(galleries.tolist())):
        test_mask = galleries == gallery
        train_mask = ~test_mask
        y_tr, y_te = y[train_mask], y[test_mask]

        fold = {"gallery": gallery,
                "n_pos": int((y_te == 1).sum()),
                "n_neg": int((y_te == 0).sum()),
                "thin": bool((y_te == 1).sum() < C.MIN_POSITIVES_PER_FOLD),
                "is_test_gallery": bool(C.TEST_GALLERY is not None and gallery == C.TEST_GALLERY),
               }
        if len(set(y_tr.tolist())) < 2 or len(set(y_te.tolist())) < 2:
            fold["skipped"] = "held-out or training split has only one class"
            folds.append(fold)
            continue

        model = make_lr(c_value)
        model.fit(X[train_mask], y_tr)
        probs = model.predict_proba(X[test_mask])[:, 1]

        rates = rates_at(y_te, probs, labeling_threshold)
        fold.update(rates)
        fold["roc_auc"] = float(roc_auc_score(y_te, probs))
        fold["pr_auc"] = float(average_precision_score(y_te, probs))
        fold["recall_ci"] = wilson_ci(rates["tp"], rates["n_pos"])
        fold["fpr_ci"] = wilson_ci(rates["fp"], rates["n_neg"])
        folds.append(fold)
    return folds


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def plot_curves(y: np.ndarray, probs: np.ndarray, t_label: float, t_deploy: float, path: Path) -> None:
    prec, rec, _thr = precision_recall_curve(y, probs)
    fpr, tpr, _ = roc_curve(y, probs)
    r_label = rates_at(y, probs, t_label)
    r_deploy = rates_at(y, probs, t_deploy)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(rec, prec, color="#2b6cb0")
    axes[0].scatter([r_label["recall"]], [r_label["precision"]], color="#c53030", zorder=3,
                    label=f"labeling thr {t_label:.3f}")
    axes[0].scatter([r_deploy["recall"]], [r_deploy["precision"]], color="#2f855a", zorder=3,
                    label=f"deploy thr {t_deploy:.3f}")
    axes[0].set_xlabel("recall")
    axes[0].set_ylabel("precision")
    axes[0].set_title("Precision-Recall (out-of-fold, selection gallery)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="lower left")

    axes[1].plot(fpr, tpr, color="#2b6cb0")
    axes[1].plot([0, 1], [0, 1], "--", color="#a0aec0", linewidth=1)
    axes[1].set_xlabel("false positive rate")
    axes[1].set_ylabel("true positive rate")
    axes[1].set_title("ROC (out-of-fold, selection gallery)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def geometry_weights(model: Pipeline, feature_set: str, geometry_names: list[str]) -> list[tuple[str, float]]:
    """Named LR coefficients for the geometry block (standardized, so comparable)."""
    if "geom" not in feature_set:
        return []
    coef = model.named_steps["lr"].coef_[0]
    geom_coef = coef[-len(geometry_names):]
    pairs = list(zip(geometry_names, (float(v) for v in geom_coef)))
    return sorted(pairs, key=lambda kv: abs(kv[1]), reverse=True)


def write_report(path: Path,
                 cache_meta: dict,
                 selection: list[dict],
                 best: dict,
                 thresholds: dict,
                 folds: list[dict],
                 geom_weights: list[tuple[str, float]],
                 ship_meta: dict,
                ) -> None:
    lines: list[str] = []
    lines.append("# Body-part fragment classifier\n")
    lines.append(f"- Backbone: `{C.BACKBONE_TAG}` (SSL pretrain, pre-finetune) — "
                 f"`{C.PRETRAIN_CKPT}`, which=`{C.BACKBONE_WHICH}`")
    lines.append(f"- Dataset: `{C.DATASET_ROOT}`")
    lines.append("- Positive class: crop is ONLY body parts (`kept_keys`) -> discard before clustering")
    lines.append(f"- Negative class: real, usable no-face person crop (`deleted_keys`)\n")

    lines.append("## Labeled data\n")
    lines.append("| gallery | positives | negatives | positive rate |")
    lines.append("|---|---|---|---|")
    for g, (npos, nneg) in cache_meta["per_gallery"].items():
        rate = npos / (npos + nneg) if (npos + nneg) else 0.0
        lines.append(f"| `{g}` | {npos} | {nneg} | {rate:.4f} |")
    lines.append("")

    lines.append(f"## Run 1 — selection (StratifiedKFold within `{cache_meta['cv_gallery']}`)\n")
    lines.append("Optimistic in absolute terms (one gallery = one venue, one set of people). "
                 "Valid for *ranking* variants. Ranked by PR-AUC — the prior is fixed within a "
                 "gallery and PR-AUC is the more sensitive metric on the minority class.\n")
    lines.append("| transform | features | model | PR-AUC | ROC-AUC |")
    lines.append("|---|---|---|---|---|")
    for r in sorted(selection, key=lambda r: r["pr_auc"], reverse=True):
        mark = " **<-**" if r is best else ""
        lines.append(f"| {r['transform']} | {r['feature_set']} | `{r['model']}` | "
                     f"{r['pr_auc']:.4f} | {r['roc_auc']:.4f}{mark} |")
    lines.append("")
    lines.append(f"**Selected:** `{best['transform']}` / `{best['feature_set']}` / `{best['model']}`\n")

    lines.append("## Thresholds\n")
    lines.append("Two gates, opposite objectives — do not collapse them into one number.\n")
    lines.append("| gate | objective | threshold | recall | precision | FPR |")
    lines.append("|---|---|---|---|---|---|")
    for gate, objective in (("labeling", f"recall >= {C.LABELING_TARGET_RECALL}"),
                            ("deploy", f"precision >= {C.DEPLOY_TARGET_PRECISION}")):
        t = thresholds[gate]
        r = thresholds[f"{gate}_rates"]
        lines.append(f"| {gate} | {objective} | {t:.4f} | {r['recall']:.4f} | "
                     f"{r['precision']:.4f} | {r['fpr']:.4f} |")
    lines.append("")
    lines.append(f"At the labeling gate the app would show "
                 f"{thresholds['labeling_rates']['tp'] + thresholds['labeling_rates']['fp']} of "
                 f"{len(thresholds['y'])} crops in the selection gallery "
                 f"({thresholds['labeling_keep_rate']:.1%} of the pool).\n")

    lines.append("## Run 2 — leave-one-gallery-out\n")
    lines.append("Recall and FPR are measured at the **labeling** threshold. FPR is tightly "
                 "estimated wherever negatives are plentiful; recall is only as good as the "
                 "positive count, hence the Wilson intervals.\n")
    lines.append("| gallery | pos | neg | ROC-AUC | recall @ labeling thr | FPR @ labeling thr | note |")
    lines.append("|---|---|---|---|---|---|---|")
    for f in folds:
        if "skipped" in f:
            lines.append(f"| `{f['gallery']}` | {f['n_pos']} | {f['n_neg']} | — | — | — | "
                         f"skipped: {f['skipped']} |")
            continue
        note = []
        if f["thin"]:
            note.append(f"THIN (<{C.MIN_POSITIVES_PER_FOLD} pos) — read as directional only")
        if f["is_test_gallery"]:
            note.append("designated test gallery")
        rlo, rhi = f["recall_ci"]
        flo, fhi = f["fpr_ci"]
        lines.append(f"| `{f['gallery']}` | {f['n_pos']} | {f['n_neg']} | {f['roc_auc']:.4f} | "
                     f"{f['recall']:.3f} [{rlo:.2f}, {rhi:.2f}] | "
                     f"{f['fpr']:.4f} [{flo:.4f}, {fhi:.4f}] | {'; '.join(note)} |")
    lines.append("")

    if geom_weights:
        lines.append("## Geometry weights (shipped model, standardized features)\n")
        lines.append("| feature | coefficient |")
        lines.append("|---|---|")
        for name, w in geom_weights:
            lines.append(f"| `{name}` | {w:+.4f} |")
        lines.append("")

    lines.append("## Run 3 — shipped model\n")
    lines.append(f"- File: `{ship_meta['path']}`")
    lines.append(f"- Trained on {ship_meta['n_train']} crops across "
                 f"{ship_meta['n_galleries']} galleries ({ship_meta['n_pos']} positives)")
    lines.append("- Not evaluated by design — run 1 selects, run 2 estimates, run 3 ships.\n")

    lines.append("![curves](curves.png)\n")

    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    transform_names = C.TRANSFORMS if C.RUN_ALL_TRANSFORMS else (C.TRANSFORM,)
    print(f"Joining per-gallery caches with labels ({', '.join(transform_names)})...")
    caches = {name: load_labeled_features(name) for name in transform_names}

    ref = caches[transform_names[0]]
    y = np.asarray(ref["label"]).astype(int)
    galleries = np.asarray(ref["gallery_id"]).astype(str)
    geometry_names = [str(v) for v in ref["geometry_names"]]

    per_gallery = {g: (int(((galleries == g) & (y == 1)).sum()),
                       int(((galleries == g) & (y == 0)).sum()))
                   for g in sorted(set(galleries.tolist()))}
    print(f"Loaded {len(y)} crops | {int(y.sum())} positives | {len(per_gallery)} galleries")
    for g, (npos, nneg) in per_gallery.items():
        print(f"  {g:>16}  pos {npos:>6}  neg {nneg:>6}")

    cv_gallery = C.CV_GALLERY or max(per_gallery, key=lambda g: per_gallery[g][0])
    if cv_gallery not in per_gallery:
        raise ValueError(f"CV_GALLERY={cv_gallery!r} is not among the labeled galleries "
                         f"{sorted(per_gallery)}")
    cv_mask = galleries == cv_gallery
    y_cv = y[cv_mask]

    print(f"\n=== Run 1: selection — StratifiedKFold within {cv_gallery} "
          f"({int(y_cv.sum())} pos / {int((y_cv == 0).sum())} neg) ===")
    selection = run_selection(caches, cv_mask, y_cv)
    overall_best = max(selection, key=lambda r: r["pr_auc"])
    print(f"\nSelection winner: {overall_best['transform']} / {overall_best['feature_set']} / "
          f"{overall_best['model']} (PR-AUC {overall_best['pr_auc']:.4f})")

    # Thresholds, LOGO and the shipped model all use the best LOGISTIC REGRESSION
    # row, not necessarily the overall winner. If kNN wins, that is a signal to
    # collect more galleries rather than a model to ship — it gives no calibrated
    # probability to threshold on and no coefficients to inspect.
    best = max((r for r in selection if r["model"].startswith("lr(")), key=lambda r: r["pr_auc"])
    if best is not overall_best:
        print(f"NOTE: {overall_best['model']} beat every LR variant. Read that as the classes "
              f"being locally clustered — more galleries will help more than more capacity. "
              f"Shipping {best['transform']} / {best['feature_set']} / {best['model']} "
              f"(PR-AUC {best['pr_auc']:.4f}) so the model stays calibrated and inspectable.")

    best_c = float(best["model"].split("C=")[1].rstrip(")"))
    X_best = build_X(caches[best["transform"]], best["feature_set"])

    t_label = threshold_at_recall(y_cv, best["probs"], C.LABELING_TARGET_RECALL)
    t_deploy = threshold_at_precision(y_cv, best["probs"], C.DEPLOY_TARGET_PRECISION)
    r_label = rates_at(y_cv, best["probs"], t_label)
    r_deploy = rates_at(y_cv, best["probs"], t_deploy)
    keep_rate = (r_label["tp"] + r_label["fp"]) / len(y_cv)
    thresholds = {"labeling": t_label,
                  "deploy": t_deploy,
                  "labeling_rates": r_label,
                  "deploy_rates": r_deploy,
                  "labeling_keep_rate": keep_rate,
                  "y": y_cv,
                 }
    print(f"Labeling threshold {t_label:.4f} -> recall {r_label['recall']:.4f}, "
          f"shows {keep_rate:.1%} of the pool")
    print(f"Deploy   threshold {t_deploy:.4f} -> precision {r_deploy['precision']:.4f}, "
          f"recall {r_deploy['recall']:.4f}")

    print(f"\n=== Run 2: leave-one-gallery-out ({len(per_gallery)} folds) ===")
    if len(per_gallery) < 2:
        print("Only one labeled gallery — skipped. Cross-gallery generalization is unmeasurable "
              "until a second gallery is labeled.")
        folds = []
    else:
        folds = run_logo(X_best, y, galleries, best_c, t_label)
        for f in folds:
            if "skipped" in f:
                print(f"  {f['gallery']:>16}  SKIPPED ({f['skipped']})")
                continue
            rlo, rhi = f["recall_ci"]
            flag = "  [THIN]" if f["thin"] else ""
            print(f"  {f['gallery']:>16}  pos {f['n_pos']:>5}  ROC-AUC {f['roc_auc']:.4f}  "
                  f"recall {f['recall']:.3f} [{rlo:.2f},{rhi:.2f}]  FPR {f['fpr']:.4f}{flag}")

    print(f"\n=== Run 3: ship (refit on all {len(y)} crops) ===")
    ship_model = make_lr(best_c)
    ship_model.fit(X_best, y)
    model_file = C.model_path()
    bundle = {"model": ship_model,
              "transform": best["transform"],
              "feature_set": best["feature_set"],
              "C": best_c,
              "labeling_threshold": t_label,
              "deploy_threshold": t_deploy,
              "geometry_names": geometry_names,
              "backbone_tag": C.BACKBONE_TAG,
              "pretrain_ckpt": str(C.PRETRAIN_CKPT),
              "backbone_which": C.BACKBONE_WHICH,
              "cls_dim": C.CLS_DIM,
              "crop_size": C.CROP_SIZE,
              "trained_on_galleries": sorted(per_gallery),
              "n_train": int(len(y)),
              "n_positives": int(y.sum()),
             }
    tmp = str(model_file) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(bundle, f)
    os.replace(tmp, model_file)
    print(f"Saved {model_file}")

    plot_curves(y_cv, best["probs"], t_label, t_deploy, C.CURVES_FILE)
    write_report(C.REPORT_FILE,
                 {"per_gallery": per_gallery, "cv_gallery": cv_gallery},
                 selection,
                 best,
                 thresholds,
                 folds,
                 geometry_weights(ship_model, best["feature_set"], geometry_names),
                 {"path": str(model_file),
                  "n_train": int(len(y)),
                  "n_galleries": len(per_gallery),
                  "n_pos": int(y.sum()),
                 },
                )
    print(f"Saved {C.REPORT_FILE}")
    print(f"Saved {C.CURVES_FILE}")


if __name__ == "__main__":
    main()
