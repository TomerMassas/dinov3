"""Train the body-part fragment classifier on the APPROVED galleries.

    python3 -m train_pictime.classifier_body_parts.train

Which galleries train the model is decided by one thing: `approved` in
config.COMPLETION_LOG, the labeling app's completion registry. Nothing else. Labels
then come from each gallery's bodyfilter_result.json (kept_keys -> 1, deleted_keys -> 0)
and features from its embedding cache, so this whole script is CPU-only and takes
seconds — embed.py did the GPU work once, per gallery, forever.

Leave-one-gallery-out does everything:

    for each C, every gallery is predicted by a model that never saw it
    -> pick C by pooled PR-AUC on those held-out predictions
    -> pick both thresholds by walking that same pooled set once
    -> refit on all the data and ship

One loop, and every number reported is cross-gallery. There is no within-gallery CV
and no single-gallery calibration anywhere, which is what used to let one positive-heavy
gallery set thresholds for the whole dataset.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_pictime.classifier_body_parts import config as C
from train_pictime.classifier_body_parts import dataset
from train_pictime.classifier_body_parts.dataset import build_X


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def approved_galleries() -> tuple[list[str], list[str]]:
    """(approved, skipped) project ids from the completion log.

    `approved is True`, not a truthy test: the string "false" is truthy. Repeated
    project ids keep the last entry, which is all the ordering care this needs — the
    app rewrites a gallery's entry in place when it is re-reviewed.
    """
    with open(C.COMPLETION_LOG) as f:
        entries = json.load(f)

    verdict: dict[str, bool] = {}
    for e in entries:
        verdict[str(e["project_id"])] = e.get("approved") is True
    approved = sorted(g for g, ok in verdict.items() if ok)
    skipped = sorted(g for g, ok in verdict.items() if not ok)
    return approved, skipped


def load_features(galleries: list[str]) -> dict:
    """Join each gallery's embedding cache with the labeling app's decisions.

    Only crops present in BOTH the cache and the labels are used; a labeled key with
    no cached embedding is counted and reported rather than silently dropped.
    """
    cls_parts, geom_parts, labels, gallery_of, keys = [], [], [], [], []
    per_gallery: dict[str, dict] = {}

    for gallery_id in galleries:
        cache = dataset.load_gallery_cache(gallery_id, C.TRANSFORM)
        index = {k: i for i, k in enumerate(cache["key"].tolist())}
        kept, deleted, _summary = dataset.load_label_keys(gallery_id)

        rows = [(k, 1) for k in sorted(kept)] + [(k, 0) for k in sorted(deleted)]
        hit = [(index[k], k, y) for k, y in rows if k in index]
        missing = len(rows) - len(hit)
        if missing:
            print(f"  {gallery_id}: {missing} labeled keys have no cached embedding "
                  f"(unresolved or unreadable at embed time)")

        n_pos = sum(y for _i, _k, y in hit)
        per_gallery[gallery_id] = {"n_pos": n_pos,
                                   "n_neg": len(hit) - n_pos,
                                   "n_cached": len(index),
                                   "n_missing": missing,
                                   "pos_rate": n_pos / max(len(hit), 1),
                                  }

        idxs = np.array([i for i, _k, _y in hit], dtype=np.int64)
        cls_parts.append(cache["cls"][idxs])
        geom_parts.append(cache["geom"][idxs])
        labels.extend(y for _i, _k, y in hit)
        keys.extend(k for _i, k, _y in hit)
        gallery_of.extend([gallery_id] * len(hit))

    if not labels:
        raise RuntimeError(f"No labeled crops found across {len(galleries)} approved "
                           f"galleries. Check that each has {C.LABELS_FILENAME} and that "
                           f"embed.py has been run.")

    return {"cls": np.concatenate(cls_parts, axis=0),
            "geom": np.concatenate(geom_parts, axis=0),
            "label": np.array(labels, dtype=np.int8),
            "gallery_id": np.array(gallery_of, dtype=object),
            "key": np.array(keys, dtype=object),
            "per_gallery": per_gallery,
            "geometry_names": list(dataset.GEOMETRY_NAMES),
           }


# ---------------------------------------------------------------------------
# Model, held-out predictions, thresholds
# ---------------------------------------------------------------------------

def make_lr(c_value: float) -> Pipeline:
    lr = LogisticRegression(C=c_value,
                            class_weight=C.CLASS_WEIGHT,
                            max_iter=C.MAX_ITER,
                            random_state=C.SEED,
                           )
    return Pipeline([("scale", StandardScaler()), ("lr", lr)])


def logo_probs(X: np.ndarray, y: np.ndarray, galleries: np.ndarray, c_value: float) -> np.ndarray:
    """P(fragment) for every crop, from a model that never saw that crop's gallery.

    Returned aligned to the input rows, so pooled metrics are the whole array and
    per-gallery metrics are a mask — no bookkeeping needed to keep them in step.
    """
    probs = np.zeros(len(y), dtype=np.float64)
    for gallery in np.unique(galleries):
        held_out = galleries == gallery
        model = make_lr(c_value)
        model.fit(X[~held_out], y[~held_out])
        probs[held_out] = model.predict_proba(X[held_out])[:, 1]
    return probs


def threshold_at_recall(y: np.ndarray, probs: np.ndarray, target_recall: float) -> float:
    """Highest threshold that still reaches target_recall — the labeling gate."""
    _prec, rec, thr = precision_recall_curve(y, probs)
    ok = np.flatnonzero(rec[:-1] >= target_recall)
    return float(thr[ok[-1]]) if len(ok) else 0.0


def threshold_at_precision(y: np.ndarray, probs: np.ndarray, target_precision: float) -> float:
    """Lowest threshold from which precision STAYS at target_precision — the deploy gate.

    Not the first threshold that clears the target: precision is noisy at the low end,
    and one lucky spike there would hand back a threshold that does not hold. Taking
    the point after the last violation is monotone-safe.
    """
    prec, _rec, thr = precision_recall_curve(y, probs)
    violations = np.flatnonzero(prec[:-1] < target_precision)
    idx = int(violations[-1]) + 1 if len(violations) else 0
    return float(thr[idx]) if idx < len(thr) else 1.0


def rates_at(y: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    """Recall / FPR / precision at one threshold."""
    pred = probs >= threshold
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    return {"n_pos": n_pos,
            "n_neg": n_neg,
            "n_shown": tp + fp,
            "recall": tp / n_pos if n_pos else float("nan"),
            "fpr": fp / n_neg if n_neg else float("nan"),
            "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
           }


def geometry_weights(model: Pipeline, geometry_names: list[str]) -> list[tuple[str, float]]:
    """Named LR coefficients for the geometry block (standardized, so comparable)."""
    if "geom" not in C.FEATURE_SET:
        return []
    geom_coef = model.named_steps["lr"].coef_[0][-len(geometry_names):]
    pairs = [(name, float(w)) for name, w in zip(geometry_names, geom_coef)]
    return sorted(pairs, key=lambda kv: abs(kv[1]), reverse=True)


# ---------------------------------------------------------------------------
# Output
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
    axes[0].set_title("Precision-Recall (leave-one-gallery-out)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="lower left")

    axes[1].plot(fpr, tpr, color="#2b6cb0")
    axes[1].plot([0, 1], [0, 1], "--", color="#a0aec0", linewidth=1)
    axes[1].set_xlabel("false positive rate")
    axes[1].set_ylabel("true positive rate")
    axes[1].set_title("ROC (leave-one-gallery-out)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_report(path: Path, meta: dict) -> None:
    L: list[str] = ["# Body-part fragment classifier", ""]
    L.append(f"*Generated {datetime.now().isoformat(timespec='seconds')} — if this looks "
             f"stale, check you re-downloaded it from the VM.*")
    L.append("")
    L.append(f"- Backbone: `{C.BACKBONE_TAG}` ({C.BACKBONE_SOURCE}), "
             f"which=`{C.BACKBONE_WHICH}`")
    L.append(f"- Checkpoint: `{C.backbone_ckpt()}`")
    L.append(f"- Dataset: `{C.DATASET_ROOT}`")
    L.append(f"- Features: `{C.FEATURE_SET}` on the `{C.TRANSFORM}` transform")
    L.append("- Positive class: crop is ONLY body parts (`kept_keys`) -> discard before clustering")
    L.append("- Negative class: real, usable no-face person crop (`deleted_keys`)")
    L.append("")

    L.append("## Galleries")
    L.append("")
    L.append(f"Trained on every gallery with `approved: true` in "
             f"`{C.COMPLETION_LOG.name}` — that is the only condition.")
    L.append("")
    L.append("| gallery | positives | negatives | positive rate | cached |")
    L.append("|---|---|---|---|---|")
    for g, s in meta["per_gallery"].items():
        L.append(f"| `{g}` | {s['n_pos']} | {s['n_neg']} | {s['pos_rate']:.4f} | "
                 f"{s['n_cached']} |")
    L.append("")
    if meta["skipped"]:
        L.append(f"Not approved, so not used: {', '.join('`' + g + '`' for g in meta['skipped'])}")
        L.append("")

    L.append("## C sweep — leave-one-gallery-out")
    L.append("")
    L.append("Every gallery scored by a model that never saw it, then pooled. `C` is picked "
             "by PR-AUC on those held-out predictions, so the choice is cross-gallery rather "
             "than a within-gallery fit.")
    L.append("")
    L.append("| C | PR-AUC | ROC-AUC |")
    L.append("|---|---|---|")
    for row in meta["sweep"]:
        mark = " **<-**" if row["c"] == meta["best_c"] else ""
        L.append(f"| {row['c']:g} | {row['pr_auc']:.4f} | {row['roc_auc']:.4f}{mark} |")
    L.append("")

    L.append("## Thresholds")
    L.append("")
    L.append(f"Both picked by walking the winning `C`'s held-out predictions once — "
             f"**{meta['n_pool']} crops** at a **{meta['pool_pos_rate']:.4f}** positive rate. "
             f"Two gates, opposite objectives; do not collapse them into one number.")
    L.append("")
    L.append("| gate | objective | threshold | recall | precision | FPR |")
    L.append("|---|---|---|---|---|---|")
    for gate, objective in (("labeling", f"recall >= {C.LABELING_TARGET_RECALL}"),
                            ("deploy", f"precision >= {C.DEPLOY_TARGET_PRECISION}")):
        r = meta[f"{gate}_rates"]
        L.append(f"| {gate} | {objective} | {meta[gate]:.4f} | {r['recall']:.4f} | "
                 f"{r['precision']:.4f} | {r['fpr']:.4f} |")
    L.append("")
    L.append(f"At the labeling gate the app would show {meta['labeling_rates']['n_shown']} of "
             f"{meta['n_pool']} crops ({meta['labeling_keep_rate']:.1%} of the pool).")
    L.append("")

    L.append("## Per gallery, held out")
    L.append("")
    L.append("Recall and FPR at the pooled labeling threshold, so individual galleries land "
             "above and below the target — expected; the target holds on the pooled set.")
    L.append("")
    L.append("| gallery | pos | neg | ROC-AUC | recall | FPR |")
    L.append("|---|---|---|---|---|---|")
    for row in meta["folds"]:
        auc = "—" if row["roc_auc"] is None else f"{row['roc_auc']:.4f}"
        L.append(f"| `{row['gallery']}` | {row['n_pos']} | {row['n_neg']} | {auc} | "
                 f"{row['recall']:.3f} | {row['fpr']:.4f} |")
    L.append("")

    if meta["geom_weights"]:
        L.append("## Geometry weights (shipped model, standardized)")
        L.append("")
        L.append("| feature | coefficient |")
        L.append("|---|---|")
        for name, w in meta["geom_weights"]:
            L.append(f"| `{name}` | {w:+.4f} |")
        L.append("")

    L.append("## Shipped model")
    L.append("")
    L.append(f"- File: `{meta['model_path']}`")
    L.append(f"- Refit on all {meta['n_pool']} crops from {len(meta['per_gallery'])} galleries "
             f"({int(meta['n_pos_total'])} positives)")
    L.append("- Not evaluated separately by design: the sweep and the thresholds above are the "
             "held-out estimates.")
    L.append("")
    L.append("![curves](curves.png)")
    L.append("")

    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    approved, skipped = approved_galleries()
    print(f"Completion log: {C.COMPLETION_LOG}")
    print(f"Approved: {len(approved)}  |  not approved: {len(skipped)}")
    if not approved:
        raise RuntimeError(f"No gallery in {C.COMPLETION_LOG} has approved: true.")
    if len(approved) < 2:
        raise RuntimeError(f"Only one approved gallery ({approved[0]}). Leave-one-gallery-out "
                           f"needs at least two — approve another before training.")

    data = load_features(approved)
    y = np.asarray(data["label"]).astype(int)
    galleries = np.asarray(data["gallery_id"]).astype(str)
    X = build_X(data, C.FEATURE_SET)

    print(f"\nLoaded {len(y)} crops | {int(y.sum())} positives "
          f"({y.mean():.4f}) | {len(data['per_gallery'])} galleries")
    print(f"{'gallery':>16} {'pos':>7} {'neg':>7} {'pos_rate':>9} {'cached':>8}")
    for g, s in data["per_gallery"].items():
        print(f"{g:>16} {s['n_pos']:>7} {s['n_neg']:>7} {s['pos_rate']:>9.4f} "
              f"{s['n_cached']:>8}")
    if skipped:
        print(f"\nNot approved, skipped: {', '.join(skipped)}")

    print(f"\n=== C sweep, leave-one-gallery-out ({len(approved)} folds each) ===")
    sweep, oof = [], {}
    for c_value in C.C_GRID:
        probs = logo_probs(X, y, galleries, c_value)
        oof[c_value] = probs
        row = {"c": c_value,
               "pr_auc": float(average_precision_score(y, probs)),
               "roc_auc": float(roc_auc_score(y, probs)),
              }
        sweep.append(row)
        print(f"  C={c_value:<7g} PR-AUC {row['pr_auc']:.4f}  ROC-AUC {row['roc_auc']:.4f}")

    best = max(sweep, key=lambda r: r["pr_auc"])
    best_c = best["c"]
    probs = oof[best_c]
    print(f"\nSelected C={best_c:g} (PR-AUC {best['pr_auc']:.4f})")

    t_label = threshold_at_recall(y, probs, C.LABELING_TARGET_RECALL)
    t_deploy = threshold_at_precision(y, probs, C.DEPLOY_TARGET_PRECISION)
    r_label = rates_at(y, probs, t_label)
    r_deploy = rates_at(y, probs, t_deploy)
    keep_rate = r_label["n_shown"] / len(y)
    print(f"Labeling threshold {t_label:.4f} -> recall {r_label['recall']:.4f}, "
          f"precision {r_label['precision']:.4f}, shows {keep_rate:.1%} of the pool")
    print(f"Deploy   threshold {t_deploy:.4f} -> precision {r_deploy['precision']:.4f}, "
          f"recall {r_deploy['recall']:.4f}")

    folds = []
    print(f"\nPer gallery, held out, at the labeling threshold:")
    for gallery in sorted(set(galleries.tolist())):
        mask = galleries == gallery
        y_g, p_g = y[mask], probs[mask]
        row = {"gallery": gallery, **rates_at(y_g, p_g, t_label)}
        # ROC-AUC is undefined on a single-class gallery — report it as absent rather
        # than crashing a run that is otherwise fine.
        row["roc_auc"] = float(roc_auc_score(y_g, p_g)) if len(set(y_g.tolist())) > 1 else None
        folds.append(row)
        auc = "     —" if row["roc_auc"] is None else f"{row['roc_auc']:.4f}"
        print(f"  {gallery:>16}  pos {row['n_pos']:>5}  ROC-AUC {auc}  "
              f"recall {row['recall']:.3f}  FPR {row['fpr']:.4f}")

    print(f"\n=== Refit on all {len(y)} crops and ship ===")
    model = make_lr(best_c)
    model.fit(X, y)
    model_file = C.model_path()
    bundle = {"model": model,
              "transform": C.TRANSFORM,
              "feature_set": C.FEATURE_SET,
              "C": best_c,
              "labeling_threshold": t_label,
              "deploy_threshold": t_deploy,
              "geometry_names": data["geometry_names"],
              "backbone_tag": C.BACKBONE_TAG,
              "backbone_source": C.BACKBONE_SOURCE,
              "backbone_ckpt": C.backbone_ckpt(),
              "backbone_which": C.BACKBONE_WHICH,
              "cls_dim": C.CLS_DIM,
              "crop_size": C.CROP_SIZE,
              # Re-checked by predict.py against the live config. Swap the backbone ckpt,
              # re-embed and forget to retrain, and the old model still fits the new
              # 384-d vectors — nothing errors, every score is silently wrong.
              "feature_signature": dataset.feature_signature(C.TRANSFORM),
              "trained_on_galleries": sorted(data["per_gallery"]),
              "n_train": int(len(y)),
              "n_positives": int(y.sum()),
             }
    tmp = str(model_file) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(bundle, f)
    os.replace(tmp, model_file)
    print(f"Saved {model_file}")

    plot_curves(y, probs, t_label, t_deploy, C.CURVES_FILE)
    write_report(C.REPORT_FILE,
                 {"per_gallery": data["per_gallery"],
                  "skipped": skipped,
                  "sweep": sweep,
                  "best_c": best_c,
                  "labeling": t_label,
                  "deploy": t_deploy,
                  "labeling_rates": r_label,
                  "deploy_rates": r_deploy,
                  "labeling_keep_rate": keep_rate,
                  "n_pool": int(len(y)),
                  "n_pos_total": int(y.sum()),
                  "pool_pos_rate": float(y.mean()),
                  "folds": folds,
                  "geom_weights": geometry_weights(model, data["geometry_names"]),
                  "model_path": str(model_file),
                 },
                )
    print(f"Saved {C.REPORT_FILE}")
    print(f"Saved {C.CURVES_FILE}")


if __name__ == "__main__":
    main()
