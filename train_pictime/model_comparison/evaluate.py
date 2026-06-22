"""Load both embedding caches, compute the comparison, write report + plots.

    python3 -m train_pictime.model_comparison.evaluate

A — embedding quality (pooled across the test set, global identity labels):
    silhouette (cosine), mAP, Rank-1/5.
B — clustering quality (each model's production-faithful clustering, per project
    vs GT, averaged): mean fracturing, %perfect, homogeneity/completeness/
    V-measure/ARI, pairwise precision/recall/F1.

Old clusters with Agglomerative(0.85); new with HDBSCAN. Not apples-to-apples by
design — it reflects production-old vs tuned-new.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from train_pictime.model_comparison import config as C
from train_pictime.model_comparison import metrics as Me
from train_pictime.model_comparison.clustering import cluster


def _load(path: Path):
    d = np.load(path, allow_pickle=True)
    keys = [(str(p), str(f), int(b)) for p, f, b in
            zip(d["project_ids"], d["filenames"], d["bbox_indices"])]
    return {
        "emb": d["embeddings"],
        "project_ids": np.array([str(p) for p in d["project_ids"]], dtype=object),
        "gt": d["gt_cluster_ids"].astype(int),
        "key_to_row": {k: i for i, k in enumerate(keys)},
    }


def _align(old, new):
    """Intersect the two caches on (project, filename, bbox_index) so both models
    are scored on exactly the same crops, in the same order."""
    common = [k for k in old["key_to_row"] if k in new["key_to_row"]]
    common.sort()
    oi = [old["key_to_row"][k] for k in common]
    ni = [new["key_to_row"][k] for k in common]
    project_ids = old["project_ids"][oi]
    gt = old["gt"][oi]
    assert np.array_equal(gt, new["gt"][ni]), "GT mismatch between caches for aligned keys"
    return old["emb"][oi], new["emb"][ni], project_ids, gt


def _global_labels(project_ids, gt):
    m, out = {}, np.empty(len(gt), dtype=np.int64)
    for i in range(len(gt)):
        k = (str(project_ids[i]), int(gt[i]))
        out[i] = m.setdefault(k, len(m))
    return out


def embedding_quality(emb, global_labels):
    return {
        "silhouette": Me.silhouette_cosine(emb, global_labels, C.SILHOUETTE_MAX_SAMPLES, C.SEED),
        **Me.query_gallery_map(emb, global_labels, C.SEED),
    }


def clustering_quality(emb, project_ids, gt, spec):
    per_proj = []
    for pid in np.unique(project_ids):
        mask = project_ids == pid
        if mask.sum() < 2:
            continue
        pred = cluster(emb[mask], spec)
        m = Me.clustering_metrics(gt[mask], pred)   # None if model assigned no crop
        if m is not None:
            per_proj.append(m)
    keys = ["fracturing_mean", "fracturing_perfect_frac", "homogeneity", "completeness",
            "v_measure", "ari", "cluster_precision", "cluster_recall", "cluster_f1",
            "n_gt_clusters", "n_pred_clusters", "cluster_count_delta"]
    if not per_proj:
        return {k: float("nan") for k in keys} | {"n_projects": 0}
    return {k: float(np.mean([p[k] for p in per_proj])) for k in keys} | {"n_projects": len(per_proj)}


# What to show (label, results-key) per group. results.json keeps the FULL set;
# these are just the displayed/plotted metrics.
DISPLAY = {
    "A": [("silhouette (cosine, higher=better)", "silhouette")],
    "B": [("mean fracturing (lower=better)", "fracturing_mean"),
          ("% perfectly grouped (higher=better)", "fracturing_perfect_frac"),
          ("cluster precision (higher=better)", "cluster_precision"),
          ("cluster recall (higher=better)", "cluster_recall"),
          ("completeness (higher=better)", "completeness"),
          ("homogeneity (higher=better)", "homogeneity"),
          ("ARI (higher=better)", "ari"),
          ("mean cluster-count Δ, pred−true (0=ideal)", "cluster_count_delta")],
}
# Shown in the table but excluded from the [0,1] bar plot — different scale
# (mean fracturing ~1-3; count delta is a signed count) would distort the [0,1] bars.
PLOT_EXCLUDE = {"cluster_count_delta", "fracturing_mean"}
GROUP_TITLE = {"A": "Embedding quality", "B": "Clustering quality"}

# One-line description per displayed metric, for the legend at the bottom of the table.
METRIC_DESC = {
    "silhouette": "mean silhouette of the true identities in the (cosine) embedding space; higher = identities form tighter, better-separated groups.",
    "mAP": "mean average precision of query→gallery retrieval by true identity; higher = same-person crops rank above everyone else.",
    "fracturing_mean": "avg number of predicted clusters a single true identity is split across (1.0 = never split); lower is better.",
    "fracturing_perfect_frac": "fraction of true identities whose crops all land in exactly one predicted cluster; higher is better.",
    "cluster_precision": "cluster purity — avg fraction of a predicted cluster that belongs to its majority identity; low = clusters mix different people.",
    "cluster_recall": "cluster coverage — avg fraction of a cluster's dominant identity captured by that cluster; low = identities split across clusters.",
    "completeness": "all crops of an identity fall in a single cluster; higher is better.",
    "homogeneity": "each predicted cluster contains crops from only one true identity; high = pure clusters (penalizes MERGING different people — the counterweight to completeness/recall).",
    "ari": "Adjusted Rand Index — chance-corrected pair agreement between predicted clusters and true identities; penalizes both merging and splitting; ~0 = random, 1 = perfect.",
    "cluster_count_delta": "mean (predicted clusters − true identities) per project; 0 = right count on average, negative = under-clusters (merges people), positive = over-clusters (fractures).",
}


def write_report(results: dict, n_crops: int, n_ids: int):
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (C.OUTPUT_DIR / "results.json").write_text(json.dumps(results, indent=2))

    col_old, col_new = "OLD (ResNet)", "NEW (ViT-S16)"
    lines = [f"# OLD ResNet vs NEW ViT-S/16 — {n_crops} crops, {n_ids} identities", ""]
    for grp in ("A", "B"):
        lines.append(f"## {GROUP_TITLE[grp]}")
        lines.append("")
        lines.append(f"| metric | {col_old} | {col_new} |")
        lines.append("|:---|:---:|:---:|")   # metric left, both model columns centered
        for label, key in DISPLAY[grp]:
            ov, nv = results[grp]["old"][key], results[grp]["new"][key]
            lines.append(f"| {label} | {ov:.4f} | {nv:.4f} |")
        lines.append("")

    # Legend — describe each displayed metric.
    lines.append("---")
    lines.append("**Legend**")
    lines.append("")
    for grp in ("A", "B"):
        for label, key in DISPLAY[grp]:
            name = label.split(" (")[0]   # strip the "(higher=better)" hint
            lines.append(f"- **{name}** — {METRIC_DESC[key]}")

    md = "\n".join(lines)
    (C.OUTPUT_DIR / "comparison.md").write_text(md + "\n")
    print("\n" + md + "\n")

    # Bar plots — one per group, only the displayed metrics.
    for grp, fname in [("A", "plot_embedding_quality.png"), ("B", "plot_clustering_quality.png")]:
        # Non-[0,1] metrics (cluster counts, mean fracturing) are table-only.
        items = [(lbl, k) for lbl, k in DISPLAY[grp] if k not in PLOT_EXCLUDE]
        labels = [lbl for lbl, _ in items]
        keys = [k for _, k in items]
        x = np.arange(len(keys))
        fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.6), 5))
        ax.bar(x - 0.2, [results[grp]["old"][k] for k in keys], 0.4, label="OLD ResNet")
        ax.bar(x + 0.2, [results[grp]["new"][k] for k in keys], 0.4, label="NEW ViT-S16")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_title(GROUP_TITLE[grp])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(C.OUTPUT_DIR / fname, dpi=120)
        plt.close()
        print(f"Wrote {C.OUTPUT_DIR / fname}")


def main():
    old, new = _load(C.OLD_CACHE), _load(C.NEW_CACHE)
    emb_old, emb_new, project_ids, gt = _align(old, new)
    glob = _global_labels(project_ids, gt)
    n_ids = len(np.unique(glob))
    print(f"Aligned {len(gt)} crops, {n_ids} identities, {len(np.unique(project_ids))} projects")

    results = {
        "A": {"old": embedding_quality(emb_old, glob), "new": embedding_quality(emb_new, glob)},
        "B": {"old": clustering_quality(emb_old, project_ids, gt, C.OLD_CLUSTER),
              "new": clustering_quality(emb_new, project_ids, gt, C.NEW_CLUSTER)},
        "n_crops": int(len(gt)), "n_identities": int(n_ids),
    }
    write_report(results, len(gt), n_ids)


if __name__ == "__main__":
    main()