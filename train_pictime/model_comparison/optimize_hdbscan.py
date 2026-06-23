"""One-param-at-a-time HDBSCAN sweep on the NEW model's cached embeddings.

Flip SWEEP_PARAM + SWEEP_VALUES below, run directly (PyCharm / `python3 ...`),
read the plot, lock the winning value into config.NEW_CLUSTER, then sweep the
next param. Every OTHER param is held at its current config.NEW_CLUSTER value, so
each run isolates one knob.

Uses the cached new_embeddings.npz (run embed.py --MODEL new once); no re-embedding.

    python3 -m train_pictime.model_comparison.optimize_hdbscan
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from train_pictime.model_comparison import config as C
from train_pictime.model_comparison.clustering import cluster
from train_pictime.model_comparison.metrics import clustering_metrics

# --- What to sweep this run (one param) ---
SWEEP_PARAM = "cluster_selection_epsilon"
SWEEP_VALUES = [0.0, 0.08, 0.09, 0.1, 0.11, 0.2, 0.5]
# Suggested value lists per param (copy into SWEEP_VALUES):
#   min_cluster_size           : [2, 3, 4, 5, 8, 10]
#   min_samples                : [1, 2, 3, 5]
#   cluster_selection_epsilon  : [0.0, 0.08, 0.09, 0.1, 0.11, 0.2, 0.5]
#   cluster_selection_method   : ["eom", "leaf"]
#   allow_single_cluster       : [True, False]

# Metric used to mark the "best" value (vertical line). fracturing_mean is minimized;
# everything else is maximized. ARI is chance-corrected and penalizes BOTH merging and
# splitting — so it isn't gamed by a high epsilon collapsing everything into one cluster
# (unlike cluster_f1 / recall / completeness). Set to "" to skip the marker.
OBJECTIVE = "ari"

# Left [0,1] axis. homogeneity is the merge counterweight: if it craters while
# completeness/recall climb, high epsilon is just merging different people.
LEFT_METRICS = ["cluster_precision", "cluster_recall", "completeness",
                "homogeneity", "fracturing_perfect_frac", "ari"]
# Right (count) axis. cluster_count_delta = pred − true identities per project:
# 0 = right count, negative = under-clustering (merging), positive = over-clustering.
RIGHT_METRICS = ["fracturing_mean", "cluster_count_delta"]
AGG_KEYS = LEFT_METRICS + RIGHT_METRICS


def eval_spec(emb, project_ids, gt, spec) -> dict:
    """Per-project cluster + clustering_metrics, averaged (predicted -1 ignored
    inside clustering_metrics). Returns mean of AGG_KEYS over projects."""
    per = []
    for pid in np.unique(project_ids):
        mask = project_ids == pid
        if mask.sum() < 2:
            continue
        m = clustering_metrics(gt[mask], cluster(emb[mask], spec))
        if m is not None:
            per.append(m)
    if not per:
        return {k: float("nan") for k in AGG_KEYS}
    return {k: float(np.mean([p[k] for p in per])) for k in AGG_KEYS}


def main():
    d = np.load(C.NEW_CACHE, allow_pickle=True)
    emb = d["embeddings"]
    project_ids = np.array([str(p) for p in d["project_ids"]], dtype=object)
    gt = d["gt_cluster_ids"].astype(int)
    print(f"Sweeping '{SWEEP_PARAM}' over {SWEEP_VALUES} "
          f"on {len(emb)} crops / {len(np.unique(project_ids))} projects")
    print(f"(other params held at config.NEW_CLUSTER: "
          f"{ {k: v for k, v in C.NEW_CLUSTER.items() if k not in ('method', SWEEP_PARAM)} })")

    rows = []
    for v in SWEEP_VALUES:
        spec = dict(C.NEW_CLUSTER)
        spec[SWEEP_PARAM] = v
        res = eval_spec(emb, project_ids, gt, spec)
        rows.append(res)
        print(f"  {SWEEP_PARAM}={v}: " + ", ".join(f"{k}={res[k]:.3f}" for k in AGG_KEYS))

    # Best value by objective (fracturing_mean minimized, else maximized)
    best_i = None
    if OBJECTIVE:
        vals = [r[OBJECTIVE] for r in rows]
        best_i = int(np.nanargmin(vals)) if OBJECTIVE == "fracturing_mean" else int(np.nanargmax(vals))
        print(f"\nBest {SWEEP_PARAM} by {OBJECTIVE}: {SWEEP_VALUES[best_i]} "
              f"({OBJECTIVE}={rows[best_i][OBJECTIVE]:.3f})")

    # --- Plot: left axis = [0,1] metrics, right axis = mean fracturing ---
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(SWEEP_VALUES))
    fig, axL = plt.subplots(figsize=(max(8, len(x) * 1.3), 6))
    axR = axL.twinx()

    for m in LEFT_METRICS:
        axL.plot(x, [r[m] for r in rows], marker="o", label=m)
    right_styles = {"fracturing_mean": "k--s", "cluster_count_delta": "C7-^"}
    for m in RIGHT_METRICS:
        axR.plot(x, [r[m] for r in rows], right_styles.get(m, "--"), label=f"{m} (right)")
    axR.axhline(0, color="C7", linewidth=0.8, alpha=0.5)   # delta=0 = predicted count matches truth

    if best_i is not None:
        axL.axvline(best_i, color="green", linestyle=":", alpha=0.7)
        axL.text(best_i, 1.01, f"best={SWEEP_VALUES[best_i]}", color="green",
                 ha="center", va="bottom", transform=axL.get_xaxis_transform())

    axL.set_xticks(x)
    axL.set_xticklabels([str(v) for v in SWEEP_VALUES])
    axL.set_xlabel(SWEEP_PARAM)
    axL.set_ylabel("score (higher better)")
    axL.set_ylim(0, 1.05)
    axR.set_ylabel("fracturing / count-delta (per project; delta 0 = ideal)")
    axL.set_title(f"HDBSCAN sweep: {SWEEP_PARAM}  (NEW model, other params fixed)")
    axL.grid(axis="y", alpha=0.3)
    # merge legends from both axes
    lines = axL.get_lines() + axR.get_lines()
    axL.legend(lines, [ln.get_label() for ln in lines], loc="best", fontsize=8)

    out_png = C.OUTPUT_DIR / f"sweep_{SWEEP_PARAM}.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"Wrote {out_png}")

    out_json = C.OUTPUT_DIR / f"sweep_{SWEEP_PARAM}.json"
    out_json.write_text(json.dumps({
        "param": SWEEP_PARAM, "values": SWEEP_VALUES,
        "fixed": {k: v for k, v in C.NEW_CLUSTER.items() if k != SWEEP_PARAM},
        "objective": OBJECTIVE,
        "best_value": (SWEEP_VALUES[best_i] if best_i is not None else None),
        "rows": rows,
    }, indent=2, default=str))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()