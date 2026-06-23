"""Histogram of GT cluster (identity) sizes across the locked test set.

Motivation: if clustering metrics improve as you raise min_cluster_size, it may be
because the test set (top-N by crops/clusters) is biased toward big clusters — so a
high floor noises out the few small identities and you score only the easy big ones.
This plot shows how many crops each true identity actually has, with reference lines
at the eval floor (MIN_GT_CLUSTER_SIZE) and at min_cluster_size values of interest,
so you can read what fraction of identities sits below each.

Reads the locked project-ids file + each project's clusters_fixed.json (raw GT, drops only
cluster_id == -1 — i.e. it shows clusters the eval's MIN_GT_CLUSTER_SIZE would later
drop, so the bias is visible).

    python3 -m train_pictime.model_comparison.gt_cluster_size_hist
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from train_pictime.model_comparison import config as C

# Histogram: sizes >= HIST_CAP roll into a final overflow bin; BIN_WIDTH = bar width.
HIST_CAP = 200
BIN_WIDTH = 2


def gt_cluster_sizes() -> list[int]:
    """# crops per GT identity (cluster_id != -1) across all selected projects."""
    with open(C.NEW_PROJECTS_FILE) as f:
        payload = json.load(f)
    pids = payload if isinstance(payload, list) else payload["project_ids"]

    sizes = []
    for pid in pids:
        cl_path = C.DATASET_ROOT / pid / C.CLUSTERS_FIXED_FILENAME
        if not cl_path.exists():
            continue
        with open(cl_path) as f:
            clusters = json.load(f)
        counts: Counter = Counter()
        for _fname, entries in clusters.items():
            for e in entries:
                cid = int(e["cluster_id"])
                if cid != -1:
                    counts[cid] += 1
        sizes.extend(counts.values())
    return sizes


def build_histogram(show: bool = True) -> Path:
    """Compute GT cluster sizes, save the histogram png to OUTPUT_DIR (returns its
    path) and print stats. show=True also displays the figure (for direct runs)."""
    sizes = np.array(gt_cluster_sizes(), dtype=int)
    if len(sizes) == 0:
        raise RuntimeError(f"No GT clusters found — is {C.NEW_PROJECTS_FILE.name} locked and the data on disk?")

    ref = C.MIN_GT_CLUSTER_SIZE

    # --- stats ---
    pcts = {p: float(np.percentile(sizes, p)) for p in (10, 25, 50, 75, 90, 95)}
    print(f"GT identities: {len(sizes)} across {C.NEW_PROJECTS_FILE.name}")
    print(f"  size  min={sizes.min()}  p10={pcts[10]:.0f}  p25={pcts[25]:.0f}  "
          f"median={pcts[50]:.0f}  mean={sizes.mean():.1f}  p75={pcts[75]:.0f}  "
          f"p90={pcts[90]:.0f}  p95={pcts[95]:.0f}  max={sizes.max()}")
    below = int((sizes < ref).sum())
    print(f"  below eval floor MIN_GT_CLUSTER_SIZE={ref}: {below} "
          f"({100.0 * below / len(sizes):.1f}%) identities (dropped from scoring)")

    # --- histogram (sizes >= HIST_CAP rolled into the final bin) ---
    clipped = np.clip(sizes, 0, HIST_CAP)
    bins = np.arange(0, HIST_CAP + BIN_WIDTH, BIN_WIDTH)
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(clipped, bins=bins, color="steelblue", edgecolor="white", alpha=0.85)

    below_pct = 100.0 * (sizes < ref).sum() / len(sizes)
    ax.axvline(ref, color="tab:red", linestyle="--", alpha=0.8,
               label=f"MIN_GT_CLUSTER_SIZE={ref} (eval floor): {below_pct:.0f}% below")

    overflow_n = int((sizes >= HIST_CAP).sum())
    if overflow_n:
        ax.plot([], [], " ",  # legend-only entry naming the final bin's true range
                label=f"final bin = {HIST_CAP}+  (range {HIST_CAP}–{sizes.max()}, n={overflow_n})")

    ax.set_xlabel(f"GT identity size (# crops; clipped at {HIST_CAP})")
    ax.set_ylabel("number of GT identities")
    ax.set_title(f"GT cluster-size distribution — {len(sizes)} identities, "
                 f"{C.NEW_PROJECTS_FILE.stem}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_png = C.OUTPUT_DIR / "gt_cluster_size_hist.png"
    plt.savefig(out_png, dpi=120)   # save BEFORE show: show() can clear the figure on some backends
    if show:
        plt.show()
    plt.close()
    print(f"Wrote {out_png}")
    return out_png


def main():
    build_histogram(show=True)


if __name__ == "__main__":
    main()