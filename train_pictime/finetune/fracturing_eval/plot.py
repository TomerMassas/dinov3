"""Fracturing eval — plotting from summary.json.

Loads the summary written by run_fracturing_eval.py and generates:
  - plot_1_histogram.png — fracturing distribution (count of GT clusters per
    fracturing number)
  - plot_2_per_rank.png — violin of sub-cluster % (rank 1 = biggest) per GT
    cluster's non-noise crops

The output dir is auto-located from FINETUNE_VERSION_DIR + best ckpt
(same convention as run_fracturing_eval.py), so this script picks up
whatever the most recent fracturing run produced for the current V<n>.

Usage:
    python3 -m train_pictime.finetune.fracturing_eval.plot
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib.pyplot as plt

from train_pictime.finetune.reeval_tiered import find_best_silhouette_ckpt
from train_pictime.finetune.fracturing_eval.config import (
    FINETUNE_VERSION_DIR,
    OUTPUT_BASE,
)


# ---------------------------------------------------------------------------
# Plot knobs (local to plot.py — output cosmetics, not metric semantics)
# ---------------------------------------------------------------------------

# Plot 2: drop violins with fewer than this many GT clusters contributing
# (sparse tails are noisy and uninformative).
MIN_SAMPLES_PER_VIOLIN = 5

# Plot 2: cap the x-axis at this rank regardless of how deep fracturing goes.
MAX_RANK_SHOWN = 8

# Plot 1: roll the tail into one "K+" bar when max fracturing > this.
HISTOGRAM_TAIL_CAP = 12


# ---------------------------------------------------------------------------
# Plot 1 — fracturing histogram
# ---------------------------------------------------------------------------

def plot_fracturing_histogram(entries: list[dict],
                              out_path: Path,
                              title_suffix: str = "",
                             ) -> None:
    counts = [e["fracturing_count"] for e in entries]
    if not counts:
        print("Plot 1: no entries — skipping")
        return

    max_frac = max(counts)
    has_zero = any(c == 0 for c in counts)
    start = 0 if has_zero else 1

    if max_frac <= HISTOGRAM_TAIL_CAP:
        bins = list(range(start, max_frac + 1))
        bar_counts = [counts.count(b) for b in bins]
        labels = [str(b) for b in bins]
    else:
        bins = list(range(start, HISTOGRAM_TAIL_CAP))
        bar_counts = [counts.count(b) for b in bins]
        tail = sum(1 for c in counts if c >= HISTOGRAM_TAIL_CAP)
        bar_counts.append(tail)
        labels = [str(b) for b in bins] + [f"{HISTOGRAM_TAIL_CAP}+"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, bar_counts, color="#4a7ab8", edgecolor="black")
    for bar, n in zip(bars, bar_counts):
        if n > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    str(n),
                    ha="center", va="bottom", fontsize=9,
                   )
    ax.set_xlabel("Fracturing count (# distinct non-noise predicted sub-clusters)")
    ax.set_ylabel("Number of GT clusters")
    ax.set_title(f"Fracturing distribution{title_suffix}")
    ax.grid(axis="y", alpha=0.3)

    n_total = len(counts)
    n_perfect = sum(1 for c in counts if c == 1)
    n_fractured = sum(1 for c in counts if c > 1)
    n_all_noise = sum(1 for c in counts if c == 0)
    ax.text(0.98, 0.95,
            f"n GT clusters: {n_total}\n"
            f"Perfectly grouped (1): {n_perfect} ({100.0*n_perfect/n_total:.1f}%)\n"
            f"Fractured (>1): {n_fractured} ({100.0*n_fractured/n_total:.1f}%)\n"
            f"All-noise (0): {n_all_noise}",
            transform=ax.transAxes,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
            fontsize=10,
           )

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 — sub-cluster % by rank
# ---------------------------------------------------------------------------

def plot_subcluster_percentages(entries: list[dict],
                                out_path: Path,
                                title_suffix: str = "",
                               ) -> None:
    rank_to_pcts: dict[int, list[float]] = defaultdict(list)
    for e in entries:
        for rank_idx, pct in enumerate(e["sub_cluster_pcts"]):
            rank_to_pcts[rank_idx + 1].append(pct)

    if not rank_to_pcts:
        print("Plot 2: no sub-clusters — skipping")
        return

    max_rank = max(rank_to_pcts.keys())
    ranks_kept: list[int] = []
    for r in range(1, min(max_rank, MAX_RANK_SHOWN) + 1):
        if len(rank_to_pcts.get(r, [])) >= MIN_SAMPLES_PER_VIOLIN:
            ranks_kept.append(r)
    if not ranks_kept:
        print(f"Plot 2: no rank has >= {MIN_SAMPLES_PER_VIOLIN} samples — skipping")
        return

    data = [rank_to_pcts[r] for r in ranks_kept]
    ns = [len(rank_to_pcts[r]) for r in ranks_kept]

    fig, ax = plt.subplots(figsize=(10, 6))
    parts = ax.violinplot(data,
                          positions=ranks_kept,
                          showmeans=False,
                          showmedians=True,
                          widths=0.7,
                         )
    for body in parts["bodies"]:
        body.set_facecolor("#4a7ab8")
        body.set_edgecolor("black")
        body.set_alpha(0.7)

    ax.set_xticks(ranks_kept)
    ax.set_xticklabels([f"rank {r}\n(n={n})" for r, n in zip(ranks_kept, ns)])
    ax.set_xlabel("Sub-cluster rank within parent GT cluster (1 = biggest)")
    ax.set_ylabel("% of GT cluster's non-noise crops")
    ax.set_title(f"Sub-cluster size distribution by rank{title_suffix}")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(50, color="gray", linestyle="--", alpha=0.4, linewidth=1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_output_dir() -> Path:
    """Locate the output dir matching FINETUNE_VERSION_DIR's best ckpt."""
    ckpt_dir = Path(FINETUNE_VERSION_DIR) / "ckpt"
    _, it, train_sil = find_best_silhouette_ckpt(ckpt_dir)
    version_name = Path(FINETUNE_VERSION_DIR).name
    return Path(OUTPUT_BASE) / f"{version_name}_iter{it}_sil{train_sil:.4f}"


def main():
    output_dir = find_output_dir()
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No summary at {summary_path}. "
                                f"Run run_fracturing_eval.py first.")

    with open(summary_path, "r") as f:
        summary = json.load(f)

    version_name = summary.get("model_version", "?")
    iteration = summary.get("iteration", "?")
    train_sil = summary.get("train_silhouette", 0.0)
    title_suffix = f" — {version_name} iter{iteration} (sil={train_sil:.3f})"

    entries = summary.get("entries", [])
    print(f"Loaded {len(entries)} GT cluster entries from {summary_path}")

    plot_fracturing_histogram(entries,
                              output_dir / "plot_1_histogram.png",
                              title_suffix,
                             )
    plot_subcluster_percentages(entries,
                                output_dir / "plot_2_per_rank.png",
                                title_suffix,
                               )


if __name__ == "__main__":
    main()