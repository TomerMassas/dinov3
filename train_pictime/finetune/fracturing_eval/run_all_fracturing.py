"""Batch fracturing eval — run the fracturing pipeline over multiple finetune
V-dirs and produce per-run outputs + a cross-run comparison.

Each V-dir is identified by reading the config wandb saved locally at
<vdir>/wandb/run-*/files/config.yaml (written by wandb.init(config=...) at
launch). That gives experiment_tag, pretrained_weights (-> which pretrain
backbone), curriculum p and face_blur — so every evaluated checkpoint is
attributed to its actual experiment, never an assumed one. A consistency
warning fires if the tag doesn't mention the backbone version found in
pretrained_weights.

Per run: best-silhouette ckpt -> embed held-out projects -> HDBSCAN ->
fracturing metrics -> summary.json + the two standard plots (same pipeline as
run_fracturing_eval.py). After all runs: a combined markdown table + an
overlay figure (fraction of GT clusters vs fracturing count; solid = V17
lineage, dashed = others; color = curriculum p).

Usage:
    python3 -m train_pictime.finetune.fracturing_eval.run_all_fracturing
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib.pyplot as plt
import torch
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

from dinov3.configs import setup_job
from train_pictime.finetune.finetune_reid import (
    CFG_PATH, REPO_ROOT, build_projection_head, load_backbone,
)
from train_pictime.finetune.reeval_tiered import find_best_silhouette_ckpt
from train_pictime.finetune.reid_dataset import get_val_transform
from train_pictime.finetune.fracturing_eval.config import (
    DATASET_ROOT,
    HELD_OUT_PROJECTS_FILE,
    MIN_GT_CLUSTER_SIZE,
    OUTPUT_BASE,
)
from train_pictime.finetune.fracturing_eval.run_fracturing_eval import (
    load_held_out_pids,
    process_project,
)
from train_pictime.finetune.fracturing_eval.plot import (
    plot_fracturing_histogram,
    plot_subcluster_percentages,
)


FINETUNE_EXPERIMENTS_BASE = Path("/data/AI/Tomer/dinov3/train_pictime/finetune_experiments")

# The finetune V-dirs to evaluate. Identity (tag / backbone / p / blur) is NOT
# assumed from this list — it is resolved per-dir from the wandb config snapshot.
RUN_DIRS = ["V39", "V40", "V41", "V42", "V43", "V44"]
RUN_DIRS = ["V44"]

# Overlay figure knobs
OVERLAY_FRACTURING_CAP = 8           # roll counts >= cap into one final point
P_COLORS = {0.5: "tab:red", 0.75: "tab:orange", 1.0: "tab:blue"}


# ---------------------------------------------------------------------------
# Identity resolution (from the local wandb config snapshot)
# ---------------------------------------------------------------------------

def resolve_run_identity(vdir: Path) -> dict:
    """Identify which experiment a finetune V-dir holds.

    Reads <vdir>/wandb/run-*/files/config.yaml — wandb's dump of the full
    reid_config passed to wandb.init(config=...). Loud-fails if the identity
    can't be established; never guesses.

    Returns dict: vdir, tag, backbone ("V17"/"V18"/"?"), p, face_blur_enabled,
    wandb_run_dir.
    """
    run_dirs = sorted((vdir / "wandb").glob("run-*"))
    if not run_dirs:
        raise RuntimeError(f"{vdir.name}: no wandb/run-* directory")
    if len(run_dirs) > 1:
        print(f"[WARN] {vdir.name}: {len(run_dirs)} wandb run dirs, using newest: {run_dirs[-1].name}")
    cfg_path = run_dirs[-1] / "files" / "config.yaml"
    if not cfg_path.exists():
        raise RuntimeError(f"{vdir.name}: missing {cfg_path}")

    with open(cfg_path) as f:
        raw = yaml.safe_load(f)

    def value_of(key, default=None):
        node = raw.get(key)
        if isinstance(node, dict) and "value" in node:
            return node["value"]
        return default

    tag = value_of("experiment_tag")
    if not tag:
        raise RuntimeError(f"{vdir.name}: no experiment_tag in {cfg_path}")

    pretrained = str(value_of("pretrained_weights", ""))
    backbone = next((seg for seg in Path(pretrained).parts if re.fullmatch(r"V\d+", seg)), "?")

    curriculum = value_of("curriculum") or {}
    p = float(curriculum.get("p_start", 1.0)) if curriculum.get("enabled", False) else 1.0
    face_blur = value_of("face_blur") or {}

    identity = {
        "vdir": vdir.name,
        "tag": str(tag),
        "backbone": backbone,
        "p": p,
        "face_blur_enabled": bool(face_blur.get("enabled", False)),
        "wandb_run_dir": run_dirs[-1].name,
    }

    # Consistency: the tag should mention the backbone it was finetuned from.
    # (Catches config mistakes like tag=v18* with pretrained_weights still V17.)
    if backbone != "?" and backbone.lower() not in identity["tag"].lower():
        print(f"[CONSISTENCY WARNING] {vdir.name}: tag '{tag}' does not mention "
              f"backbone {backbone} (pretrained_weights={pretrained})")

    return identity


# ---------------------------------------------------------------------------
# Cross-run comparison outputs
# ---------------------------------------------------------------------------

def _run_stats(entries: list[dict]) -> dict:
    counts = [e["fracturing_count"] for e in entries]
    n = len(counts)
    rank1 = sorted(e["sub_cluster_pcts"][0] for e in entries if e["sub_cluster_pcts"])
    return {
        "n_gt_clusters": n,
        "perfect": sum(1 for c in counts if c == 1),
        "fractured": sum(1 for c in counts if c > 1),
        "all_noise": sum(1 for c in counts if c == 0),
        "mean_fracturing": (sum(counts) / n) if n else 0.0,
        "max_fracturing": max(counts) if counts else 0,
        "median_rank1_pct": rank1[len(rank1) // 2] if rank1 else 0.0,
    }


def write_comparison_table(per_run: dict[str, tuple[dict, list[dict]]],
                           out_path: Path,
                          ) -> None:
    lines = [
        "| run | backbone | p | ft_blur | n GT | perfect | fractured | all-noise | mean frac | max | median rank-1 % |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, (ident, entries) in per_run.items():
        s = _run_stats(entries)
        n = max(s["n_gt_clusters"], 1)
        lines.append(f"| {ident['tag']} ({ident['vdir']}) | {ident['backbone']} | {ident['p']:.2f} "
                     f"| {ident['face_blur_enabled']} | {s['n_gt_clusters']} "
                     f"| {s['perfect']} ({100.0 * s['perfect'] / n:.1f}%) "
                     f"| {s['fractured']} ({100.0 * s['fractured'] / n:.1f}%) "
                     f"| {s['all_noise']} | {s['mean_fracturing']:.2f} | {s['max_fracturing']} "
                     f"| {s['median_rank1_pct']:.1f} |")
    table = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table + "\n")
    print("\n" + table + "\n")
    print(f"Wrote {out_path}")


def plot_overlay(per_run: dict[str, tuple[dict, list[dict]]],
                 out_path: Path,
                ) -> None:
    """Fraction of GT clusters per fracturing count, one line per run.
    Linestyle: solid = V17 lineage, dashed = everything else. Color = p."""
    fig, ax = plt.subplots(figsize=(11, 6))
    xs = list(range(0, OVERLAY_FRACTURING_CAP + 1))

    for label, (ident, entries) in per_run.items():
        counts = [e["fracturing_count"] for e in entries]
        n = max(len(counts), 1)
        fracs = [sum(1 for c in counts if c == x) / n for x in xs[:-1]]
        fracs.append(sum(1 for c in counts if c >= OVERLAY_FRACTURING_CAP) / n)

        style = "-" if ident["backbone"] == "V17" else "--"
        color = P_COLORS.get(ident["p"], "gray")
        ax.plot(xs,
                fracs,
                style,
                color=color,
                marker="o",
                markersize=4,
                label=f"{ident['tag']} ({ident['vdir']})",
               )

    labels = [str(x) for x in xs[:-1]] + [f"{OVERLAY_FRACTURING_CAP}+"]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Fracturing count (# distinct non-noise predicted sub-clusters)")
    ax.set_ylabel("Fraction of GT clusters")
    ax.set_title("Fracturing distribution — all runs (solid=V17 lineage, dashed=other; color=p)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = OmegaConf.load(CFG_PATH)
    setup_job(output_dir=None, seed=cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Resolve all identities FIRST and show the mapping (the "which run is
    # this" gate). Any unresolvable dir aborts before GPU work starts. ---
    identities = []
    for name in RUN_DIRS:
        vdir = FINETUNE_EXPERIMENTS_BASE / name
        ident = resolve_run_identity(vdir)
        ckpt_path, it, train_sil = find_best_silhouette_ckpt(vdir / "ckpt")
        ident.update({"ckpt_path": str(ckpt_path), "iteration": int(it), "train_sil": float(train_sil)})
        identities.append(ident)

    print("\n===== Run mapping (from wandb config snapshots) =====")
    for ident in identities:
        print(f"  {ident['vdir']}: tag={ident['tag']:<20} backbone={ident['backbone']} "
              f"p={ident['p']:.2f} ft_blur={ident['face_blur_enabled']} "
              f"ckpt=iter{ident['iteration']} sil={ident['train_sil']:.4f}")
    print("=====================================================\n")

    pids = load_held_out_pids(HELD_OUT_PROJECTS_FILE)
    print(f"Held-out projects: {len(pids)} from {HELD_OUT_PROJECTS_FILE}")

    # --- Load the backbone ONCE. The DCP source here is irrelevant: every run's
    # finetune ckpt below contains the FULL backbone state dict and is loaded
    # with strict=True, overwriting every weight — including for runs that were
    # finetuned from a different pretrain than cfg.pretrained_weights. ---
    print("\nLoading backbone (construction source only — weights replaced per run)...")
    pretrain_cfg_path = str(REPO_ROOT / cfg.pretrain_config)
    backbone, embed_dim = load_backbone(pretrain_cfg_path, cfg.pretrained_weights, cfg.backbone_which)
    backbone.cuda()
    proj_head = build_projection_head(embed_dim, cfg.proj_hidden_dim, cfg.proj_output_dim).cuda()
    transform = get_val_transform()

    per_run: dict[str, tuple[dict, list[dict]]] = {}

    for ident in identities:
        label = f"{ident['vdir']}_{ident['tag']}_iter{ident['iteration']}"
        print(f"\n===== {label} =====")

        state = torch.load(ident["ckpt_path"], map_location="cuda")
        backbone.load_state_dict(state["backbone_state_dict"], strict=True)
        proj_head.load_state_dict(state["proj_head_state_dict"], strict=True)
        backbone.eval()
        proj_head.eval()
        del state
        torch.cuda.empty_cache()

        all_entries: list[dict] = []
        status_counts: dict[str, int] = defaultdict(int)
        for pid in tqdm(pids, desc=label):
            try:
                status, entries = process_project(pid,
                                                  DATASET_ROOT,
                                                  backbone,
                                                  proj_head,
                                                  transform,
                                                  device,
                                                  MIN_GT_CLUSTER_SIZE,
                                                 )
            except Exception as e:
                tqdm.write(f"[{pid}] ERROR: {e!r}")
                status_counts["error"] += 1
                continue
            status_counts[status] += 1
            all_entries.extend(entries)

        out_dir = Path(OUTPUT_BASE) / label
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            **ident,
            "held_out_projects_file": HELD_OUT_PROJECTS_FILE,
            "dataset_root": DATASET_ROOT,
            "min_gt_cluster_size": MIN_GT_CLUSTER_SIZE,
            "status_counts": dict(status_counts),
            "n_gt_clusters": len(all_entries),
            "entries": all_entries,
        }
        summary_path = out_dir / "summary.json"
        tmp = str(summary_path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(summary, f, indent=2)
        os.replace(tmp, summary_path)

        title_suffix = f" — {ident['tag']} ({ident['vdir']} iter{ident['iteration']})"
        plot_fracturing_histogram(all_entries,
                                  out_dir / "plot_1_histogram.png",
                                  title_suffix,
                                 )
        plot_subcluster_percentages(all_entries,
                                    out_dir / "plot_2_per_rank.png",
                                    title_suffix,
                                   )
        stats = _run_stats(all_entries)
        print(f"{label}: n={stats['n_gt_clusters']}, perfect={stats['perfect']}, "
              f"fractured={stats['fractured']}, mean={stats['mean_fracturing']:.2f}")
        per_run[label] = (ident, all_entries)

    # --- Cross-run comparison ---
    comparison_dir = Path(OUTPUT_BASE) / "comparison"
    write_comparison_table(per_run, comparison_dir / "comparison_table.md")
    plot_overlay(per_run, comparison_dir / "overlay_fracturing.png")


if __name__ == "__main__":
    main()