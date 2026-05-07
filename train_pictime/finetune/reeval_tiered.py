"""Re-evaluate finetuned checkpoints at multiple eval tiers (top-50/75/100).

Loads a fresh V11 backbone once, then per run swaps in the saved finetune
state and runs tiered ReID eval on the v2 val pool. Prints a markdown row per
run for easy copy-paste into experiment.md.

Edit RUNS below to point each label at its V<n> output directory, then:
    python3 -m train_pictime.finetune.reeval_tiered
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from dinov3.configs import setup_job

from train_pictime.finetune.finetune_reid import (
    REPO_ROOT, CFG_PATH, load_backbone, build_projection_head,
)
from train_pictime.finetune.reid_dataset import load_index, train_val_split
from train_pictime.finetune.reid_evaluator import ReIDEvaluator


# ---------------------------------------------------------------------------
# Configure: fill V<n> for each run
# ---------------------------------------------------------------------------

RUNS: dict[str, str] = {
    "blue": "/data/AI/Tomer/dinov3/train_pictime/finetune_experiments/V14",
    "pink": "/data/AI/Tomer/dinov3/train_pictime/finetune_experiments/V13",
    "red":  "/data/AI/Tomer/dinov3/train_pictime/finetune_experiments/V15",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CKPT_PATTERN = re.compile(r"ckpt_iter(\d+)_sil([-\d.]+)\.pt$")


def find_best_silhouette_ckpt(ckpt_dir: Path) -> tuple[Path, int, float]:
    """Pick the highest-silhouette ckpt in V<n>/ckpt/ by parsing filenames."""
    candidates: list[tuple[float, int, Path]] = []
    for p in ckpt_dir.glob("ckpt_iter*_sil*.pt"):
        m = CKPT_PATTERN.search(p.name)
        if m is None:
            continue
        it = int(m.group(1))
        sil = float(m.group(2))
        candidates.append((sil, it, p))
    if not candidates:
        raise FileNotFoundError(f"No ckpt_iter*_sil*.pt found in {ckpt_dir}")
    sil, it, path = max(candidates, key=lambda x: x[0])
    return path, it, sil


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = OmegaConf.load(CFG_PATH)
    setup_job(output_dir=None, seed=cfg.seed)

    # Val pool — same logic as finetune_reid.py
    print("Loading index...")
    index_path = Path(cfg.data_base_path) / cfg.get("reid_index_filename", "reid_index.npz")
    image_paths, bboxes, bbox_indices, project_ids, cluster_ids = load_index(index_path)

    # Drop cluster_id == -1 globally — symmetric to finetune_reid.py.
    keep = cluster_ids != -1
    n_dropped = int((~keep).sum())
    image_paths   = image_paths[keep]
    bboxes        = bboxes[keep]
    bbox_indices  = bbox_indices[keep]
    project_ids   = project_ids[keep]
    cluster_ids   = cluster_ids[keep]
    print(f"Filtered {n_dropped} cluster_id=-1 samples globally; {len(image_paths)} samples remain")

    unique_projects = sorted(set(project_ids))
    _, val_project_ids = train_val_split(unique_projects, cfg.val_ratio, cfg.seed)
    val_mask = np.isin(project_ids, val_project_ids)
    print(f"Val pool: {len(val_project_ids)} projects, {int(val_mask.sum())} samples")

    eval_tiers = list(cfg.get("eval_tiers", [1.0]))
    centroid_filename = cfg.get("eval_centroid_distances_filename", None)
    print(f"Eval tiers: {eval_tiers} | centroid file: {centroid_filename}")

    # Build evaluator once; reuse across all runs (val pool is shared).
    evaluator = ReIDEvaluator(
        image_paths=image_paths[val_mask],
        bboxes=bboxes[val_mask],
        bbox_indices=bbox_indices[val_mask],
        project_ids=project_ids[val_mask],
        cluster_ids=cluster_ids[val_mask],
        seed=cfg.seed,
        min_k=cfg.K,
        silhouette_max_samples=cfg.silhouette_max_samples,
        centroid_distances_filename=centroid_filename,
        eval_tiers=eval_tiers,
    )

    # Load V11 architecture once; swap state dicts per run (FSDP2 dance is heavy).
    print("\nLoading V11 backbone (one-time)...")
    pretrain_cfg_path = str(REPO_ROOT / cfg.pretrain_config)
    backbone, embed_dim = load_backbone(
        pretrain_cfg_path, cfg.pretrained_weights, cfg.backbone_which,
    )
    backbone.cuda()
    print(f"Backbone embed_dim={embed_dim}")

    tier_suffixes = [f"top{int(round(t * 100))}" for t in sorted(eval_tiers)]
    rows: list[str] = []

    for label, run_dir in RUNS.items():
        run_path = Path(run_dir)
        ckpt_dir = run_path / "ckpt"
        if not ckpt_dir.exists():
            print(f"\n[{label}] SKIPPING — no ckpt dir at {ckpt_dir}")
            continue

        try:
            ckpt_path, it, train_sil = find_best_silhouette_ckpt(ckpt_dir)
        except FileNotFoundError as e:
            print(f"\n[{label}] SKIPPING — {e}")
            continue

        print(f"\n===== {label}: {ckpt_path.name} (train sil={train_sil:.4f}) =====")

        state = torch.load(ckpt_path, map_location="cuda")
        backbone.load_state_dict(state["backbone_state_dict"], strict=True)

        proj_head = build_projection_head(
            embed_dim, cfg.proj_hidden_dim, cfg.proj_output_dim,
        ).cuda()
        proj_head.load_state_dict(state["proj_head_state_dict"], strict=True)
        backbone.eval()
        proj_head.eval()

        metrics = evaluator.maybe_eval(
            backbone, proj_head, iteration=it, wandb_run=None,
        )

        del proj_head, state
        torch.cuda.empty_cache()

        if metrics is None:
            print(f"[{label}] eval returned None — skipping row")
            continue

        cells = [label, str(it)]
        for s in tier_suffixes:
            cells += [
                f"{metrics.get(f'mAP_{s}', float('nan')):.4f}",
                f"{metrics.get(f'rank1_{s}', float('nan')):.4f}",
                f"{metrics.get(f'silhouette_{s}', float('nan')):.4f}",
            ]
        n_id = metrics.get(f"n_identities_{tier_suffixes[0]}", float("nan"))
        cells += [f"{int(n_id)}" if n_id == n_id else "nan"]  # NaN check
        rows.append("| " + " | ".join(cells) + " |")

    # Markdown table — copy-paste into experiment.md
    header_cols = ["run", "iter"]
    for s in tier_suffixes:
        header_cols += [f"mAP_{s}", f"R1_{s}", f"sil_{s}"]
    header_cols += [f"n_id"]
    header = "| " + " | ".join(header_cols) + " |"
    sep = "|" + "|".join(["---"] * len(header_cols)) + "|"

    print("\n\n===== Markdown table =====")
    print(header)
    print(sep)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
