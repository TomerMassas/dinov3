import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from dinov3.configs import setup_config, setup_job
from dinov3.configs.config import DinoV3SetupArgs
from dinov3.train.ssl_meta_arch import SSLMetaArch
from dinov3.checkpointer import load_checkpoint
from dinov3.train.cosine_lr_scheduler import CosineScheduler

from train_pictime.wandb_logger import init_wandb, log_wandb
from train_pictime.finetune.supcon_loss import SupConLoss
from train_pictime.finetune.reid_dataset import (
    load_project, build_global_identity_map, ReIDCropDataset,
    PKBatchSampler, train_val_split, get_train_transform, get_val_transform,
)
from train_pictime.finetune.reid_evaluator import ReIDEvaluator

REPO_ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = Path(__file__).parent / "reid_config.yaml"


# ---------------------------------------------------------------------------
# Version directories (same pattern as train_dino_grad_accum.py)
# ---------------------------------------------------------------------------

def _find_version_dirs(base_dir: Path):
    versions = []
    if base_dir.exists():
        for folder in base_dir.iterdir():
            if folder.is_dir() and folder.name.startswith("V"):
                try:
                    versions.append((int(folder.name[1:]), folder))
                except ValueError:
                    continue
    versions.sort()
    return versions


def add_version_suffix(output_dir: str) -> str:
    base_dir = Path(output_dir)
    versions = _find_version_dirs(base_dir)
    next_version = (versions[-1][0] + 1) if versions else 1
    return str(base_dir / f"V{next_version}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_backbone(pretrain_cfg_path: str, ckpt_path: str, which: str = "teacher"):
    """Load a pretrained DINOv3 backbone from a DCP checkpoint.

    Builds SSLMetaArch the same way the pretrain script does, loads the
    checkpoint, then extracts the teacher/student backbone.
    """
    setup_args = DinoV3SetupArgs(
        config_file=pretrain_cfg_path,
        output_dir="/tmp/finetune_dummy",
        opts=[],
    )
    cfg = setup_config(setup_args, strict_cfg=False)

    # Build model on meta device, then materialize on CUDA (same as pretrain script)
    with torch.device("meta"):
        model = SSLMetaArch(cfg)
    model.prepare_for_distributed_training()
    model._apply(
        lambda t: torch.full_like(
            t,
            fill_value=math.nan if t.dtype.is_floating_point else (2 ** (t.dtype.itemsize * 8 - 1)),
            device="cuda",
        ),
        recurse=True,
    )
    model.init_weights()

    # Load DCP checkpoint
    load_checkpoint(ckpt_path, model=model)
    print(f"Loaded checkpoint from {ckpt_path}")

    # Extract backbone
    mdl = model.teacher if which == "teacher" else model.student
    backbone = mdl["backbone"]
    embed_dim = backbone.embed_dim

    # Detach from SSLMetaArch to free memory
    del model
    torch.cuda.empty_cache()

    return backbone, embed_dim


def build_projection_head(embed_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(embed_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, output_dim),
    )


# ---------------------------------------------------------------------------
# Freeze / Unfreeze
# ---------------------------------------------------------------------------

def freeze_backbone(backbone: nn.Module):
    backbone.requires_grad_(False)
    backbone.eval()


def maybe_unfreeze(backbone, proj_head, optimizer, iteration, cfg):
    """Unfreeze last N blocks at the configured iteration. Returns new optimizer or None."""
    if cfg.unfreeze_after <= 0 or iteration != cfg.unfreeze_after:
        return None

    n = cfg.unfreeze_n_blocks
    print(f"[iter {iteration}] Unfreezing last {n} blocks + final norm")

    # Unfreeze last N blocks
    for block in backbone.blocks[-n:]:
        block.requires_grad_(True)
    # Unfreeze final layer norm
    if hasattr(backbone, "norm"):
        backbone.norm.requires_grad_(True)

    # Rebuild optimizer with 2 param groups
    unfrozen_params = [p for p in backbone.parameters() if p.requires_grad]
    new_optimizer = torch.optim.AdamW([
        {"params": proj_head.parameters(), "lr": cfg.lr},
        {"params": unfrozen_params, "lr": cfg.lr_backbone},
    ], weight_decay=cfg.weight_decay)

    return new_optimizer


# ---------------------------------------------------------------------------
# Best-checkpoint tracking
# ---------------------------------------------------------------------------

class BestCheckpointTracker:
    """Track and keep only the best N checkpoints by a metric (higher = better)."""

    def __init__(self, ckpt_dir: Path, max_keep: int = 3):
        self.ckpt_dir = ckpt_dir
        self.max_keep = max_keep
        self.entries: list[tuple[float, Path]] = []  # (metric_value, path)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def maybe_save(self, metric_value: float, iteration: int, backbone, proj_head, optimizer):
        """Save if metric is better than worst kept checkpoint."""
        if len(self.entries) >= self.max_keep:
            worst_val, worst_path = min(self.entries, key=lambda x: x[0])
            if metric_value <= worst_val:
                return  # not good enough
            # Remove worst
            self.entries.remove((worst_val, worst_path))
            if worst_path.exists():
                worst_path.unlink()

        path = self.ckpt_dir / f"ckpt_iter{iteration}_mAP{metric_value:.4f}.pt"
        torch.save({
            "iteration": iteration,
            "mAP": metric_value,
            "backbone_state_dict": backbone.state_dict(),
            "proj_head_state_dict": proj_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, path)
        self.entries.append((metric_value, path))
        print(f"[Checkpoint] Saved {path.name} (mAP={metric_value:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = OmegaConf.load(CFG_PATH)

    # Initialize distributed (single-process) — needed for DCP checkpoint loading
    setup_job(output_dir=None, seed=cfg.seed)

    # Resolve paths relative to repo root
    pretrain_cfg_path = str(REPO_ROOT / cfg.pretrain_config)

    # Version dir
    output_dir = add_version_suffix(cfg.output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    # W&B
    run_name = f"finetune_reid_P{cfg.P}_K{cfg.K}_lr{cfg.lr:g}"
    run = init_wandb(
        OmegaConf.create({"dummy": True}),  # flat config for wandb
        output_dir=output_dir,
        run_name=run_name,
    )

    # Model
    print("Loading pretrained backbone...")
    backbone, embed_dim = load_backbone(pretrain_cfg_path, cfg.pretrained_weights, cfg.backbone_which)
    backbone.cuda()
    if cfg.freeze_backbone:
        freeze_backbone(backbone)
    print(f"Backbone embed_dim={embed_dim}, frozen={cfg.freeze_backbone}")

    proj_head = build_projection_head(embed_dim, cfg.proj_hidden_dim, cfg.proj_output_dim).cuda()

    # Data
    print("Loading dataset...")
    data_base = Path(cfg.data_base_path)
    project_dirs = sorted([d for d in data_base.iterdir() if d.is_dir()])
    print(f"Found {len(project_dirs)} projects")

    train_dirs, val_dirs = train_val_split(project_dirs, cfg.val_ratio, cfg.seed)
    print(f"Train: {len(train_dirs)} projects, Val: {len(val_dirs)} projects")

    # Load train samples
    train_samples = []
    for d in tqdm(train_dirs, desc="Loading train projects", file=sys.stdout):
        train_samples.extend(load_project(d))
    print(f"Train samples: {len(train_samples)}")

    id_map, train_labels = build_global_identity_map(train_samples)
    train_dataset = ReIDCropDataset(train_samples, train_labels, transform=get_train_transform(), min_k=cfg.K)
    print(f"Valid projects for sampling: {len(train_dataset.valid_projects)}")
    print(f"Valid identities (>= {cfg.K} samples): {len([v for v in train_dataset.identity_to_indices.values() if len(v) >= cfg.K])}")

    # Compute iterations
    num_valid_identities = len([v for v in train_dataset.identity_to_indices.values() if len(v) >= cfg.K])
    iters_per_epoch = max(1, num_valid_identities // cfg.P)
    total_iters = cfg.num_epochs * iters_per_epoch
    print(f"Iters per epoch: {iters_per_epoch}, Total iters: {total_iters}")

    sampler = PKBatchSampler(train_dataset, P=cfg.P, K=cfg.K, num_batches=total_iters, seed=cfg.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Eval
    evaluator = ReIDEvaluator(
        val_project_dirs=val_dirs,
        eval_every=cfg.eval_every,
        seed=cfg.seed,
        min_k=cfg.K,
    )

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(proj_head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    lr_schedule = CosineScheduler(
        base_value=cfg.lr,
        final_value=cfg.min_lr,
        total_iters=total_iters,
        warmup_iters=cfg.warmup_iters,
        start_warmup_value=0,
    )

    # Loss
    criterion = SupConLoss(temperature=cfg.temperature)

    # Checkpointing
    ckpt_tracker = BestCheckpointTracker(Path(output_dir) / "ckpt", max_keep=cfg.ckpt_max_keep)

    # Training loop
    pbar = tqdm(
        total=total_iters,
        desc="Finetuning",
        unit="iter",
        file=sys.stdout,
        mininterval=300.0,
        dynamic_ncols=False,
        ncols=100,
    )

    for it, (images, labels) in enumerate(train_loader):
        # LR schedule
        lr = float(lr_schedule[it])
        for pg in optimizer.param_groups:
            lr_mult = pg.get("lr_multiplier", 1.0)
            pg["lr"] = lr * lr_mult

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # Forward
        with torch.no_grad() if cfg.freeze_backbone and (cfg.unfreeze_after <= 0 or it < cfg.unfreeze_after) else torch.enable_grad():
            embs = backbone(images)
            embs = embs["x_norm_clstoken"] if isinstance(embs, dict) else embs

        proj = proj_head(embs)
        proj = F.normalize(proj, dim=-1)

        loss = criterion(proj, labels)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if cfg.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(proj_head.parameters(), max_norm=cfg.clip_grad)

        if not math.isfinite(loss.item()):
            print(f"[WARN] NaN/Inf loss at iter {it}, skipping update")
            optimizer.zero_grad(set_to_none=True)
            pbar.update(1)
            continue

        optimizer.step()

        # Maybe unfreeze
        new_opt = maybe_unfreeze(backbone, proj_head, optimizer, it, cfg)
        if new_opt is not None:
            optimizer = new_opt
            # Rebuild LR schedule for remaining iters
            lr_schedule = CosineScheduler(
                base_value=cfg.lr,
                final_value=cfg.min_lr,
                total_iters=total_iters - it,
                warmup_iters=0,
                start_warmup_value=cfg.lr,
            )

        # Log
        if it % 10 == 0:
            log_wandb(run, {
                "train/iter": it,
                "train/loss": loss.item(),
                "train/lr": lr,
            }, step=it)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.6f}"})
        pbar.update(1)

        # Eval + checkpoint
        if cfg.ckpt_every > 0 and it > 0 and it % cfg.ckpt_every == 0:
            metrics = evaluator.maybe_eval(backbone, proj_head, it, run)
            if metrics is not None:
                ckpt_tracker.maybe_save(metrics["mAP"], it, backbone, proj_head, optimizer)

    pbar.close()

    # Final eval
    metrics = evaluator.maybe_eval(backbone, proj_head, total_iters, run)
    if metrics is not None:
        ckpt_tracker.maybe_save(metrics["mAP"], total_iters, backbone, proj_head, optimizer)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
