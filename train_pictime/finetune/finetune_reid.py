import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np

from dinov3.configs import setup_config, setup_job
from dinov3.configs.config import DinoV3SetupArgs
from dinov3.train.ssl_meta_arch import SSLMetaArch
from dinov3.checkpointer import load_checkpoint
from dinov3.train.cosine_lr_scheduler import CosineScheduler

from train_pictime.wandb_logger import init_wandb, log_wandb
from train_pictime.finetune.supcon_loss import SupConLoss
from train_pictime.finetune.reid_dataset import (
    load_index, build_global_identity_map, ReIDCropDataset,
    PKBatchSampler, train_val_split, get_train_transform,
)
from train_pictime.finetune.reid_evaluator import ReIDEvaluator
from train_pictime.run_name import arch_to_tag

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

    DCP requires the FSDP2-wrapped SSLMetaArch to load. We use that only to
    read the weights, then copy the full (unsharded) state dict into a fresh,
    *unwrapped* backbone with fp32 params. Finetune then runs with autocast(bf16)
    for compute speed while params/grads/optimizer state stay fp32 (standard
    mixed-precision recipe — avoids update swamping on small LR_backbone).
    """
    from dinov3.models import build_model_from_cfg
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions

    dummy_dir = "/tmp/finetune_dummy"
    Path(dummy_dir).mkdir(parents=True, exist_ok=True)
    setup_args = DinoV3SetupArgs(
        config_file=pretrain_cfg_path,
        output_dir=dummy_dir,
        opts=[],
    )
    cfg = setup_config(setup_args, strict_cfg=False)

    # Build FSDP2-wrapped model on meta → materialize on CUDA (same as pretrain script)
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

    load_checkpoint(ckpt_path, model=model)
    print(f"Loaded checkpoint from {ckpt_path}")

    # Extract full (unsharded) state dict from the wrapped backbone
    wrapped_backbone = (model.teacher if which == "teacher" else model.student)["backbone"]
    full_sd = get_model_state_dict(
        wrapped_backbone,
        options=StateDictOptions(full_state_dict=True, cpu_offload=False),
    )

    # Defensive: strip wrapper prefixes that can leak through (AC / torch.compile)
    prefix_strip = ["_orig_mod.", "_checkpoint_wrapped_module."]
    cleaned_sd = {}
    for k, v in full_sd.items():
        new_k = k
        for p in prefix_strip:
            new_k = new_k.replace(p, "")
        cleaned_sd[new_k] = v

    # Free the FSDP2-wrapped model before building the fresh one
    del model, wrapped_backbone, full_sd
    torch.cuda.empty_cache()

    # Fresh, unwrapped backbone — no FSDP / no compile / no AC, fp32 params on CUDA
    fresh_backbone, embed_dim = build_model_from_cfg(cfg, only_teacher=True)
    fresh_backbone.to_empty(device="cuda")
    fresh_backbone.load_state_dict(cleaned_sd, strict=True)  # shouts on any key mismatch

    # Loud assertion: if for any reason params aren't fp32, stop now rather than
    # silently finetune with the wrong dtype
    param_dtypes = {p.dtype for p in fresh_backbone.parameters()}
    if param_dtypes != {torch.float32}:
        raise RuntimeError(
            f"Expected all backbone params to be float32, got dtypes: {param_dtypes}"
        )

    import shutil
    shutil.rmtree(dummy_dir, ignore_errors=True)

    return fresh_backbone, embed_dim


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
        {"params": unfrozen_params, "lr": cfg.lr_backbone, "lr_multiplier": cfg.lr_backbone / cfg.lr},
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

        path = self.ckpt_dir / f"ckpt_iter{iteration}_sil{metric_value:.4f}.pt"
        torch.save({
            "iteration": iteration,
            "silhouette": metric_value,
            "backbone_state_dict": backbone.state_dict(),
            "proj_head_state_dict": proj_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, path)
        self.entries.append((metric_value, path))
        print(f"[Checkpoint] Saved {path.name} (silhouette={metric_value:.4f})")


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

    # W&B — derive arch tag from pretrain config
    pretrain_cfg = OmegaConf.load(pretrain_cfg_path)
    arch_tag = arch_to_tag(pretrain_cfg.student.arch) + str(int(pretrain_cfg.student.patch_size))
    batch_size = cfg.P * cfg.K
    frozen_tag = "_frozen" if cfg.freeze_backbone and cfg.unfreeze_after <= 0 else ""
    tag_suffix = f"_{cfg.experiment_tag}" if cfg.get("experiment_tag") else ""
    run_name = f"finetune_reid_{arch_tag}_bs{batch_size}{frozen_tag}{tag_suffix}"
    run = init_wandb(
                        cfg,
                        output_dir=output_dir,
                        run_name=run_name,
                        project=cfg.wandb_project,
                        group=cfg.get("experiment_group"),
                    )

    # Model
    print("Loading pretrained backbone...")
    backbone, embed_dim = load_backbone(pretrain_cfg_path, cfg.pretrained_weights, cfg.backbone_which)
    backbone.cuda()
    if cfg.freeze_backbone:
        freeze_backbone(backbone)
    print(f"Backbone embed_dim={embed_dim}, frozen={cfg.freeze_backbone}")

    proj_head = build_projection_head(embed_dim, cfg.proj_hidden_dim, cfg.proj_output_dim).cuda()

    # Data — load from pre-built index (created by build_index.py)
    print("Loading index...")
    index_path = Path(cfg.data_base_path) / "reid_index.npz"
    image_paths, bboxes, project_ids, cluster_ids = load_index(index_path)
    ########################################################################################## TODO comment out
    # tmp_trip = 10000
    # image_paths, bboxes, project_ids, cluster_ids = image_paths[:tmp_trip], bboxes[:tmp_trip], project_ids[:tmp_trip], cluster_ids[:tmp_trip]
    ##############################################################################################################################
    print(f"Total samples: {len(image_paths)}")

    # Split by project
    unique_projects = sorted(set(project_ids))
    train_project_ids, val_project_ids = train_val_split(unique_projects, cfg.val_ratio, cfg.seed)
    train_mask = np.isin(project_ids, train_project_ids)
    val_mask = np.isin(project_ids, val_project_ids)
    print(f"Train: {len(train_project_ids)} projects ({train_mask.sum()} samples), "
          f"Val: {len(val_project_ids)} projects ({val_mask.sum()} samples)")

    # Train dataset
    train_labels = build_global_identity_map(project_ids[train_mask], cluster_ids[train_mask])
    train_dataset = ReIDCropDataset(
                                    image_paths[train_mask], bboxes[train_mask], project_ids[train_mask],
                                    train_labels, transform=get_train_transform(), min_k=cfg.K,
                                )
    print(f"Valid projects for sampling: {len(train_dataset.valid_projects)}")
    print(f"Valid identities (>= {cfg.K} samples): "
          f"{len([v for v in train_dataset.identity_to_indices.values() if len(v) >= cfg.K])}")

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
        image_paths=image_paths[val_mask],
        bboxes=bboxes[val_mask],
        project_ids=project_ids[val_mask],
        cluster_ids=cluster_ids[val_mask],
        seed=cfg.seed,
        min_k=cfg.K,
        silhouette_max_samples=cfg.silhouette_max_samples,
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

        # Forward — autocast(bf16) for matmul speed; params + grads stay fp32
        backbone_frozen = cfg.freeze_backbone and (cfg.unfreeze_after <= 0 or it < cfg.unfreeze_after)
        grad_ctx = torch.no_grad() if backbone_frozen else torch.enable_grad()
        with grad_ctx, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            embs = backbone(images)
            embs = embs["x_norm_clstoken"] if isinstance(embs, dict) else embs

        proj = proj_head(embs.float())  # bf16 → fp32 before proj head
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
            metrics_log = {
                "train/iter": it,
                "train/loss": loss.item(),
                "train/lr": lr,
            }
            if len(optimizer.param_groups) > 1:
                metrics_log["train/lr_backbone"] = optimizer.param_groups[1]["lr"]
            log_wandb(run, metrics_log, step=it)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.6f}"})
        pbar.update(1)

        # Eval + checkpoint
        if cfg.ckpt_every > 0 and it > 0 and it % cfg.ckpt_every == 0:
            metrics = evaluator.maybe_eval(backbone, proj_head, it, run)
            if metrics is not None and "silhouette" in metrics:
                ckpt_tracker.maybe_save(metrics["silhouette"], it, backbone, proj_head, optimizer)

    pbar.close()

    # Final eval
    metrics = evaluator.maybe_eval(backbone, proj_head, total_iters, run)
    if metrics is not None and "silhouette" in metrics:
        ckpt_tracker.maybe_save(metrics["silhouette"], total_iters, backbone, proj_head, optimizer)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
