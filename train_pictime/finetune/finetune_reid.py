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
from pytorch_metric_learning.losses import ArcFaceLoss
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
    full_sd = get_model_state_dict(wrapped_backbone,
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
        raise RuntimeError(f"Expected all backbone params to be float32, got dtypes: {param_dtypes}")

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


def maybe_unfreeze(backbone, proj_head, criterion, optimizer, iteration, cfg):
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

    # Rebuild optimizer. For ArcFace: 3 groups (head / classifier / backbone).
    # For SupCon: 2 groups (head+criterion / backbone) — criterion has no params.
    unfrozen_params = [p for p in backbone.parameters() if p.requires_grad]
    if cfg.get("loss", "supcon") == "arcface" and list(criterion.parameters()):
        head_params = list(proj_head.parameters())
        classifier_params = list(criterion.parameters())
        classifier_lr = float(cfg.get("arcface_classifier_lr", cfg.lr))
        classifier_wd = float(cfg.get("arcface_classifier_wd", cfg.weight_decay))
        new_optimizer = torch.optim.AdamW([
            {"params": head_params, "lr": cfg.lr,
             "weight_decay": cfg.weight_decay, "lr_multiplier": 1.0},
            {"params": classifier_params, "lr": classifier_lr,
             "weight_decay": classifier_wd, "lr_multiplier": classifier_lr / cfg.lr},
            {"params": unfrozen_params, "lr": cfg.lr_backbone,
             "weight_decay": cfg.weight_decay, "lr_multiplier": cfg.lr_backbone / cfg.lr},
        ])
    else:
        head_and_criterion_params = list(proj_head.parameters()) + list(criterion.parameters())
        new_optimizer = torch.optim.AdamW([
            {"params": head_and_criterion_params, "lr": cfg.lr},
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

    def maybe_save(self, metric_value: float, iteration: int, backbone, proj_head, criterion, optimizer):
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
            "criterion_state_dict": criterion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, path)
        self.entries.append((metric_value, path))
        print(f"[Checkpoint] Saved {path.name} (silhouette={metric_value:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_finetune(cfg):
    """Run one finetune training using a fully-resolved OmegaConf cfg.

    Caller must have already called `setup_job()` exactly once for the
    process — distributed init is per-process, not per-run.
    """
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
    index_path = Path(cfg.data_base_path) / cfg.get("reid_index_filename", "reid_index.npz")
    image_paths, bboxes, bbox_indices, project_ids, cluster_ids = load_index(index_path)
    ########################################################################################## TODO comment out
    # tmp_trip = 10000
    # image_paths, bboxes, bbox_indices, project_ids, cluster_ids = image_paths[:tmp_trip], bboxes[:tmp_trip], bbox_indices[:tmp_trip], project_ids[:tmp_trip], cluster_ids[:tmp_trip]
    ##############################################################################################################################

    # Drop cluster_id == -1 globally — HDBSCAN noise (or reviewer-flagged ambiguous);
    # too dirty to train or evaluate on. Keeps vanilla and curriculum on the same
    # identity pool so iter counts and metrics are comparable.
    keep = cluster_ids != -1
    n_dropped = int((~keep).sum())
    image_paths   = image_paths[keep]
    bboxes        = bboxes[keep]
    bbox_indices  = bbox_indices[keep]
    project_ids   = project_ids[keep]
    cluster_ids   = cluster_ids[keep]
    print(f"Filtered {n_dropped} cluster_id=-1 samples globally; {len(image_paths)} samples remain")
    print(f"Total samples: {len(image_paths)}")

    # Split by project
    unique_projects = sorted(set(project_ids))
    train_project_ids, val_project_ids = train_val_split(unique_projects, cfg.val_ratio, cfg.seed)
    train_mask = np.isin(project_ids, train_project_ids)
    val_mask = np.isin(project_ids, val_project_ids)
    print(f"Train: {len(train_project_ids)} projects ({train_mask.sum()} samples), "
          f"Val: {len(val_project_ids)} projects ({val_mask.sum()} samples)")

    # Curriculum config
    curriculum_cfg = cfg.get("curriculum", None)
    curriculum_enabled = bool(curriculum_cfg and curriculum_cfg.get("enabled", False))

    centroid_distances_filename = (curriculum_cfg.centroid_distances_filename if curriculum_enabled else None)

    # Train dataset
    train_labels = build_global_identity_map(project_ids[train_mask], cluster_ids[train_mask])
    num_classes = int(train_labels.max()) + 1  # labels are contiguous 0..N-1 by construction
    print(f"Num classes: {num_classes}")
    train_dataset = ReIDCropDataset(
        image_paths[train_mask], bboxes[train_mask], bbox_indices[train_mask],
        project_ids[train_mask], cluster_ids[train_mask], train_labels,
        transform=get_train_transform(), min_k=cfg.K,
        centroid_distances_filename=centroid_distances_filename,
    )
    print(f"Valid projects for sampling: {len(train_dataset.valid_projects)}")
    print(f"Valid identities (>= {cfg.K} samples): "
          f"{len([v for v in train_dataset.identity_to_indices.values() if len(v) >= cfg.K])}")

    # Compute iterations
    num_valid_identities = len([v for v in train_dataset.identity_to_indices.values() if len(v) >= cfg.K])
    iters_per_epoch = max(1, num_valid_identities // cfg.P)
    total_iters = cfg.num_epochs * iters_per_epoch
    print(f"Iters per epoch: {iters_per_epoch}, Total iters: {total_iters}")

    sampler = PKBatchSampler(
        train_dataset, P=cfg.P, K=cfg.K, num_batches=total_iters, seed=cfg.seed,
        curriculum_p_start=(curriculum_cfg.p_start if curriculum_enabled else 1.0),
        curriculum_p_end=(curriculum_cfg.p_end if curriculum_enabled else 1.0),
        curriculum_end_frac=(curriculum_cfg.end_frac if curriculum_enabled else 0.3),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Eval — val_mask uses the ORIGINAL split (curriculum filter is train-only).
    # Tiered eval: when eval_centroid_distances_filename is set, the evaluator
    # drops cluster_id=-1 (no centroid) and computes per-tier metrics.
    evaluator = ReIDEvaluator(
        image_paths=image_paths[val_mask],
        bboxes=bboxes[val_mask],
        bbox_indices=bbox_indices[val_mask],
        project_ids=project_ids[val_mask],
        cluster_ids=cluster_ids[val_mask],
        seed=cfg.seed,
        min_k=cfg.K,
        silhouette_max_samples=cfg.silhouette_max_samples,
        centroid_distances_filename=cfg.get("eval_centroid_distances_filename", None),
        eval_tiers=cfg.get("eval_tiers", None),
    )

    # Loss
    if cfg.get("loss", "supcon") == "arcface":
        criterion = ArcFaceLoss(
            num_classes=num_classes,
            embedding_size=cfg.proj_output_dim,
            margin=cfg.arcface_m_deg,
            scale=cfg.arcface_s,
        ).cuda()
        print(f"Loss: ArcFace (s={cfg.arcface_s}, m={cfg.arcface_m_deg}°)")
    else:
        criterion = SupConLoss(temperature=cfg.temperature)
        print(f"Loss: SupCon (tau={cfg.temperature})")

    # ArcFace margin warmup: pytorch-metric-learning converts margin to radians in init_margin()
    # and uses self.margin directly each forward — no precomputation, so runtime mutation works.
    arcface_target_margin_rad = None
    if cfg.get("loss", "supcon") == "arcface":
        arcface_target_margin_rad = float(criterion.margin)  # already radians after init
        if cfg.get("arcface_margin_warmup_iters", 0) > 0:
            criterion.margin = 0.0

    # Optimizer + scheduler. For ArcFace we split classifier into its own param group
    # (separate LR and weight_decay from head). criterion.parameters() is empty for SupCon.
    if cfg.get("loss", "supcon") == "arcface" and list(criterion.parameters()):
        head_params = list(proj_head.parameters())
        classifier_params = list(criterion.parameters())
        classifier_lr = float(cfg.get("arcface_classifier_lr", cfg.lr))
        classifier_wd = float(cfg.get("arcface_classifier_wd", cfg.weight_decay))
        optimizer = torch.optim.AdamW([
            {"params": head_params, "lr": cfg.lr,
             "weight_decay": cfg.weight_decay, "lr_multiplier": 1.0},
            {"params": classifier_params, "lr": classifier_lr,
             "weight_decay": classifier_wd, "lr_multiplier": classifier_lr / cfg.lr},
        ])
        print(f"Optimizer: head lr={cfg.lr} wd={cfg.weight_decay} | "
              f"classifier lr={classifier_lr} wd={classifier_wd}")
    else:
        optimizer = torch.optim.AdamW(
            list(proj_head.parameters()) + list(criterion.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

    lr_schedule = CosineScheduler(
        base_value=cfg.lr,
        final_value=cfg.min_lr,
        total_iters=total_iters,
        warmup_iters=cfg.warmup_iters,
        start_warmup_value=0,
    )

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

        # ArcFace margin warmup — ramp from 0 to target over arcface_margin_warmup_iters
        if arcface_target_margin_rad is not None:
            warmup_iters = cfg.get("arcface_margin_warmup_iters", 0)
            if warmup_iters > 0 and it < warmup_iters:
                criterion.margin = (it / warmup_iters) * arcface_target_margin_rad
            else:
                criterion.margin = arcface_target_margin_rad

        loss = criterion(proj, labels)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        head_grad_norm = None
        bb_grad_norm = None
        classifier_grad_norm = None
        if cfg.clip_grad > 0:
            head_grad_norm = torch.nn.utils.clip_grad_norm_(list(proj_head.parameters()), max_norm=cfg.clip_grad)
            backbone_trainable = [p for p in backbone.parameters() if p.requires_grad]
            if backbone_trainable:
                bb_grad_norm = torch.nn.utils.clip_grad_norm_(backbone_trainable, max_norm=cfg.clip_grad)
            classifier_params = [p for p in criterion.parameters() if p.requires_grad]
            if classifier_params:
                classifier_grad_norm = torch.nn.utils.clip_grad_norm_(classifier_params, max_norm=cfg.clip_grad)

        if not math.isfinite(loss.item()):
            print(f"[WARN] NaN/Inf loss at iter {it}, skipping update")
            optimizer.zero_grad(set_to_none=True)
            pbar.update(1)
            continue

        optimizer.step()

        # Maybe unfreeze (scheduler is shared base curve; no rebuild needed)
        new_opt = maybe_unfreeze(backbone, proj_head, criterion, optimizer, it, cfg)
        if new_opt is not None:
            optimizer = new_opt

        # Log
        if it % 10 == 0:
            metrics_log = {
                "train/iter": it,
                "train/loss": loss.item(),
                "train/lr": lr,
            }
            # Curriculum p — duplicates the formula in PKBatchSampler.__iter__; keep aligned.
            if curriculum_enabled:
                ramp_end = max(1, int(total_iters * curriculum_cfg.end_frac))
                if it < ramp_end:
                    cur_p = curriculum_cfg.p_start + (curriculum_cfg.p_end - curriculum_cfg.p_start) * (it / ramp_end)
                else:
                    cur_p = curriculum_cfg.p_end
                metrics_log["train/curriculum_p"] = cur_p
            if arcface_target_margin_rad is not None:
                metrics_log["train/arcface_margin_deg"] = float(np.degrees(criterion.margin))
                # With ArcFace, param_groups[1] is the classifier (frozen path);
                # in Mode-C ArcFace it will be param_groups[2] — revisit then.
                metrics_log["train/lr_classifier"] = optimizer.param_groups[1]["lr"]
            elif len(optimizer.param_groups) > 1:
                metrics_log["train/lr_backbone"] = optimizer.param_groups[1]["lr"]
            if head_grad_norm is not None:
                metrics_log["train/grad_norm_head"] = float(head_grad_norm)
            if bb_grad_norm is not None:
                metrics_log["train/grad_norm_backbone"] = float(bb_grad_norm)
            if classifier_grad_norm is not None:
                metrics_log["train/grad_norm_classifier"] = float(classifier_grad_norm)
            log_wandb(run, metrics_log, step=it)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.6f}"})
        pbar.update(1)

        # Eval + checkpoint
        if cfg.ckpt_every > 0 and it > 0 and it % cfg.ckpt_every == 0:
            metrics = evaluator.maybe_eval(backbone, proj_head, it, run)
            if metrics is not None and "silhouette" in metrics:
                ckpt_tracker.maybe_save(metrics["silhouette"], it, backbone, proj_head, criterion, optimizer)

    pbar.close()

    # Final eval
    metrics = evaluator.maybe_eval(backbone, proj_head, total_iters, run)
    if metrics is not None and "silhouette" in metrics:
        ckpt_tracker.maybe_save(metrics["silhouette"], total_iters, backbone, proj_head, criterion, optimizer)

    if run is not None:
        run.finish()


def main():
    cfg = OmegaConf.load(CFG_PATH)
    setup_job(output_dir=None, seed=cfg.seed)
    run_finetune(cfg)


if __name__ == "__main__":
    main()
