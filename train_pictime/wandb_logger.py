from __future__ import annotations
import os
from typing import Any, Dict, Optional

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


def _arch_to_tag(arch: str) -> str:
    # cfg.student.arch is typically: vit_large / vit_base / vit_small
    a = (arch or "").lower()
    if "large" in a:
        return "vitl"
    if "base" in a:
        return "vitb"
    if "small" in a:
        return "vits"
    return a.replace("_", "")


def make_run_name(cfg, prefix: str = "pictime") -> str:
    arch_tag = _arch_to_tag(cfg.student.arch)
    ps = int(cfg.student.patch_size)
    bs = int(cfg.train.batch_size_per_gpu)
    lc = int(cfg.crops.local_crops_number)
    lr = float(cfg.optim.lr)
    return f"{prefix}_{arch_tag}{ps}_bs{bs}_lc{lc}_lr{lr:g}"


def init_wandb(cfg, output_dir: str, run_name: Optional[str] = None) -> Any:
    if wandb is None:
        return None

    from omegaconf import OmegaConf

    project = os.environ.get("WANDB_PROJECT", "person-reid-dinov3")
    entity = os.environ.get("WANDB_ENTITY", None)

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        dir=output_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    return run


def log_wandb(run: Any, metrics: Dict[str, float], step: int) -> None:
    if run is None:
        return
    run.log(metrics, step=step)
