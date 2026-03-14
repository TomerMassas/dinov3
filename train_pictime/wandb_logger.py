from __future__ import annotations
import os
from typing import Any, Dict, Optional
from typing import Iterable, Tuple
import torch  # add this import at top
import wandb



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
    wandb.define_metric("train/iter")
    wandb.define_metric("*", step_metric="train/iter")
    return run


def log_wandb(run: Any, metrics: Dict[str, float], step: int) -> None:
    if run is None:
        return
    run.log(metrics, step=step)



def _as_float(v):
    # Accept tensors, numpy scalars, ints, floats
    if torch.is_tensor(v):
        # only log scalars
        return float(v.detach().item()) if v.numel() == 1 else None
    try:
        return float(v)
    except Exception:
        return None


def prefix_dict(d: Dict[str, Any], prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        fv = _as_float(v)
        if fv is not None:
            out[f"{prefix}{k}"] = fv
    return out


def log_prefixed(run: Any, step: int, items: Iterable[Tuple[str, Dict[str, Any]]]) -> None:
    """
    items: iterable of (prefix, metrics_dict)
    Logs everything as ONE wandb.log call (best practice).
    """
    if run is None:
        return
    merged: Dict[str, float] = {}
    for prefix, d in items:
        merged.update(prefix_dict(d, prefix))
    run.log(merged, step=step)


def log_prefixed_variant(
    run: Any,
    step: int,
    prefix: str,
    variant_to_dict: Dict[str, Dict[str, Any]],
) -> None:
    """
    Logs metrics so that variants (raw/ctr) share the same base name and differ by suffix.
    Example key: f"{prefix}{metric_name}/{variant}"
    """
    if run is None:
        return
    merged: Dict[str, float] = {}
    for variant, d in variant_to_dict.items():
        for k, v in d.items():
            fv = _as_float(v)
            if fv is not None:
                merged[f"{prefix}{k}/{variant}"] = fv
    run.log(merged, step=step)


def log_paired(
    run: Any,
    step: int,
    prefix: str,
    teacher_dict: Dict[str, Any],
    student_dict: Dict[str, Any],
) -> None:
    """
    Log teacher & student metrics so they appear on the SAME W&B graph.
    Key format: {prefix}{metric}/teacher and {prefix}{metric}/student
    W&B auto-groups these under {prefix}{metric}.
    """
    if run is None:
        return
    merged: Dict[str, float] = {}
    for k, v in teacher_dict.items():
        fv = _as_float(v)
        if fv is not None:
            merged[f"{prefix}{k}/teacher"] = fv
    for k, v in student_dict.items():
        fv = _as_float(v)
        if fv is not None:
            merged[f"{prefix}{k}/student"] = fv
    if merged:
        run.log(merged, step=step)


def log_paired_variant(
    run: Any,
    step: int,
    prefix: str,
    teacher_variants: Dict[str, Dict[str, Any]],
    student_variants: Dict[str, Dict[str, Any]],
) -> None:
    """
    Like log_paired but with raw/ctr sub-variants.
    Key format: {prefix}{metric}/{variant}/teacher
    """
    if run is None:
        return
    merged: Dict[str, float] = {}
    for variant, d in teacher_variants.items():
        for k, v in d.items():
            fv = _as_float(v)
            if fv is not None:
                merged[f"{prefix}{k}/{variant}/teacher"] = fv
    for variant, d in student_variants.items():
        for k, v in d.items():
            fv = _as_float(v)
            if fv is not None:
                merged[f"{prefix}{k}/{variant}/student"] = fv
    if merged:
        run.log(merged, step=step)