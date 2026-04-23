from __future__ import annotations
import os
from typing import Any, Dict, Optional
from typing import Iterable, Tuple
import torch  # add this import at top
import wandb



def find_wandb_run_id_by_name(run_name: str, project: Optional[str] = None, entity: Optional[str] = None) -> Optional[str]:
    """Query W&B API for a run with the given name. Returns the latest run's ID, or None."""
    api = wandb.Api()
    project = project or os.environ.get("WANDB_PROJECT", "person-reid-dinov3")
    entity = entity or os.environ.get("WANDB_ENTITY", None)
    path = f"{entity}/{project}" if entity else project
    try:
        runs = api.runs(path, filters={"display_name": run_name}, order="-created_at")
        for run in runs:
            print(f"Found W&B run '{run_name}' with id={run.id}")
            return run.id
    except Exception as e:
        print(f"WARNING: failed to query W&B API for run '{run_name}': {e}")
    return None


def init_wandb(cfg, output_dir: str, run_name: Optional[str] = None, resume_id: Optional[str] = None, project: Optional[str] = None, group: Optional[str] = None) -> Any:
    if wandb is None:
        return None

    from omegaconf import OmegaConf

    project = project or os.environ.get("WANDB_PROJECT", "person-reid-dinov3")
    entity = os.environ.get("WANDB_ENTITY", None)

    resume_kwargs = {}
    if resume_id is not None:
        resume_kwargs = {"id": resume_id, "resume": "must"}

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        group=group,
        dir=output_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
        **resume_kwargs,
    )
    wandb.define_metric("train/iter")
    wandb.define_metric("*", step_metric="train/iter")
    return run


def log_wandb(run: Any, metrics: Dict[str, float], step: int) -> None:
    if run is None:
        return
    metrics["train/iter"] = step
    if step % 100 == 0:   # TEMP diagnostic for missing train-loss charts — remove after verification
        print(f"[W&B LOG @ iter {step}] keys={sorted(metrics.keys())}")
    run.log(metrics)



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
    merged["train/iter"] = step
    run.log(merged)


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
    merged["train/iter"] = step
    run.log(merged)


_paired_history: Dict[str, Dict[str, list]] = {}


def log_paired(
    run: Any,
    step: int,
    prefix: str,
    teacher_dict: Dict[str, Any],
    student_dict: Dict[str, Any],
) -> None:
    """
    Log teacher & student metrics on the SAME W&B chart via line_series.
    Accumulates history in _paired_history and re-logs the full chart each call.
    """
    if run is None:
        return
    payload: Dict[str, Any] = {}
    for k in teacher_dict:
        fv_t = _as_float(teacher_dict[k])
        fv_s = _as_float(student_dict.get(k))
        if fv_t is None and fv_s is None:
            continue
        chart_key = f"{prefix}{k}"
        if chart_key not in _paired_history:
            _paired_history[chart_key] = {"steps": [], "teacher": [], "student": []}
        hist = _paired_history[chart_key]
        hist["steps"].append(step)
        hist["teacher"].append(fv_t)
        hist["student"].append(fv_s)
        payload[chart_key] = wandb.plot.line_series(
            xs=hist["steps"],
            ys=[hist["teacher"], hist["student"]],
            keys=["teacher", "student"],
            title=chart_key,
            xname="train/iter",
        )
    if payload:
        payload["train/iter"] = step
        run.log(payload)


def log_paired_variant(
    run: Any,
    step: int,
    prefix: str,
    teacher_variants: Dict[str, Dict[str, Any]],
    student_variants: Dict[str, Dict[str, Any]],
) -> None:
    """
    Like log_paired but with raw/ctr sub-variants.
    One chart per metric, with a line per (variant, model) combo.
    """
    if run is None:
        return
    # Collect all metric names across variants
    all_metrics: set = set()
    for d in list(teacher_variants.values()) + list(student_variants.values()):
        all_metrics.update(d.keys())

    variant_names = list(teacher_variants.keys())
    series_keys = [f"teacher/{v}" for v in variant_names] + [f"student/{v}" for v in variant_names]

    payload: Dict[str, Any] = {}
    for k in all_metrics:
        chart_key = f"{prefix}{k}"
        if chart_key not in _paired_history:
            _paired_history[chart_key] = {"steps": []}
            for sk in series_keys:
                _paired_history[chart_key][sk] = []
        hist = _paired_history[chart_key]
        hist["steps"].append(step)
        for v in variant_names:
            hist[f"teacher/{v}"].append(_as_float(teacher_variants.get(v, {}).get(k)))
            hist[f"student/{v}"].append(_as_float(student_variants.get(v, {}).get(k)))
        payload[chart_key] = wandb.plot.line_series(
            xs=hist["steps"],
            ys=[hist[sk] for sk in series_keys],
            keys=series_keys,
            title=chart_key,
            xname="train/iter",
        )
    if payload:
        payload["train/iter"] = step
        run.log(payload)