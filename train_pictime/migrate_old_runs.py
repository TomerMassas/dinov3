"""
Migrate old W&B runs to the new logging format.

Old format:
  - Train: flat keys like `ibot_loss`, `dino_global_crops_loss`, `koleo_loss`
  - Eval:  flat keys like `eval/proto/teacher/X`, `eval/rank/teacher/X/raw`, etc.

New format:
  - Train: `train/ibot_loss`, `train/dino_global`, `train/koleo`
  - Eval:  `wandb.plot.line_series` charts grouping teacher/student on the same graph

Usage:
  python train_pictime/migrate_old_runs.py --project person-reid-dinov3 --run-name pictime_vits16_bs16_lc8_lr0.0005
  # or pass --run-id <id> directly
"""
from __future__ import annotations
import argparse
import re
from collections import defaultdict

import wandb


# ── Train key renames (flat → flat) ──────────────────────────────────────
TRAIN_RENAMES = {
    "ibot_loss": "train/ibot_loss",
    "dino_global_crops_loss": "train/dino_global",
    "koleo_loss": "train/koleo",
}


# ── Eval paired groups (old flat keys → line_series charts) ──────────────
# Old keys follow these patterns (prefix = "eval/"):
#   log_prefixed:         eval/<group>/teacher/<metric>  eval/<group>/student/<metric>
#   log_prefixed_variant: eval/<group>/teacher/<metric>/<variant>  (variant = raw|ctr)
#
# New log_paired produces one line_series chart per metric at key = eval/<group>/<metric>
# New log_paired_variant produces chart at key = eval/<group>/<metric> with lines teacher/raw, teacher/ctr, student/raw, student/ctr

PAIRED_GROUPS = {"proto", "views"}  # use log_paired (teacher + student)
VARIANT_GROUPS = {"rank", "geom"}   # use log_paired_variant (teacher/raw, teacher/ctr, student/raw, student/ctr)
EVAL_PREFIX = "eval/"


def fetch_runs(api, project, entity, run_name=None, run_id=None):
    """Return list of matching runs."""
    if run_id:
        return [api.run(f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}")]
    filters = {"display_name": run_name} if run_name else {}
    runs = api.runs(f"{entity}/{project}" if entity else project, filters=filters)
    return list(runs)


def classify_eval_key(key):
    """
    Classify an old eval key. Returns (group_type, group, metric, role, variant) or None.

    Examples:
      eval/proto/teacher/utilization → ('paired', 'proto', 'utilization', 'teacher', None)
      eval/rank/teacher/eff_rank/raw → ('variant', 'rank', 'eff_rank', 'teacher', 'raw')
    """
    if not key.startswith(EVAL_PREFIX):
        return None
    rest = key[len(EVAL_PREFIX):]
    parts = rest.split("/")

    if len(parts) < 3:
        return None

    group = parts[0]
    role = parts[1]  # teacher or student
    if role not in ("teacher", "student"):
        return None

    if group in VARIANT_GROUPS and len(parts) == 4:
        metric = parts[2]
        variant = parts[3]
        return ("variant", group, metric, role, variant)
    elif group in PAIRED_GROUPS and len(parts) == 3:
        metric = parts[2]
        return ("paired", group, metric, role, None)
    elif group in VARIANT_GROUPS and len(parts) == 3:
        # non-centered fallback (no variant suffix)
        metric = parts[2]
        return ("paired", group, metric, role, None)
    elif len(parts) == 3:
        metric = parts[2]
        return ("paired", group, metric, role, None)

    return None


def migrate_run(run):
    print(f"\n{'='*60}")
    print(f"Migrating run: {run.name} (id={run.id})")
    print(f"{'='*60}")

    # First pass: discover keys from run summary (no full scan needed)
    all_keys = set(run.summary.keys())
    # Also check history keys from the run object
    if hasattr(run, 'history_keys') and callable(run.history_keys):
        try:
            hk = run.history_keys()
            if isinstance(hk, dict) and "keys" in hk:
                all_keys.update(hk["keys"].keys())
        except Exception:
            pass

    train_keys_found = {old: new for old, new in TRAIN_RENAMES.items() if old in all_keys}
    if train_keys_found:
        print(f"  Train keys to rename: {train_keys_found}")

    # Build eval key classifications
    eval_classifications = {}
    for key in sorted(all_keys):
        c = classify_eval_key(key)
        if c:
            eval_classifications[key] = c

    if eval_classifications:
        print(f"  Eval keys to migrate: {len(eval_classifications)}")

    if not train_keys_found and not eval_classifications:
        print("  Nothing to migrate.")
        return

    # Build the set of keys we actually need to fetch
    needed_keys = set(train_keys_found.keys()) | set(eval_classifications.keys()) | {"train/iter", "_step"}
    print(f"  Streaming history (fetching {len(needed_keys)} keys only)...")

    # ── Stream history and build charts + train renames in one pass ──
    charts = defaultdict(lambda: {"steps": []})
    row_count = 0

    for row in run.scan_history(keys=list(needed_keys), page_size=10000):
        row_count += 1
        if row_count % 10000 == 0:
            print(f"    ...processed {row_count} rows")

        step = row.get("train/iter") or row.get("_step")
        if step is None:
            continue

        # Train renames — collect into per-metric series for line_series charts
        if train_keys_found:
            for old_key, new_key in train_keys_found.items():
                val = row.get(old_key)
                if val is not None:
                    chart = charts[new_key]
                    if "value" not in chart:
                        chart["value"] = []
                    chart["steps"].append(step)
                    chart["value"].append(float(val))

        # Eval charts
        charts_updated = set()
        for key, classification in eval_classifications.items():
            val = row.get(key)
            if val is None:
                continue

            group_type, group, metric, role, variant = classification
            chart_key = f"{EVAL_PREFIX}{group}/{metric}"

            if group_type == "paired":
                series_name = role
            else:
                series_name = f"{role}/{variant}"

            chart = charts[chart_key]
            if series_name not in chart:
                chart[series_name] = []

            charts_updated.add(chart_key)
            chart[series_name].append(float(val))

        for ck in charts_updated:
            charts[ck]["steps"].append(step)

    print(f"  Streamed {row_count} rows total")

    # Pad shorter series with None
    for chart_key, chart in charts.items():
        n_steps = len(chart["steps"])
        for k, v in chart.items():
            if k == "steps":
                continue
            while len(v) < n_steps:
                v.append(None)

    # ── Resume run and log new data ──
    resumed = wandb.init(
        project=run.project,
        entity=run.entity,
        id=run.id,
        resume="must",
    )

    # Log all charts (both train renames and eval)
    if charts:
        print(f"  Logging {len(charts)} line_series charts...")
        for chart_key, chart in sorted(charts.items()):
            steps = chart["steps"]
            if not steps:
                continue
            series_names = [k for k in chart if k != "steps"]
            ys = [chart[s] for s in series_names]

            last_step = steps[-1]
            line_chart = wandb.plot.line_series(
                xs=steps,
                ys=ys,
                keys=series_names,
                title=chart_key,
                xname="train/iter",
            )
            resumed.log({chart_key: line_chart}, step=last_step)
            print(f"    {chart_key}: {len(steps)} points, series={series_names}")

    resumed.finish()
    print(f"  Done migrating {run.name}")


def main():
    parser = argparse.ArgumentParser(description="Migrate old W&B runs to new logging format")
    parser.add_argument("--project", default="person-reid-dinov3")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--run-name", default=None, help="Run display name to find")
    parser.add_argument("--run-id", default=None, help="Exact run ID")
    parser.add_argument("--dry-run", action="store_true", help="Just print what would be done")
    args = parser.parse_args()

    api = wandb.Api()
    runs = fetch_runs(api, args.project, args.entity, args.run_name, args.run_id)
    print(f"Found {len(runs)} matching run(s)")

    if args.dry_run:
        for r in runs:
            print(f"  Would migrate: {r.name} (id={r.id})")
        return

    for r in runs:
        migrate_run(r)


def dry_run_debug():
    """Dry run for PyCharm debug mode — prints what would be migrated without writing anything."""
    project = "person-reid-dinov3"
    run_name = "pictime_vits16_bs16_lc8_lr0.0005"

    api = wandb.Api()
    runs = fetch_runs(api, project, entity=None, run_name=run_name)
    print(f"Found {len(runs)} matching run(s)")

    for run in runs:
        print(f"\n{'='*60}")
        print(f"Run: {run.name} (id={run.id})")
        print(f"{'='*60}")

        history = list(run.scan_history())
        if not history:
            print("  No history found.")
            continue
        print(f"  History rows: {len(history)}")

        all_keys = set()
        for row in history:
            all_keys.update(row.keys())

        # Train keys
        train_keys_found = {old: new for old, new in TRAIN_RENAMES.items() if old in all_keys}
        print(f"\n  Train keys to rename ({len(train_keys_found)}):")
        for old, new in train_keys_found.items():
            print(f"    {old} → {new}")

        # Eval keys
        print(f"\n  Eval keys to migrate:")
        for key in sorted(all_keys):
            c = classify_eval_key(key)
            if c:
                print(f"    {key} → {c}")

        # Show all keys for reference
        print(f"\n  All logged keys ({len(all_keys)}):")
        for key in sorted(all_keys):
            print(f"    {key}")


if __name__ == "__main__":
    import sys
    sys.argv = ["migrate_old_runs.py", "--run-name", "pictime_vits16_bs16_lc8_lr0.0005"]
    main()