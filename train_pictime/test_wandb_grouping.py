"""
Throwaway script to test different W&B approaches for plotting
teacher & student metrics on the SAME chart.

Run:  python train_pictime/test_wandb_grouping.py
Then check the W&B run to see which approach gives paired graphs.
"""

import math
import wandb

NUM_STEPS = 50
PROJECT = "wandb-grouping-test"


def fake_metrics(step: int, offset: float = 0.0):
    """Generate fake sinusoidal metrics to distinguish teacher vs student."""
    return {
        "view_top1": 0.7 + 0.1 * math.sin(step / 5) + offset,
        "view_top5": 0.85 + 0.05 * math.sin(step / 7) + offset,
        "mAP": 0.6 + 0.15 * math.sin(step / 4) + offset,
    }


# ── Approach 1: Current approach (suffix /teacher /student) ──────────────
# Expected: SEPARATE charts (this is what we have now and want to fix)
def test_suffix_approach():
    run = wandb.init(project=PROJECT, name="approach1_suffix", reinit=True)
    wandb.define_metric("train/iter")
    wandb.define_metric("*", step_metric="train/iter")

    for step in range(NUM_STEPS):
        t = fake_metrics(step, offset=0.02)
        s = fake_metrics(step, offset=-0.02)
        payload = {"train/iter": step}
        for k, v in t.items():
            payload[f"eval/views/{k}/teacher"] = v
        for k, v in s.items():
            payload[f"eval/views/{k}/student"] = v
        run.log(payload, step=step)

    run.finish()


# ── Approach 2: wandb.plot.line_series ───────────────────────────────────
# Logs a custom Vega chart with both lines on one panel.
def test_line_series_approach():
    run = wandb.init(project=PROJECT, name="approach2_line_series", reinit=True)

    xs = list(range(NUM_STEPS))
    metrics = ["view_top1", "view_top5", "mAP"]

    for metric_name in metrics:
        teacher_ys = [fake_metrics(s, 0.02)[metric_name] for s in xs]
        student_ys = [fake_metrics(s, -0.02)[metric_name] for s in xs]

        wandb.log({
            f"eval/views/{metric_name}": wandb.plot.line_series(
                xs=xs,
                ys=[teacher_ys, student_ys],
                keys=["teacher", "student"],
                title=f"eval/views/{metric_name}",
                xname="train/iter",
            )
        })

    run.finish()


# ── Approach 3: Custom wandb.Table + custom vega chart ───────────────────
# Logs a table per metric, then uses wandb.plot.line to render it.
def test_table_approach():
    run = wandb.init(project=PROJECT, name="approach3_table", reinit=True)

    metrics = ["view_top1", "view_top5", "mAP"]

    for metric_name in metrics:
        table = wandb.Table(columns=["step", "model", metric_name])
        for step in range(NUM_STEPS):
            t_val = fake_metrics(step, 0.02)[metric_name]
            s_val = fake_metrics(step, -0.02)[metric_name]
            table.add_data(step, "teacher", t_val)
            table.add_data(step, "student", s_val)

        wandb.log({
            f"eval/views/{metric_name}": wandb.plot.line(
                table,
                x="step",
                y=metric_name,
                stroke="model",
                title=f"eval/views/{metric_name}",
            )
        })

    run.finish()


# ── Approach 4: Using our actual log_paired / log_paired_variant ──────
def test_actual_log_paired():
    from train_pictime.wandb_logger import log_paired, log_paired_variant, _paired_history
    _paired_history.clear()  # reset accumulated history

    run = wandb.init(project=PROJECT, name="approach4_log_paired", reinit=True)
    wandb.define_metric("train/iter")
    wandb.define_metric("*", step_metric="train/iter")

    for step in range(NUM_STEPS):
        t = fake_metrics(step, offset=0.02)
        s = fake_metrics(step, offset=-0.02)
        log_paired(run, step=step, prefix="eval/views/", teacher_dict=t, student_dict=s)

    run.finish()


def test_actual_log_paired_variant():
    from train_pictime.wandb_logger import log_paired_variant, _paired_history
    _paired_history.clear()

    run = wandb.init(project=PROJECT, name="approach5_log_paired_variant", reinit=True)
    wandb.define_metric("train/iter")
    wandb.define_metric("*", step_metric="train/iter")

    for step in range(NUM_STEPS):
        t_raw = fake_metrics(step, offset=0.02)
        t_ctr = fake_metrics(step, offset=0.04)
        s_raw = fake_metrics(step, offset=-0.02)
        s_ctr = fake_metrics(step, offset=-0.04)
        log_paired_variant(run, step=step, prefix="eval/rank/",
                           teacher_variants={"raw": t_raw, "ctr": t_ctr},
                           student_variants={"raw": s_raw, "ctr": s_ctr})

    run.finish()


if __name__ == "__main__":
    print("=== Approach 4: log_paired (incremental line_series) ===")
    test_actual_log_paired()

    print("=== Approach 5: log_paired_variant (incremental line_series) ===")
    test_actual_log_paired_variant()

    print("\nDone. Check W&B project 'wandb-grouping-test' to compare.")
