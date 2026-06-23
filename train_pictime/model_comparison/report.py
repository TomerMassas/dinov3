"""Re-render the comparison table + plots from the existing results.json —
without re-embedding or re-clustering. Use this to iterate on the table layout /
legend / displayed metrics (edit DISPLAY, GROUP_TITLE, METRIC_DESC in evaluate.py)
and just re-run this.

    python3 -m train_pictime.model_comparison.report

NOTE: results.json must already exist from an evaluate.py run that produced the
CURRENT metric keys (cluster_precision/cluster_recall, noise-excluded). If it's
from an older run, run evaluate.py once to refresh it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train_pictime.model_comparison import config as C
from train_pictime.model_comparison.evaluate import write_report, DISPLAY, REPORT_GROUPS


def main():
    results_path = C.OUTPUT_DIR / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"No {results_path} — run evaluate.py once first.")
    with open(results_path) as f:
        results = json.load(f)

    # Guard against a stale results.json (older metric set) — fail with guidance
    # rather than a cryptic KeyError mid-render.
    missing = [key for grp in REPORT_GROUPS for _, key in DISPLAY[grp]
               if key not in results.get(grp, {}).get("old", {})]
    if missing:
        raise RuntimeError(
            f"results.json is stale — missing metric(s) {missing}. "
            f"Run evaluate.py once to recompute it with the current metric set."
        )

    write_report(results, results["n_crops"], results["n_identities"])


if __name__ == "__main__":
    main()