"""Lock the comparison test set = the TOP_N approved projects with the highest
num_crops / num_clusters (mean cluster size) — the galleries with the biggest
per-identity clusters, read from completion_log.json.

Run:
    python3 -m train_pictime.model_comparison.prepare_eval_set
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train_pictime.model_comparison import config as C


def main():
    with open(C.COMPLETION_LOG) as f:
        log = json.load(f)

    # Approved, valid ratio, dedup by project_id (keep highest ratio if repeated).
    by_pid: dict[str, dict] = {}
    skipped_unapproved = skipped_badratio = 0
    for e in log:
        if not e.get("approved", False):
            skipped_unapproved += 1
            continue
        nc, nk = e.get("num_crops", 0), e.get("num_clusters", 0)
        if not nk or nk <= 0:
            skipped_badratio += 1
            continue
        ratio = nc / nk
        pid = str(e["project_id"])
        if pid not in by_pid or ratio > by_pid[pid]["ratio"]:
            by_pid[pid] = {"project_id": pid, "num_crops": nc, "num_clusters": nk, "ratio": ratio}

    ranked = sorted(by_pid.values(), key=lambda r: r["ratio"], reverse=True)
    print(f"Approved+valid projects: {len(ranked)} "
          f"(skipped {skipped_unapproved} unapproved, {skipped_badratio} num_clusters<=0)")

    # Keep only projects whose GT + detections actually exist on disk, until TOP_N.
    selected, missing = [], []
    for r in ranked:
        proj = C.DATASET_ROOT / r["project_id"]
        if (proj / C.CLUSTERS_FIXED_FILENAME).exists() and (proj / C.DETECTIONS_FILENAME).exists():
            selected.append(r)
            if len(selected) >= C.TOP_N:
                break
        else:
            missing.append(r["project_id"])
    if missing:
        print(f"[WARN] {len(missing)} high-ranked projects skipped (no clusters_fixed/detections on disk), "
              f"e.g. {missing[:5]}")
    if len(selected) < C.TOP_N:
        print(f"[WARN] only {len(selected)} projects available (< TOP_N={C.TOP_N})")

    lo, hi = selected[-1]["ratio"], selected[0]["ratio"]
    print(f"Selected {len(selected)} projects; mean-cluster-size ratio range "
          f"[{lo:.1f} .. {hi:.1f}] (crops/clusters)")

    payload = {
        "project_ids": [r["project_id"] for r in selected],
        "count": len(selected),
        "selection": "top_by_crops_per_cluster",
        "top_n": C.TOP_N,
        "ratio_range": [lo, hi],
        "details": selected,
    }
    tmp = str(C.NEW_PROJECTS_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    Path(tmp).replace(C.NEW_PROJECTS_FILE)
    print(f"Locked {len(selected)} projects -> {C.NEW_PROJECTS_FILE}")


if __name__ == "__main__":
    main()