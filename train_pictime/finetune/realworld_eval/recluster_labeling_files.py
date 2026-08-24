"""Re-cluster the labeling files from the cached embeddings — no GPU, no model, no images.

`prep_labeling_files.py` already wrote `embeddings_<tag>.npz` per project (filenames,
bbox_indices, embeddings). Clustering and the centroid distances are pure functions of those
three arrays, so changing the HDBSCAN params does NOT require re-embedding: this script loads
each cache, re-runs HDBSCAN with the current config, and rewrites only

    clusters_<tag>.json        {filename: [{bbox_index, cluster_id}, ...]}
    crop_distances_<tag>.json  {cluster_id: [{filename, bbox_index, distance}, ...]}

`embeddings_<tag>.npz` is never touched. Minutes instead of hours, and any param set is
seconds away, so trying one is cheap and reverting is cheap.

The HDBSCAN params and the distance math are IMPORTED from cluster_test_set /
prep_labeling_files rather than restated, so this script cannot drift from the pipeline that
produced the caches. (That pulls torch in as a side effect of those imports, but no model is
built and no checkpoint is read.)

Projects with a `clusters_fixed.json` are skipped — reviewer truth is never overwritten.
Projects with no cache are reported, not silently passed over: they were never embedded, and
only `prep_labeling_files.py` can fix that.

Edit the HDBSCAN_* constants in realworld_eval/config.py, then:

    python3 -m train_pictime.finetune.realworld_eval.recluster_labeling_files
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from tqdm import tqdm

from train_pictime.finetune.realworld_eval.config import (
    DATASET_ROOT, FINETUNE_VERSION_DIR, HDBSCAN_ALLOW_SINGLE_CLUSTER,
    HDBSCAN_CLUSTER_SELECTION_EPSILON, HDBSCAN_CLUSTER_SELECTION_METHOD, HDBSCAN_METRIC,
    HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES,
)
from train_pictime.finetune.realworld_eval.cluster_test_set import cluster_embeddings_hdbscan
from train_pictime.finetune.realworld_eval.prep_labeling_files import (
    CLUSTERS_FIXED_FILENAME, _save_json, compute_crop_distances,
)

# Restrict to specific project ids for a first look; empty list = every project in DATASET_ROOT.
ONLY_PROJECTS: list[str] = [] #["52472251", "52544230"]


# Max tolerated |‖v‖ - 1| before a cache is refused. See assert_unit_norm.
NORM_TOLERANCE = 1e-4


# ---------------------------------------------------------------------------
# Cache -> labels
# ---------------------------------------------------------------------------

def load_cache(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """filenames were saved with dtype=object, so allow_pickle is required to read them back."""
    with np.load(path, allow_pickle=True) as data:
        filenames = [str(f) for f in data["filenames"]]
        bbox_indices = data["bbox_indices"]
        embeddings = data["embeddings"].astype(np.float32)
    if not (len(filenames) == len(bbox_indices) == len(embeddings)):
        raise ValueError(f"ragged cache: {len(filenames)} filenames, {len(bbox_indices)} "
                         f"bbox_indices, {len(embeddings)} embeddings")
    return filenames, bbox_indices, embeddings


def assert_unit_norm(embeddings: np.ndarray) -> float:
    """Refuse a cache that is not L2-normalized. Returns the observed max drift.

    HDBSCAN_CLUSTER_SELECTION_EPSILON is an ABSOLUTE distance, and the value in config is the
    euclidean equivalent of a cosine threshold: d_euclid = sqrt(2 * d_cos). That identity holds
    only on unit vectors, so clustering an unnormalized cache would silently apply a different
    threshold than the one the params were tuned at — wrong output, no error. Fail loudly.
    """
    if len(embeddings) == 0:
        return 0.0
    drift = float(np.max(np.abs(np.linalg.norm(embeddings, axis=1) - 1.0)))
    if drift > NORM_TOLERANCE:
        raise ValueError(f"embeddings are not L2-normalized (max |norm-1| = {drift:.2e} > "
                         f"{NORM_TOLERANCE:.0e}); EPSILON={HDBSCAN_CLUSTER_SELECTION_EPSILON} "
                         f"encodes a cosine threshold and is only valid on unit vectors")
    return drift


def clusters_dict_from_arrays(filenames: list[str],
                              bbox_indices: np.ndarray,
                              labels: np.ndarray,
                             ) -> dict:
    """Same schema and grouping as cluster_test_set.build_clusters_dict, from cache arrays.

    prep_labeling_files writes filenames/bbox_indices in embedding order (built from
    valid_idx in order), so iterating them here reproduces that function's output exactly.
    """
    result: dict[str, list[dict]] = {}
    for i, filename in enumerate(filenames):
        result.setdefault(filename, []).append({"bbox_index": int(bbox_indices[i]),
                                                "cluster_id": int(labels[i]),
                                               })
    return result


# ---------------------------------------------------------------------------
# Before / after reporting
# ---------------------------------------------------------------------------

def stats_from_labels(labels) -> tuple[int, int]:
    """(n_clusters, n_noise). cluster_id -1 is noise and is never counted as a cluster."""
    ids = [int(label) for label in labels]
    return len(set(ids) - {-1}), sum(1 for i in ids if i == -1)


def stats_from_clusters_file(path: Path) -> tuple[int, int] | None:
    """Same stats read out of an existing clusters_<tag>.json, or None if unreadable.

    Read BEFORE the rewrite: both param sets share the `v52` tag, so this per-project delta is
    the only comparison available without pointing the config at a differently-named dir.
    """
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return None
    ids = [int(entry["cluster_id"]) for rows in data.values() for entry in rows]
    return len(set(ids) - {-1}), sum(1 for i in ids if i == -1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tag = Path(FINETUNE_VERSION_DIR).name.lower()
    emb_name = f"embeddings_{tag}.npz"
    cl_name = f"clusters_{tag}.json"
    dist_name = f"crop_distances_{tag}.json"

    print(f"Dataset:  {DATASET_ROOT}")
    print(f"Reading:  {emb_name}   (never modified)")
    print(f"Writing:  {cl_name} | {dist_name}")
    print(f"\nHDBSCAN   min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}  "
          f"min_samples={HDBSCAN_MIN_SAMPLES}  "
          f"epsilon={HDBSCAN_CLUSTER_SELECTION_EPSILON}  "
          f"method={HDBSCAN_CLUSTER_SELECTION_METHOD}  "
          f"allow_single={HDBSCAN_ALLOW_SINGLE_CLUSTER}  metric={HDBSCAN_METRIC}")
    if HDBSCAN_METRIC == "euclidean":
        print(f"          epsilon {HDBSCAN_CLUSTER_SELECTION_EPSILON} == cosine distance "
              f"{HDBSCAN_CLUSTER_SELECTION_EPSILON ** 2 / 2:.6g} on L2-normalized embeddings\n")

    if ONLY_PROJECTS:
        project_dirs = [os.path.join(DATASET_ROOT, pid) for pid in ONLY_PROJECTS]
        print(f"ONLY_PROJECTS is set — {len(project_dirs)} project(s) only\n")
    else:
        project_dirs = sorted(e.path for e in os.scandir(DATASET_ROOT) if e.is_dir())

    processed = skipped_fixed = no_cache = errors = 0
    total_crops = total_clusters = total_noise = 0
    was_clusters = was_noise = 0          # totals over the projects we actually rewrote
    max_drift = 0.0

    for project_dir in tqdm(project_dirs, desc="Projects"):
        pdir = Path(project_dir)
        if (pdir / CLUSTERS_FIXED_FILENAME).exists():
            skipped_fixed += 1
            continue

        emb_p, cl_p, dist_p = pdir / emb_name, pdir / cl_name, pdir / dist_name
        if not emb_p.exists():
            no_cache += 1
            continue

        try:
            filenames, bbox_indices, embeddings = load_cache(emb_p)
            if len(embeddings) == 0:
                no_cache += 1
                continue
            max_drift = max(max_drift, assert_unit_norm(embeddings))

            before = stats_from_clusters_file(cl_p)
            labels = cluster_embeddings_hdbscan(embeddings)
            n_clusters, n_noise = stats_from_labels(labels)

            _save_json(cl_p, clusters_dict_from_arrays(filenames, bbox_indices, labels))
            _save_json(dist_p, compute_crop_distances(embeddings, filenames, bbox_indices, labels))

            if before is not None:
                was_clusters += before[0]
                was_noise += before[1]
                if before != (n_clusters, n_noise):
                    tqdm.write(f"[{pdir.name}] clusters {before[0]} -> {n_clusters}, "
                               f"noise {before[1]} -> {n_noise}  (of {len(labels)} crops)")
            total_crops += len(labels)
            total_clusters += n_clusters
            total_noise += n_noise
            processed += 1
        except Exception as e:
            tqdm.write(f"[{pdir.name}] ERROR: {e!r}\n{traceback.format_exc()}")
            errors += 1

    print(f"\n===== Summary =====")
    print(f"Re-clustered:                    {processed}")
    print(f"Skipped (clusters_fixed present): {skipped_fixed}")
    print(f"Skipped (no cached embeddings):   {no_cache}   <- run prep_labeling_files for these")
    print(f"Errors:                           {errors}")
    if processed > 0:
        print(f"\nMean clusters / project: {total_clusters / processed:.2f}")
        print(f"Mean crops / project:    {total_crops / processed:.1f}")
        print(f"Noise fraction:          {total_noise / max(total_crops, 1):.3f}")
        print(f"Max |norm-1| seen:       {max_drift:.2e}  (tolerance {NORM_TOLERANCE:.0e})")
        print(f"\nBefore -> after over the {processed} rewritten projects:")
        print(f"  clusters: {was_clusters} -> {total_clusters}")
        print(f"  noise:    {was_noise} -> {total_noise}")


if __name__ == "__main__":
    main()
