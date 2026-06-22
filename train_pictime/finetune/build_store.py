"""Build / incrementally update the project-keyed finetune store.

The store (reid_store.pkl) replaces the flat reid_index_*.npz. It is a dict
keyed by project_id; each project's record holds its crop rows + per-crop
centroid distances + a `labeled` flag:

    { "meta":     {embeddings_filename, clusters_base_filename, ...},
      "projects": { "<pid>": {labeled, filenames, bboxes, bbox_indices,
                              cluster_ids, distances}, ... } }

A record is ALWAYS a full rebuild from its source clusters file (never a
patch) so reviewer deletions / reassignments are handled automatically.

Two phases, both idempotent (safe to re-run):
  1. BASELINE  — any project not yet in the store gets a record from the
                 HDBSCAN baseline (clusters_v3.json), labeled=False.
  2. INCORPORATE — projects in labeled_projects.json that aren't yet
                 labeled=True get rebuilt from clusters_fixed.json (labeled=True).
                 Labels are append-only, so this is a simple set difference.

Distances are computed from the per-project embeddings (which never change).
A per-project crop_distances JSON is ALSO written (same content) so the
eval / fracturing paths that read those files keep working unchanged.

Run (base build is implicit on first run; incremental on later runs):
    python3 -m train_pictime.finetune.build_store
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm


DATASET_ROOT = Path("/data/AI/Tomer/person_reid/dataset_utils/dataset_finetune/Portraits[26]")
STORE_PATH = DATASET_ROOT / "reid_store.pkl"
LABELED_PROJECTS_FILE = Path(__file__).parent / "labeled_projects.json"

EMBEDDINGS_FILENAME = "embeddings_v3.npz"
CLUSTERS_BASE_FILENAME = "clusters_v3.json"        # HDBSCAN baseline (unlabeled)
CLUSTERS_FIXED_FILENAME = "clusters_fixed.json"    # reviewer truth (labeled)
DETECTIONS_FILENAME = "detections.json"
DISTANCES_OUT_FILENAME = "crop_distances_v3.json"  # written for eval/fracturing compat

MODEL_SOURCE = "v17_ckpt19750"


# ---------------------------------------------------------------------------
# Per-project record
# ---------------------------------------------------------------------------

def _compute_distances(embeddings, cluster_ids):
    """Per-crop cosine distance to its own cluster centroid. cluster_id == -1
    rows get NaN (no centroid; they're dropped before curriculum anyway)."""
    distances = np.full(len(cluster_ids), np.nan, dtype=np.float32)
    for cid in set(int(c) for c in cluster_ids):
        if cid == -1:
            continue
        mask = cluster_ids == cid
        embs = embeddings[mask]                       # [m, D], L2-normed
        centroid = embs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        distances[mask] = 1.0 - embs @ centroid       # cosine distance
    return distances


def build_project_record(project_dir: Path, clusters_filename: str, labeled: bool):
    """Build one project's record from detections + the given clusters file +
    embeddings. Returns (record_dict, distances_json) or (None, None) to skip.

    Raises on a genuine inconsistency (a clustered crop missing from embeddings)
    — caught per-project by the caller so one bad project doesn't kill the run.
    """
    det_path = project_dir / DETECTIONS_FILENAME
    cl_path = project_dir / clusters_filename
    emb_path = project_dir / EMBEDDINGS_FILENAME
    if not det_path.exists() or not cl_path.exists() or not emb_path.exists():
        return None, None

    with open(det_path) as f:
        detections = json.load(f)
    with open(cl_path) as f:
        clusters = json.load(f)

    emb_data = np.load(emb_path)
    emb_lookup = {(str(emb_data["filenames"][i]), int(emb_data["bbox_indices"][i])): i
                  for i in range(len(emb_data["filenames"]))}
    embeddings = emb_data["embeddings"]

    filenames, bboxes, bbox_indices, cluster_ids, emb_rows = [], [], [], [], []
    for fname, entries in clusters.items():
        det_list = detections.get(fname)
        if det_list is None:
            continue
        for entry in entries:
            bbox_idx = int(entry["bbox_index"])
            cid = int(entry["cluster_id"])
            if bbox_idx >= len(det_list):
                continue
            key = (str(fname), bbox_idx)
            if cid != -1 and key not in emb_lookup:
                raise RuntimeError(f"{project_dir.name}: clustered crop {key} missing from embeddings")
            filenames.append(str(fname))
            bboxes.append(det_list[bbox_idx]["bbox"])
            bbox_indices.append(bbox_idx)
            cluster_ids.append(cid)
            emb_rows.append(emb_lookup.get(key, -1))

    if not filenames:
        return None, None

    cluster_ids = np.array(cluster_ids, dtype=np.int32)
    # Gather embeddings aligned to our rows (rows with cid==-1 may have emb_row -1;
    # _compute_distances ignores them via the cid==-1 mask before indexing).
    row_embs = np.zeros((len(filenames), embeddings.shape[1]), dtype=embeddings.dtype)
    for i, r in enumerate(emb_rows):
        if r >= 0:
            row_embs[i] = embeddings[r]
    distances = _compute_distances(row_embs, cluster_ids)

    record = {
        "labeled":      labeled,
        "filenames":    np.array(filenames, dtype=object),
        "bboxes":       np.array(bboxes, dtype=np.float32),
        "bbox_indices": np.array(bbox_indices, dtype=np.int32),
        "cluster_ids":  cluster_ids,
        "distances":    distances,
    }

    # Same distances as a per-project JSON (grouped by cluster, sorted ascending)
    # for the eval / fracturing paths that still read crop_distances files.
    distances_json: dict[str, list] = {}
    for i in range(len(filenames)):
        cid = int(cluster_ids[i])
        if cid == -1:
            continue
        distances_json.setdefault(str(cid), []).append(
            {"filename": filenames[i], "bbox_index": int(bbox_indices[i]), "distance": float(distances[i])}
        )
    for cid in distances_json:
        distances_json[cid].sort(key=lambda e: e["distance"])

    return record, distances_json


def _save_distances_json(project_dir: Path, distances_json: dict):
    path = project_dir / DISTANCES_OUT_FILENAME
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(distances_json, f)
    os.replace(tmp, path)


def _save_store(store: dict):
    tmp = str(STORE_PATH) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, STORE_PATH)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if STORE_PATH.exists():
        with open(STORE_PATH, "rb") as f:
            store = pickle.load(f)
        print(f"Loaded store with {len(store['projects'])} projects from {STORE_PATH}")
    else:
        store = {"meta": {"embeddings_filename": EMBEDDINGS_FILENAME,
                          "clusters_base_filename": CLUSTERS_BASE_FILENAME,
                          "model_source": MODEL_SOURCE},
                 "projects": {}}
        print("No existing store — base build from scratch.")

    projects = store["projects"]
    all_dirs = [e for e in sorted(os.scandir(DATASET_ROOT), key=lambda e: e.name) if e.is_dir()]

    # --- Phase 1: baseline (clusters_v3) for any project not yet in the store ---
    n_base, n_base_err, n_base_skip = 0, 0, 0
    for entry in tqdm(all_dirs, desc="Phase 1: baseline"):
        pid = entry.name
        if pid in projects:
            continue
        try:
            rec, dist_json = build_project_record(Path(entry.path), CLUSTERS_BASE_FILENAME, labeled=False)
            if rec is None:
                n_base_skip += 1
                continue
            projects[pid] = rec
            _save_distances_json(Path(entry.path), dist_json)
            n_base += 1
        except Exception as e:
            tqdm.write(f"[baseline ERROR] {pid}: {e!r}")
            n_base_err += 1
    print(f"Phase 1: +{n_base} baseline records, {n_base_skip} skipped, {n_base_err} errors")

    # --- Phase 2: incorporate newly-labeled projects (delta) ---
    n_inc, n_inc_err, n_inc_skip = 0, 0, 0
    if LABELED_PROJECTS_FILE.exists():
        with open(LABELED_PROJECTS_FILE) as f:
            payload = json.load(f)
        labeled_list = payload if isinstance(payload, list) else payload.get("project_ids", [])
        delta = [pid for pid in labeled_list if not projects.get(pid, {}).get("labeled", False)]
        print(f"Labeled projects: {len(labeled_list)}, already incorporated: "
              f"{len(labeled_list) - len(delta)}, delta to process: {len(delta)}")
        for pid in tqdm(delta, desc="Phase 2: incorporate"):
            proj_dir = DATASET_ROOT / pid
            try:
                rec, dist_json = build_project_record(proj_dir, CLUSTERS_FIXED_FILENAME, labeled=True)
                if rec is None:
                    n_inc_skip += 1
                    continue
                projects[pid] = rec                  # replaces the baseline record (handles deletions)
                _save_distances_json(proj_dir, dist_json)
                n_inc += 1
            except Exception as e:
                tqdm.write(f"[incorporate ERROR] {pid}: {e!r}")
                n_inc_err += 1
        print(f"Phase 2: +{n_inc} incorporated, {n_inc_skip} skipped, {n_inc_err} errors")
    else:
        print(f"No {LABELED_PROJECTS_FILE} — skipping Phase 2 (run scan_labeled_projects.py first).")

    _save_store(store)
    n_labeled = sum(1 for r in projects.values() if r["labeled"])
    print(f"\nSaved store: {len(projects)} projects ({n_labeled} labeled, "
          f"{len(projects) - n_labeled} baseline) -> {STORE_PATH}")


if __name__ == "__main__":
    main()