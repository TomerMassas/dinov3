import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm


def load_clusters(project_dir):
    """Resolve clusters file: clusters_fixed.json wins, else per-model clusters file.
    Returns parsed dict, or None if neither file exists.
    """
    fixed_path = os.path.join(project_dir, CLUSTERS_FIXED_FILENAME)
    if os.path.exists(fixed_path):
        with open(fixed_path) as f:
            return json.load(f)
    cl_path = os.path.join(project_dir, CLUSTERS_FILENAME)
    if os.path.exists(cl_path):
        with open(cl_path) as f:
            return json.load(f)
    return None


def build_centroids_for_project(project_dir):
    """Compute per-cluster centroid distances for one project.

    Returns dict keyed by cluster_id (str) -> list of {filename, bbox_index, distance},
    sorted ascending by distance. Excludes cluster_id == -1.

    Crashes loudly if any cluster entry can't be matched to an embedding row.
    """
    emb_path = os.path.join(project_dir, EMBEDDINGS_FILENAME)
    data = np.load(emb_path)
    filenames = data["filenames"]
    bbox_indices = data["bbox_indices"]
    embeddings = data["embeddings"]

    # (filename, bbox_index) -> row index
    row_lookup = {}
    for i in range(len(filenames)):
        key = (str(filenames[i]), int(bbox_indices[i]))
        row_lookup[key] = i

    clusters = load_clusters(project_dir)
    if clusters is None:
        raise RuntimeError(f"No clusters file in {project_dir}")

    # Group cluster entries by cluster_id
    cluster_entries = {}  # cluster_id (int) -> list of (filename, bbox_index)
    for fname, entries in clusters.items():
        for entry in entries:
            cid = int(entry["cluster_id"])
            if cid == -1:
                continue
            cluster_entries.setdefault(cid, []).append((str(fname), int(entry["bbox_index"])))

    result = {}
    for cid, entries in cluster_entries.items():
        rows = []
        for key in entries:
            if key not in row_lookup:
                raise RuntimeError(
                    f"Cluster entry {key} (cluster_id={cid}) not found in embeddings "
                    f"of {project_dir}"
                )
            rows.append(row_lookup[key])

        embs = embeddings[rows]                              # (M, D), L2-normed
        centroid = embs.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        distances = 1.0 - embs @ centroid                    # cosine distance

        sorted_entries = sorted(
            (
                {"filename": entries[i][0], "bbox_index": entries[i][1], "distance": float(distances[i])}
                for i in range(len(entries))
            ),
            key=lambda e: e["distance"],
        )
        result[str(cid)] = sorted_entries

    return result


def save_distances(project_dir, distances_dict):
    """Atomic save of crop_distances_*.json."""
    path = os.path.join(project_dir, DISTANCES_FILENAME)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(distances_dict, f)
    os.replace(tmp, path)


def main():
    project_dirs = sorted(e.path for e in os.scandir(DATASET_ROOT) if e.is_dir())

    skipped, processed, errors = 0, 0, 0
    total_clusters, total_crops = 0, 0

    for project_dir in tqdm(project_dirs, desc="Building centroids"):
        emb_path = os.path.join(project_dir, EMBEDDINGS_FILENAME)
        out_path = os.path.join(project_dir, DISTANCES_FILENAME)

        if not os.path.exists(emb_path):
            continue
        if os.path.exists(out_path) and not FORCE:
            skipped += 1
            continue

        try:
            result = build_centroids_for_project(project_dir)
            save_distances(project_dir, result)
            processed += 1
            total_clusters += len(result)
            total_crops += sum(len(v) for v in result.values())
        except Exception as e:
            tqdm.write(f"Error in {project_dir}: {e}")
            errors += 1

    print(f"\nDone. Processed: {processed}, Skipped (already done): {skipped}, Errors: {errors}")
    print(f"Total clusters: {total_clusters}, Total crops: {total_crops}")


MODEL_SOURCE = "v11_ckpt13k"  # "foundation_b16" | "v11_ckpt13k"

# Maps each backbone source to its input/output filenames — NEVER use ternaries here.
EMBEDDINGS_FILENAME_BY_MODEL = {
    "foundation_b16": "embeddings.npz",
    "v11_ckpt13k":    "embeddings_v2.npz",
}
CLUSTERS_FILENAME_BY_MODEL = {
    "foundation_b16": "clusters.json",
    "v11_ckpt13k":    "clusters_v2.json",
}
DISTANCES_FILENAME_BY_MODEL = {
    "foundation_b16": "crop_distances.json",
    "v11_ckpt13k":    "crop_distances_v2.json",
}
EMBEDDINGS_FILENAME = EMBEDDINGS_FILENAME_BY_MODEL[MODEL_SOURCE]
CLUSTERS_FILENAME = CLUSTERS_FILENAME_BY_MODEL[MODEL_SOURCE]
DISTANCES_FILENAME = DISTANCES_FILENAME_BY_MODEL[MODEL_SOURCE]

# Reviewer-corrected clusters take precedence — never overwrite this file's labels.
CLUSTERS_FIXED_FILENAME = "clusters_fixed.json"

DATASET_ROOT = Path("/data/AI/Tomer/person_reid/dataset_utils/dataset_finetune/Portraits[26]")
FORCE = False

if __name__ == "__main__":
    main()
