import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    from hdbscan import HDBSCAN



def cluster_project(embeddings_path, min_cluster_size, min_samples):
    """Load embeddings and run HDBSCAN. Returns (result_dict, stats_dict)."""
    data = np.load(embeddings_path)
    filenames = data["filenames"]
    bbox_indices = data["bbox_indices"]
    embeddings = data["embeddings"]

    n = len(embeddings)
    if n == 0:
        return {}, {"n_clusters": 0, "n_noise": 0, "n_total": 0}

    if n < 2:
        labels = np.array([-1])
    else:
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(embeddings)

    # Build output dict grouped by filename
    result = {}
    for i in range(n):
        fname = str(filenames[i])
        if fname not in result:
            result[fname] = []
        result[fname].append({
            "bbox_index": int(bbox_indices[i]),
            "cluster_id": int(labels[i]),
        })

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    return result, {"n_clusters": n_clusters, "n_noise": n_noise, "n_total": n}


def save_clusters(project_dir, clusters_dict):
    """Atomic save of clusters.json."""
    path = os.path.join(project_dir, CLUSTERS_FILENAME)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(clusters_dict, f)
    os.replace(tmp, path)


def main():
    project_dirs = sorted(e.path for e in os.scandir(DATASET_ROOT) if e.is_dir())

    skipped, processed, errors = 0, 0, 0
    total_clusters, total_noise, total_embs = 0, 0, 0

    for project_dir in tqdm(project_dirs, desc="Clustering"):
        emb_path = os.path.join(project_dir, EMBEDDINGS_FILENAME)
        clusters_path = os.path.join(project_dir, CLUSTERS_FILENAME)

        if not os.path.exists(emb_path):
            continue
        if os.path.exists(clusters_path) and not FORCE:
            skipped += 1
            continue

        try:
            result, stats = cluster_project(emb_path, MIN_CLUSTER_SIZE, MIN_SAMPLES)
            save_clusters(project_dir, result)
            processed += 1
            total_clusters += stats["n_clusters"]
            total_noise += stats["n_noise"]
            total_embs += stats["n_total"]
        except Exception as e:
            tqdm.write(f"Error in {project_dir}: {e}")
            errors += 1

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}, Errors: {errors}")
    print(f"Total embeddings: {total_embs}, Clusters: {total_clusters}, Noise points: {total_noise}")



EMBEDDINGS_FILENAME = "embeddings.npz"
CLUSTERS_FILENAME = "clusters.json"
DATASET_ROOT = Path("/data/AI/Tomer/person_reid/dataset_utils/dataset_finetune/Portraits[26]")
# DATASET_ROOT = Path("/data/AI/Tomer/UI_dataset_view/data") # for testing
MIN_CLUSTER_SIZE = 3
MIN_SAMPLES = None
FORCE = False
if __name__ == "__main__":
    main()