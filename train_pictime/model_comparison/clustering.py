"""Per-project clustering for each model.

OLD (faithful production, IdentityClustering body path): AgglomerativeClustering
   distance_threshold=0.85, average linkage, on a precomputed euclidean distance
   matrix of L2-normalized embeddings, then a centroid post-merge (cosine < 0.2).
   NOTE: the centroid-merge replication is best-effort — the production invocation
   for the body modality wasn't fully traced. Set merge_centroid_cosine=None in
   config to disable it if it skews results.

NEW (tuned): HDBSCAN (same params as train_pictime/cluster_embeddings.py).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize

# Prefer the standalone `hdbscan` package (production IdentityClustering uses it):
# sklearn's HDBSCAN has a bug where cluster_selection_epsilon>0 + allow_single_cluster=True
# crashes in traverse_upwards ("only 0-dimensional arrays..."). The hdbscan package
# handles that combination, and avoids sklearn's `copy` FutureWarning spam.
try:
    from hdbscan import HDBSCAN
    _HDBSCAN_PKG = "hdbscan"
except ImportError:
    from sklearn.cluster import HDBSCAN
    _HDBSCAN_PKG = "sklearn"


def cluster_agglomerative(embs, distance_threshold=0.85, linkage="average",
                          merge_centroid_cosine=0.2, **_ignored):
    n = len(embs)
    if n < 2:
        return np.zeros(n, dtype=int)
    v = normalize(embs)                       # production: normalize, then euclidean
    dist = cdist(v, v, metric="euclidean")
    labels = AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage=linkage,
        distance_threshold=distance_threshold,
    ).fit_predict(dist)
    if merge_centroid_cosine is not None:
        labels = _merge_by_centroid(embs, labels, merge_centroid_cosine)
    return labels


def _merge_by_centroid(embs, labels, thresh):
    """Mirror IdentityClustering.merge_clusters: raw-vector centroids, cosine
    distance, pairs below thresh merged into the lower-indexed cluster."""
    uniq = list(np.unique(labels))
    if len(uniq) < 2:
        return labels
    centers = np.array([embs[labels == u].mean(axis=0) for u in uniq])
    d = cosine_distances(centers)
    out = labels.copy()
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            if d[i, j] < thresh:
                out[out == uniq[j]] = uniq[i]
    return out


def cluster_hdbscan(embs, min_cluster_size=3, min_samples=None,
                    cluster_selection_epsilon=0.0, cluster_selection_method="eom",
                    allow_single_cluster=False, metric="euclidean", **_ignored):
    n = len(embs)
    if n < 2:
        return np.array([-1] * n, dtype=int)
    # sklearn's HDBSCAN crashes on epsilon>0 + allow_single_cluster=True — guard with a
    # clear message instead of the cryptic "only 0-dimensional arrays" TypeError.
    if _HDBSCAN_PKG == "sklearn" and cluster_selection_epsilon > 0 and allow_single_cluster:
        raise RuntimeError(
            "sklearn HDBSCAN can't combine cluster_selection_epsilon>0 with "
            "allow_single_cluster=True (known bug). Install the hdbscan package "
            "(`pip install hdbscan`) — it handles this and is what production uses."
        )
    return HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=float(cluster_selection_epsilon),
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=allow_single_cluster,
        metric=metric,
    ).fit_predict(embs)


def cluster(embs, spec: dict):
    """Dispatch on spec['method']."""
    method = spec["method"]
    if method == "agglomerative":
        return cluster_agglomerative(embs, **{k: v for k, v in spec.items() if k != "method"})
    if method == "hdbscan":
        return cluster_hdbscan(embs, **{k: v for k, v in spec.items() if k not in ("method", "dist_func")})
    raise ValueError(f"Unknown clustering method: {method}")