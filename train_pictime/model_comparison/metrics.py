"""Metrics for the comparison.

A — embedding quality (clustering-independent, vs GT labels): silhouette (cosine),
    mAP + Rank-1/5 (query/gallery ReID protocol).
B — clustering quality (predicted clusters vs GT, per project): mean fracturing,
    homogeneity / completeness / V-measure / ARI, pairwise precision / recall / F1.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    silhouette_score,
)


# ---------------------------------------------------------------------------
# A — embedding quality
# ---------------------------------------------------------------------------

def silhouette_cosine(embs: np.ndarray, labels: np.ndarray, max_samples: int, seed: int) -> float:
    """Silhouette with cosine metric (both models comparable: old raw-2048 cosine
    == euclidean-on-normalized; new is already L2-normed). NaN if degenerate."""
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
        return float("nan")
    if max_samples and len(embs) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(embs), max_samples, replace=False)
        embs, labels = embs[idx], labels[idx]
        if len(np.unique(labels)) < 2:
            return float("nan")
    return float(silhouette_score(embs, labels, metric="cosine"))


def compute_cmc_map(query_embs, query_labels, gallery_embs, gallery_labels) -> dict:
    """CMC (Rank-1/5/10) + mAP. Embeddings are L2-normalized here, then cosine
    similarity. (Copied from reid_evaluator.compute_cmc_map.)"""
    import torch.nn.functional as F
    q = F.normalize(torch.as_tensor(query_embs, dtype=torch.float32), dim=1)
    g = F.normalize(torch.as_tensor(gallery_embs, dtype=torch.float32), dim=1)
    ql = torch.as_tensor(query_labels)
    gl = torch.as_tensor(gallery_labels)

    sim = q @ g.T
    indices = sim.argsort(dim=1, descending=True)
    sorted_labels = gl.unsqueeze(0).expand_as(sim).gather(1, indices)
    matches = sorted_labels == ql.unsqueeze(1)

    Q = q.shape[0]
    cmc = torch.zeros(g.shape[0])
    all_ap = []
    for i in range(Q):
        mp = matches[i].nonzero(as_tuple=False).squeeze(1)
        if mp.numel() == 0:
            continue
        cmc[mp[0].item():] += 1
        n_correct = mp.numel()
        precisions = torch.arange(1, n_correct + 1, dtype=torch.float32) / (mp.float() + 1)
        all_ap.append(precisions.mean().item())

    cmc = cmc / Q
    return {
        "rank1": cmc[0].item() if len(cmc) else 0.0,
        "rank5": cmc[min(4, len(cmc) - 1)].item() if len(cmc) > 4 else (cmc[-1].item() if len(cmc) else 0.0),
        "mAP": (sum(all_ap) / len(all_ap)) if all_ap else 0.0,
    }


def query_gallery_map(embs: np.ndarray, labels: np.ndarray, seed: int) -> dict:
    """Pool across projects with global identity labels; one random query per
    identity (>=2 samples), rest gallery. Singletons stay as gallery distractors."""
    rng = np.random.default_rng(seed)
    q_idx, g_idx = [], []
    for lab in np.unique(labels):
        members = np.where(labels == lab)[0]
        if len(members) >= 2:
            q = rng.choice(members)
            q_idx.append(q)
            g_idx.extend([m for m in members if m != q])
        else:
            g_idx.extend(members.tolist())
    if not q_idx:
        return {"rank1": float("nan"), "rank5": float("nan"), "mAP": float("nan")}
    return compute_cmc_map(embs[q_idx], labels[q_idx], embs[g_idx], labels[g_idx])


# ---------------------------------------------------------------------------
# B — clustering quality (per project, gt has no -1; pred may have -1)
# ---------------------------------------------------------------------------

def _cluster_prf(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    """Cluster-level precision/recall (NOT pairwise). Each predicted cluster is
    matched to its dominant (majority) GT identity, then:
        precision(c) = |c ∩ dominant| / |c|            (cluster purity)
        recall(c)    = |c ∩ dominant| / |dominant|     (how much of that identity
                                                        this cluster captured)
    Averaged (macro) over predicted clusters. A merged cluster (2 people) → low
    precision; a fractured identity (split across clusters) → low recall on each
    piece. pred has no -1 here (filtered upstream)."""
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    gt_sizes = {int(g): int((gt == g).sum()) for g in np.unique(gt)}
    precisions, recalls = [], []
    for c in np.unique(pred):
        members = gt[pred == c]
        vals, counts = np.unique(members, return_counts=True)
        top = int(counts.argmax())
        dom, dom_count = int(vals[top]), int(counts[top])
        precisions.append(dom_count / len(members))
        recalls.append(dom_count / gt_sizes[dom])
    p = float(np.mean(precisions)) if precisions else 0.0
    r = float(np.mean(recalls)) if recalls else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1


def clustering_metrics(gt: np.ndarray, pred: np.ndarray) -> dict | None:
    """All clustering-vs-GT metrics for one project.

    Predicted noise (-1) is IGNORED entirely: those crops are dropped first and
    every metric is computed on the assigned crops only (so noise neither helps
    nor hurts any score). Returns None if the model assigned no crop.
    """
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    keep = pred != -1
    gt, pred = gt[keep], pred[keep]
    if len(gt) == 0:
        return None

    hom, comp, vms = homogeneity_completeness_v_measure(gt, pred)
    ari = adjusted_rand_score(gt, pred)
    # fracturing: per GT identity, # distinct predicted clusters (no -1 left)
    fracs = [int(len(np.unique(pred[gt == g]))) for g in np.unique(gt)]
    p_, r_, f_ = _cluster_prf(gt, pred)
    return {
        "homogeneity": hom, "completeness": comp, "v_measure": vms, "ari": ari,
        "fracturing_mean": float(np.mean(fracs)) if fracs else 0.0,
        "fracturing_perfect_frac": float(np.mean([f == 1 for f in fracs])) if fracs else 0.0,
        "cluster_precision": p_, "cluster_recall": r_, "cluster_f1": f_,
        "n_gt_clusters": int(len(np.unique(gt))),
        "n_pred_clusters": int(len(np.unique(pred))),
        # predicted clusters - true identities. 0 = right count; <0 = under-cluster
        # (merges people); >0 = over-cluster (fractures). Averaged across projects.
        "cluster_count_delta": int(len(np.unique(pred)) - len(np.unique(gt))),
    }