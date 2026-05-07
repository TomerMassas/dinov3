from __future__ import annotations

from typing import Dict, Iterable, Tuple
import torch
import torch.nn.functional as F


def _percentiles(x: torch.Tensor, ps=(0.5, 0.9, 0.95)) -> Dict[str, float]:
    x = x.flatten()
    x_sorted, _ = torch.sort(x)
    n = x_sorted.numel()
    out = {}
    for p in ps:
        idx = int(p * (n - 1))
        out[f"p{int(p*100):02d}"] = float(x_sorted[idx].item())
    return out


def random_pair_cosine_distance_stats(E: torch.Tensor, num_pairs: int = 50000, seed: int = 0) -> Dict[str, float]:
    """
    Uniformity / global scale proxy:
    sample random pairs and compute cosine distance d = 1 - cos.
    """
    if E.is_cuda:
        E = E.cpu()
    E = F.normalize(E.float(), dim=-1)
    N = E.shape[0]

    g = torch.Generator().manual_seed(seed)
    i = torch.randint(0, N, (num_pairs,), generator=g)
    j = torch.randint(0, N, (num_pairs,), generator=g)
    same = (i == j)
    if same.any():
        j[same] = (j[same] + 1) % N

    cos = (E[i] * E[j]).sum(dim=-1).clamp(-1.0, 1.0)
    dist = (1.0 - cos).clamp_min(0.0)

    out = {"rand_dist_mean": float(dist.mean().item()),
           "rand_dist_std": float(dist.std(unbiased=False).item()),}

    out.update({f"rand_dist_{k}": v for k, v in _percentiles(dist, ps=(0.5, 0.9, 0.95)).items()})
    return out


@torch.no_grad()
def knn_k_distance_stats(E: torch.Tensor,
                         ks: Iterable[int] = (1, 5, 10),
                         chunk: int = 512,
                         device: str = "cuda",
                        ) -> Dict[str, float]:
    """
    For each point, compute distance to its k-th nearest neighbor (cosine distance).
    Uses chunked matmul: sims = Q @ E^T, then topk.
    """
    ks = sorted(set(int(k) for k in ks))
    max_k = max(ks)

    # Move to compute device (GPU is much faster for N up to ~20k)
    E_dev = E.to(device, non_blocking=True)
    E_dev = F.normalize(E_dev.float(), dim=-1)  # float32 for stable dot products
    N, D = E_dev.shape
    ET = E_dev.T  # [D, N]

    # We will store kth neighbor distances for each k: [N]
    dk = {k: torch.empty((N,), device="cpu", dtype=torch.float32) for k in ks}

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        Q = E_dev[start:end]                       # [c, D]
        sims = Q @ ET                               # [c, N]

        # mask self-similarities (diagonal block)
        rows = torch.arange(end - start, device=sims.device)
        cols = torch.arange(start, end, device=sims.device)
        sims[rows, cols] = -1e9

        top = torch.topk(sims, k=max_k, dim=1).values.clamp(-1.0, 1.0)  # [c, max_k]
        dist_top = (1.0 - top).clamp_min(0.0)

        for k in ks:
            # k-th nearest => index k-1 in topk list
            dk[k][start:end] = dist_top[:, k - 1].detach().cpu()

    out: Dict[str, float] = {}
    for k in ks:
        x = dk[k]
        out[f"knn_d{k}_mean"] = float(x.mean().item())
        out[f"knn_d{k}_std"] = float(x.std(unbiased=False).item())
        out.update({f"knn_d{k}_{pname}": pval for pname, pval in _percentiles(x, ps=(0.5, 0.9, 0.95)).items()})

    return out


def geometry_pack(E: torch.Tensor,
                  num_pairs: int,
                  ks=(1, 5, 10),
                  device: str = "cuda",
                  center_and_renorm: bool = False,
                 ) -> Dict[str, float]:

    if center_and_renorm:
        if E.is_cuda:
            E = E.cpu()
        E = E.float()
        E = F.normalize(E - E.mean(dim=0, keepdim=True), dim=-1)

    out = {}
    out.update(random_pair_cosine_distance_stats(E, num_pairs=num_pairs, seed=0))
    out.update(knn_k_distance_stats(E, ks=ks, chunk=512, device=device))
    return out