from __future__ import annotations

from typing import Dict
import torch
import torch.nn.functional as F


def embedding_variance_and_effective_rank(E: torch.Tensor, center_and_renorm: bool = False) -> Dict[str, float]:
    """
    E: [N, D] embeddings (often already L2-normalized). CPU preferred.
    If center_and_renorm=True, applies: E <- normalize(E - mean(E)).
    Returns variance stats + effective rank measures computed from covariance eigenvalues.
    """
    assert E.ndim == 2, f"Expected [N, D], got {tuple(E.shape)}"
    if E.is_cuda:
        E = E.cpu()

    E = E.float()

    if center_and_renorm:
        E = F.normalize(E - E.mean(dim=0, keepdim=True), dim=-1)

    N, D = E.shape

    # Per-dimension std
    dim_std = E.std(dim=0, unbiased=False)
    dim_std_sorted, _ = torch.sort(dim_std)

    out: Dict[str, float] = {"dim_std_mean": float(dim_std.mean().item()),
                            "dim_std_median": float(dim_std.median().item()),
                            "dim_std_p05": float(dim_std_sorted[int(0.05 * (D - 1))].item()),
                            "dim_std_p95": float(dim_std_sorted[int(0.95 * (D - 1))].item()),}

    # Covariance eigenvalues (note: covariance itself centers again; that's fine)
    X = E - E.mean(dim=0, keepdim=True)
    C = (X.T @ X) / max(N - 1, 1)

    try:
        evals = torch.linalg.eigvalsh(C).clamp_min(0.0)
    except torch._C._LinAlgError:
        out.update({"eff_rank_pr": 0.0, "eff_rank_entropy": 0.0, "top1_var_ratio": 0.0})
        return out
    evals, _ = torch.sort(evals, descending=True)

    s1 = float(evals.sum().item())
    s2 = float((evals * evals).sum().item())

    if s1 <= 0 or s2 <= 0:
        out.update({"eff_rank_pr": 0.0, "eff_rank_entropy": 0.0, "top1_var_ratio": 0.0})
        return out

    out["eff_rank_pr"] = float((s1 * s1) / s2)

    p = (evals / evals.sum()).clamp_min(1e-12)
    out["eff_rank_entropy"] = float(torch.exp(-(p * p.log()).sum()).item())

    out["top1_var_ratio"] = float((evals[0] / evals.sum()).item())
    return out