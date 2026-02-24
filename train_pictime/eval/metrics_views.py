from __future__ import annotations

from typing import Dict, Literal, Tuple, List

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from train_pictime.eval.embed import load_paths, get_backbone


def _two_view_transforms(img_size: int = 224):
    """
    Two *light* stochastic views. We want invariance checks, not full DINO multi-crop.
    """
    common_norm = transforms.Normalize(
                                        mean=(0.485, 0.456, 0.406),
                                        std=(0.229, 0.224, 0.225),
                                    )

    #TODO You can tune these later. Keep it mild for person reID.
    view = transforms.Compose([
                                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor(),
                                common_norm,
                            ])
    return view, view


@torch.no_grad()
def extract_two_view_embeddings(
    model,
    paths: List[str],
    which: Literal["teacher", "student"],
    batch_size: int = 64,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: (E_a, E_b) both [N, D] float32 on CPU, L2-normalized.
    """
    tfm_a, tfm_b = _two_view_transforms(img_size=224)

    backbone = get_backbone(model, which=which)
    was_training = backbone.training
    backbone.eval()
    try:
        outs_a: List[torch.Tensor] = []
        outs_b: List[torch.Tensor] = []

        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]

            imgs_a = torch.stack([tfm_a(Image.open(p).convert("RGB")) for p in batch_paths]).to(device, non_blocking=True)
            imgs_b = torch.stack([tfm_b(Image.open(p).convert("RGB")) for p in batch_paths]).to(device, non_blocking=True)

            out_a = backbone(imgs_a)
            out_b = backbone(imgs_b)

            emb_a = out_a["x_norm_clstoken"] if isinstance(out_a, dict) else out_a
            emb_b = out_b["x_norm_clstoken"] if isinstance(out_b, dict) else out_b

            emb_a = F.normalize(emb_a.float(), dim=-1)
            emb_b = F.normalize(emb_b.float(), dim=-1)

            outs_a.append(emb_a.cpu())
            outs_b.append(emb_b.cpu())

        return torch.cat(outs_a, dim=0), torch.cat(outs_b, dim=0)
    finally:
        backbone.train(was_training)

def uniformity_random_pairs(E: torch.Tensor, num_pairs: int = 20000, seed: int = 0) -> dict:
    """
    Uniformity proxy: random-pair cosine distance stats.
    Returns mean/p50/p95 of (1 - cosine similarity).
    we want this to be reasonably high (close to 1) and not too concentrated (p50 not too close to 0).
    """
    E = F.normalize(E, dim=-1)
    N = E.shape[0]
    g = torch.Generator()
    g.manual_seed(seed)

    i = torch.randint(0, N, (num_pairs,), generator=g)
    j = torch.randint(0, N, (num_pairs,), generator=g)
    # avoid i==j (optional)
    same = (i == j)
    if same.any():
        j[same] = (j[same] + 1) % N

    cos = (E[i] * E[j]).sum(dim=-1)            # [num_pairs]
    dist = 1.0 - cos                            # cosine distance

    p50 = dist.kthvalue(int(0.50 * dist.numel())).values.item()
    p95 = dist.kthvalue(int(0.95 * dist.numel())).values.item()
    return {
        "uniform_dist_mean": float(dist.mean().item()),
        "uniform_dist_p50": float(p50),
        "uniform_dist_p95": float(p95),
    }

def alignment_cosine_distance(Ea: torch.Tensor, Eb: torch.Tensor) -> float:
    """
    Mean cosine distance between paired views: mean(1 - cos(Ea[i], Eb[i])).
    Lower is better.
    """
    Ea = F.normalize(Ea, dim=-1)
    Eb = F.normalize(Eb, dim=-1)
    pos_cos = (Ea * Eb).sum(dim=-1)  # [N]
    return float((1.0 - pos_cos).mean().item())

def view_consistency_topk(Ea: torch.Tensor, Eb: torch.Tensor, ks=(1, 5), chunk: int = 512) -> Dict[str, float]:
    """
    For each i, find nearest neighbors of Ea[i] among all Eb[*] by cosine similarity.
    Report Top-1 and Top-5 accuracy.
    top1 is strict view consistency: the two views of the same item should be closest.
    top5 is more relaxed: the two views should be among the 5 closest.
    both are in [0, 1], higher is better.
    """
    Ea = F.normalize(Ea, dim=-1)
    Eb = F.normalize(Eb, dim=-1)
    N = Ea.shape[0]
    assert Eb.shape[0] == N

    # Compute similarities in chunks to keep memory bounded.
    topk_hits = {k: 0 for k in ks}
    Eb_t = Eb.T  # [D, N]

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        sims = Ea[start:end] @ Eb_t  # [chunk, N]
        # get top max_k indices
        max_k = max(ks)
        top_idx = torch.topk(sims, k=max_k, dim=1).indices  # [chunk, max_k]

        gt = torch.arange(start, end).unsqueeze(1)  # [chunk, 1]
        for k in ks:
            hit = (top_idx[:, :k] == gt).any(dim=1).sum().item()
            topk_hits[k] += int(hit)

    return {f"top{k}": topk_hits[k] / N for k in ks}

def pos_neg_gap_hard(Ea: torch.Tensor, Eb: torch.Tensor, chunk: int = 512) -> dict:
    """
    Computes hard-negative gap stats: s_pos - max_{j!=i} s_neg.
    s_pos is the cosine similarity of the paired views (Ea[i], Eb[i]).
    s_neg is the max cosine similarity of Ea[i] to any other Eb[j] where j != i (the hardest negative).
    Returns mean/median/p10/p90 of the gap.
    We want the gap to be positive (s_pos > s_neg) and reasonably large (values are in [-2, 2])
    """
    Ea = F.normalize(Ea, dim=-1)
    Eb = F.normalize(Eb, dim=-1)
    N = Ea.shape[0]
    assert Eb.shape[0] == N

    EbT = Eb.T  # [D, N]
    gaps = []

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        sims = Ea[start:end] @ EbT  # [chunk, N]

        # mask diagonal (the true positive) so it can't be chosen as negative
        rows = torch.arange(end - start)
        cols = torch.arange(start, end)
        sims[rows, cols] = -1e9

        s_neg = sims.max(dim=1).values  # [chunk]
        s_pos = (Ea[start:end] * Eb[start:end]).sum(dim=-1)  # [chunk]
        gaps.append((s_pos - s_neg).cpu())

    g = torch.cat(gaps, dim=0)  # [N]
    g_sorted, _ = torch.sort(g)
    p10 = g_sorted[int(0.10 * (N - 1))].item()
    p50 = g_sorted[int(0.50 * (N - 1))].item()
    p90 = g_sorted[int(0.90 * (N - 1))].item()

    return {
        "gap_mean": float(g.mean().item()),
        "gap_median": float(p50),
        "gap_p10": float(p10),
        "gap_p90": float(p90),
    }

def anisotropy_stats(E: torch.Tensor) -> Dict[str, float]:
    """
    Basic anisotropy: how aligned vectors are with the mean direction.
    Reports mean/median cosine similarity to mean vector (after L2 norm).
    we want this to be reasonably low (not close to 1), indicating vectors are not all collapsed in the same direction.
    """
    E0 = F.normalize(E, dim=-1)
    mu = F.normalize(E0.mean(dim=0, keepdim=True), dim=-1)  # [1, D]
    cos_to_mu = (E0 * mu).sum(dim=-1)  # [N]
    return {
        "cos_to_mean_mean": float(cos_to_mu.mean().item()),
        "cos_to_mean_median": float(cos_to_mu.median().item()),
    }


def evaluate_views_pack(
        model,
        paths: List[str],
        which: Literal["teacher", "student"],
        batch_size: int = 64,
        device: str = "cuda",
    ) -> Dict[str, float]:

    Ea, Eb = extract_two_view_embeddings(model,
                                         paths=paths,
                                         which=which,
                                         batch_size=batch_size,
                                         device=device,)

    out: Dict[str, float] = {}

    # alignment
    out["alignment"] = alignment_cosine_distance(Ea, Eb)

    # uniformity
    out.update(uniformity_random_pairs(Ea, num_pairs=20000, seed=0))

    # view consistency
    topk = view_consistency_topk(Ea, Eb, ks=(1, 5))
    out.update({f"view_{k}": v for k, v in topk.items()})
    out.update(pos_neg_gap_hard(Ea, Eb, chunk=512))

    # anisotropy on single-view (use Ea)
    out.update({f"anis_{k}": v for k, v in anisotropy_stats(Ea).items()})

    # mean-centering anisotropy check
    Ea_centered = F.normalize(Ea - Ea.mean(dim=0, keepdim=True), dim=-1)
    out.update({f"anis_centered_{k}": v for k, v in anisotropy_stats(Ea_centered).items()})

    return out
