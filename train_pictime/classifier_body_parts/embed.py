"""Build the per-gallery embedding cache — the ONLY GPU step, run once per gallery.

For every project dir under DATASET_ROOT that has a bodyfilter_baseline.json, embeds
that gallery's whole baseline pool (the post-face-filter candidates) and writes, into
the project dir, one cache per transform:

    classifier_embeddings_v18_letterbox.npz      key[], cls[N,384], geom[N,G] + fingerprint
    classifier_embeddings_v18_warp.npz
    classifier_embeddings_v18_reid_val.npz

Both train.py and predict.py then run entirely off these caches, on CPU. Nothing
upstream of the logistic regression ever changes: the backbone is frozen, so for a
fixed (crop, transform) the CLS vector is deterministic, and geometry comes from
detections.json + image dims. Only the LR and its two thresholds change per round.

Uses the V18 SSL pretrain backbone with NO finetune ckpt applied —
finetune_reid.load_backbone gives exactly that, since prep_labeling_files only
becomes "the finetune model" by overwriting the state dict afterwards.

Skips galleries whose cache already exists with a matching fingerprint, so re-running
after adding a gallery only embeds the new one. All transforms are cached in ONE pass
over the images, so the transform ablation costs one image decode instead of three.

    python3 -m train_pictime.classifier_body_parts.embed
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from dinov3.configs import setup_job
from train_pictime.finetune.finetune_reid import load_backbone
from train_pictime.classifier_body_parts import config as C
from train_pictime.classifier_body_parts.dataset import (
    GEOMETRY_DIM, GalleryImageDataset, cache_is_valid, discover_all_projects, identity_collate,
    load_baseline_keys, load_gallery_pool, save_gallery_cache,
)


def load_classifier_backbone(device: str):
    """The backbone named by config.BACKBONE_SOURCE. Returns (backbone, embed_dim).

    "finetune" is the deployed ReID backbone, so production can serve the body
    embedding and this classifier from ONE forward pass. The projection head is
    deliberately never built: we take the 384-d pre-head CLS, because the head is
    where SupCon's nuisance-invariance lives.
    """
    if C.BACKBONE_SOURCE == "pretrain":
        print(f"Backbone:  SSL pretrain (pre-finetune)")
        print(f"           {C.PRETRAIN_CKPT}  (which={C.BACKBONE_WHICH})")
        backbone, embed_dim = load_backbone(str(C.PRETRAIN_CFG),
                                            C.PRETRAIN_CKPT,
                                            C.BACKBONE_WHICH,
                                           )
    elif C.BACKBONE_SOURCE == "finetune":
        # Base arch + weights come from reid_config so they cannot drift from whatever
        # the finetune run actually started from; the ckpt then overwrites the backbone.
        cfg = OmegaConf.load(C.REID_CONFIG)
        base = Path(cfg.pretrain_config)
        if not base.is_absolute():
            base = C.DINOV3_REPO / cfg.pretrain_config
        print(f"Backbone:  finetuned ReID (deployed) — projection head NOT built")
        print(f"           base:  {base.name} @ {cfg.pretrained_weights}")
        print(f"           ckpt:  {C.FINETUNE_CKPT}  (which={cfg.backbone_which})")
        backbone, embed_dim = load_backbone(str(base),
                                            cfg.pretrained_weights,
                                            cfg.backbone_which,
                                           )
        state = torch.load(C.FINETUNE_CKPT, map_location=device)
        if "backbone_state_dict" not in state:
            raise KeyError(f"{C.FINETUNE_CKPT} has no 'backbone_state_dict' "
                           f"(keys: {sorted(state)[:8]}) — is this a finetune ckpt?")
        backbone.load_state_dict(state["backbone_state_dict"], strict=True)
        del state
        torch.cuda.empty_cache()
    else:
        raise ValueError(f"BACKBONE_SOURCE must be 'pretrain' or 'finetune', "
                         f"got {C.BACKBONE_SOURCE!r}")

    backbone.to(device)
    backbone.eval()
    return backbone, embed_dim


@torch.no_grad()
def embed_rows(backbone, rows: list[dict], transform_names: tuple[str, ...], device: str):
    """Forward every row through the backbone once per transform.

    Returns ({transform_name: cls [N,384]}, geometry [N,G], valid [N]).
    Crops are accumulated across images so the GPU still sees full batches even
    though the DataLoader yields one image at a time.
    """
    dataset = GalleryImageDataset(rows, transform_names)
    loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=C.NUM_WORKERS,
                        collate_fn=identity_collate,
                       )

    n = len(rows)
    cls = {name: np.zeros((n, C.CLS_DIM), dtype=np.float32) for name in transform_names}
    geom = np.zeros((n, GEOMETRY_DIM), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)

    pending_rows: list[int] = []
    pending = {name: [] for name in transform_names}

    def flush():
        if not pending_rows:
            return
        idxs = np.array(pending_rows, dtype=np.int64)
        for name in transform_names:
            batch = torch.stack(pending[name]).to(device, non_blocking=True)
            out = backbone(batch)
            out = out["x_norm_clstoken"] if isinstance(out, dict) else out
            cls[name][idxs] = out.float().cpu().numpy()
            pending[name].clear()
        pending_rows.clear()

    for row_idxs, crops, g, v in loader:
        geom[row_idxs] = g
        valid[row_idxs] = v
        for j, row_idx in enumerate(row_idxs):
            if not v[j]:
                continue
            pending_rows.append(row_idx)
            for name in transform_names:
                pending[name].append(crops[name][j])
            if len(pending_rows) >= C.BATCH_SIZE:
                flush()
    flush()
    return cls, geom, valid


def embed_gallery(backbone, gallery_id: str, transform_names: tuple[str, ...], device: str) -> dict:
    """Embed one gallery's baseline pool and write one cache per transform."""
    keys, shape = load_baseline_keys(gallery_id)
    rows, unresolved = load_gallery_pool(gallery_id, keys)
    if not rows:
        return {"baseline": len(keys), "cached": 0, "unresolved": len(unresolved),
                "invalid": 0, "shape": shape}

    cls, geom, valid = embed_rows(backbone, rows, transform_names, device)
    keep = np.flatnonzero(valid)
    kept_keys = [rows[i]["key"] for i in keep]
    provenance = {"n_baseline": len(keys),
                  "n_unresolved": len(unresolved),
                  "n_invalid": int(len(rows) - len(keep)),
                  "baseline_shape": shape,
                 }
    for name in transform_names:
        save_gallery_cache(gallery_id, name, kept_keys, cls[name][keep], geom[keep], provenance)

    return {"baseline": len(keys), "cached": len(keep), "unresolved": len(unresolved),
            "invalid": int(len(rows) - len(keep)), "shape": shape}


def main():
    # load_backbone materializes the fresh backbone with .to_empty(device="cuda"),
    # so CPU is not an option here — fail with that reason rather than an opaque
    # FSDP error 30 seconds in.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required: load_backbone builds an FSDP2-wrapped SSLMetaArch "
                           "on CUDA. Run this on the VM; train.py and predict.py are CPU-only.")
    device = "cuda"
    setup_job(output_dir=None, seed=C.SEED)

    transform_names = (C.TRANSFORM,)
    projects = discover_all_projects()
    print(f"\nDataset:    {C.DATASET_ROOT}")
    print(f"Projects:   {len(projects)}")
    print(f"Transforms: {', '.join(transform_names)}")
    print(f"Writing:    <project>/{C.embed_cache_name('<transform>')}\n")

    # Decide the work list before loading the backbone — if everything is already
    # cached there is no reason to spend a minute building an FSDP2 model.
    todo: list[str] = []
    skipped_cached = skipped_no_baseline = skipped_no_dets = 0
    for gallery_id in projects:
        pdir = C.DATASET_ROOT / gallery_id
        if not (pdir / C.DETECTIONS_FILENAME).exists():
            skipped_no_dets += 1
            continue
        if not (pdir / C.DETECTIONS_BASELINE_FILENAME).exists():
            skipped_no_baseline += 1
            continue
        if not C.EMBED_FORCE and all(cache_is_valid(gallery_id, t) for t in transform_names):
            skipped_cached += 1
            continue
        todo.append(gallery_id)

    print(f"To embed: {len(todo)}  |  already cached: {skipped_cached}  |  "
          f"no baseline: {skipped_no_baseline}  |  no detections: {skipped_no_dets}")
    if not todo:
        print("\nNothing to do — every project with a baseline pool already has a valid cache.")
        return

    backbone, embed_dim = load_classifier_backbone(device)
    if embed_dim != C.CLS_DIM:
        raise RuntimeError(f"Backbone embed_dim={embed_dim} but config.CLS_DIM={C.CLS_DIM} — "
                           f"arch drift between the pretrain cfg and this config")

    done = errors = 0
    tot_baseline = tot_cached = tot_unresolved = tot_invalid = 0
    shapes: set[str] = set()

    for gallery_id in tqdm(todo, desc="Galleries"):
        try:
            stats = embed_gallery(backbone, gallery_id, transform_names, device)
            shapes.add(stats["shape"])
            tot_baseline += stats["baseline"]
            tot_cached += stats["cached"]
            tot_unresolved += stats["unresolved"]
            tot_invalid += stats["invalid"]
            done += 1
            if stats["unresolved"] or stats["invalid"]:
                tqdm.write(f"[{gallery_id}] {stats['unresolved']} keys unresolved against "
                           f"detections.json, {stats['invalid']} crops unreadable/degenerate")
        except Exception as e:
            tqdm.write(f"[{gallery_id}] ERROR: {e!r}")
            errors += 1

    print(f"\n===== Summary =====")
    print(f"Embedded:      {done} galleries ({errors} errors)")
    print(f"Baseline keys: {tot_baseline}")
    print(f"Cached crops:  {tot_cached}")
    print(f"Dropped:       {tot_unresolved} unresolved + {tot_invalid} invalid")
    if shapes:
        print(f"Baseline file shape(s): {', '.join(sorted(shapes))}")
    print(f"\nNext: python3 -m train_pictime.classifier_body_parts.train   (CPU)")


if __name__ == "__main__":
    main()
