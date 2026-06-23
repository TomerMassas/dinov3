"""Embed the GT crops of the locked test set with ONE model, cache to npz.

Run once per model: flip MODEL below between "new" and "old" and run the file
directly (PyCharm / `python3 embed.py`) — no CLI args. Kept separate so the old
ResNet (person-reID reid_src) can run in whatever env imports it; the npz caches
are then consumed by evaluate.py.

Crop set is identical across models: same projects, same bboxes, same GT filter
(drop cluster_id == -1, drop GT clusters < MIN_GT_CLUSTER_SIZE). Each cached row is
keyed by (project_id, filename, bbox_index, gt_cluster_id) so evaluate.py aligns the
two models exactly.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_pictime.model_comparison import config as C
from train_pictime.model_comparison import models as M

# Which model to embed this run: "new" (DINOv3 ViT-S/16) | "old" (ResNet CTL).
# Flip and re-run to produce the other cache.
MODEL = "new" # "new", "old"


def build_entries() -> list[tuple]:
    """(project_id, filename, bbox, bbox_index, gt_cluster_id) for every surviving GT crop."""
    with open(C.NEW_PROJECTS_FILE) as f:
        payload = json.load(f)
    pids = payload if isinstance(payload, list) else payload["project_ids"]

    entries = []
    for pid in tqdm(pids, desc="Scanning projects"):
        proj = C.DATASET_ROOT / pid
        det_path = proj / C.DETECTIONS_FILENAME
        cl_path = proj / C.CLUSTERS_FIXED_FILENAME
        if not det_path.exists() or not cl_path.exists():
            continue
        with open(det_path) as f:
            detections = json.load(f)
        with open(cl_path) as f:
            clusters = json.load(f)

        # GT cluster sizes (drop -1, then drop clusters < MIN_GT_CLUSTER_SIZE)
        sizes = defaultdict(int)
        for fname, ents in clusters.items():
            for e in ents:
                if int(e["cluster_id"]) != -1:
                    sizes[int(e["cluster_id"])] += 1
        keep_cids = {c for c, n in sizes.items() if n >= C.MIN_GT_CLUSTER_SIZE}

        for fname, ents in clusters.items():
            det_list = detections.get(fname)
            if det_list is None:
                continue
            for e in ents:
                cid = int(e["cluster_id"])
                bidx = int(e["bbox_index"])
                if cid not in keep_cids or bidx >= len(det_list):
                    continue
                entries.append((pid, fname, det_list[bidx]["bbox"], bidx, cid))
    return entries


def main():
    assert MODEL in ("old", "new"), f"MODEL must be 'old' or 'new', got {MODEL!r}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    entries = build_entries()
    print(f"[{MODEL}] {len(entries)} GT crops across the locked test set")

    if MODEL == "old":
        model = M.load_old_model(device)
        transform = M.old_transform()
        forward = M.old_forward
        out_path = C.OLD_CACHE
    else:
        model = M.load_new_model(device)
        transform = M.new_transform()
        forward = M.new_forward
        out_path = C.NEW_CACHE

    loader = DataLoader(M.CropDataset(entries, transform),
                        batch_size=C.BATCH_SIZE, shuffle=False,
                        num_workers=C.NUM_WORKERS, pin_memory=True)

    embs_by_row: dict[int, np.ndarray] = {}
    for idxs, batch, valid in tqdm(loader, desc=f"Embedding ({MODEL})"):
        vmask = valid.bool()
        if not vmask.any():
            continue
        out = forward(model, batch[vmask].to(device, non_blocking=True)).cpu().numpy()
        for row, emb in zip(idxs[vmask].tolist(), out):
            embs_by_row[row] = emb

    rows = sorted(embs_by_row.keys())
    embeddings = np.stack([embs_by_row[r] for r in rows]).astype(np.float32)
    project_ids = np.array([entries[r][0] for r in rows], dtype=object)
    filenames = np.array([entries[r][1] for r in rows], dtype=object)
    bbox_indices = np.array([entries[r][3] for r in rows], dtype=np.int32)
    gt_cluster_ids = np.array([entries[r][4] for r in rows], dtype=np.int32)

    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = str(out_path) + ".tmp.npz"
    np.savez(tmp, embeddings=embeddings, project_ids=project_ids, filenames=filenames,
             bbox_indices=bbox_indices, gt_cluster_ids=gt_cluster_ids)
    Path(tmp).replace(out_path)
    print(f"Saved {len(rows)} embeddings (dim={embeddings.shape[1]}) -> {out_path}")


if __name__ == "__main__":
    main()