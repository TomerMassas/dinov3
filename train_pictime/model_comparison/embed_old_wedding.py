"""Embed specific wedding projects with the OLD ResNet model (2048-d, un-normalized),
over ALL detected crops (no GT / clusters_fixed needed). One-off helper for
comparing the old model against the new on real wedding galleries.

Per project under WEDDING_ROOT, writes into the project dir:
    embeddings_resnet.npz   filenames, bbox_indices, embeddings [N, 2048]  (raw, NOT normalized)

Reuses the old-model loader/transform/forward from model_comparison.models (same as
the model-comparison OLD arm). Reads detections.json + images/ per project. Skip-if-exists.

Prereqs (same as the model-comparison old arm): reid_src importable, yacs installed,
config.OLD_WEIGHTS pointing at the real reid_model.pt.

    python3 -m train_pictime.model_comparison.embed_old_wedding
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from train_pictime.model_comparison import config as C
from train_pictime.model_comparison.models import load_old_model, old_forward, old_transform

# --- What to embed ---
WEDDING_ROOT = Path("/data/AI/Tomer/person_reid/dataset_utils/dataset_finetune/Wedding[1]")
PROJECT_IDS = ["52544230", "52544256"]
DETECTIONS_FILENAME = "detections.json"
OUT_NAME = "embeddings_resnet.npz"
FORCE = False


class OldCropDataset(Dataset):
    """All detected crops for ONE project. Invalid-crop placeholder matches the old
    transform's output shape (OLD_RESIZE_HW) so batches collate cleanly."""

    def __init__(self, project_dir: str, detections: dict, transform):
        self.project_dir = project_dir
        self.transform = transform
        self.placeholder = torch.zeros(3, *C.OLD_RESIZE_HW)
        self.entries: list[tuple[str, int, list]] = []
        for fname, dets in detections.items():
            if not os.path.exists(os.path.join(project_dir, C.IMAGES_SUBDIR, fname)):
                continue
            for bbox_idx, det in enumerate(dets):
                self.entries.append((fname, bbox_idx, det["bbox"]))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        fname, _bbox_idx, bbox = self.entries[idx]
        img_path = os.path.join(self.project_dir, C.IMAGES_SUBDIR, fname)
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            x1, y1, x2, y2 = bbox
            crop = img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
            if crop.size[0] < 4 or crop.size[1] < 4:
                return idx, self.placeholder, False
            return idx, self.transform(crop), True
        except Exception:
            return idx, self.placeholder, False


def _save_npz(path: Path, filenames, bbox_indices, embeddings) -> None:
    tmp = str(path) + ".tmp.npz"
    np.savez(tmp,
             filenames=np.array(filenames, dtype=object),
             bbox_indices=np.array(bbox_indices, dtype=np.int32),
             embeddings=embeddings.astype(np.float32))
    os.replace(tmp, path)


@torch.no_grad()
def embed_project(model, dataset: OldCropDataset, device: str):
    loader = DataLoader(dataset, batch_size=C.BATCH_SIZE, shuffle=False,
                        num_workers=C.NUM_WORKERS, pin_memory=True)
    embs, valid_idx = [], []
    for idxs, batch, valid in loader:
        vmask = valid.bool()
        if not vmask.any():
            continue
        out = old_forward(model, batch[vmask].to(device, non_blocking=True)).cpu().numpy()
        embs.append(out)
        valid_idx.extend(int(i) for i in idxs[vmask].tolist())
    if not embs:
        return np.empty((0, C.OLD_EMB_DIM), dtype=np.float32), []
    return np.concatenate(embs, axis=0), valid_idx


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading old ResNet model...")
    model = load_old_model(device)
    transform = old_transform()

    for pid in PROJECT_IDS:
        pdir = WEDDING_ROOT / pid
        out_path = pdir / OUT_NAME
        det_path = pdir / DETECTIONS_FILENAME
        if not det_path.exists():
            print(f"[{pid}] no {DETECTIONS_FILENAME} — run detection first. Skipping.")
            continue
        if out_path.exists() and not FORCE:
            print(f"[{pid}] {OUT_NAME} exists — skipping.")
            continue

        with open(det_path) as f:
            detections = json.load(f)
        dataset = OldCropDataset(str(pdir), detections, transform)
        if len(dataset) == 0:
            print(f"[{pid}] no crops — skipping.")
            continue

        embeddings, valid_idx = embed_project(model, dataset, device)
        if len(embeddings) == 0:
            print(f"[{pid}] no valid crops — skipping.")
            continue

        filenames = [dataset.entries[i][0] for i in valid_idx]
        bbox_idxs = [dataset.entries[i][1] for i in valid_idx]
        _save_npz(out_path, filenames, bbox_idxs, embeddings)
        print(f"[{pid}] saved {len(embeddings)} embeddings (dim={embeddings.shape[1]}) -> {out_path}")


if __name__ == "__main__":
    main()