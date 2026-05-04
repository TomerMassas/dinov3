import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dinov3.hub.backbones import dinov3_vitb16
from dinov3.configs import setup_job
from train_pictime.finetune.finetune_reid import load_backbone




def get_weights_path():
    root_path = Path(__file__).resolve().parents[1]
    return str(root_path / "dinov3" / "weights" / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")


def load_model(device="cuda"):
    weights_path = get_weights_path()
    model = dinov3_vitb16(pretrained=True, weights=weights_path)
    model = model.eval().to(device)
    return model


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


def debug_visualize(img_rgb, boxes_normalized):
    """Visualize original image with bboxes and each crop side by side.

    Args:
        img_rgb: numpy array (H, W, 3) RGB image
        boxes_normalized: list of [x1, y1, x2, y2] in normalized [0,1] coords
    """
    if len(boxes_normalized) == 0:
        return

    h, w = img_rgb.shape[:2]
    n_boxes = len(boxes_normalized)
    fig, axes = plt.subplots(1, n_boxes + 1, figsize=(5 * (n_boxes + 1), 5))
    if n_boxes == 1:
        axes = [axes[0], axes[1]]

    # Original image with all bboxes drawn
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Original ({n_boxes} bbox{'es' if n_boxes > 1 else ''})")
    axes[0].axis('off')
    for box in boxes_normalized:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1 * w, y1 * h), (x2 - x1) * w, (y2 - y1) * h,
            linewidth=2, edgecolor='lime', facecolor='none'
        )
        axes[0].add_patch(rect)

    # Each crop
    for i, box in enumerate(boxes_normalized):
        x1, y1, x2, y2 = box
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        crop = img_rgb[py1:py2, px1:px2]
        axes[i + 1].imshow(crop)
        axes[i + 1].set_title(f"Crop {i + 1}")
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show()


def crop_bbox(img_pil, bbox_normalized):
    """Crop a normalized [0,1] bbox from a PIL image, return as PIL Image."""
    w, h = img_pil.size
    x1, y1, x2, y2 = bbox_normalized
    px1 = int(x1 * w)
    py1 = int(y1 * h)
    px2 = int(x2 * w)
    py2 = int(y2 * h)
    # Clamp to image bounds
    px1 = max(0, min(px1, w))
    py1 = max(0, min(py1, h))
    px2 = max(0, min(px2, w))
    py2 = max(0, min(py2, h))
    return img_pil.crop((px1, py1, px2, py2))


def count_total_bboxes(detections):
    """Count total number of bboxes across all images in a detections dict."""
    return sum(len(dets) for dets in detections.values())


def should_skip_project(project_dir, detections):
    """Skip if embeddings.npz exists and has the same number of entries as total bboxes."""
    embeddings_path = os.path.join(project_dir, EMBEDDINGS_FILENAME)
    if not os.path.exists(embeddings_path):
        return False
    try:
        data = np.load(embeddings_path)
        return len(data["embeddings"]) == count_total_bboxes(detections)
    except Exception:
        return False


class CropDataset(Dataset):
    """Flat dataset of all person crops across all projects.

    Items are ordered by project so that with shuffle=False, all crops from
    one project arrive before the next — enabling per-project saving.
    """

    def __init__(self, entries, tfm):
        """
        Args:
            entries: list of (project_dir, img_name, bbox_idx, bbox) tuples
            tfm: torchvision transform for eval preprocessing
        """
        self.entries = entries
        self.tfm = tfm

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        project_dir, img_name, bbox_idx, bbox = self.entries[idx]
        img_path = os.path.join(project_dir, "images", img_name)
        try:
            img_pil = Image.open(img_path).convert("RGB")
            crop_pil = crop_bbox(img_pil, bbox)
            if crop_pil.size[0] < 4 or crop_pil.size[1] < 4:
                return idx, torch.zeros(3, 224, 224), False
            return idx, self.tfm(crop_pil), True
        except Exception:
            return idx, torch.zeros(3, 224, 224), False


def save_project(project_dir, filenames, bbox_indices, embeddings):
    """Atomic save of embeddings.npz for one project."""
    if len(filenames) == 0:
        return
    emb_array = np.concatenate(embeddings, axis=0)
    embeddings_path = os.path.join(project_dir, EMBEDDINGS_FILENAME)
    tmp_path = embeddings_path + ".tmp.npz"
    np.savez(tmp_path,
             filenames=np.array(filenames),
             bbox_indices=np.array(bbox_indices),
             embeddings=emb_array)
    os.replace(tmp_path, embeddings_path)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DETECTIONS_FILENAME = "detections.json"

MODEL_SOURCE = "v11_ckpt13k"  # "foundation_b16" | "v11_ckpt13k"

# Maps each backbone source to its output filename — NEVER use a ternary here.
# Adding a new backbone? Add a row.
EMBEDDINGS_FILENAME_BY_MODEL = {
    "foundation_b16": "embeddings.npz",
    "v11_ckpt13k":    "embeddings_v2.npz",
}
EMBEDDINGS_FILENAME = EMBEDDINGS_FILENAME_BY_MODEL[MODEL_SOURCE]

# Paths used only when MODEL_SOURCE == "v11_ckpt13k"
V11_CKPT_PATH = "/data/AI/Tomer/dinov3/train_pictime/experiments/V11/ckpt/13000"
PRETRAIN_CFG_PATH = str(Path(__file__).resolve().parents[1] / "train_pictime" / "pictime_vitl_im1k_lin834.yaml")

MAX_BATCH_SIZE = 64
NUM_WORKERS = 8
DEBUG = False

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if MODEL_SOURCE == "foundation_b16":
        print("Loading ViT-B/16 foundation model...")
        model = load_model(device=device)
    elif MODEL_SOURCE == "v11_ckpt13k":
        print("Loading V11 ViT-S/16 backbone (DCP)...")
        setup_job(output_dir=None, seed=42)  # required for DCP load
        model, embed_dim = load_backbone(PRETRAIN_CFG_PATH, V11_CKPT_PATH, which="teacher")
        model = model.eval().to(device)
        print(f"V11 backbone loaded, embed_dim={embed_dim}")
    else:
        raise ValueError(f"Unknown MODEL_SOURCE: {MODEL_SOURCE}")

    tfm = get_eval_transform()

    dataset_root = Path("/data/AI/Tomer/person_reid/dataset_utils/dataset_finetune/Portraits[26]")
    project_dirs = sorted([entry.path for entry in os.scandir(dataset_root) if entry.is_dir()])

    # --- Phase 1: collect all crop entries from non-skipped projects ---
    entries = []  # (project_dir, img_name, bbox_idx, bbox)
    project_ranges = {}  # project_dir -> (start_idx, end_idx) in entries list
    total_skipped = 0

    print("Scanning projects...")
    for project_dir in tqdm(project_dirs, desc="Scanning"):
        detections_path = os.path.join(project_dir, DETECTIONS_FILENAME)
        if not os.path.exists(detections_path):
            continue
        with open(detections_path, 'r') as f:
            detections = json.load(f)
        if should_skip_project(project_dir, detections):
            total_skipped += 1
            continue

        start = len(entries)
        for img_name, dets in detections.items():
            img_path = os.path.join(project_dir, "images", img_name)
            if not os.path.exists(img_path):
                continue
            for bbox_idx, det in enumerate(dets):
                entries.append((project_dir, img_name, bbox_idx, det["bbox"]))
        end = len(entries)
        if end > start:
            project_ranges[project_dir] = (start, end)

    print(f"Skipped {total_skipped} already-done projects. "
          f"{len(project_ranges)} projects remaining, {len(entries)} crops to process.")

    if len(entries) == 0:
        print("Nothing to do.")
        sys.exit(0)

    # --- Phase 2: DataLoader + batched GPU inference ---
    crop_dataset = CropDataset(entries, tfm)
    loader = DataLoader(
        crop_dataset,
        batch_size=MAX_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
    )

    # Build sorted list of (project_dir, start, end) for flushing
    project_order = sorted(project_ranges.items(), key=lambda x: x[1][0])

    # Accumulators for current project
    proj_idx = 0
    cur_project, (cur_start, cur_end) = project_order[proj_idx]
    cur_filenames = []
    cur_bbox_indices = []
    cur_embeddings = []

    total_processed = 0
    total_errors = 0

    with torch.no_grad():
        for batch_indices, batch_tensors, batch_valid in tqdm(loader, desc="Extracting"):
            # Run valid crops through GPU
            valid_mask = batch_valid.bool()
            if valid_mask.any():
                valid_tensors = batch_tensors[valid_mask].to(device, non_blocking=True)
                out = model(valid_tensors)
                embs = F.normalize(out.float(), dim=-1).cpu().numpy()

            # Distribute results back per-entry
            emb_offset = 0
            for i in range(len(batch_indices)):
                global_idx = batch_indices[i].item()
                is_valid = batch_valid[i].item()

                # Flush completed projects as we pass their range
                while global_idx >= cur_end:
                    save_project(cur_project, cur_filenames, cur_bbox_indices, cur_embeddings)
                    if cur_filenames:
                        total_processed += 1
                    cur_filenames = []
                    cur_bbox_indices = []
                    cur_embeddings = []
                    proj_idx += 1
                    cur_project, (cur_start, cur_end) = project_order[proj_idx]

                if is_valid:
                    project_dir, img_name, bbox_idx, _ = entries[global_idx]
                    cur_filenames.append(img_name)
                    cur_bbox_indices.append(bbox_idx)
                    cur_embeddings.append(embs[emb_offset:emb_offset + 1])
                    emb_offset += 1

    # Flush last project
    save_project(cur_project, cur_filenames, cur_bbox_indices, cur_embeddings)
    if cur_filenames:
        total_processed += 1

    print(f"\nDone. Processed: {total_processed}, Skipped: {total_skipped}, Errors: {total_errors}")