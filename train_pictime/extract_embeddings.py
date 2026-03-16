import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dinov3.hub.backbones import dinov3_vitb16




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


@torch.no_grad()
def extract_project_embeddings(model, project_dir, detections, tfm, device):
    """Extract embeddings for all detected bboxes in a project.

    Returns:
        filenames: list of str (image filename for each embedding)
        bbox_indices: list of int (bbox index within that image)
        embeddings: numpy array [M, D] float32, L2-normalized
    """
    images_dir = os.path.join(project_dir, "images")
    filenames = []
    bbox_indices = []
    crops = []

    for img_name, dets in detections.items():
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            continue

        if len(dets) == 0:
            continue

        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            tqdm.write(f"Warning: Failed to load {img_path}: {e}")
            continue

        if DEBUG:
            boxes = [d["bbox"] for d in dets]
            debug_visualize(np.asarray(img_pil), boxes)

        for bbox_idx, det in enumerate(dets):
            bbox = det["bbox"]
            crop_pil = crop_bbox(img_pil, bbox)
            # Skip tiny crops
            if crop_pil.size[0] < 4 or crop_pil.size[1] < 4:
                continue
            crops.append(tfm(crop_pil))
            filenames.append(img_name)
            bbox_indices.append(bbox_idx)

    if len(crops) == 0:
        return filenames, bbox_indices, np.zeros((0, 768), dtype=np.float32)

    # Batch inference
    all_embs = []
    for i in range(0, len(crops), MAX_BATCH_SIZE):
        batch = torch.stack(crops[i:i + MAX_BATCH_SIZE]).to(device, non_blocking=True)
        out = model(batch)
        emb = F.normalize(out.float(), dim=-1)
        all_embs.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embs, axis=0)
    return filenames, bbox_indices, embeddings



IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DETECTIONS_FILENAME = "detections.json"
EMBEDDINGS_FILENAME = "embeddings.npz"
MAX_BATCH_SIZE = 64
DEBUG = False


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Loading ViT-B/16 model...")
    model = load_model(device=device)
    tfm = get_eval_transform()

    dataset_root = Path("/data/AI/Tomer/person_reid/dataset_utils/dataset_finetune/Portraits[26]")
    project_dirs = sorted([entry.path for entry in os.scandir(dataset_root) if entry.is_dir()])

    total_processed = 0
    total_skipped = 0
    total_errors = 0

    for project_dir in tqdm(project_dirs, desc="Projects"):
        try:
            detections_path = os.path.join(project_dir, DETECTIONS_FILENAME)
            if not os.path.exists(detections_path):
                continue

            with open(detections_path, 'r') as f:
                detections = json.load(f)

            if should_skip_project(project_dir, detections):
                total_skipped += 1
                continue

            filenames, bbox_indices, embeddings = extract_project_embeddings(model, project_dir, detections, tfm, device)

            if len(filenames) == 0:
                continue

            # Atomic write
            embeddings_path = os.path.join(project_dir, EMBEDDINGS_FILENAME)
            tmp_path = embeddings_path + ".tmp.npz"
            np.savez(tmp_path,
                     filenames=np.array(filenames),
                     bbox_indices=np.array(bbox_indices),
                     embeddings=embeddings)
            os.replace(tmp_path, embeddings_path)

            total_processed += 1

        except Exception as e:
            tqdm.write(f"Error processing {project_dir}: {e}")
            total_errors += 1

    print(f"\nDone. Processed: {total_processed}, Skipped: {total_skipped}, Errors: {total_errors}")