"""Model loaders + preprocessing for the comparison.

OLD: person-reID reid_src CTLModel (ResNet50, 2048-d, not normalized), production
     transform Resize(256,128) + ImageNet norm. Imported from the person-reID repo
     via sys.path; if that env's deps aren't importable here, run `embed.py --model
     old` in an env where reid_src imports (it only needs to write the npz cache).

NEW: DINOv3 ViT-S/16 finetune (deployed V44). Backbone arch from the pictime cfg +
     projection head; weights from the slim ckpt (backbone_state_dict +
     proj_head_state_dict). Output L2-normalized 128-d. Transform Resize(256) +
     CenterCrop(224) + ImageNet norm (get_val_transform).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from train_pictime.model_comparison import config as C

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Shared crop dataset — identical pixels for both models; only the transform differs
# ---------------------------------------------------------------------------

class CropDataset(Dataset):
    """entries: list of (project_id, filename, bbox[xyxy norm], bbox_index, gt_cluster_id).
    Returns (row_index, transformed_crop, is_valid)."""

    def __init__(self, entries, transform):
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        pid, fname, bbox, _bbox_idx, _gid = self.entries[idx]
        img_path = C.DATASET_ROOT / pid / C.IMAGES_SUBDIR / fname
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            x1, y1, x2, y2 = bbox
            crop = img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
            if crop.size[0] < 4 or crop.size[1] < 4:
                return idx, torch.zeros(3, 224, 224), False
            return idx, self.transform(crop), True
        except Exception:
            return idx, torch.zeros(3, 224, 224), False


# ---------------------------------------------------------------------------
# OLD model
# ---------------------------------------------------------------------------

def old_transform():
    return T.Compose([
        T.Resize(C.OLD_RESIZE_HW),                    # (H, W) = (256, 128), production SIZE_TEST
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_old_model(device="cuda"):
    """Build the production ResNet CTL model + load reid_model weights. Imports
    reid_src from the person-reID repo (added to sys.path)."""
    sys.path.insert(0, str(C.PERSON_REID_REPO))
    from reid_src.config import _C as reid_cfg   # noqa: E402
    from reid_src.ctl_model import CTLModel       # noqa: E402

    reid_cfg.merge_from_list(["MODEL.PRETRAIN_PATH", str(C.OLD_WEIGHTS)])
    model = CTLModel(reid_cfg)
    model.load_state_dict(torch.load(str(C.OLD_WEIGHTS), map_location="cpu"))
    model.eval().to(device)
    return model


@torch.no_grad()
def old_forward(model, batch):
    """Returns [B, 2048] raw embeddings (production does not L2-normalize here)."""
    out = model(batch)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.float()


# ---------------------------------------------------------------------------
# NEW model
# ---------------------------------------------------------------------------

def new_transform():
    from train_pictime.finetune.reid_dataset import get_val_transform
    return get_val_transform()


def _build_projection_head(embed_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Linear(embed_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, output_dim),
    )


def load_new_model(device="cuda"):
    """Build the ViT-S/16 backbone arch from the pictime cfg and load the slim V44
    ckpt (backbone + proj head). Returns (backbone, proj_head)."""
    from dinov3.configs import setup_config, setup_job
    from dinov3.configs.config import DinoV3SetupArgs
    from dinov3.models import build_model_from_cfg

    # setup_config -> apply_scaling_rules_to_cfg asserts distributed.is_enabled(),
    # so enable the (single-process) distributed env first, like the finetune script.
    # It also write_config()s into output_dir, so that dir must exist.
    dummy_dir = Path("/tmp/cmp_dummy")
    dummy_dir.mkdir(parents=True, exist_ok=True)
    setup_job(output_dir=None, seed=C.SEED)
    setup_args = DinoV3SetupArgs(config_file=str(C.PICTIME_CFG), output_dir=str(dummy_dir), opts=[])
    cfg = setup_config(setup_args, strict_cfg=False)
    backbone, embed_dim = build_model_from_cfg(cfg, only_teacher=True)
    backbone.to_empty(device=device)

    state = torch.load(str(C.NEW_CKPT), map_location=device)
    backbone.load_state_dict(state["backbone_state_dict"], strict=True)
    proj_head = _build_projection_head(embed_dim, C.PROJ_HIDDEN_DIM, C.PROJ_OUTPUT_DIM).to(device)
    proj_head.load_state_dict(state["proj_head_state_dict"], strict=True)
    backbone.eval()
    proj_head.eval()
    return backbone, proj_head


@torch.no_grad()
def new_forward(model, batch):
    """model is (backbone, proj_head). Returns [B, 128] L2-normalized embeddings."""
    backbone, proj_head = model
    feat = backbone(batch)
    feat = feat["x_norm_clstoken"] if isinstance(feat, dict) else feat
    emb = proj_head(feat.float())
    return F.normalize(emb, dim=-1)