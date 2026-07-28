"""Labels, geometry/context features, crop transforms, and the per-gallery
embedding cache for the body-part fragment classifier.

Labels come from the labeling app's per-gallery bodyfilter_result.json:

    kept_keys     -> label 1  (crop is ONLY body parts -> discard)
    deleted_keys  -> label 0  (real, usable person crop, no visible face)

`kept_keys | deleted_keys` is exactly the set of crops the reviewer was SHOWN.
Everything else in the gallery is UNLABELED and must never become a negative:
from round 2 on, the app only shows crops the classifier flagged, so treating the
remainder as negatives would train the model to reproduce its own blind spots.

Crop keys are "<filename>_<detection_index>", and filenames themselves contain
underscores (the "_rot*" variants), so keys MUST be split with rpartition("_").
A naive split("_") fails only on rotated images, silently dropping a systematic
subset instead of raising.

The embedding cache is written once per gallery and never recomputed: the backbone
is frozen, so for a fixed (crop, transform) the 384-d CLS is deterministic, and
geometry comes from detections.json + image dims. Only the logistic regression and
its thresholds change between rounds. Every cache carries a fingerprint of
everything it depends on, verified on load — a silently stale cache would produce
wrong scores with no error anywhere.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from train_pictime.extract_embeddings import crop_bbox
from train_pictime.finetune.reid_dataset import IMAGENET_MEAN, IMAGENET_STD, get_val_transform
from train_pictime.classifier_body_parts import config as C

EPS = 1e-6
IMAGENET_FILL = tuple(int(round(m * 255)) for m in IMAGENET_MEAN)   # (124, 116, 104)

GEOMETRY_NAMES = ("log_aspect",         # log(h_px / w_px): hand ~0, leg >0, wide sliver <0
                  "sqrt_rel_area",      # sqrt of bbox area as a fraction of the frame
                  "log_crop_px",        # absolute size — a 20 px crop is unusable whatever it shows
                  "conf",               # YOLO confidence: informative as a feature, useless as a gate
                  "touch_left",         # the four border flags: truncation -> partial body
                  "touch_top",
                  "touch_right",
                  "touch_bottom",
                  "max_iou_sibling",    # overlap with other detections -> occlusion / merged people
                  "log_n_dets",         # crowd density in this frame
                  "area_rank",          # 0 = biggest box in the frame, 1 = smallest
                  "center_y",           # vertical position; fragments skew to frame edges
                 )
GEOMETRY_DIM = len(GEOMETRY_NAMES)
BORDER_EPS = 0.01


# ---------------------------------------------------------------------------
# Keys, detections, project discovery
# ---------------------------------------------------------------------------

def parse_crop_key(key: str) -> tuple[str, int]:
    """'10035892690_rot3.jpg_1' -> ('10035892690_rot3.jpg', 1).

    rpartition, NOT split: filenames contain underscores.
    """
    filename, sep, idx = key.rpartition("_")
    if not sep or not filename or not idx.isdigit():
        raise ValueError(f"Malformed crop key: {key!r}")
    return filename, int(idx)


def crop_key(filename: str, bbox_index: int) -> str:
    """Inverse of parse_crop_key — the labeling app's key format."""
    return f"{filename}_{bbox_index}"


def load_detections(gallery_id: str) -> dict:
    with open(C.DATASET_ROOT / gallery_id / C.DETECTIONS_FILENAME) as f:
        return json.load(f)


def detections_signature(gallery_id: str) -> str:
    """Content signature of detections.json, part of the cache fingerprint.

    Re-running detection changes bboxes, which changes both the crops and every
    geometry feature — so an embedding cache built against the old file is wrong.

    Hashed content, NOT size+mtime: mtime changes on rsync, backup restore or a move
    to another machine without the bboxes changing at all, and that would invalidate
    every cache in the dataset for no reason. detections.json is a few hundred KB, so
    hashing it is well under a millisecond per gallery.
    """
    path = C.DATASET_ROOT / gallery_id / C.DETECTIONS_FILENAME
    digest = hashlib.sha1(path.read_bytes()).hexdigest()[:16]
    return f"{path.stat().st_size}-{digest}"


def discover_all_projects() -> list[str]:
    """Every project dir under DATASET_ROOT, labeled or not."""
    return sorted(e.name for e in os.scandir(C.DATASET_ROOT) if e.is_dir())


def discover_galleries() -> list[str]:
    """Project ids that have a saved bodyfilter_result.json — i.e. the labeled set.

    Only galleries the reviewer clicked Save/Done on have that file.
    """
    if C.GALLERIES is not None:
        return list(C.GALLERIES)
    return sorted(p for p in discover_all_projects()
                  if (C.DATASET_ROOT / p / C.LABELS_FILENAME).exists())


# ---------------------------------------------------------------------------
# Labels and the baseline pool
# ---------------------------------------------------------------------------

def load_label_keys(gallery_id: str) -> tuple[set[str], set[str], dict]:
    """(kept, deleted, summary) raw key sets from bodyfilter_result.json.

    Raises if a key is in both (an app bug). Nothing else in the gallery is
    labeled — see the module docstring.
    """
    with open(C.DATASET_ROOT / gallery_id / C.LABELS_FILENAME) as f:
        labels = json.load(f)
    kept = set(labels.get("kept_keys", []))
    deleted = set(labels.get("deleted_keys", []))
    overlap = kept & deleted
    if overlap:
        raise ValueError(f"[{gallery_id}] {len(overlap)} keys are in BOTH kept_keys and "
                         f"deleted_keys, e.g. {sorted(overlap)[:5]} — labeling app bug")
    summary = {"gallery_id": gallery_id,
               "n_positives": len(kept),
               "n_negatives": len(deleted),
               "reviewer": labels.get("reviewer"),
               "saved_at": labels.get("saved_at"),
               "finished_batches": len(labels.get("finished_batches", [])),
              }
    return kept, deleted, summary


def load_baseline_keys(gallery_id: str) -> tuple[list[str], str]:
    """Crop keys from bodyfilter_baseline.json — the post-face-filter candidate pool.

    Tolerant of a few plausible shapes since the file is written by the labeling
    app; returns which shape it matched so a format change is visible immediately
    rather than silently producing an empty pool.
    """
    path = C.DATASET_ROOT / gallery_id / C.DETECTIONS_BASELINE_FILENAME
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return [str(k) for k in data], "list-of-keys"

    if isinstance(data, dict):
        for field in ("baseline_keys", "keys", "crop_keys", "candidate_keys", "pool_keys"):
            if isinstance(data.get(field), list):
                return [str(k) for k in data[field]], f"dict['{field}']"
        # {filename: [0, 1, 2]} or {filename: [{...}, {...}]}
        if data and all(isinstance(v, list) for v in data.values()):
            keys: list[str] = []
            for filename, entries in data.items():
                for i, entry in enumerate(entries):
                    idx = entry if isinstance(entry, int) else entry.get("bbox_index", i)
                    keys.append(crop_key(filename, int(idx)))
            return keys, "dict[filename] -> list"

    raise ValueError(f"Unrecognized {C.DETECTIONS_BASELINE_FILENAME} shape in {gallery_id}: "
                     f"top level is {type(data).__name__}"
                     + (f" with keys {sorted(data)[:6]}" if isinstance(data, dict) else "")
                     + ". Add its shape to load_baseline_keys.")


def rows_from_keys(gallery_id: str,
                   detections: dict,
                   keys,
                  ) -> tuple[list[dict], list[str]]:
    """Build embed-ready rows from crop keys.

    Siblings are ALL detections in the frame, not just baseline ones — the crowding
    features describe the real scene, and face-bearing neighbours still occlude.
    Returns (rows, unresolved_keys); unresolved keys are never silently absorbed.
    """
    rows: list[dict] = []
    unresolved: list[str] = []
    for key in keys:
        filename, bbox_index = parse_crop_key(key)
        dets = detections.get(filename)
        if dets is None or bbox_index >= len(dets):
            unresolved.append(key)
            continue
        det = dets[bbox_index]
        rows.append({"gallery_id": gallery_id,
                     "key": key,
                     "filename": filename,
                     "bbox_index": bbox_index,
                     "bbox": [float(v) for v in det["bbox"]],
                     "conf": float(det.get("conf", 1.0)),
                     "siblings": [[float(v) for v in d["bbox"]] for d in dets],
                    })
    return rows, unresolved


def load_gallery_pool(gallery_id: str, keys) -> tuple[list[dict], list[str]]:
    """Embed-ready rows for the given keys of one gallery."""
    return rows_from_keys(gallery_id, load_detections(gallery_id), keys)


# ---------------------------------------------------------------------------
# Per-gallery embedding cache
# ---------------------------------------------------------------------------

def gallery_cache_path(gallery_id: str, transform: str) -> Path:
    return C.DATASET_ROOT / gallery_id / C.embed_cache_name(transform)


def cache_fingerprint(gallery_id: str, transform: str) -> str:
    """JSON signature of everything the cached vectors depend on.

    Checked on every load. If any of these change the cached CLS/geometry are no
    longer what the current code would produce, and using them would be silently
    wrong rather than loudly broken.
    """
    return json.dumps({"backbone_tag": C.BACKBONE_TAG,
                       "pretrain_ckpt": str(C.PRETRAIN_CKPT),
                       "backbone_which": C.BACKBONE_WHICH,
                       "transform": transform,
                       "crop_size": C.CROP_SIZE,
                       "geometry_names": list(GEOMETRY_NAMES),
                       "detections": detections_signature(gallery_id),
                      },
                      sort_keys=True,
                     )


def cache_is_valid(gallery_id: str, transform: str) -> bool:
    """True if a cache exists and its fingerprint matches the current config."""
    path = gallery_cache_path(gallery_id, transform)
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=True) as data:
            return str(data["fingerprint"]) == cache_fingerprint(gallery_id, transform)
    except Exception:
        return False


def load_gallery_cache(gallery_id: str, transform: str) -> dict:
    """Load one gallery's cache, verifying the fingerprint. Raises on staleness."""
    path = gallery_cache_path(gallery_id, transform)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run embed.py first")

    with np.load(path, allow_pickle=True) as data:
        found = str(data["fingerprint"])
        expected = cache_fingerprint(gallery_id, transform)
        if found != expected:
            raise RuntimeError(f"Stale embedding cache {path}\n"
                               f"  cached:   {found}\n"
                               f"  expected: {expected}\n"
                               f"Delete it (or set EMBED_FORCE = True) and re-run embed.py.")
        return {"key": np.asarray(data["key"]).astype(str),
                "cls": np.asarray(data["cls"]),
                "geom": np.asarray(data["geom"]),
                "provenance": json.loads(str(data["provenance"])),
               }


def save_gallery_cache(gallery_id: str,
                       transform: str,
                       keys: list[str],
                       cls: np.ndarray,
                       geom: np.ndarray,
                       extra: dict,
                      ) -> Path:
    """Atomic write of one gallery's cache, stamped with the current fingerprint."""
    path = gallery_cache_path(gallery_id, transform)
    tmp = str(path) + ".tmp.npz"
    np.savez(tmp,
             key=np.array(keys, dtype=object),
             cls=cls.astype(np.float32),
             geom=geom.astype(np.float32),
             fingerprint=np.array(cache_fingerprint(gallery_id, transform), dtype=object),
             geometry_names=np.array(GEOMETRY_NAMES, dtype=object),
             provenance=np.array(json.dumps(extra, sort_keys=True), dtype=object),
            )
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# Geometry / context features
# ---------------------------------------------------------------------------

def _iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def geometry_features(bbox: list[float],
                      bbox_index: int,
                      siblings: list[list[float]],
                      conf: float,
                      img_w: int,
                      img_h: int,
                     ) -> np.ndarray:
    """The GEOMETRY_NAMES vector for one crop. bbox coords are normalized [0,1].

    Aspect ratio and absolute size are computed in PIXELS, not normalized units —
    normalized coords are distorted by the frame's own aspect ratio.
    """
    x1, y1, x2, y2 = bbox
    rel_w = max(x2 - x1, EPS)
    rel_h = max(y2 - y1, EPS)
    px_w = max(rel_w * img_w, 1.0)
    px_h = max(rel_h * img_h, 1.0)
    this_area = rel_w * rel_h

    n_dets = max(len(siblings), 1)
    n_bigger = sum(1 for i, b in enumerate(siblings)
                   if i != bbox_index and (b[2] - b[0]) * (b[3] - b[1]) > this_area)
    area_rank = n_bigger / max(n_dets - 1, 1)
    max_iou = max((_iou(bbox, b) for i, b in enumerate(siblings) if i != bbox_index), default=0.0)

    return np.array([math.log(px_h / px_w),
                     math.sqrt(this_area),
                     math.log(math.sqrt(px_w * px_h)),
                     conf,
                     float(x1 <= BORDER_EPS),
                     float(y1 <= BORDER_EPS),
                     float(x2 >= 1.0 - BORDER_EPS),
                     float(y2 >= 1.0 - BORDER_EPS),
                     max_iou,
                     math.log1p(n_dets),
                     area_rank,
                     0.5 * (y1 + y2),
                    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class PadToSquare:
    """Pad a PIL crop to square with the ImageNet mean colour, preserving aspect.

    Mean-coloured padding is ~zero after Normalize, and it leaves the crop's shape
    visible to the ViT: a leg stays a thin sliver inside the square instead of
    being stretched into a full frame.
    """

    def __init__(self, fill=IMAGENET_FILL):
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        side = max(w, h)
        if w == h:
            return img
        canvas = Image.new("RGB", (side, side), self.fill)
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        return canvas


def get_transform(name: str):
    """One of config.TRANSFORMS. All three output CROP_SIZE x CROP_SIZE."""
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if name == "letterbox":
        return transforms.Compose([PadToSquare(),
                                   transforms.Resize((C.CROP_SIZE, C.CROP_SIZE)),
                                   transforms.ToTensor(),
                                   normalize,
                                  ])
    if name == "warp":
        return transforms.Compose([transforms.Resize((C.CROP_SIZE, C.CROP_SIZE)),
                                   transforms.ToTensor(),
                                   normalize,
                                  ])
    if name == "reid_val":
        return get_val_transform()
    raise ValueError(f"Unknown transform {name!r} (expected one of {C.TRANSFORMS})")


# ---------------------------------------------------------------------------
# Crop dataset
# ---------------------------------------------------------------------------

class GalleryImageDataset(Dataset):
    """One item = one IMAGE and every requested crop inside it.

    Grouping by image matters: wedding JPEGs are large and several detections
    share one file, so decoding once per image instead of once per crop per
    transform is the difference between minutes and tens of minutes.

    Returns (row_indices [K], {transform_name: [K,3,S,S]}, geometry [K,G], valid [K]).
    Invalid crops (unreadable image, degenerate bbox) get a zero placeholder sized
    to the active transform's output and valid=False, so a batch mixing valid and
    invalid crops can never fail to collate.
    """

    def __init__(self, rows: list[dict], transform_names: tuple[str, ...]):
        self.rows = rows
        self.transform_names = tuple(transform_names)
        self.transforms = {name: get_transform(name) for name in self.transform_names}

        groups: dict[tuple[str, str], list[int]] = {}
        for i, row in enumerate(rows):
            groups.setdefault((row["gallery_id"], row["filename"]), []).append(i)
        self.groups = [(key, idxs) for key, idxs in groups.items()]

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int):
        (gallery_id, filename), row_idxs = self.groups[idx]
        img_path = C.DATASET_ROOT / gallery_id / C.IMAGES_SUBDIR / filename

        placeholder = torch.zeros(3, C.CROP_SIZE, C.CROP_SIZE)
        k = len(row_idxs)
        crops = {name: [placeholder] * k for name in self.transform_names}
        geom = np.zeros((k, GEOMETRY_DIM), dtype=np.float32)
        valid = np.zeros(k, dtype=bool)

        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception:
            return row_idxs, {n: torch.stack(v) for n, v in crops.items()}, geom, valid

        img_w, img_h = img_pil.size
        for j, row_idx in enumerate(row_idxs):
            row = self.rows[row_idx]
            geom[j] = geometry_features(row["bbox"],
                                        row["bbox_index"],
                                        row["siblings"],
                                        row["conf"],
                                        img_w,
                                        img_h,
                                       )
            try:
                crop_pil = crop_bbox(img_pil, row["bbox"])
                if crop_pil.size[0] < 4 or crop_pil.size[1] < 4:
                    continue
                for name in self.transform_names:
                    crops[name][j] = self.transforms[name](crop_pil)
                valid[j] = True
            except Exception:
                continue

        return row_idxs, {n: torch.stack(v) for n, v in crops.items()}, geom, valid


def identity_collate(batch):
    """DataLoader collate for GalleryImageDataset (batch_size=1): pass the item through."""
    return batch[0]
