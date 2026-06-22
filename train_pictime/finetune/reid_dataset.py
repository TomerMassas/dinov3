from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, Sampler
from torchvision import transforms


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ReIDSample(NamedTuple):
    image_path: str
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 normalized [0,1]
    project_id: str
    cluster_id: int


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_index(index_path: str | Path):
    """Load pre-built .npz index (created by build_index.py).

    Returns:
        image_paths:  np.ndarray of str [N]
        bboxes:       np.ndarray float32 [N, 4]
        bbox_indices: np.ndarray int32 [N]
        project_ids:  np.ndarray of str [N]
        cluster_ids:  np.ndarray int32 [N]
    """
    import numpy as np
    data = np.load(index_path, allow_pickle=True)
    return (
        data["image_paths"],
        data["bboxes"],
        data["bbox_indices"],
        data["project_ids"],
        data["cluster_ids"],
    )


def load_store(store_path: str | Path, data_base_path: str | Path):
    """Load the project-keyed store (reid_store.pkl, built by build_store.py)
    and assemble the flat arrays the dataset/evaluator expect.

    Unlike load_index, this also returns per-crop `distances` (centroid distances
    carried in the store, so curriculum sorting needs no per-project file opens)
    and the set of labeled project_ids (reviewer-trusted clusters_fixed).

    Returns:
        image_paths, bboxes, bbox_indices, project_ids, cluster_ids,
        distances (float32 [N]), labeled_projects (set[str])
    """
    import pickle
    import numpy as np

    with open(store_path, "rb") as f:
        store = pickle.load(f)

    data_base = str(data_base_path)
    image_paths, bboxes, bbox_indices, project_ids, cluster_ids, distances = [], [], [], [], [], []
    labeled_projects: set[str] = set()

    for pid, rec in store["projects"].items():
        n = len(rec["filenames"])
        if n == 0:
            continue
        if rec["labeled"]:
            labeled_projects.add(str(pid))
        for fn in rec["filenames"]:
            image_paths.append(f"{data_base}/{pid}/images/{fn}")
        project_ids.extend([str(pid)] * n)
        bboxes.append(rec["bboxes"])
        bbox_indices.append(rec["bbox_indices"])
        cluster_ids.append(rec["cluster_ids"])
        distances.append(rec["distances"])

    return (
        np.array(image_paths, dtype=object),
        np.concatenate(bboxes) if bboxes else np.empty((0, 4), dtype=np.float32),
        np.concatenate(bbox_indices) if bbox_indices else np.empty(0, dtype=np.int32),
        np.array(project_ids, dtype=object),
        np.concatenate(cluster_ids) if cluster_ids else np.empty(0, dtype=np.int32),
        np.concatenate(distances) if distances else np.empty(0, dtype=np.float32),
        labeled_projects,
    )


def load_project(project_dir: Path) -> list[ReIDSample]:
    """Load samples from a single project directory.

    Reads clusters_fixed.json (preferred) or clusters.json, paired with
    detections.json for bbox coordinates.
    """
    det_path = project_dir / "detections.json"
    if not det_path.exists():
        return []

    cluster_path = project_dir / "clusters_fixed.json"
    if not cluster_path.exists():
        cluster_path = project_dir / "clusters.json"
    if not cluster_path.exists():
        return []

    with open(det_path) as f:
        detections = json.load(f)
    with open(cluster_path) as f:
        clusters = json.load(f)

    project_id = project_dir.name
    samples = []

    for fname, cluster_entries in clusters.items():
        if fname not in detections:
            continue
        det_list = detections[fname]  # list of {"bbox": [x1,y1,x2,y2]}

        for entry in cluster_entries:
            bbox_idx = entry["bbox_index"]
            cluster_id = entry["cluster_id"]

            if bbox_idx >= len(det_list):
                continue

            bbox = tuple(det_list[bbox_idx]["bbox"])
            image_path = str(project_dir / fname)
            samples.append(ReIDSample(image_path, bbox, project_id, cluster_id))

    return samples


def build_global_identity_map(project_ids, cluster_ids):
    """Assign global integer IDs from (project_id, cluster_id) pairs.

    Args:
        project_ids: np.ndarray of str [N]
        cluster_ids: np.ndarray of int [N]

    Returns:
        labels: np.ndarray int32 [N] — global identity ID per sample
    """
    import numpy as np
    id_map: dict[tuple[str, int], int] = {}
    labels = np.empty(len(project_ids), dtype=np.int32)
    for i in range(len(project_ids)):
        key = (str(project_ids[i]), int(cluster_ids[i]))
        if key not in id_map:
            id_map[key] = len(id_map)
        labels[i] = id_map[key]
    return labels


# ---------------------------------------------------------------------------
# Face blur (Trial 3)
# ---------------------------------------------------------------------------

def _build_face_lookup(image_paths,
                       project_ids,
                       faces_filename,
                      ):
    """Walk unique projects, load each <project_dir>/<faces_filename>.

    Returns:
        face_lookup: dict[(project_id, filename), list[(x1,y1,x2,y2)]]
                     — keys exist only for images that have >= 1 face detected
        stats: dict with counts (n_projects, n_missing, n_keys, n_faces)
    """
    project_to_first_idx: dict[str, int] = {}
    for i in range(len(image_paths)):
        pid = str(project_ids[i])
        if pid not in project_to_first_idx:
            project_to_first_idx[pid] = i

    face_lookup: dict[tuple[str, str], list[tuple[float, float, float, float]]] = {}
    n_missing = 0
    n_faces = 0

    for pid, idx in project_to_first_idx.items():
        proj_dir = Path(str(image_paths[idx])).parent.parent
        faces_path = proj_dir / faces_filename
        if not faces_path.exists():
            n_missing += 1
            continue
        with open(faces_path) as f:
            raw = json.load(f)
        for fname, dets in raw.items():
            bboxes = [tuple(d["bbox"]) for d in dets]
            if bboxes:
                face_lookup[(pid, fname)] = bboxes
                n_faces += len(bboxes)

    stats = {
        "n_projects": len(project_to_first_idx),
        "n_missing":  n_missing,
        "n_keys":     len(face_lookup),
        "n_faces":    n_faces,
    }
    return face_lookup, stats


def apply_face_blur(crop,
                    body_bbox,
                    faces,
                    img_size,
                    sigma_factor,
                   ):
    """Gaussian-blur each face region inside a body crop (in place via paste).

    Args:
        crop:         PIL.Image — body crop (already cropped from full image)
        body_bbox:    (x1, y1, x2, y2) normalized [0,1] full-image xyxy of the body crop
        faces:        list of (x1, y1, x2, y2) normalized [0,1] full-image xyxy face boxes
        img_size:     (W, H) of the original full image in pixels
        sigma_factor: PIL GaussianBlur radius = sigma_factor * min(face_w_px, face_h_px)

    Returns:
        The crop (same object, mutated in place). Faces outside the crop bounds
        are clipped; faces with <2px in either dim after clipping are skipped.
    """
    if not faces:
        return crop

    W, H = img_size
    bx1, by1, _, _ = body_bbox
    bx1_px = bx1 * W
    by1_px = by1 * H
    crop_W, crop_H = crop.size

    for fx1, fy1, fx2, fy2 in faces:
        lx1 = int(round(fx1 * W - bx1_px))
        ly1 = int(round(fy1 * H - by1_px))
        lx2 = int(round(fx2 * W - bx1_px))
        ly2 = int(round(fy2 * H - by1_px))

        lx1 = max(0, lx1)
        ly1 = max(0, ly1)
        lx2 = min(crop_W, lx2)
        ly2 = min(crop_H, ly2)

        face_w_px = lx2 - lx1
        face_h_px = ly2 - ly1
        if face_w_px < 2 or face_h_px < 2:
            continue

        radius = sigma_factor * min(face_w_px, face_h_px)
        face_patch = crop.crop((lx1, ly1, lx2, ly2)).filter(ImageFilter.GaussianBlur(radius=radius))
        crop.paste(face_patch, (lx1, ly1))

    return crop


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ReIDCropDataset(Dataset):
    """Dataset that returns cropped person bboxes with global identity labels."""

    def __init__(self,
                 image_paths,
                 bboxes,
                 bbox_indices,
                 project_ids,
                 cluster_ids,
                 labels,
                 transform=None,
                 min_k: int = 1,
                 centroid_distances_filename: str | None = None,
                 sample_distances=None,
                 labeled_projects=None,
                 face_blur_enabled: bool = False,
                 face_blur_sigma_factor: float = 0.3,
                 face_blur_faces_filename: str = "faces.json",
                ):
        """
        Args:
            image_paths:  np.ndarray of str [N]
            bboxes:       np.ndarray float32 [N, 4]
            bbox_indices: np.ndarray int32 [N] — index of the bbox in detections.json for that image
            project_ids:  np.ndarray of str [N]
            cluster_ids:  np.ndarray int32 [N] — per-project cluster id (for curriculum lookup)
            labels:       np.ndarray int32 [N] — global identity IDs
            centroid_distances_filename:
                If given, load each project's `<project_dir>/<filename>` and build
                `identity_to_sorted_indices` (per-identity dataset indices ordered by
                ascending cosine distance to the cluster centroid). Required by
                PKBatchSampler when curriculum is active. None disables curriculum.
            face_blur_enabled:
                When True, build a per-image face-bbox lookup and Gaussian-blur each
                face region inside the body crop in __getitem__. Default False keeps
                vanilla behavior (no lookup built, no blur applied).
            face_blur_sigma_factor:
                PIL GaussianBlur radius = sigma_factor * min(face_w_px, face_h_px) in
                original-image pixels. 0.3 ≈ heavy blur that destroys identity.
            face_blur_faces_filename:
                Per-project filename to load face bboxes from (same shape as
                detections.json). Loaded at init only when face_blur_enabled=True.
        """
        self.image_paths = image_paths
        self.bboxes = bboxes
        self.bbox_indices = bbox_indices
        self.project_ids = project_ids
        self.labels = labels
        self.transform = transform

        # Bulk-convert to Python lists ONCE. The per-sample loops below would
        # otherwise do numpy scalar indexing (labels[idx] / project_ids[idx]) per
        # element — at N~5M that's minutes of Python↔numpy overhead. .tolist() is
        # a single C call; iterating the lists is plain-Python fast.
        labels_list = labels.tolist()
        project_ids_list = [str(p) for p in project_ids]

        # Build lookups
        self.identity_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, gid in enumerate(labels_list):
            self.identity_to_indices[gid].append(idx)

        # Map project_id -> identity IDs that have >= min_k samples
        self.project_to_identities: dict[str, list[int]] = defaultdict(list)
        seen = set()
        for idx, gid in enumerate(labels_list):
            if gid not in seen and len(self.identity_to_indices[gid]) >= min_k:
                self.project_to_identities[project_ids_list[idx]].append(gid)
                seen.add(gid)

        # Projects that have at least 1 valid identity
        self.valid_projects = [p for p, ids in self.project_to_identities.items() if len(ids) > 0]

        # Labeled identities (from reviewer-trusted projects) — used by PKBatchSampler
        # for per-project curriculum p (labeled -> p_labeled, else p_unlabeled).
        self.labeled_identities: set[int] = set()
        if labeled_projects is not None:
            lp = set(str(x) for x in labeled_projects)
            for idx, gid in enumerate(labels_list):
                if project_ids_list[idx] in lp:
                    self.labeled_identities.add(gid)

        # Curriculum: per-identity dataset indices sorted by ascending centroid distance.
        # Prefer in-memory `sample_distances` (from the store — no per-project file
        # opens, fast init); else fall back to the per-project distances files
        # (the evaluator path); else disabled.
        if sample_distances is not None:
            self.identity_to_sorted_indices = self._build_sorted_indices_from_distances(sample_distances)
        elif centroid_distances_filename is None:
            self.identity_to_sorted_indices: dict[int, list[int]] | None = None
        else:
            self.identity_to_sorted_indices = self._build_sorted_indices(
                image_paths, bbox_indices, project_ids, cluster_ids, labels,
                centroid_distances_filename,
            )

        # Face blur: per-image face-bbox lookup (only built when enabled)
        self.face_blur_enabled = face_blur_enabled
        self.face_blur_sigma_factor = face_blur_sigma_factor
        if face_blur_enabled:
            self.face_lookup, stats = _build_face_lookup(image_paths,
                                                        project_ids,
                                                        face_blur_faces_filename,
                                                       )
            n_loaded = stats["n_projects"] - stats["n_missing"]
            print(f"[face_blur] Loaded {n_loaded}/{stats['n_projects']} projects "
                  f"({stats['n_missing']} missing {face_blur_faces_filename}), "
                  f"{stats['n_keys']} (project, file) keys, {stats['n_faces']} total faces, "
                  f"sigma_factor={face_blur_sigma_factor}")
        else:
            self.face_lookup = None

    def _build_sorted_indices_from_distances(self, sample_distances) -> dict[int, list[int]]:
        """Per-identity dataset indices sorted by ascending centroid distance,
        using an in-memory per-sample distance array (carried in the store).
        No file I/O — replaces the 52K-file-open init when training from the store.
        """
        # Convert to a Python list once — sorted()'s key would otherwise do a
        # numpy scalar fetch per comparison (O(N log N) numpy __getitem__ calls).
        dist = sample_distances.tolist() if hasattr(sample_distances, "tolist") else sample_distances
        result: dict[int, list[int]] = {}
        for gid, idxs in self.identity_to_indices.items():
            result[gid] = sorted(idxs, key=dist.__getitem__)
        return result

    @staticmethod
    def _build_sorted_indices(image_paths,
                              bbox_indices,
                              project_ids,
                              cluster_ids,
                              labels,
                              centroid_distances_filename,
                             ) -> dict[int, list[int]]:
        """For each global identity, return its dataset indices sorted by ascending
        cosine distance to the cluster centroid (read from per-project distances file).

        Crashes loudly if any identity is missing from the distances file or any
        (filename, bbox_index) entry can't be matched to a dataset row.
        """
        # Reverse lookup: (project_id, filename, bbox_index) -> dataset idx
        rev_lookup: dict[tuple[str, str, int], int] = {}
        for i in range(len(image_paths)):
            fname = Path(str(image_paths[i])).name
            key = (str(project_ids[i]), fname, int(bbox_indices[i]))
            rev_lookup[key] = i

        # First-sample-per-project (used to derive the project directory path)
        project_to_first_idx: dict[str, int] = {}
        for i in range(len(image_paths)):
            pid = str(project_ids[i])
            if pid not in project_to_first_idx:
                project_to_first_idx[pid] = i

        # Identity -> (project_id, cluster_id)
        identity_to_proj_cluster: dict[int, tuple[str, int]] = {}
        for i in range(len(labels)):
            gid = int(labels[i])
            if gid not in identity_to_proj_cluster:
                identity_to_proj_cluster[gid] = (str(project_ids[i]), int(cluster_ids[i]))

        project_distances_cache: dict[str, dict[int, list]] = {}
        identity_to_sorted_indices: dict[int, list[int]] = {}

        for gid, (proj_id, cluster_id) in identity_to_proj_cluster.items():
            if proj_id not in project_distances_cache:
                proj_dir = Path(str(image_paths[project_to_first_idx[proj_id]])).parent.parent
                with open(proj_dir / centroid_distances_filename) as f:
                    raw = json.load(f)
                project_distances_cache[proj_id] = {int(k): v for k, v in raw.items()}

            cd = project_distances_cache[proj_id]
            if cluster_id not in cd:
                raise RuntimeError(
                    f"Identity {gid} (project={proj_id}, cluster={cluster_id}) "
                    f"has no entry in {centroid_distances_filename}"
                )

            sorted_idxs: list[int] = []
            for entry in cd[cluster_id]:
                key = (proj_id, entry["filename"], int(entry["bbox_index"]))
                if key not in rev_lookup:
                    raise RuntimeError(
                        f"Centroid entry {key} (identity={gid}) not found in dataset"
                    )
                sorted_idxs.append(rev_lookup[key])
            identity_to_sorted_indices[gid] = sorted_idxs

        return identity_to_sorted_indices

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = int(self.labels[idx])

        img = Image.open(str(self.image_paths[idx])).convert("RGB")
        w, h = img.size
        x1, y1, x2, y2 = self.bboxes[idx]
        crop = img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))

        if self.face_blur_enabled:
            fname = Path(str(self.image_paths[idx])).name
            pid = str(self.project_ids[idx])
            faces = self.face_lookup.get((pid, fname), [])
            if faces:
                crop = apply_face_blur(crop,
                                       (x1, y1, x2, y2),
                                       faces,
                                       (w, h),
                                       self.face_blur_sigma_factor,
                                      )

        if self.transform is not None:
            crop = self.transform(crop)

        return crop, label


# ---------------------------------------------------------------------------
# PK Batch Sampler
# ---------------------------------------------------------------------------

class PKBatchSampler(Sampler):
    """Samples P projects, 1 identity per project, K crops per identity.

    All negatives are cross-project (guaranteed different people).
    Identities with < K samples are never selected.

    Curriculum (optional): when `curriculum_p_start != 1.0` or `curriculum_p_end != 1.0`,
    each identity's pool is restricted to the closest fraction `p(t)` of its crops to
    the cluster centroid. `p(t)` ramps linearly from `curriculum_p_start` to
    `curriculum_p_end` over the first `curriculum_end_frac * num_batches` iterations,
    then stays at `curriculum_p_end`. `max(K, ceil(p * len))` ensures the pool is
    always large enough to sample K crops.
    """

    def __init__(self,
                 dataset: ReIDCropDataset,
                 P: int,
                 K: int,
                 num_batches: int,
                 seed: int = 42,
                 curriculum_p_start: float = 1.0,
                 curriculum_p_end: float = 1.0,
                 curriculum_end_frac: float = 0.3,
                 per_project_p: bool = False,
                 p_labeled: float = 1.0,
                 p_unlabeled: float = 1.0,
                ):
        self.dataset = dataset
        self.P = P
        self.K = K
        self.num_batches = num_batches
        self.rng = random.Random(seed)
        self.p_start = curriculum_p_start
        self.p_end = curriculum_p_end
        self.end_frac = curriculum_end_frac
        # Per-project mode: static p per identity by labeled-ness (no time ramp).
        # labeled (reviewer-trusted) -> p_labeled; else -> p_unlabeled.
        self.per_project_p = per_project_p
        self.p_labeled = p_labeled
        self.p_unlabeled = p_unlabeled

        if len(dataset.valid_projects) < P:
            raise ValueError(
                f"Need at least P={P} projects with valid identities, "
                f"but only {len(dataset.valid_projects)} available."
            )

        if per_project_p:
            curriculum_active = (p_labeled != 1.0) or (p_unlabeled != 1.0)
        else:
            curriculum_active = (curriculum_p_start != 1.0) or (curriculum_p_end != 1.0)
        if curriculum_active and dataset.identity_to_sorted_indices is None:
            raise ValueError(
                "Curriculum is active but the dataset has no centroid distances. "
                "Pass `sample_distances` (store) or `centroid_distances_filename` to ReIDCropDataset."
            )
        self._use_sorted = dataset.identity_to_sorted_indices is not None

    def __iter__(self):
        ramp_end_iter = max(1, int(self.num_batches * self.end_frac))
        for t in range(self.num_batches):
            # Ramp mode: global p(t). Per-project mode resolves p per identity below.
            if t < ramp_end_iter:
                p_ramp = self.p_start + (self.p_end - self.p_start) * (t / ramp_end_iter)
            else:
                p_ramp = self.p_end

            projects = self.rng.sample(self.dataset.valid_projects, self.P)
            batch = []
            for proj in projects:
                identity = self.rng.choice(self.dataset.project_to_identities[proj])
                if self._use_sorted:
                    if self.per_project_p:
                        p = self.p_labeled if identity in self.dataset.labeled_identities else self.p_unlabeled
                    else:
                        p = p_ramp
                    sorted_idx = self.dataset.identity_to_sorted_indices[identity]
                    pool_size = max(self.K, math.ceil(p * len(sorted_idx)))
                    pool = sorted_idx[:pool_size]
                    chosen = self.rng.sample(pool, self.K)
                else:
                    indices = self.dataset.identity_to_indices[identity]
                    chosen = self.rng.sample(indices, self.K)
                batch.extend(chosen)
            yield batch

    def __len__(self):
        return self.num_batches


# ---------------------------------------------------------------------------
# Train / Val split
# ---------------------------------------------------------------------------

def train_val_split(items: list, val_ratio: float, seed: int) -> tuple[list, list]:
    """Deterministic split into train and val."""
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    n_val = max(1, int(len(items) * val_ratio))
    return items[n_val:], items[:n_val]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
