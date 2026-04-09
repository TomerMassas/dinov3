from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import torch
from PIL import Image
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


def build_global_identity_map(samples: list[ReIDSample]) -> tuple[dict[tuple[str, int], int], list[int]]:
    """Assign global integer IDs from (project_id, cluster_id) pairs.

    Returns:
        id_map: {(project_id, cluster_id): global_int_id}
        labels: list of global_int_id for each sample (same order as input)
    """
    id_map: dict[tuple[str, int], int] = {}
    labels = []
    for s in samples:
        key = (s.project_id, s.cluster_id)
        if key not in id_map:
            id_map[key] = len(id_map)
        labels.append(id_map[key])
    return id_map, labels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ReIDCropDataset(Dataset):
    """Dataset that returns cropped person bboxes with global identity labels."""

    def __init__(self, samples: list[ReIDSample], labels: list[int], transform=None, min_k: int = 1):
        self.samples = samples
        self.labels = labels
        self.transform = transform

        # Build lookups
        self.identity_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            self.identity_to_indices[label].append(idx)

        # Map project_id -> set of identity IDs that have >= min_k samples
        self.project_to_identities: dict[str, list[int]] = defaultdict(list)
        seen = set()
        for idx, s in enumerate(samples):
            gid = labels[idx]
            if gid not in seen and len(self.identity_to_indices[gid]) >= min_k:
                self.project_to_identities[s.project_id].append(gid)
                seen.add(gid)

        # Projects that have at least 1 valid identity
        self.valid_projects = [p for p, ids in self.project_to_identities.items() if len(ids) > 0]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        label = self.labels[idx]

        img = Image.open(s.image_path).convert("RGB")
        w, h = img.size
        x1, y1, x2, y2 = s.bbox
        crop = img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))

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
    """

    def __init__(self, dataset: ReIDCropDataset, P: int, K: int, num_batches: int, seed: int = 42):
        self.dataset = dataset
        self.P = P
        self.K = K
        self.num_batches = num_batches
        self.rng = random.Random(seed)

        if len(dataset.valid_projects) < P:
            raise ValueError(
                f"Need at least P={P} projects with valid identities, "
                f"but only {len(dataset.valid_projects)} available."
            )

    def __iter__(self):
        for _ in range(self.num_batches):
            projects = self.rng.sample(self.dataset.valid_projects, self.P)
            batch = []
            for proj in projects:
                identity = self.rng.choice(self.dataset.project_to_identities[proj])
                indices = self.dataset.identity_to_indices[identity]
                chosen = self.rng.sample(indices, self.K)
                batch.extend(chosen)
            yield batch

    def __len__(self):
        return self.num_batches


# ---------------------------------------------------------------------------
# Train / Val split
# ---------------------------------------------------------------------------

def train_val_split(
    project_dirs: list[Path], val_ratio: float, seed: int
) -> tuple[list[Path], list[Path]]:
    """Deterministic split of project directories into train and val."""
    rng = random.Random(seed)
    dirs = list(project_dirs)
    rng.shuffle(dirs)
    n_val = max(1, int(len(dirs) * val_ratio))
    return dirs[n_val:], dirs[:n_val]


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
