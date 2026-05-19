from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Optional, Any
from PIL import Image
from torch.utils.data import Dataset

from train_pictime.finetune.reid_dataset import apply_face_blur


def _build_face_lookup_from_paths(image_paths,
                                  faces_filename,
                                 ):
    """Walk unique project dirs derived from image_paths, load each
    <project_dir>/<faces_filename>. Mirrors the finetune _build_face_lookup
    schema but walks a flat path list instead of a project_ids array.

    Project dir is `Path(p).parent.parent` (each pretrain path is
    <project_dir>/bbox_images/<filename>).

    Returns:
        face_lookup: dict[(project_id, filename), list[(x1,y1,x2,y2)]]
                     — keys exist only for images that have >= 1 face detected
        stats: dict with counts (n_projects, n_missing, n_keys, n_faces)
    """
    project_dirs: dict[str, Path] = {}
    for p in image_paths:
        proj_dir = Path(p).parent.parent
        pid = proj_dir.name
        if pid not in project_dirs:
            project_dirs[pid] = proj_dir

    face_lookup: dict[tuple[str, str], list[tuple[float, float, float, float]]] = {}
    n_missing = 0
    n_faces = 0

    for pid, proj_dir in project_dirs.items():
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
        "n_projects": len(project_dirs),
        "n_missing":  n_missing,
        "n_keys":     len(face_lookup),
        "n_faces":    n_faces,
    }
    return face_lookup, stats


class PicTimeImageDataset(Dataset):
    """
    Dataset backed by a text file with one absolute image path per line.
    Returns (image, 0). Label is unused for SSL.
    """

    def __init__(self,
                 file_images_paths: str = None,  # passed from "PicTime:extra=..."
                 transform: Optional[Any] = None,
                 target_transform: Optional[Any] = None,
                 transforms: Optional[Any] = None,
                 face_blur_enabled: bool = False,
                 face_blur_sigma_factor: float = 0.3,
                 face_blur_faces_filename: str = "faces.json",
                ):
        if file_images_paths is None:
            raise ValueError("Expected dataset_path like PicTime:extra=/path/to/paths.txt")

        self.images_txt_file_path = Path(file_images_paths)
        with self.images_txt_file_path.open("r", encoding="utf-8") as f:
            self.images_paths: List[str] = [ln.strip() for ln in f if ln.strip()]
        print(f"Loaded {len(self.images_paths)} image paths from {self.images_txt_file_path}")
        # DINOv3/torchvision-style transform plumbing
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms  # if set, should be called as transforms(img, target)

        # Face blur: per-image face-bbox lookup (only built when enabled)
        self.face_blur_enabled = face_blur_enabled
        self.face_blur_sigma_factor = face_blur_sigma_factor
        if face_blur_enabled:
            self.face_lookup, stats = _build_face_lookup_from_paths(self.images_paths,
                                                                    face_blur_faces_filename,
                                                                   )
            n_loaded = stats["n_projects"] - stats["n_missing"]
            print(f"[face_blur] Loaded {n_loaded}/{stats['n_projects']} projects "
                  f"({stats['n_missing']} missing {face_blur_faces_filename}), "
                  f"{stats['n_keys']} (project, file) keys, {stats['n_faces']} total faces, "
                  f"sigma_factor={face_blur_sigma_factor}")
        else:
            self.face_lookup = None

    def __len__(self) -> int:
        return len(self.images_paths)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        p = self.images_paths[idx]
        img = Image.open(p).convert("RGB") #TODO maybe change to numpy array for better performance
        target = 0

        if self.face_blur_enabled:
            pid = Path(p).parent.parent.name
            fname = Path(p).name
            faces = self.face_lookup.get((pid, fname), [])
            if faces:
                apply_face_blur(img,
                                (0.0, 0.0, 1.0, 1.0),
                                faces,
                                img.size,
                                self.face_blur_sigma_factor,
                               )

        # If `transforms` is provided, it takes precedence (torchvision convention)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)


        return img, target




if __name__ == "__main__":
    txt = "/data/AI/Tomer/dinov3/train_pictime/train_paths.txt"
    txt = "/data/AI/Tomer/dinov3/train_pictime/val_paths_100K.txt"

    ds = PicTimeImageDataset(txt)

    print("N =", len(ds))
    img, _ = ds[0]
    print("first image size =", img.size, "mode =", img.mode)