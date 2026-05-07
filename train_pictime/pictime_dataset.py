from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Any
from PIL import Image
from torch.utils.data import Dataset


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

    def __len__(self) -> int:
        return len(self.images_paths)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        p = self.images_paths[idx]
        img = Image.open(p).convert("RGB") #TODO maybe change to numpy array for better performance
        target = 0

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