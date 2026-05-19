"""Debug visualization for pretrain face-blur.

Builds PicTimeImageDataset with face_blur_enabled=True (no transform), picks
N_SHOW indices that have >=1 face according to the face lookup, and shows the
resulting blurred PIL images via plt.show().

Used to eyeball-verify the lookup is finding faces and the blur is landing on
them before launching a face-blurred pretrain run.
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt

from train_pictime.pictime_dataset import PicTimeImageDataset


PATHS_FILE = "/data/AI/Tomer/dinov3/train_pictime/val_paths_100K.txt"
FACES_FILENAME = "faces.json"
SIGMA_FACTOR = 0.3
N_SHOW = 16
SEED = 0


def main():
    ds = PicTimeImageDataset(file_images_paths=PATHS_FILE,
                             transform=None,
                             face_blur_enabled=True,
                             face_blur_sigma_factor=SIGMA_FACTOR,
                             face_blur_faces_filename=FACES_FILENAME,
                            )

    # Reverse-index: for each (pid, fname) in face_lookup, find the dataset idx.
    rev: dict[tuple[str, str], int] = {}
    for i, p in enumerate(ds.images_paths):
        pid = Path(p).parent.parent.name
        fname = Path(p).name
        key = (pid, fname)
        if key in ds.face_lookup and key not in rev:
            rev[key] = i

    if not rev:
        print("No faces found in any sample — check faces.json availability.")
        return

    rng = random.Random(SEED)
    pool = list(rev.values())
    rng.shuffle(pool)
    picks = pool[:N_SHOW]
    print(f"Showing {len(picks)} samples with faces (out of {len(pool)} face-bearing candidates).")

    cols = 4
    rows = (len(picks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for ax, idx in zip(axes, picks):
        img, _ = ds[idx]
        ax.imshow(img)
        ax.set_title(f"idx={idx}", fontsize=8)
        ax.axis("off")

    for ax in axes[len(picks):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()