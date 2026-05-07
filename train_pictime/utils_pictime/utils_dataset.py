from __future__ import annotations

import random
from pathlib import Path
from typing import Union


def write_random_subset_paths(src_txt: Union[str, Path],
                              dst_txt: Union[str, Path],
                              k: int = 5000,
                              seed: int = 42,
                             ) -> None:
    """
    Read all image paths from src_txt (one path per line), randomly sample k unique paths,
    and write them (one per line) to dst_txt.

    Notes:
    - This loads all lines into memory (OK for a one-time step; you already do this in training).
    - If k >= number of lines, it will write all paths shuffled.
    """
    src_txt = Path(src_txt)
    dst_txt = Path(dst_txt)

    with src_txt.open("r", encoding="utf-8") as f:
        paths = [ln.strip() for ln in f if ln.strip()]

    if not paths:
        raise ValueError(f"No paths found in: {src_txt}")

    rng = random.Random(seed)
    n = len(paths)

    if k >= n:
        rng.shuffle(paths)
        sample = paths
    else:
        sample = rng.sample(paths, k)

    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    with dst_txt.open("w", encoding="utf-8") as f:
        f.write("\n".join(sample) + "\n")


def split_train_val_paths(src_txt: Union[str, Path],
                          train_txt: Union[str, Path],
                          val_txt: Union[str, Path],
                          val_k: int = 5000,
                          seed: int = 42,
                         ) -> None:
    """
    Read all image paths from src_txt, randomly select val_k unique paths for validation,
    and write the rest to train_txt.

    Args:
        src_txt: Source file with one image path per line.
        train_txt: Output file for training paths (all paths except val).
        val_txt: Output file for validation paths (val_k paths).
        val_k: Number of paths to use for validation.
        seed: Random seed for reproducibility.

    Notes:
    - This loads all lines into memory.
    - If val_k >= number of lines, raises ValueError.
    """
    src_txt = Path(src_txt)
    train_txt = Path(train_txt)
    val_txt = Path(val_txt)

    with src_txt.open("r", encoding="utf-8") as f:
        paths = [ln.strip() for ln in f if ln.strip()]

    if not paths:
        raise ValueError(f"No paths found in: {src_txt}")

    n = len(paths)
    if val_k >= n:
        raise ValueError(f"val_k ({val_k}) must be less than total paths ({n})")

    rng = random.Random(seed)
    rng.shuffle(paths)

    val_paths = paths[:val_k]
    train_paths = paths[val_k:]

    train_txt.parent.mkdir(parents=True, exist_ok=True)
    val_txt.parent.mkdir(parents=True, exist_ok=True)

    with train_txt.open("w", encoding="utf-8") as f:
        f.write("\n".join(train_paths) + "\n")

    with val_txt.open("w", encoding="utf-8") as f:
        f.write("\n".join(val_paths) + "\n")


if __name__ == "__main__":
    # # smoke run
    # write_random_subset_paths(
    #     src_txt="/data/AI/Tomer/person_reid/dataset_utils/train_images_paths.txt",
    #     dst_txt="/data/AI/Tomer/person_reid/dataset_utils/tiny_train_images_path.txt",
    #     k=5000,
    #     seed=123,
    # )

    # # UVAL
    # write_random_subset_paths(
    #     src_txt="/data/AI/Tomer/person_reid/dataset_utils/train_images_paths.txt",
    #     dst_txt="/data/AI/Tomer/dinov3/train_pictime/uval_paths_100K.txt",
    #     k=100000,
    #     seed=11,
    # )

    # Split train/val
    split_train_val_paths(src_txt="/data/AI/Tomer/person_reid/dataset_utils/train_images_paths.txt",
                          train_txt="/data/AI/Tomer/dinov3/train_pictime/train_paths.txt",
                          val_txt="/data/AI/Tomer/dinov3/train_pictime/val_paths_100K.txt",
                          val_k=100000,
                          seed=11,
                         )
