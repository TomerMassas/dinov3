"""Body-part fragment classifier -- inference.

Decides, per crop, whether the crop shows only body parts (a hand, a leg, a torso
sliver) and should be discarded before ReID clustering.

The caller supplies crop embeddings and nothing else: no images, no bounding boxes,
no detection metadata. They must be the 384-d pre-projection-head CLS vectors from the
same ReID backbone the classifier was fitted against -- meta["backbone_ckpt"] records
which one.

Reads the .npz written by train.py, so this module needs numpy alone: no sklearn, no
pickle, nothing to keep version-matched. Self-contained -- copy it into the serving
repo as-is.

    flt = CropFilter()
    drop = flt.should_discard(embeddings)     # bool per crop, True = discard
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Path to model_<backbone_tag>.npz.
MODEL_PATH = ""

# None -> the deploy threshold stored in the file, calibrated for precision >= 0.95:
# the gate that decides what is dropped before clustering. Set a number here only to
# trade precision for recall deliberately.
THRESHOLD = None


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Overflow-free logistic -- both branches keep the exp() argument <= 0."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


class CropFilter:
    """Scores crop embeddings and decides which crops to discard."""

    def __init__(self, model_path=MODEL_PATH, threshold: float | None = THRESHOLD):
        if not str(model_path):
            raise ValueError("MODEL_PATH is empty -- set it to the classifier .npz")
        w = np.load(Path(model_path))

        self._mean = np.asarray(w["mean"], dtype=np.float64)
        self._scale = np.asarray(w["scale"], dtype=np.float64)
        self._coef = np.asarray(w["coef"], dtype=np.float64)
        self._intercept = float(w["intercept"])
        self.dim = int(w["cls_dim"])
        if not (self._mean.shape == self._scale.shape == self._coef.shape == (self.dim,)):
            raise ValueError(f"weights disagree with cls_dim={self.dim}: "
                             f"mean{self._mean.shape} scale{self._scale.shape} "
                             f"coef{self._coef.shape}")

        self.threshold = (float(w["deploy_threshold"]) if threshold is None
                          else float(threshold))
        self.meta = {"backbone_tag": str(w["backbone_tag"]),
                     "backbone_ckpt": str(w["backbone_ckpt"]),
                     "transform": str(w["transform"]),
                     "feature_set": str(w["feature_set"]),
                     "dim": self.dim,
                     "labeling_threshold": float(w["labeling_threshold"]),
                     "deploy_threshold": float(w["deploy_threshold"]),
                     "n_train": int(w["n_train"]),
                    }

    def score(self, embeddings) -> np.ndarray:
        """P(fragment) per crop, shape (n_crops,), aligned to the input rows."""
        X = np.asarray(embeddings, dtype=np.float64)
        if X.size == 0:
            return np.empty(0, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"embeddings must be 2-D (n_crops, {self.dim}), "
                             f"got shape {X.shape}")
        if X.shape[1] != self.dim:
            raise ValueError(f"embeddings have {X.shape[1]} dims, model expects {self.dim}")
        if not np.isfinite(X).all():
            # A non-finite embedding scores NaN, compares False against the threshold,
            # and is silently kept. Fail instead of filtering nothing.
            raise ValueError("embeddings contain NaN or inf")

        return sigmoid(((X - self._mean) / self._scale) @ self._coef + self._intercept)

    def should_discard(self, embeddings) -> np.ndarray:
        """True where the crop should be dropped."""
        return self.score(embeddings) >= self.threshold
