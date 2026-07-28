"""Score every project with the trained classifier and write the per-project decision
file the labeling UI reads.

Pure CPU and seconds long: it reads the per-gallery embedding caches embed.py already
built, so there is no backbone, no CUDA, and no image I/O. Run it after each train
phase — only the logistic regression and its thresholds change between rounds.

For every project dir under DATASET_ROOT with a valid embedding cache, writes into
the project dir:

    classifier_scores_v18.json
        model            provenance — which model/ckpt/thresholds produced this file
        labeling_threshold / deploy_threshold
        show_keys        p_fragment >= labeling_threshold, minus already-reviewed crops
        audit_keys       RANDOM_QUOTA crops sampled from the SUPPRESSED pool
        scores           EVERY scored key -> p_fragment (including already-reviewed
                         ones, so precision/recall per gallery is computable from
                         this file alone)

The app should display  show_keys ∪ audit_keys  and sort every displayed crop into
kept_keys / deleted_keys as it does today — audit crops included, so an audited crop
the reviewer deletes becomes an ordinary reviewed negative. Crops in NEITHER list stay
unlabeled and must never become negatives: they are the classifier's own predictions,
and feeding them back would train it on its own blind spots (README §6).

Already-reviewed crops (kept_keys | deleted_keys) are subtracted from both display
lists, so a labeled gallery can be re-scored with a newer model without ever showing
the same crop twice. Set EXCLUDE_REVIEWED = False to disable.

audit_keys exists for two reasons: it keeps ordinary negatives flowing in from every
new venue (without it `deleted_keys` degenerates to boundary negatives only as the
model improves), and because the above-threshold set is a census while the suppressed
pool is sampled, it gives the one direct read on false negatives:

    estimated missed fragments = (fragments found among audit_keys) / sampling_fraction

    python3 -m train_pictime.classifier_body_parts.predict
"""
from __future__ import annotations

import json
import os
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from tqdm import tqdm

from train_pictime.classifier_body_parts import config as C
from train_pictime.classifier_body_parts.dataset import (
    cache_is_valid, discover_all_projects, load_gallery_cache,
)
# build_X is imported rather than reimplemented: the feature column order MUST match
# training exactly, and a silent divergence here would be invisible at runtime.
from train_pictime.classifier_body_parts.train import build_X


# ---------------------------------------------------------------------------
# Model + baseline loading
# ---------------------------------------------------------------------------

def load_bundle() -> tuple[dict, Path]:
    """The pickled bundle from train.py: model, transform, feature_set, thresholds.

    train.py always writes the same filename, so there is nothing to disambiguate and
    predict always picks up the newest model. MODEL_PATH pins an archived one instead.
    """
    path = Path(C.MODEL_PATH) if C.MODEL_PATH else C.model_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run train.py first")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle, path


def load_reviewed_keys(gallery_dir: Path) -> set[str]:
    """kept_keys | deleted_keys — every crop the reviewer has already judged.

    Subtracted from the display lists so re-scoring a labeled gallery with a newer
    model never shows the same crop twice. Read raw rather than via
    dataset.load_label_keys: that one raises on kept/deleted overlap, which is right
    for training but not for a display filter — and here a key that no longer
    resolves should still count as reviewed.
    """
    path = gallery_dir / C.LABELS_FILENAME
    if not path.exists():
        return set()
    with open(path) as f:
        labels = json.load(f)
    return set(labels.get("kept_keys", [])) | set(labels.get("deleted_keys", []))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_gallery(bundle: dict, gallery_id: str) -> tuple[list[str], np.ndarray, dict]:
    """Read the gallery's embedding cache and apply the LR. Pure CPU, no images.

    The cache's fingerprint is verified on load, so a cache built with a different
    backbone / transform / geometry set / detections.json raises instead of quietly
    producing wrong scores.
    """
    cache = load_gallery_cache(gallery_id, bundle["transform"])
    X = build_X(cache, bundle["feature_set"])
    probs = bundle["model"].predict_proba(X)[:, 1]
    return cache["key"].tolist(), probs, cache["provenance"]


def build_output(bundle: dict,
                 model_path: Path,
                 gallery_id: str,
                 scored_keys: list[str],
                 probs: np.ndarray,
                 provenance: dict,
                 reviewed: set[str],
                ) -> dict:
    """Assemble the JSON the UI reads.

    Every baseline crop is scored and appears in `scores`, including ones already
    reviewed — that is what lets you measure real precision/recall per gallery from
    this file alone. But already-reviewed crops are excluded from show_keys and
    audit_keys, so re-scoring a labeled gallery with a newer model never puts the
    same crop in front of the reviewer twice.
    """
    t_label = float(bundle["labeling_threshold"])
    skip = reviewed if C.EXCLUDE_REVIEWED else set()

    above_all = [k for k, p in zip(scored_keys, probs) if p >= t_label]
    below_all = [k for k, p in zip(scored_keys, probs) if p < t_label]
    above = [k for k in above_all if k not in skip]
    below = [k for k in below_all if k not in skip]

    # Seeded per gallery so a re-run with the SAME model redraws the same sample.
    # A new model reshuffles it anyway — the suppressed pool itself changes — which
    # is correct: a new model has new blind spots to audit.
    rng = random.Random(f"{C.SEED}-{gallery_id}")
    quota = min(C.RANDOM_QUOTA, len(below))
    audit = sorted(rng.sample(below, quota)) if quota else []
    fraction = quota / len(below) if below else 0.0

    return {"model": {"file": str(model_path),
                      "backbone_tag": bundle["backbone_tag"],
                      "pretrain_ckpt": bundle["pretrain_ckpt"],
                      "backbone_which": bundle["backbone_which"],
                      "transform": bundle["transform"],
                      "feature_set": bundle["feature_set"],
                      "C": bundle["C"],
                      "trained_on_galleries": bundle["trained_on_galleries"],
                      "n_train": bundle["n_train"],
                      "n_positives": bundle["n_positives"],
                     },
            "labeling_threshold": t_label,
            "deploy_threshold": float(bundle["deploy_threshold"]),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "already_labeled": bool(reviewed),
            "exclude_reviewed": C.EXCLUDE_REVIEWED,
            "counts": {"baseline": provenance.get("n_baseline", len(scored_keys)),
                       "scored": len(scored_keys),
                       "unresolved": provenance.get("n_unresolved", 0),
                       "invalid_crops": provenance.get("n_invalid", 0),
                       "already_reviewed": len(reviewed),
                       "above_threshold": len(above_all),      # before the reviewed cut
                       "suppressed": len(below_all),           # before the reviewed cut
                       "above_threshold_new": len(above),      # what actually gets shown
                       "suppressed_new": len(below),           # the pool audit samples from
                       "audit_quota": quota,
                       "displayed": len(above) + quota,
                      },
            "audit": {"n_sampled": quota,
                      "n_suppressed_pool": len(below),
                      "sampling_fraction": fraction,
                      "ui_batch_size": C.UI_BATCH_SIZE,
                      "random_batches": C.RANDOM_BATCHES,
                      "note": "estimated missed fragments = (fragments kept among audit_keys) "
                              "/ sampling_fraction",
                     },
            "show_keys": sorted(above),
            "audit_keys": audit,
            "scores": {k: round(float(p), 5) for k, p in zip(scored_keys, probs)},
           }


def save_json(path: Path, obj: dict) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    bundle, model_path = load_bundle()
    out_name = C.SCORES_FILENAME[C.BACKBONE_TAG]
    print(f"Model:      {model_path}")
    print(f"            {bundle['transform']} / {bundle['feature_set']} / C={bundle['C']:g}, "
          f"trained on {bundle['n_train']} crops from {len(bundle['trained_on_galleries'])} galleries")
    print(f"Thresholds: labeling {bundle['labeling_threshold']:.4f} | "
          f"deploy {bundle['deploy_threshold']:.4f}")
    print(f"Dataset:    {C.DATASET_ROOT}")
    print(f"Reading:    <project>/{C.embed_cache_name(bundle['transform'])}")
    print(f"Writing:    <project>/{out_name}")
    print(f"Display:    show_keys + {C.RANDOM_BATCHES} random batches of "
          f"{C.UI_BATCH_SIZE} from the suppressed pool")
    print(f"Excluding already-reviewed crops: {C.EXCLUDE_REVIEWED}\n")

    if bundle["backbone_tag"] != C.BACKBONE_TAG:
        raise RuntimeError(f"Model was trained on backbone '{bundle['backbone_tag']}' but config "
                           f"says '{C.BACKBONE_TAG}' — the output filename would misattribute it")

    projects = discover_all_projects()
    print(f"{len(projects)} project dirs found\n")

    written = skipped_done = skipped_no_cache = errors = 0
    tot_baseline = tot_shown = tot_audit = 0

    for gallery_id in tqdm(projects, desc="Projects"):
        pdir = C.DATASET_ROOT / gallery_id
        out_path = pdir / out_name

        if not cache_is_valid(gallery_id, bundle["transform"]):
            # Either never embedded, or the cache is stale against the current
            # backbone / transform / geometry set / detections.json. Either way,
            # embed.py owns fixing it.
            skipped_no_cache += 1
            continue
        if out_path.exists() and not C.PREDICT_FORCE:
            skipped_done += 1
            continue

        try:
            scored_keys, probs, provenance = score_gallery(bundle, gallery_id)
            if not scored_keys:
                tqdm.write(f"[{gallery_id}] empty cache — skipped")
                skipped_no_cache += 1
                continue

            out = build_output(bundle,
                               model_path,
                               gallery_id,
                               scored_keys,
                               probs,
                               provenance,
                               reviewed=load_reviewed_keys(pdir),
                              )
            save_json(out_path, out)

            tot_baseline += out["counts"]["baseline"]
            tot_shown += out["counts"]["above_threshold_new"]
            tot_audit += out["counts"]["audit_quota"]
            written += 1
            if out["counts"]["displayed"] == 0 and out["counts"]["already_reviewed"]:
                tqdm.write(f"[{gallery_id}] fully reviewed — nothing left to display")
        except Exception as e:
            tqdm.write(f"[{gallery_id}] ERROR: {e!r}")
            errors += 1

    print(f"\n===== Summary =====")
    print(f"Written:                    {written}")
    print(f"Skipped (already scored):   {skipped_done}")
    print(f"Skipped (no valid cache):   {skipped_no_cache}")
    print(f"Errors:                     {errors}")
    if skipped_no_cache:
        print(f"\n{skipped_no_cache} projects have no valid embedding cache for "
              f"'{bundle['transform']}' — run embed.py to add them.")
    if written:
        displayed = tot_shown + tot_audit
        print(f"\nBaseline crops:  {tot_baseline}")
        print(f"To display:      {displayed}  ({tot_shown} scored + {tot_audit} random)")
        print(f"Reviewer sees:   {displayed / max(tot_baseline, 1):.1%} of the pool "
              f"-> {tot_baseline / max(displayed, 1):.1f}x less labeling")


if __name__ == "__main__":
    main()
