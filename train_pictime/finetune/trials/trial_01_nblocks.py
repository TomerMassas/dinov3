"""Trial 1 — n_blocks sweep on V11/ckpt/13000.

Sweeps unfreeze_n_blocks ∈ {0, 2, 4, 6, 8, 12}, holding all other knobs
fixed (see experiment.md → Trial 1). Runs sequentially in one Python
process. Each run gets its own V<n+1> output dir + W&B run.

Usage:
    python3 train_pictime/finetune/trials/trial_01_nblocks.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from copy import deepcopy

import torch
from omegaconf import OmegaConf

from dinov3.configs import setup_job
from train_pictime.finetune.finetune_reid import CFG_PATH, run_finetune


N_BLOCKS_VALUES = [0, 2, 4, 6, 8, 12]
EXPERIMENT_GROUP = "trial_n_blocks"


def main():
    base_cfg = OmegaConf.load(CFG_PATH)
    setup_job(output_dir=None, seed=base_cfg.seed)

    for n_blocks in N_BLOCKS_VALUES:
        cfg = deepcopy(base_cfg)
        cfg.unfreeze_n_blocks = n_blocks
        cfg.unfreeze_after = 0 if n_blocks == 0 else 2000   # n_blocks=0 -> Mode A
        cfg.curriculum.enabled = False
        cfg.experiment_tag = f"nblocks{n_blocks}_v11ckpt13k"
        cfg.experiment_group = EXPERIMENT_GROUP

        print(f"\n{'='*70}\nTrial 1 - n_blocks={n_blocks}\n{'='*70}\n")
        try:
            run_finetune(cfg)
        except Exception as e:
            print(f"[Trial 1] n_blocks={n_blocks} CRASHED: {e!r}\nContinuing sweep.")
        finally:
            torch.cuda.empty_cache()

    print("Trial 1 - n_blocks sweep complete.")


if __name__ == "__main__":
    main()
