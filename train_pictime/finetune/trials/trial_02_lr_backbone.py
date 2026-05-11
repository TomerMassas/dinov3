"""Trial 2 - lr_backbone x n_blocks sweep on V11/ckpt/13000.

Sweeps lr_backbone in {1e-5, 5e-5, 1e-4, 5e-4} across n_blocks in
{2, 6, 12}, holding all other knobs fixed (see experiment.md -> Trial 2).
12 runs total, sequential single-process. Each run gets its own V<n+1>
output dir + W&B run.

Usage:
    python3 train_pictime/finetune/trials/trial_02_lr_backbone.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from copy import deepcopy

import torch
from omegaconf import OmegaConf

from dinov3.configs import setup_job
from train_pictime.finetune.finetune_reid import CFG_PATH, run_finetune


N_BLOCKS_VALUES = [2, 6, 12]
LR_BACKBONE_VALUES = [1e-5, 5e-5, 1e-4, 5e-4]
EXPERIMENT_GROUP = "trial_lrbb_x_nblocks"


def _fmt_lr(lr):
    """Format LR as '1e-5' / '5e-5' (drops leading zero in exponent)."""
    s = f"{lr:.0e}"                         # '1e-05'
    mantissa, exp = s.split("e")
    return f"{mantissa}e{int(exp):d}"       # '1e-5'


def main():
    base_cfg = OmegaConf.load(CFG_PATH)
    setup_job(output_dir=None, seed=base_cfg.seed)

    for n_blocks in N_BLOCKS_VALUES:
        for lr_backbone in LR_BACKBONE_VALUES:
            cfg = deepcopy(base_cfg)
            cfg.unfreeze_n_blocks = n_blocks
            cfg.unfreeze_after = 2000
            cfg.lr_backbone = lr_backbone
            cfg.curriculum.enabled = False
            lr_tag = _fmt_lr(lr_backbone)
            cfg.experiment_tag = f"nblocks{n_blocks}_lrbb{lr_tag}_v11ckpt13k"
            cfg.experiment_group = EXPERIMENT_GROUP

            print(f"\n{'='*70}\nTrial 2 - n_blocks={n_blocks}, lr_backbone={lr_tag}\n{'='*70}\n")
            try:
                run_finetune(cfg)
            except Exception as e:
                print(f"[Trial 2] n_blocks={n_blocks} lr_backbone={lr_tag} CRASHED: {e!r}\nContinuing sweep.")
            finally:
                torch.cuda.empty_cache()

    print("Trial 2 - lr_backbone x n_blocks sweep complete.")


if __name__ == "__main__":
    main()
