import math
import torch
import torch.distributed as dist
import torch.compiler

torch.compiler.set_stance("force_eager")
from functools import partial
import argparse
import os
import sys
from types import SimpleNamespace
from pathlib import Path
from tqdm import tqdm

from dinov3.configs import setup_config, setup_job
from dinov3.configs.config import DinoV3SetupArgs
from dinov3.data import MaskingGenerator, SamplerType, collate_data_and_cast, make_data_loader
from dinov3.train.cosine_lr_scheduler import CosineScheduler
from dinov3.train.ssl_meta_arch import SSLMetaArch
from dinov3.checkpointer import save_checkpoint, keep_last_n_checkpoints

from train_pictime.run_name import make_run_name
from train_pictime.pictime_dataset import PicTimeImageDataset
from train_pictime.wandb_logger import init_wandb, log_wandb
from train_pictime.eval.evaluator import Evaluator, load_eval_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_setup_args(args) -> DinoV3SetupArgs:
    return DinoV3SetupArgs(
        config_file=args.config_file,
        output_dir=args.output_dir,
        opts=[],
    )


def parse_args():
    debug_defaults = dict(
        config_file=str(REPO_ROOT / "train_pictime/pictime_vitl_im1k_lin834.yaml"),
        output_dir="/data/AI/Tomer/dinov3/train_pictime/experiments",
        train_list="/data/AI/Tomer/dinov3/train_pictime/train_paths.txt",
        pretrained="/data/AI/Tomer/dinov3/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    )
    use_debug = (os.environ.get("DEBUG_PICTIME", "0") == "1") or (len(sys.argv) == 1)
    p = argparse.ArgumentParser("PicTime DINOv3 wrapper")
    p.add_argument("--config-file", required=not use_debug)
    p.add_argument("--output-dir", required=not use_debug)
    p.add_argument("--train-list", required=not use_debug)
    p.add_argument("--pretrained", required=not use_debug)

    if use_debug:
        args = SimpleNamespace(**debug_defaults)
    else:
        args = p.parse_args()
    return args


def add_version_suffix(args):
    base_dir = Path(args.output_dir)
    existing_versions = []
    if base_dir.exists():
        for folder in base_dir.iterdir():
            if folder.is_dir() and folder.name.startswith("V"):
                try:
                    version_num = int(folder.name[1:])
                    existing_versions.append(version_num)
                except ValueError:
                    continue
    next_version = max(existing_versions, default=0) + 1
    args.output_dir = str(base_dir / f"V{next_version}")
    return args


def build_model(cfg):
    with torch.device("meta"):
        model = SSLMetaArch(cfg)
    model.prepare_for_distributed_training()
    model._apply(
        lambda t: torch.full_like(
            t,
            fill_value=math.nan if t.dtype.is_floating_point else (2 ** (t.dtype.itemsize * 8 - 1)),
            device="cuda",
        ),
        recurse=True,
    )
    return model


# NOTE (Modified): Added total_iters argument.
# We must pass the EXACT number of optimizer updates from main() to here.
# Previously, this function used a default calculation that was ~16x shorter than the actual training loop,
# causing the LR to hit 0 early and degrade metrics.
def build_schedulers(cfg, total_iters):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH

    # NOTE: Calculate scaling factor to adjust warmup/freeze periods proportionally.
    # Config defines warmup in 'epochs' relative to OFFICIAL_EPOCH_LENGTH.
    # Since our total_iters is much larger (due to infinite dataset logic), we scale warmup to match.
    default_iters = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH
    scale_factor = total_iters / default_iters if default_iters > 0 else 1.0

    warmup_iters = int(cfg.optim.warmup_epochs * OFFICIAL_EPOCH_LENGTH * scale_factor)
    teacher_warmup_iters = int(cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH * scale_factor)
    freeze_iters = int(cfg.optim.freeze_last_layer_epochs * OFFICIAL_EPOCH_LENGTH * scale_factor)

    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=teacher_warmup_iters,
        warmup_iters=teacher_warmup_iters,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    # NOTE: Use total_iters (updates) passed from main
    lr = dict(
        base_value=cfg.optim.lr,
        final_value=cfg.optim.min_lr,  # This will now be reached at the END of the long loop, not beginning
        total_iters=total_iters,
        warmup_iters=warmup_iters,
        start_warmup_value=0,
        trunc_extra=cfg.optim.schedule_trunc_extra,
    )
    wd = dict(
        base_value=cfg.optim.weight_decay,
        final_value=cfg.optim.weight_decay_end,
        total_iters=total_iters,
        trunc_extra=cfg.optim.schedule_trunc_extra,
    )
    momentum = dict(
        base_value=cfg.teacher.momentum_teacher,
        final_value=cfg.teacher.final_momentum_teacher,
        total_iters=total_iters,
        trunc_extra=cfg.optim.schedule_trunc_extra,
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    mom_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)

    last_layer_lr_schedule = CosineScheduler(**lr)
    last_layer_lr_schedule.schedule[:freeze_iters] = 0

    return lr_schedule, wd_schedule, mom_schedule, teacher_temp_schedule, last_layer_lr_schedule


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for pg in optimizer.param_groups:
        is_last_layer = pg.get("is_last_layer", False)
        lr_mult = pg.get("lr_multiplier", 1.0)
        wd_mult = pg.get("wd_multiplier", 1.0)
        pg["weight_decay"] = wd * wd_mult
        pg["lr"] = (last_layer_lr if is_last_layer else lr) * lr_mult


def build_data_loader(cfg, model, train_list: str, start_iter: int = 0):
    img_size = cfg.crops.global_crops_size
    patch_size = int(cfg.student.patch_size * cfg.crops.teacher_to_student_resolution_scale)
    n_tokens = (img_size // patch_size) ** 2

    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * (img_size // patch_size) * (img_size // patch_size),
    )

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        dtype=dtype_map[cfg.compute_precision.param_dtype],
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        random_circular_shift=cfg.ibot.mask_random_circular_shift,
        local_batch_size=cfg.train.batch_size_per_gpu,
    )

    aug = model.build_data_augmentation_dino(cfg)
    dataset = PicTimeImageDataset(file_images_paths=train_list, transform=aug)
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=cfg.train.seed + start_iter + 1,
        sampler_type=SamplerType.INFINITE,
        sampler_advance=start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return data_loader


def to_device(batch, device):
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.to(device, non_blocking=True)
    return batch


def safety_check_eval(cfg, evaluator: Evaluator):
    size_of_val_set = len(evaluator.uval_paths)
    max_pack_size = max(int(cfg.sizes.geom), int(cfg.sizes.rank), int(cfg.sizes.proto), int(cfg.sizes.views))
    if size_of_val_set < max_pack_size:
        print(f"VAL SET ERROR: {size_of_val_set} < {max_pack_size}")
        raise Exception


def main():
    args = parse_args()
    args = add_version_suffix(args)
    setup_job(output_dir=args.output_dir, seed=0)
    setup_args = DinoV3SetupArgs(config_file=args.config_file, output_dir=args.output_dir, opts=[])
    cfg = setup_config(setup_args, strict_cfg=False)

    cfg.train.output_dir = args.output_dir
    cfg.student.pretrained_weights = args.pretrained

    target_batch_size = 256  # also used below for grad accumulation
    run_name = make_run_name(cfg, effective_bs=target_batch_size)
    print("Run name:", run_name)

    run = init_wandb(cfg, output_dir=args.output_dir, run_name=run_name)
    eval_cfg = load_eval_config(str(Path(__file__).resolve().parent / "eval/eval_config.yaml"))
    evaluator = Evaluator(eval_cfg, cfg, wandb_run=run)
    safety_check_eval(eval_cfg.cfg, evaluator)

    model = build_model(cfg)
    model.init_weights()
    model.train()

    # Data
    data_loader = build_data_loader(cfg, model, train_list=args.train_list, start_iter=0)
    it_data = iter(data_loader)

    # --- Option A: Gradient Accumulation Setup ---
    # NOTE: DINOv3 is unstable at small batch sizes (e.g., 16).
    # We use gradient accumulation to simulate a larger "Target" batch size (e.g., 64).
    gpu_batch_size = cfg.train.batch_size_per_gpu

    # Calculate how many forward/backwards per optimizer step
    accum_steps = max(1, target_batch_size // gpu_batch_size)

    # NOTE: Recalculate max_iters (Optimizer Updates).
    # Original logic: epochs * official_len * gpu_batch_size (huge number).
    # New logic:      Same huge number of images seen, but divided by accum_steps,
    #                 because we only step the optimizer once every 'accum_steps' samples.
    # This keeps the total training time roughly the same but stabilizes gradients.
    max_iters = (cfg.optim.epochs * cfg.train.OFFICIAL_EPOCH_LENGTH * cfg.train.batch_size_per_gpu) // accum_steps

    print(f"--- Option A Config ---")
    print(f"Physical GPU Batch Size: {gpu_batch_size}")
    print(f"Accumulation Steps:      {accum_steps}")
    print(f"Effective Batch Size:    {gpu_batch_size * accum_steps} (Target: {target_batch_size})")
    print(f"Total Optimizer Updates: {max_iters}")

    # Optim + schedules
    optimizer = torch.optim.AdamW(model.get_params_groups(), betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))

    # NOTE: Pass max_iters to build_schedulers to ensure LR decay is spread over the WHOLE training run.
    lr_s, wd_s, mom_s, ttemp_s, lastlr_s = build_schedulers(cfg, total_iters=max_iters)

    pbar = tqdm(
        total=max_iters,
        desc="Training",
        unit="iter",
        file=sys.stdout,
        mininterval=300.0,
        dynamic_ncols=False,
        ncols=100,
    )

    optimizer.zero_grad(set_to_none=True)

    # Loop over optimizer updates
    for it in range(max_iters):

        lr = float(lr_s[it])
        wd = float(wd_s[it])
        mom = float(mom_s[it])
        teacher_temp = float(ttemp_s[it])
        last_layer_lr = float(lastlr_s[it])

        apply_optim_scheduler(optimizer, lr=lr, wd=wd, last_layer_lr=last_layer_lr)

        # NOTE: Gradient Accumulation Loop
        accum_loss = 0.0
        avg_metrics = {}

        for micro_step in range(accum_steps):
            data = to_device(next(it_data), torch.device("cuda"))
            # NOTE: Inform model of effective size so internal logic (if any) is correct
            data["global_batch_size"] = gpu_batch_size * accum_steps

            # Forward + Backward
            # Gradients are accumulated in .grad attributes (PyTorch default behavior)
            total_loss, metrics_dict = model.forward_backward(data, teacher_temp=teacher_temp, iteration=it)

            accum_loss += total_loss.item()

            # Aggregate metrics
            if micro_step == 0:
                # Detach items to avoid keeping graph in memory
                avg_metrics = {k: float(v.item()) if torch.is_tensor(v) else float(v) for k, v in metrics_dict.items()}
            else:
                for k, v in metrics_dict.items():
                    val = float(v.item()) if torch.is_tensor(v) else float(v)
                    if k in avg_metrics:
                        avg_metrics[k] += val

        # End of accumulation loop, prepare for update

        # NOTE: Normalize Gradients & Loss
        # Since we summed gradients over 'accum_steps', we must divide by 'accum_steps'
        # to effectively average the gradients (simulating mean over large batch).
        if accum_steps > 1:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(accum_steps)

            # Normalize display metrics
            accum_loss /= accum_steps
            avg_metrics = {k: v / accum_steps for k, v in avg_metrics.items()}

        if cfg.optim.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.optim.clip_grad)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.update_ema(mom)

        # Eval
        evaluator.maybe_eval(model, it)

        # Log
        if it % 10 == 0:
            log_wandb(run, {
                "train/iter": it,
                "train/total_loss": accum_loss,
                "train/ibot_loss": avg_metrics.get("ibot_loss", 0.0),
                "train/dino_global": avg_metrics.get("dino_global_crops_loss", 0.0),
                "train/koleo": avg_metrics.get("koleo_loss", 0.0),
                "train/lr": lr,
                "train/wd": wd,
                "train/mom": mom,
            }, step=it)

        pbar.set_postfix({'loss': f'{accum_loss:.4f}', 'lr': f'{lr:.6f}'})
        pbar.update(1)

        # Checkpoints
        ckpt_period = cfg.checkpointing.period
        if ckpt_period > 0 and it > 0 and it % ckpt_period == 0:
            ckpt_dir = Path(args.output_dir) / "ckpt"
            torch.cuda.synchronize()
            save_checkpoint(
                ckpt_dir / str(it),
                iteration=it,
                model=model,
                optimizer=optimizer,
                overwrite=True,
            )
            keep_last_n_checkpoints(ckpt_dir, cfg.checkpointing.max_to_keep)

    pbar.close()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
