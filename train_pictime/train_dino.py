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



from dinov3.configs import setup_config, setup_job
from dinov3.configs.config import DinoV3SetupArgs
from dinov3.data import MaskingGenerator, SamplerType, collate_data_and_cast, make_data_loader
from dinov3.train.cosine_lr_scheduler import CosineScheduler
from dinov3.train.ssl_meta_arch import SSLMetaArch



from train_pictime.run_name import make_run_name
from train_pictime.pictime_dataset import PicTimeImageDataset
from train_pictime.eval.metrics_prototype import prototype_utilization
from train_pictime.eval.embed import extract_embeddings
from train_pictime.eval.metrics_views import evaluate_views_pack
from train_pictime.eval.metrics_rank import embedding_variance_and_effective_rank
from train_pictime.eval.metrics_geometry import geometry_pack
from train_pictime.wandb_logger import init_wandb, log_wandb, log_prefixed, log_prefixed_variant
from train_pictime.eval.evaluator import Evaluator, load_eval_config

REPO_ROOT = Path(__file__).resolve().parents[1]  # .../dinov3 (repo root)


def make_setup_args(args) -> DinoV3SetupArgs:
    return DinoV3SetupArgs(
        config_file=args.config_file,
        output_dir=args.output_dir,
        opts=[],  # you said you don’t want CLI overrides
        # pretrained_weights: leave None here; we set cfg.student.pretrained_weights later
    )

def parse_args():
    """
    CLI parser with a debug fallback.
    Debug mode activates if:
      - DEBUG_PICTIME=1, OR
      - no CLI args were provided (len(sys.argv)==1)
    """
    debug_defaults = dict(
                        config_file=str(REPO_ROOT / "train_pictime/pictime_vitl_im1k_lin834.yaml"),
                        output_dir="/data/AI/Tomer/person_reid/body_embedding_src/experiments/pictime_wrapper_debug",
                        train_list="/data/AI/Tomer/person_reid/dataset_utils/tiny_train_images_path.txt",
                        pretrained="/data/AI/Tomer/dinov3/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
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

    # Optional: log what happened
    print(f"[parse_args] use_debug={use_debug} argv={sys.argv}")
    return args

def build_model(cfg):
    with torch.device("meta"):
        model = SSLMetaArch(cfg)

    model.prepare_for_distributed_training()

    # Fill params with NaNs to detect missing init (same idea as official train.py)
    model._apply(
        lambda t: torch.full_like(
            t,
            fill_value=math.nan if t.dtype.is_floating_point else (2 ** (t.dtype.itemsize * 8 - 1)),
            device="cuda",
        ),
        recurse=True,
    )
    return model

def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH

    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    lr = dict(
        base_value=cfg.optim.lr,
        final_value=cfg.optim.min_lr,
        total_iters=cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim.warmup_epochs * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
        trunc_extra=cfg.optim.schedule_trunc_extra,
    )
    wd = dict(
        base_value=cfg.optim.weight_decay,
        final_value=cfg.optim.weight_decay_end,
        total_iters=cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH,
        trunc_extra=cfg.optim.schedule_trunc_extra,
    )
    momentum = dict(
        base_value=cfg.teacher.momentum_teacher,
        final_value=cfg.teacher.final_momentum_teacher,
        total_iters=cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH,
        trunc_extra=cfg.optim.schedule_trunc_extra,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    mom_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)

    # last-layer schedule = lr schedule, but frozen for N epochs
    last_layer_lr_schedule = CosineScheduler(**lr)
    freeze_iters = cfg.optim.freeze_last_layer_epochs * OFFICIAL_EPOCH_LENGTH
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
    # exactly like dinov3/train/train.py, but dataset comes from our filelist
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

    # infinite sampler (so we can run N iterations without worrying about epoch end)
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

def safety_check_eval(cfg, evaluator:Evaluator):
    size_of_val_set = len(evaluator.uval_paths)
    max_pack_size = max(int(cfg.sizes.geom), int(cfg.sizes.rank), int(cfg.sizes.proto), int(cfg.sizes.views))
    if size_of_val_set < max_pack_size:
        print(f"size of val set ({size_of_val_set}) is smaller than max pack size ({max_pack_size}), Please adjust cfg.sizes or provide a larger val set.")
        raise Exception


def main():
    args = parse_args()

    # sets up distributed (even for world_size=1) + logging dirs + seed
    setup_job(output_dir=args.output_dir, seed=0)

    # Use real setup args type (clean, no PyCharm warnings)
    setup_args = DinoV3SetupArgs(config_file=args.config_file, output_dir=args.output_dir, opts=[])
    cfg = setup_config(setup_args, strict_cfg=False)
    global_batch_size = cfg.train.batch_size_per_gpu
    # Explicit I/O overrides (YAML remains truth for training hyperparams)
    cfg.train.output_dir = args.output_dir
    cfg.student.pretrained_weights = args.pretrained

    print("Run name:", make_run_name(cfg, prefix="pictime"))
    print("train.output_dir:", cfg.train.output_dir)
    print("student.pretrained_weights:", cfg.student.pretrained_weights)

    run = init_wandb(cfg, output_dir=args.output_dir, run_name=make_run_name(cfg, prefix="pictime"))
    eval_cfg = load_eval_config(str(Path(__file__).resolve().parent / "eval/eval_config.yaml"))
    evaluator = Evaluator(eval_cfg, cfg, wandb_run=run)
    safety_check_eval(eval_cfg.cfg, evaluator)

    model = build_model(cfg)
    model.init_weights()
    model.train()

    # Data
    data_loader = build_data_loader(cfg, model, train_list=args.train_list, start_iter=0)
    it_data = iter(data_loader)

    # Optim + schedules
    optimizer = torch.optim.AdamW(model.get_params_groups(), betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))
    lr_s, wd_s, mom_s, ttemp_s, lastlr_s = build_schedulers(cfg)

    # Run just 10 iters for now
    # TODO move later the train loop into a separate function
    max_iters = min(10, cfg.optim.epochs * cfg.train.OFFICIAL_EPOCH_LENGTH)

    for it in range(max_iters):
        lr = float(lr_s[it])
        wd = float(wd_s[it])
        mom = float(mom_s[it])
        teacher_temp = float(ttemp_s[it])
        last_layer_lr = float(lastlr_s[it])

        apply_optim_scheduler(optimizer, lr=lr, wd=wd, last_layer_lr=last_layer_lr)

        # data = next(it_data)
        data = to_device(next(it_data), torch.device("cuda"))
        data["global_batch_size"] = global_batch_size

        optimizer.zero_grad(set_to_none=True)
        total_loss, metrics_dict = model.forward_backward(data, teacher_temp=teacher_temp, iteration=it)

        # optional grad clip (keeps identical behavior to official trainer)
        if cfg.optim.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.optim.clip_grad)

        total_loss = total_loss.detach()
        metrics_dict = {k: torch.as_tensor(v).detach() for k, v in metrics_dict.items()}

        optimizer.step()
        model.update_ema(mom)
        evaluator.maybe_eval(model, it)

        # log metrics
        md = {k: float(v.item()) for k, v in metrics_dict.items()}
        if it % 10 == 0:
            log_wandb(run, {
                "train/iter": it,
                "train/total_loss": float(total_loss.item()),
                "train/ibot_loss": float(md.get("ibot_loss", 0.0)),
                "train/dino_global": float(md.get("dino_global_crops_loss", 0.0)),
                "train/koleo": float(md.get("koleo_loss", 0.0)),
                "train/lr": float(lr),
                "train/wd": float(wd),
                "train/mom": float(mom),
            }, step=it)

    print("10-iter wrapper smoke OK")


    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    if run is not None:
        run.finish()



if __name__ == "__main__":
    main()
