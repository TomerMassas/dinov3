import logging

import torch
import torch.distributed.tensor
from torch.distributed.device_mesh import DeviceMesh

import dinov3.distributed as distributed

logger = logging.getLogger("dinov3")


def load_foundation_into_backbone(model, pth_path: str) -> None:
    """Load a flat LVD-142M-style foundation .pth into the student backbone
    (and re-sync the EMA teacher). Heads stay at random init — they are what
    gets learned during continued pretrain.

    Mirrors init_fsdp_model_from_checkpoint's FSDP2 sharding but targets the
    backbone submodule only, since the foundation .pth has no head weights.
    """
    logger.info(f"Loading FOUNDATION weights from {pth_path}")
    chkpt = torch.load(pth_path, map_location="cpu")

    for wrap in ("teacher", "model", "state_dict"):
        if isinstance(chkpt, dict) and wrap in chkpt and isinstance(chkpt[wrap], dict):
            chkpt = chkpt[wrap]
            break
    chkpt = {k.replace("module.", "").replace("backbone.", ""): v for k, v in chkpt.items()}

    process_group = distributed.get_process_subgroup()
    world_mesh = DeviceMesh.from_group(process_group, "cuda")
    keys_not_sharded = ("rope_embed.periods", "qkv.bias_mask")
    sharded = {
        key: (
            torch.distributed.tensor.distribute_tensor(tensor, world_mesh, src_data_rank=None)
            if not any(nk in key for nk in keys_not_sharded)
            else tensor
        )
        for key, tensor in chkpt.items()
    }

    missing, unexpected = model.student["backbone"].load_state_dict(sharded, strict=True)
    logger.info(f"Foundation load done — missing: {len(missing)}, unexpected: {len(unexpected)}")

    model.model_ema.load_state_dict(model.student.state_dict())