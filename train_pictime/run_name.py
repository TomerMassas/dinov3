


def arch_to_tag(arch: str) -> str:
    a = (arch or "").lower()
    if "large" in a:
        return "vitl"
    if "base" in a:
        return "vitb"
    if "small" in a:
        return "vits"
    return a.replace("_", "")

def make_run_name(cfg, prefix: str = "person_reid", effective_bs: int | None = None) -> str:
    arch_tag = arch_to_tag(cfg.student.arch)
    ps = int(cfg.student.patch_size)
    ebs = effective_bs if effective_bs is not None else int(cfg.train.batch_size_per_gpu)
    lr = float(cfg.optim.lr)
    return f"{prefix}_{arch_tag}{ps}_effbs{ebs}_lr{lr:g}"
