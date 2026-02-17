


def arch_to_tag(arch: str) -> str:
    a = (arch or "").lower()
    if "large" in a:
        return "vitl"
    if "base" in a:
        return "vitb"
    if "small" in a:
        return "vits"
    return a.replace("_", "")

def make_run_name(cfg, prefix: str = "pictime") -> str:
    arch_tag = arch_to_tag(cfg.student.arch)
    ps = int(cfg.student.patch_size)
    bs = int(cfg.train.batch_size_per_gpu)
    lc = int(cfg.crops.local_crops_number)
    lr = float(cfg.optim.lr)
    return f"{prefix}_{arch_tag}{ps}_bs{bs}_lc{lc}_lr{lr:g}"
