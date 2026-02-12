import os, sys
repo_root = r"C:\Users\TomerMassas\Documents\GitHub\dinov3"
bad_path = os.path.join(repo_root, "dinov3")  # this is the one that makes `logging` collide
# Remove the bad path if present
sys.path = [p for p in sys.path if os.path.normcase(p) != os.path.normcase(bad_path)]
# Ensure repo root is present (safe)
if os.path.normcase(repo_root) not in map(os.path.normcase, sys.path):
    sys.path.insert(0, repo_root)


import torch

repo_dir = r"C:\Users\TomerMassas\Documents\GitHub\dinov3"
ckpt = r".\weights\dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"



m = torch.hub.load(repo_dir, "dinov3_vitl16", source="local", weights=ckpt)
device = "cuda" if torch.cuda.is_available() else "cpu"
m = m.to(device)
m.eval()


import glob
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms



tfm = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225)),
                    ])

paths = glob.glob(r"C:\Users\TomerMassas\Documents\GitHub\person-reID\dataset_utils\dataset_pretrain\15982458\bbox_images\*.jpg", recursive=True)[:64]
imgs = torch.stack([tfm(Image.open(p).convert("RGB")) for p in paths]).to(device)

with torch.no_grad():
    e = m(imgs)                      # [B, 1024]
e = F.normalize(e, dim=-1)

sim = e @ e.T
off_diag = sim[~torch.eye(sim.size(0), dtype=torch.bool, device=device)]

print("off-diag cos sim min/mean/p95/max:",
      off_diag.min().item(),
      off_diag.mean().item(),
      off_diag.kthvalue(int(0.95 * off_diag.numel())).values.item(),
      off_diag.max().item())

with torch.no_grad():
    e = m(imgs).float().cpu()
print("per-dim std: mean/min/max",
      e.std(dim=0).mean().item(),
      e.std(dim=0).min().item(),
      e.std(dim=0).max().item())
print("vector norm: mean/min/max",
      e.norm(dim=1).mean().item(),
      e.norm(dim=1).min().item(),
      e.norm(dim=1).max().item())

print()

