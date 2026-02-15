import os.path
from pathlib import Path
import torch
import glob
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from dinov3.models.vision_transformer import vit_large

ckpt_path = os.path.join("./dinov3", "weights","dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
model = vit_large()  # or vit_base/vit_small etc
state = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(state, strict=False)  # strict depends on ckpt format
model.eval().cuda()


# repo_root = Path(__file__).resolve().parent
ckpt = os.path.join("./dinov3", "weights","dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")

m = torch.load(ckpt)
# m = torch.hub.load(str(repo_root), "dinov3_vitl16", source="local", weights=str(ckpt))
device = "cuda" if torch.cuda.is_available() else "cpu"
m = m.to(device).eval()
m.eval()


tfm = transforms.Compose([  transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225)),])

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

