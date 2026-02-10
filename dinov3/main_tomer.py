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
m.eval()
print("Loaded OK")
