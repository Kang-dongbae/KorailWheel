import os, glob, re, random
from collections import defaultdict, Counter
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import config as cfg

# =========================
# 1) Dataset & Utils
# =========================
def get_wheel_id(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    return re.sub(r"\(\d+\)$", "", re.split(r"__y", stem)[0]) # Simple split for ID

def pad_to_square(img: Image.Image, size=256, fill=0) -> Image.Image:
    w, h = img.size
    scale = size / h
    new_w = max(1, int(round(w * scale)))
    img_rs = img.resize((new_w, size), Image.BILINEAR)
    if new_w >= size:
        left = (new_w - size) // 2
        return img_rs.crop((left, 0, left + size, size))
    pad_left = (size - new_w) // 2
    img_np = np.array(img_rs)
    if img_np.ndim == 2: img_np = img_np[:, :, None]
    if img_np.shape[2] == 1: img_np = np.repeat(img_np, 3, axis=2)
    out = np.full((size, size, 3), fill, dtype=np.uint8)
    out[:, pad_left:pad_left+new_w, :] = img_np
    return Image.fromarray(out)

class SquarePadTransform:
    def __init__(self, size): self.size = size
    def __call__(self, img): return pad_to_square(img, size=self.size, fill=0)

class TileDataset(Dataset):
    def __init__(self, split, augment=False):
        self.files = sorted(glob.glob(os.path.join(cfg.OUT_TILE_ROOT, split, "images", "*.jpg")))
        aug = [T.RandomApply([T.ColorJitter(0.15, 0.15, 0.1, 0.1)], p=0.5), T.RandomGrayscale(p=0.05)] if augment else []
        self.tf = T.Compose(aug + [SquarePadTransform(cfg.AE_IMG_SIZE), T.ToTensor()])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx): return self.tf(Image.open(self.files[idx]).convert("RGB")), self.files[idx]

# =========================
# 2) Model & Loss
# =========================
class WeightedMSELoss(nn.Module):
    def __init__(self, thr=0.02, bg_weight=0.02, eps=1e-8):
        super().__init__()
        self.thr, self.bg_weight, self.eps = thr, bg_weight, eps
    def forward(self, y, x):
        mask = (x.mean(dim=1, keepdim=True) > self.thr).float()
        w = mask + (1.0 - mask) * self.bg_weight
        return ((y - x) ** 2 * w).sum() / (w.sum() + self.eps)

class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(True)
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.dec(self.enc(x))

# =========================
# 3) Training & Scoring
# =========================
def run_epoch(loader, model, crit, opt=None, train=True):
    model.train(train)
    total, n = 0.0, 0
    for x, _ in loader:
        x = x.to(cfg.DEVICE)
        if train:
            y = model(x)
            loss = crit(y, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        else:
            with torch.no_grad():
                y = model(x)
                loss = crit(y, x)
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(1, n)

@torch.no_grad()
def score_loader(loader, model):
    model.eval()
    scores, fpaths = [], []
    for x, fps in loader:
        x = x.to(cfg.DEVICE)
        y = model(x)
        mask = (x.mean(dim=1, keepdim=True) > cfg.MASK_THRESHOLD).float()
        w = mask + (1.0 - mask) * cfg.BG_WEIGHT
        se = (y - x) ** 2
        s = ((se * w).sum(dim=(1,2,3)) / (w.sum(dim=(1,2,3)) + 1e-8)).cpu().numpy()
        scores.extend(s.tolist()); fpaths.extend(fps)
    return np.array(scores, dtype=np.float32), fpaths

def aggregate_scores(scores, fpaths):
    d = defaultdict(list)
    for s, fp in zip(scores, fpaths):
        d[get_wheel_id(fp)].append((float(s), fp))
    
    # Filter by required tiles
    if cfg.REQUIRE_TILES:
        d = {k:v for k,v in d.items() if len(v) == cfg.REQUIRE_TILES}
    
    out = {}
    for wid, items in d.items():
        vals = [v for v, _ in items]
        out[wid] = max(vals) if cfg.AGG_METHOD == "max" else float(np.mean(vals))
    return out, d

def save_vis(model, wheel_items, wheel_scores, prefix):
    out_dir = os.path.join(cfg.OUT_TILE_ROOT, f"ae_vis_{prefix}")
    cfg.ensure_dir(out_dir)
    top = sorted(wheel_scores.items(), key=lambda kv: kv[1], reverse=True)[:cfg.TOPK_VIS]
    
    for wid, wscore in top:
        _, fp = max(wheel_items[wid], key=lambda t: t[0]) # Best tile
        img = Image.open(fp).convert("RGB")
        x_raw = SquarePadTransform(cfg.AE_IMG_SIZE)(img)
        x = T.ToTensor()(x_raw).unsqueeze(0).to(cfg.DEVICE)
        y = model(x).clamp(0,1)[0].detach().cpu()
        
        # Make Canvas (Input | Recon | Err)
        x_np = np.array(x_raw)
        y_np = (y.permute(1,2,0).numpy() * 255).astype(np.uint8)
        e_np = np.mean(np.abs(x_np.astype(float) - y_np.astype(float)), axis=2)
        e_np = (e_np / e_np.max() * 255).astype(np.uint8)
        e_np = np.stack([e_np]*3, axis=2)
        
        Image.fromarray(np.concatenate([x_np, y_np, e_np], axis=1)).save(os.path.join(out_dir, f"{prefix}_{wid}_{wscore:.4f}.png"))
    print(f"Saved visualization to {out_dir}")

# =========================
# 4) Main
# =========================
if __name__ == "__main__":
    torch.manual_seed(cfg.SEED)
    cfg.ensure_dir(cfg.RUNS_DIR)

    train_loader = DataLoader(TileDataset("train", True), batch_size=cfg.AE_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(TileDataset("valid", False), batch_size=cfg.AE_BATCH_SIZE, num_workers=0)
    test_loader  = DataLoader(TileDataset("test",  False), batch_size=cfg.AE_BATCH_SIZE, num_workers=0)

    model = ConvAE().to(cfg.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.AE_LR)
    crit = WeightedMSELoss(cfg.MASK_THRESHOLD, cfg.BG_WEIGHT)

    # Train
    best_val, save_path = 1e9, os.path.join(cfg.RUNS_DIR, "ae_best.pt")
    for ep in range(1, cfg.AE_EPOCHS + 1):
        tr = run_epoch(train_loader, model, crit, opt, True)
        va = run_epoch(val_loader,   model, crit, None, False)
        print(f"Epoch {ep:02d} | Train: {tr:.6f} | Val: {va:.6f}")
        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), save_path)
    
    print(f"Training Done. Best Val: {best_val:.6f}")
    
    # Evaluate
    model.load_state_dict(torch.load(save_path))
    print("Evaluating...")
    
    val_s, val_fp = score_loader(val_loader, model)
    test_s, test_fp = score_loader(test_loader, model)
    
    val_w, val_items = aggregate_scores(val_s, val_fp)
    test_w, test_items = aggregate_scores(test_s, test_fp)
    
    thr = float(np.quantile(list(val_w.values()), cfg.VAL_Q))
    test_flags = [1 if s >= thr else 0 for s in test_w.values()]
    
    print(f"Threshold (Val Q={cfg.VAL_Q}): {thr:.6f}")
    print(f"Test Flagged Rate: {np.mean(test_flags):.3f} ({sum(test_flags)}/{len(test_flags)})")
    
    save_vis(model, val_items, val_w, "valid")
    save_vis(model, test_items, test_w, "test")