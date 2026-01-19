# train_ta.py
from pathlib import Path
import sys
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm

# =========================================================
# USER SETTINGS
# =========================================================
STEP = 2  # 1=in-domain normal-only, 2=external evaluation

# in-domain tiles (your current structure)
DATA_ROOT = Path(r"C:\Dev\KorailWheel\data\data_tiles")
TRAIN_DIR = DATA_ROOT / "train" / "images"
VAL_DIR   = DATA_ROOT / "valid" / "images"
TEST_DIR  = DATA_ROOT / "test" / "images"

# external tiles (prepare later)
# recommended structure:
#   C:\Dev\KorailWheel\data\st2_external\good\images
#   C:\Dev\KorailWheel\data\st2_external\defect\images
EXTERNAL_ROOT = Path(r"C:\Dev\KorailWheel\data\data_tiles\external_test")
EXT_GOOD_DIR   = EXTERNAL_ROOT / "normal" / "images"
EXT_DEFECT_DIR = EXTERNAL_ROOT / "defect" / "images"

OUT_DIR = Path(r"C:\Dev\KorailWheel\runs\st2_patchcore")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# HYPERPARAMETERS (keep here for reproducibility)
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512
BATCH = 32
NUM_WORKERS = 4

BACKBONE = "wide_resnet50_2"       # strong default for AD
HOOK_LAYERS = ["layer2", "layer3"] # multiscale patch features

PATCHES_PER_IMAGE = 256   # memory control: sample patches per image
CORESET_RATIO = 0.10      # memory bank reduction (random)
PROJ_DIM = 256            # random projection dim (reduces memory + speeds KNN)

VAL_PERCENTILE = 0.995    # threshold from val normal scores (top 0.5%)
SEED = 0



# =========================================================
# OPTIONAL: FAISS for fast KNN
# =========================================================
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================================================
# DATA
# =========================================================
def list_images(folder: Path):
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT])

def preprocess_bgr(img_bgr, size=256):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    # ImageNet norm
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std  = np.array([0.229, 0.224, 0.225], np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # CHW
    return img

class TileFolder(Dataset):
    def __init__(self, folder: Path):
        self.paths = list_images(folder)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        x = preprocess_bgr(img, IMG_SIZE)
        return torch.from_numpy(x), str(p)


# =========================================================
# FEATURE EXTRACTOR (hooks)
# =========================================================
class FeatureHook(nn.Module):
    def __init__(self, backbone_name="wide_resnet50_2", layers=("layer2", "layer3")):
        super().__init__()
        if backbone_name == "wide_resnet50_2":
            m = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        elif backbone_name == "resnet50":
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.backbone = m.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.layers = list(layers)
        self._feats = {}
        for name, module in self.backbone.named_children():
            if name in self.layers:
                module.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name):
        def fn(_, __, out):
            self._feats[name] = out
        return fn

    def forward(self, x):
        _ = self.backbone(x)
        return [self._feats[k] for k in self.layers]

def embed_from_feats(feats):
    # feats: list of [B,C,H,W] -> upsample to max H,W and concat channels
    H = max([f.shape[-2] for f in feats])
    W = max([f.shape[-1] for f in feats])
    ups = []
    for f in feats:
        if f.shape[-2:] != (H, W):
            ups.append(torch.nn.functional.interpolate(f, size=(H, W), mode="bilinear", align_corners=False))
        else:
            ups.append(f)
    emb = torch.cat(ups, dim=1)     # [B,Csum,H,W]
    emb = emb.permute(0, 2, 3, 1)   # [B,H,W,Csum]
    emb = emb.reshape(emb.shape[0], -1, emb.shape[-1])  # [B,P,D0]
    return emb

def make_random_projection(in_dim, out_dim, device, seed=0):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    W = torch.randn(in_dim, out_dim, generator=g, device=device) / np.sqrt(out_dim)
    return W

def coreset_subsample(bank_np, ratio=0.1, seed=0):
    rng = np.random.default_rng(seed)
    n = bank_np.shape[0]
    m = max(1, int(n * ratio))
    idx = rng.choice(n, size=m, replace=False)
    return bank_np[idx]


# =========================================================
# KNN distance (min squared L2)
# =========================================================
def knn_min_dist(query_np, bank_np, chunk=4096):
    # query_np: [P,D], bank_np: [M,D] -> returns [P] min squared L2
    if HAS_FAISS:
        index = faiss.IndexFlatL2(bank_np.shape[1])
        index.add(bank_np.astype(np.float32))
        D, _ = index.search(query_np.astype(np.float32), 1)
        return D.reshape(-1)  # already squared L2

    # torch fallback
    q = torch.from_numpy(query_np).to(DEVICE)
    b = torch.from_numpy(bank_np).to(DEVICE)
    mins = []
    with torch.no_grad():
        for i in range(0, q.shape[0], chunk):
            qq = q[i:i+chunk]
            d = torch.cdist(qq, b)             # [c,M]
            mins.append(d.min(dim=1).values)   # [c]
    return (torch.cat(mins).cpu().numpy().astype(np.float32) ** 2)


# =========================================================
# SCORING
# =========================================================
def score_folder(feat_model, Wproj, bank_np, folder: Path, out_csv: Path):
    ds = TileFolder(folder)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    rows = []
    feat_model.eval()
    with torch.no_grad():
        for x, paths in tqdm(dl, desc=f"score:{folder.name}"):
            x = x.to(DEVICE)
            feats = feat_model(x)
            emb = embed_from_feats(feats)               # [B,P,D0]
            emb = (emb @ Wproj).cpu().numpy().astype(np.float32)  # [B,P,Dp]

            for i in range(emb.shape[0]):
                q = emb[i]                              # [P,Dp]
                d2 = knn_min_dist(q, bank_np)           # [P]
                score = float(np.max(d2))               # image-level: max patch anomaly
                rows.append((paths[i], score))

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("image,score\n")
        for p, s in rows:
            f.write(f"{p},{s}\n")

    scores = np.array([s for _, s in rows], dtype=np.float32)
    return scores


# =========================================================
# METRICS (no sklearn)
# =========================================================
def auroc(scores, labels):
    # labels: 1=defect, 0=good
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    P = labels.sum()
    N = len(labels) - P
    if P == 0 or N == 0:
        return float("nan")

    order = np.argsort(-scores)  # desc
    y = labels[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)

    tpr = tps / P
    fpr = fps / N

    # add (0,0)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    return float(np.trapz(tpr, fpr))

def average_precision(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    P = labels.sum()
    if P == 0:
        return float("nan")

    order = np.argsort(-scores)
    y = labels[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)

    precision = tps / (tps + fps + 1e-12)
    recall = tps / P

    # AP = sum over recall increments of precision
    # ensure recall starts at 0
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    ap = np.sum((recall - recall_prev) * precision)
    return float(ap)

def recall_at_threshold(scores_defect, thr):
    scores_defect = np.asarray(scores_defect, dtype=np.float64)
    return float((scores_defect > thr).mean())


# =========================================================
# STEP 1: IN-DOMAIN NORMAL-ONLY
# =========================================================
def step1_train_val_test():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_dl = DataLoader(TileFolder(TRAIN_DIR), batch_size=BATCH, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)

    feat_model = FeatureHook(BACKBONE, HOOK_LAYERS).to(DEVICE)

    # build raw bank from train normals
    bank_chunks = []
    feat_model.eval()
    with torch.no_grad():
        for x, _ in tqdm(train_dl, desc="build_bank"):
            x = x.to(DEVICE)
            feats = feat_model(x)
            emb = embed_from_feats(feats)  # [B,P,D0]
            B, P, D0 = emb.shape
            for i in range(B):
                e = emb[i]  # [P,D0]
                if PATCHES_PER_IMAGE < P:
                    idx = torch.randperm(P, device=e.device)[:PATCHES_PER_IMAGE]
                    e = e[idx]
                bank_chunks.append(e.detach().cpu())
    raw_bank = torch.cat(bank_chunks, dim=0).numpy().astype(np.float32)  # [N,D0]

    # projection
    Wproj = make_random_projection(raw_bank.shape[1], PROJ_DIM, DEVICE, seed=SEED)
    raw_bank_t = torch.from_numpy(raw_bank).to(DEVICE)
    bank_proj = (raw_bank_t @ Wproj).cpu().numpy().astype(np.float32)    # [N,Dp]

    # coreset
    bank = coreset_subsample(bank_proj, CORESET_RATIO, seed=SEED)

    # save artifacts
    np.save(OUT_DIR / "bank.npy", bank)
    np.save(OUT_DIR / "Wproj.npy", Wproj.detach().cpu().numpy())

    meta = {
        "step": 1,
        "data_root": str(DATA_ROOT),
        "train": str(TRAIN_DIR),
        "valid": str(VAL_DIR),
        "test": str(TEST_DIR),
        "backbone": BACKBONE,
        "hook_layers": HOOK_LAYERS,
        "img_size": IMG_SIZE,
        "batch": BATCH,
        "patches_per_image": PATCHES_PER_IMAGE,
        "coreset_ratio": CORESET_RATIO,
        "proj_dim": PROJ_DIM,
        "val_percentile": VAL_PERCENTILE,
        "seed": SEED,
        "has_faiss": HAS_FAISS,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # score val -> threshold
    val_scores = score_folder(feat_model, Wproj, bank, VAL_DIR, OUT_DIR / "val_scores.csv")
    thr = float(np.quantile(val_scores, VAL_PERCENTILE))
    (OUT_DIR / "threshold.txt").write_text(f"{thr}\n", encoding="utf-8")

    # score test -> FPR
    test_scores = score_folder(feat_model, Wproj, bank, TEST_DIR, OUT_DIR / "test_scores.csv")
    fpr = float((test_scores > thr).mean())

    summary = (
        f"STEP1 in-domain normal-only\n"
        f"VAL threshold (q={VAL_PERCENTILE}) = {thr:.6f}\n"
        f"TEST FPR (normal-only) = {fpr:.4f}\n"
        f"VAL score: mean={val_scores.mean():.6f}, std={val_scores.std(ddof=1):.6f}\n"
        f"TEST score: mean={test_scores.mean():.6f}, std={test_scores.std(ddof=1):.6f}\n"
        f"Saved: {OUT_DIR}\n"
    )
    (OUT_DIR / "step1_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    if not HAS_FAISS:
        print("NOTE: faiss not found -> torch.cdist fallback (slower).")


# =========================================================
# STEP 2: EXTERNAL EVAL (good/defect tiles)
# =========================================================
def step2_external_eval():
    bank = np.load(OUT_DIR / "bank.npy").astype(np.float32)
    Wproj_np = np.load(OUT_DIR / "Wproj.npy").astype(np.float32)
    Wproj = torch.from_numpy(Wproj_np).to(DEVICE)

    thr = float((OUT_DIR / "threshold.txt").read_text(encoding="utf-8").strip())

    feat_model = FeatureHook(BACKBONE, HOOK_LAYERS).to(DEVICE)

    have_good = EXT_GOOD_DIR.exists() and len(list_images(EXT_GOOD_DIR)) > 0
    have_def  = EXT_DEFECT_DIR.exists() and len(list_images(EXT_DEFECT_DIR)) > 0

    if not have_good and not have_def:
        print("STEP2 external: no images found.")
        print("Expected:")
        print("  ", EXT_GOOD_DIR)
        print("  ", EXT_DEFECT_DIR)
        return

    scores_good = None
    scores_def  = None

    if have_good:
        scores_good = score_folder(feat_model, Wproj, bank, EXT_GOOD_DIR, OUT_DIR / "external_good_scores.csv")
    if have_def:
        scores_def = score_folder(feat_model, Wproj, bank, EXT_DEFECT_DIR, OUT_DIR / "external_defect_scores.csv")

    lines = []
    lines.append("STEP2 external evaluation")
    lines.append(f"threshold(from step1 val) = {thr:.6f}")

    if scores_good is not None:
        fpr_ext = float((scores_good > thr).mean())
        lines.append(f"external GOOD count={len(scores_good)}, FPR@thr={fpr_ext:.4f}")

    if scores_def is not None:
        rec_ext = recall_at_threshold(scores_def, thr)
        lines.append(f"external DEFECT count={len(scores_def)}, Recall@thr={rec_ext:.4f}")

    # if both exist -> AUROC/AUPRC
    if (scores_good is not None) and (scores_def is not None):
        scores = np.concatenate([scores_good, scores_def], axis=0)
        labels = np.concatenate([np.zeros_like(scores_good, dtype=np.int32),
                                 np.ones_like(scores_def, dtype=np.int32)], axis=0)
        auc = auroc(scores, labels)
        ap  = average_precision(scores, labels)
        lines.append(f"AUROC={auc:.4f}")
        lines.append(f"AUPRC(AP)={ap:.4f}")

    report = "\n".join(lines) + "\n"
    (OUT_DIR / "step2_external_report.txt").write_text(report, encoding="utf-8")
    print(report)


def main():
    # allow CLI override: python train_ta.py 1 or 2
    step = STEP
    if len(sys.argv) >= 2:
        step = int(sys.argv[1])

    if step == 1:
        step1_train_val_test()
    elif step == 2:
        step2_external_eval()
    else:
        raise ValueError("step must be 1 or 2")


if __name__ == "__main__":
    main()
