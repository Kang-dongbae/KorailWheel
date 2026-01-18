# tread_yolo.py
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from config import ROOT, DATA_YAML, TEST_IMAGES, TEST_LABELS, RUNS_DIR, ART_DIR

# -------------------------
# Stage1 hyperparams (ONLY here)
# -------------------------
STEP = 2  # 1=train+val, 2=test(eval+infer overlays)

PRETRAIN = "yolov8n-seg.pt"   # or yolov8s-seg.pt
IMG_SIZE = 1024
EPOCHS = 150
BATCH = 8
DEVICE = "0"                 # "cpu" or "0"
CONF = 0.25
MAX_DET = 10

RUN_NAME = "tread_seg_stage1"
ART_DIR.mkdir(parents=True, exist_ok=True)


def train_and_validate() -> Path:
    model = YOLO(PRETRAIN)
    r = model.train(
        data=str(DATA_YAML),
        task="segment",
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        device=DEVICE,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
    )
    run_dir = Path(r.save_dir)
    return run_dir / "weights" / "best.pt"


def _img_list(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])


def _read_polys_yolo_seg(label_path: Path):
    # YOLO-seg line: cls x y w h x1 y1 x2 y2 ... (normalized polygon)
    if not label_path.exists():
        return []
    txt = label_path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    polys = []
    for line in txt.splitlines():
        p = line.strip().split()
        coords = np.array(list(map(float, p[5:])), dtype=np.float32)
        polys.append(coords.reshape(-1, 2))
    return polys


def _polys_to_mask(polys_norm, w, h):
    m = np.zeros((h, w), dtype=np.uint8)
    if not polys_norm:
        return m
    pts_list = []
    for poly in polys_norm:
        pts = poly.copy()
        pts[:, 0] = np.clip(pts[:, 0] * w, 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1] * h, 0, h - 1)
        pts_list.append(pts.astype(np.int32))
    cv2.fillPoly(m, pts_list, 1)
    return m


def _dice_iou(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    inter = int((pred & gt).sum())
    union = int((pred | gt).sum())
    ps = int(pred.sum())
    gs = int(gt.sum())
    iou = inter / union if union else (1.0 if ps == 0 and gs == 0 else 0.0)
    dice = (2 * inter) / (ps + gs) if (ps + gs) else 1.0
    return dice, iou


def _pred_union_mask(model: YOLO, img_path: Path):
    res = model.predict(
        source=str(img_path),
        task="segment",
        imgsz=IMG_SIZE,
        conf=CONF,
        max_det=MAX_DET,
        device=DEVICE,
        retina_masks=True,   
        verbose=False,
    )[0]

    h, w = res.orig_shape
    out = np.zeros((h, w), dtype=np.uint8)
    if res.masks is None:
        return out

    masks = res.masks.data.cpu().numpy()  # (n, mh, mw)
    if masks.ndim == 3 and masks.shape[1] == h and masks.shape[2] == w:
        return (masks > 0).any(axis=0).astype(np.uint8)

    for poly in res.masks.xy:  # list of (N,2) float coords in original image space
        if poly is None or len(poly) == 0:
            continue
        pts = np.round(poly).astype(np.int32)
        cv2.fillPoly(out, [pts], 1)

    return out


def test_eval_and_overlay(weights: Path):
    model = YOLO(str(weights))

    overlays = ART_DIR / "test_overlays"
    overlays.mkdir(parents=True, exist_ok=True)

    rows = []
    for img_path in tqdm(_img_list(TEST_IMAGES), desc="test"):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # GT mask from label polygons
        label_path = TEST_LABELS / f"{img_path.stem}.txt"
        gt = _polys_to_mask(_read_polys_yolo_seg(label_path), w, h)

        # Pred mask
        pred = _pred_union_mask(model, img_path)

        dice, iou = _dice_iou(pred, gt)
        rows.append({"image": str(img_path), "dice": dice, "iou": iou})

        # overlay: GT(초록), Pred(빨강)
        vis = img.copy()
        gt_c, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pr_c, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, gt_c, -1, (0, 255, 0), 2)
        cv2.drawContours(vis, pr_c, -1, (0, 0, 255), 2)
        cv2.imwrite(str(overlays / f"{img_path.stem}.jpg"), vis)

    df = pd.DataFrame(rows)
    df.to_csv(ART_DIR / "test_metrics.csv", index=False, encoding="utf-8-sig")

    summary = (
        f"Dice mean={df.dice.mean():.4f}, std={df.dice.std(ddof=1):.4f}\n"
        f"IoU  mean={df.iou.mean():.4f}, std={df.iou.std(ddof=1):.4f}\n"
        f"metrics: {ART_DIR / 'test_metrics.csv'}\n"
        f"overlays: {overlays}\n"
    )
    (ART_DIR / "test_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


def main():
    if STEP == 1:
        best = train_and_validate()
        print("best weights:", best)
    elif STEP == 2:
        # 기본 weight 경로(학습 결과) 자동 추정
        default = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
        test_eval_and_overlay(default)
    else:
        raise ValueError("STEP must be 1 or 2")


if __name__ == "__main__":
    main()
