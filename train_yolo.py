# train_yolo.py
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

# =========================
# FIXED PATHS (USER)
# =========================
SYNTH_ROOT = Path(r"C:\Dev\KorailWheel\data\data_tiles_synth")  # train/valid/test with images/labels
RUNS_ROOT  = Path(r"C:\Dev\KorailWheel\runs\synth_yolo")

# =========================
# YOLO SETTINGS
# =========================
PRETRAIN = "yolov8n-seg.pt"
IMG_SIZE = 512
EPOCHS   = 80
BATCH    = 32
DEVICE   = "0"     # "cpu" or "0"
CONF     = 0.25
MAX_DET  = 10

DATA_YAML = RUNS_ROOT / "dataset_synth.yaml"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _img_list(folder: Path):
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])


def _write_dataset_yaml():
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    content = f"""# auto-generated
path: {SYNTH_ROOT.as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 1
names: ["defect"]
"""
    DATA_YAML.write_text(content, encoding="utf-8")


def _read_yolo_seg_polys(label_path: Path):
    # YOLO-seg line: cls x y w h x1 y1 x2 y2 ...
    if not label_path.exists():
        return []
    txt = label_path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    polys = []
    for line in txt.splitlines():
        p = line.strip().split()
        if len(p) <= 5:
            continue
        coords = np.array(list(map(float, p[5:])), dtype=np.float32)
        if coords.size >= 6:
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


def _dice_iou(pred01, gt01):
    pred = (pred01 > 0).astype(np.uint8)
    gt   = (gt01 > 0).astype(np.uint8)
    inter = int((pred & gt).sum())
    union = int((pred | gt).sum())
    ps = int(pred.sum())
    gs = int(gt.sum())
    iou  = inter / union if union else (1.0 if ps == 0 and gs == 0 else 0.0)
    dice = (2 * inter) / (ps + gs) if (ps + gs) else 1.0
    return float(dice), float(iou)


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

    # fast path when masks already full-res
    try:
        md = res.masks.data
        if md is not None:
            md = md.cpu().numpy()
            if md.ndim == 3 and md.shape[1] == h and md.shape[2] == w:
                return (md > 0).any(axis=0).astype(np.uint8)
    except Exception:
        pass

    for poly in res.masks.xy:
        if poly is None or len(poly) < 3:
            continue
        pts = np.round(poly).astype(np.int32)
        cv2.fillPoly(out, [pts], 1)
    return out


def _overlay(img_bgr, gt01=None, pred01=None):
    vis = img_bgr.copy()
    if gt01 is not None:
        gt_c, _ = cv2.findContours((gt01 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, gt_c, -1, (0, 255, 0), 2)  # GT green
    if pred01 is not None:
        pr_c, _ = cv2.findContours((pred01 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, pr_c, -1, (0, 0, 255), 2)  # Pred red
    return vis


def train() -> Path:
    _write_dataset_yaml()
    model = YOLO(PRETRAIN)
    r = model.train(
        data=str(DATA_YAML),
        task="segment",
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        device=DEVICE,
        project=str(RUNS_ROOT / "internal"),
        name="train",
        exist_ok=True,
    )
    run_dir = Path(r.save_dir)
    best = run_dir / "weights" / "best.pt"
    print("best weights:", best)
    return best


def eval_internal(best_weights: Path):
    model = YOLO(str(best_weights))

    test_img_dir = SYNTH_ROOT / "test" / "images"
    test_lab_dir = SYNTH_ROOT / "test" / "labels"

    out_dir = RUNS_ROOT / "internal" / "eval_test"
    ov_dir  = out_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    ov_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for img_path in tqdm(_img_list(test_img_dir), desc="internal:test"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_polys = _read_yolo_seg_polys(test_lab_dir / f"{img_path.stem}.txt")
        gt = _polys_to_mask(gt_polys, w, h)

        pred = _pred_union_mask(model, img_path)

        dice, iou = _dice_iou(pred, gt)
        rows.append({
            "image": str(img_path),
            "dice": dice,
            "iou": iou,
            "gt_pixels": int(gt.sum()),
            "pred_pixels": int(pred.sum()),
        })

        vis = _overlay(img, gt01=gt, pred01=pred)
        cv2.imwrite(str(ov_dir / f"{img_path.stem}.jpg"), vis)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "test_metrics.csv", index=False, encoding="utf-8-sig")

    summary = (
        f"INTERNAL synth-test\n"
        f"count={len(df)}\n"
        f"dice mean={df['dice'].mean():.4f} std={df['dice'].std(ddof=1):.4f}\n"
        f"iou  mean={df['iou'].mean():.4f} std={df['iou'].std(ddof=1):.4f}\n"
        f"saved: {out_dir}\n"
    )
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


def main():
    best = train()
    eval_internal(best)


if __name__ == "__main__":
    main()
