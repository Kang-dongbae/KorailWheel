# external_inference.py
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
EXT_ROOT  = Path(r"C:\Dev\KorailWheel\data\data_tiles\external_test")
RUNS_ROOT = Path(r"C:\Dev\KorailWheel\runs\synth_yolo")

# weights produced by train_yolo.py
WEIGHTS = RUNS_ROOT / "internal" / "train" / "weights" / "best.pt"

# =========================
# INFER SETTINGS
# =========================
IMG_SIZE = 512
DEVICE   = "0"
CONF     = 0.001
MAX_DET  = 10

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _img_list(folder: Path):
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])


def _pred_union_mask_and_score(model: YOLO, img_path: Path):
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
    mask = np.zeros((h, w), dtype=np.uint8)

    score = 0.0
    if res.boxes is not None and res.boxes.conf is not None and len(res.boxes.conf) > 0:
        score = float(res.boxes.conf.max().item())

    if res.masks is None:
        return mask, score

    try:
        md = res.masks.data
        if md is not None:
            md = md.cpu().numpy()
            if md.ndim == 3 and md.shape[1] == h and md.shape[2] == w:
                mask = (md > 0).any(axis=0).astype(np.uint8)
                return mask, score
    except Exception:
        pass

    for poly in res.masks.xy:
        if poly is None or len(poly) < 3:
            continue
        pts = np.round(poly).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)

    return mask, score


def _overlay_pred(img_bgr, pred01):
    vis = img_bgr.copy()
    pr_c, _ = cv2.findContours((pred01 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, pr_c, -1, (0, 0, 255), 2)
    return vis


def infer_split(model: YOLO, cls_name: str):
    in_dir = EXT_ROOT / cls_name / "images"
    if not in_dir.exists():
        print("[SKIP] not found:", in_dir)
        return

    out_dir = RUNS_ROOT / "external" / f"infer_{cls_name}"
    ov_dir  = out_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    ov_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for img_path in tqdm(_img_list(in_dir), desc=f"external:{cls_name}"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        pred, score = _pred_union_mask_and_score(model, img_path)
        pred_area_frac = float(pred.sum() / (h * w))

        rows.append({
            "image": str(img_path),
            "score": score,
            "pred_area_frac": pred_area_frac,
            "pred_pixels": int(pred.sum()),
        })

        vis = _overlay_pred(img, pred)
        cv2.imwrite(str(ov_dir / f"{img_path.stem}.jpg"), vis)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{cls_name}_pred.csv", index=False, encoding="utf-8-sig")
    print(f"[DONE] {cls_name}: {len(df)} -> {out_dir}")


def main():
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"best.pt not found: {WEIGHTS}\n먼저 python train_yolo.py 실행하세요.")

    model = YOLO(str(WEIGHTS))
    infer_split(model, "normal")
    infer_split(model, "defect")


if __name__ == "__main__":
    main()
